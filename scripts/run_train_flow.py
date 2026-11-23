"""Step 4: Latent 2D Rectified Flow Refiner over the Clinical State Matrix U in R^{R x K}.

Trains a lightweight AdaLN-Zero DiT velocity network strictly on Source ICD-9 train,
conditioned on both the deterministic prior U_init(x) from Step 2/3 AND the note chunks H(x).
Runs a strict falsification sweep across N in {0, 2, 4, 8} Euler steps with paired
bootstrap 95% CI & p-value against N=0 (deterministic Step 3).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ocsf.compiler import (
    AlignmentStats,
    apply_or_groups,
    compile_drg_operator,
    compile_target_operator,
)
from ocsf.data import (
    DEFAULT_DATA_ROOT,
    iter_source_icd9,
    iter_target_eval,
    load_codebook,
    load_source_subject_ids,
    target_paths,
)
from ocsf.metrics import (
    calibrate_source_threshold,
    evaluate_answer_groups,
    evaluate_multilabel,
    paired_bootstrap_ci,
)
from ocsf.models import (
    StateFieldFlowRefiner,
    StateFieldPredictor,
    build_batch_target_state,
    predict_state_matrices,
)
from ocsf.state_matrix import ChannelBank, compile_deconvolution_operator, l2_normalize


ALL_TARGETS = ("mimiciv-icd10", "mimiciv-ms-drg", "codiesp", "elcardiocc-2026", "chip-cdn")


@torch.inference_mode()
def refine_states(
    refiner: StateFieldFlowRefiner,
    u_init: np.ndarray,
    chunks_mmap: np.ndarray,
    lengths: np.ndarray,
    n_steps: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    if n_steps == 0:
        return u_init
    refiner.eval()
    n = u_init.shape[0]
    out = np.empty_like(u_init)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        u0 = torch.from_numpy(u_init[s:e]).to(device)
        ch = torch.from_numpy(np.array(chunks_mmap[s:e], dtype=np.float32)).to(device)
        ln = torch.from_numpy(lengths[s:e]).to(device)
        mask = torch.arange(ch.shape[1], device=device)[None, :] < ln[:, None]
        out[s:e] = refiner.sample_euler(u0, ch, mask, n_steps=n_steps).cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ocsf-dir", type=Path, required=True, help="Output directory from run_train_ocsf.py")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lam", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--steps-grid", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--targets", nargs="+", default=list(ALL_TARGETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-target", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    cache_root = args.cache_dir / args.backbone
    cb_dir = cache_root / "codebooks"
    sp_dir = cache_root / "splits"

    # 1. Load frozen ChannelBank + AlignmentStats + Step 2/3 StateFieldPredictor
    bank_npz = np.load(args.ocsf_dir / "channel_bank.npz")
    bank = ChannelBank(anchors=bank_npz["anchors"])
    align = AlignmentStats(source_mean=bank_npz["source_mean"])
    predictor = StateFieldPredictor(torch.from_numpy(bank.anchors)).to(dev)
    ckpt = torch.load(args.ocsf_dir / "best_state_predictor.pt", map_location=dev, weights_only=False)
    predictor.load_state_dict(ckpt["model_state"])
    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad_(False)

    # 2. Precompute U_init on Source ICD-9 train + val once
    src_cb = load_codebook(args.data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_map = {e.key: i for i, e in enumerate(src_cb)}
    z_src = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))
    src_op = compile_deconvolution_operator(z_src, bank, lam=args.lam, self_normalize=True)

    train_rows = list(iter_source_icd9("train", args.data_root, limit=args.limit_train))
    val_rows = list(iter_source_icd9("validation", args.data_root, limit=args.limit_val))
    train_labels = [[src_map[k] for k in r.labels if k in src_map] for r in train_rows]
    val_labels = [r.labels for r in val_rows]

    chunks_tr = np.load(sp_dir / "source_train_chunks.npy", mmap_mode="r")
    lengths_tr = np.load(sp_dir / "source_train_lengths.npy")
    chunks_va = np.load(sp_dir / "source_validation_chunks.npy", mmap_mode="r")
    lengths_va = np.load(sp_dir / "source_validation_lengths.npy")

    u_init_va = predict_state_matrices(predictor, chunks_va, lengths_va, device=dev)

    # 3. Train Rectified Flow velocity network strictly on Source ICD-9 train (on-the-fly U_init -> 0 GB train RAM)
    # Path matches StateFieldFlowRefiner.sample_euler: t=0 is u_start, t=1 is u_clean
    dim = z_src.shape[1]
    coord_noise_std = args.noise_std / math.sqrt(float(dim))
    refiner = StateFieldFlowRefiner(dim=dim).to(dev)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    z_a_gpu = torch.from_numpy(z_src).to(dev)
    code_ch_gpu = torch.from_numpy(src_op.assign).to(dev)

    best_val_r = -1.0
    best_flow_ckpt = args.ocsf_dir / "best_flow_refiner.pt"
    rng = np.random.default_rng(args.seed)
    n_train = len(train_rows)

    for ep in range(1, args.epochs + 1):
        t_ep = time.perf_counter()
        refiner.train()
        order = rng.permutation(n_train)
        tot_loss = 0.0
        n_batches = 0
        for s in range(0, n_train, args.batch_size):
            idx = order[s : s + args.batch_size]
            ch = torch.from_numpy(np.array(chunks_tr[idx], dtype=np.float32)).to(dev)
            ln = torch.from_numpy(lengths_tr[idx]).to(dev)
            mask = torch.arange(ch.shape[1], device=dev)[None, :] < ln[:, None]
            with torch.no_grad():
                u_init = predictor(ch, mask)

            u_clean, _ = build_batch_target_state(
                [train_labels[i] for i in idx], z_a_gpu, code_ch_gpu, bank.n_channels
            )
            u_start = u_init + coord_noise_std * torch.randn_like(u_init)
            t = torch.rand(len(idx), device=dev)
            t_view = t[:, None, None]
            u_t = (1.0 - t_view) * u_start + t_view * u_clean
            target_v = u_clean - u_start

            opt.zero_grad(set_to_none=True)
            pred_v = refiner(u_t, u_init, t, ch, mask)
            active = (u_clean.norm(dim=-1) > 1e-4).float()
            weights = 1.0 + 4.0 * active
            loss = (weights * (pred_v - target_v).pow(2).sum(dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.detach())
            n_batches += 1

        sched.step()
        u_va_ref = refine_states(refiner, u_init_va, chunks_va, lengths_va, n_steps=4, device=dev)
        va_scores = src_op.score_batch(u_va_ref)
        va_m, _ = evaluate_multilabel(va_scores, val_labels, src_map, threshold=0.5)
        va_r = va_m["r@|y|"]
        marker = ""
        if va_r > best_val_r:
            best_val_r = va_r
            torch.save({"model_state": refiner.state_dict(), "epoch": ep, "val_r_at_y": va_r}, best_flow_ckpt)
            marker = " *best*"
        print(
            f"[flow] ep={ep:02d}/{args.epochs} loss={tot_loss/n_batches:.4f} "
            f"src_val(N=4) R@|y|={va_r:.4f} ({time.perf_counter() - t_ep:.1f}s){marker}"
        )

    # 4. Falsification Sweep: N = 0 vs N in {2, 4, 8} across Source val + all frozen targets
    refiner.load_state_dict(torch.load(best_flow_ckpt, map_location=dev, weights_only=False)["model_state"])
    source_subjects = load_source_subject_ids(args.data_root)
    drg_protos = dict(np.load(cb_dir / "drg_prototypes.npz")) if (cb_dir / "drg_prototypes.npz").exists() else {}

    target_cache: dict[str, dict] = {}
    for tgt in args.targets:
        tpaths = target_paths(tgt, args.data_root)
        cb = load_codebook(tpaths["codebook"])
        code_map = {e.key: i for i, e in enumerate(cb)}
        rows = list(iter_target_eval(tgt, args.data_root, source_subjects=source_subjects, limit=args.limit_target))
        z_t = np.load(cb_dir / f"{tgt}.npy")
        anchors_override = align.align_anchors_for_target(bank.anchors, z_t, centroid_shift_min_codes=1000)
        if tgt == "mimiciv-ms-drg" and drg_protos:
            op = compile_drg_operator(
                cb, z_t, drg_protos, bank, cms_dir=tpaths["cms_dir"], alignment=align, lam=args.lam
            )
        else:
            op = compile_target_operator(z_t, bank, alignment=align, lam=args.lam, self_normalize=True)
        chunks_t = np.load(sp_dir / f"target_{tgt}_chunks.npy", mmap_mode="r")
        lengths_t = np.load(sp_dir / f"target_{tgt}_lengths.npy")
        u_init_t = predict_state_matrices(
            predictor, chunks_t, lengths_t, device=dev, anchors_override=anchors_override
        )
        target_cache[tgt] = {
            "tpaths": tpaths,
            "code_map": code_map,
            "rows": rows,
            "op": op,
            "u_init": u_init_t,
            "chunks": chunks_t,
            "lengths": lengths_t,
        }

    sweep_results: dict[str, dict[str, dict[str, float]]] = {}
    per_note_by_step: dict[int, dict[str, np.ndarray]] = {}
    src_gold_idx = [[src_map[k] for k in r.labels if k in src_map] for r in val_rows]

    for n_steps in args.steps_grid:
        tag = f"N{n_steps}"
        sweep_results[tag] = {}
        per_note_by_step[n_steps] = {}

        u_va_s = refine_states(refiner, u_init_va, chunks_va, lengths_va, n_steps=n_steps, device=dev)
        src_scores = src_op.score_batch(u_va_s)
        tau_a, _ = calibrate_source_threshold(src_scores, src_gold_idx)
        m_src, pn_src = evaluate_multilabel(src_scores, val_labels, src_map, threshold=tau_a)
        m_src["tau_A"] = tau_a
        sweep_results[tag]["source_icd9_val"] = m_src
        per_note_by_step[n_steps]["source_icd9_val"] = pn_src

        for tgt in args.targets:
            tc = target_cache[tgt]
            t0 = time.perf_counter()
            u_ref = refine_states(
                refiner,
                tc["u_init"],
                tc["chunks"],
                tc["lengths"],
                n_steps=n_steps,
                device=dev,
            )
            scores = tc["op"].score_batch(u_ref)
            infer_ms = (time.perf_counter() - t0) * 1000.0 / max(len(tc["rows"]), 1)

            code_map = tc["code_map"]
            if tgt == "codiesp" and tc["tpaths"]["groups"].exists():
                scores, code_map = apply_or_groups(scores, code_map, tc["tpaths"]["groups"])

            if tgt in {"mimiciv-ms-drg", "elcardiocc-2026"}:
                m, pn = evaluate_answer_groups(scores, [r.label_groups for r in tc["rows"]], code_map)
            else:
                m, pn = evaluate_multilabel(scores, [r.labels for r in tc["rows"]], code_map, threshold=tau_a)
            m["refine_plus_score_ms_per_note"] = infer_ms
            sweep_results[tag][tgt] = m
            per_note_by_step[n_steps][tgt] = pn

            if tgt == "mimiciv-icd10" and tc["tpaths"]["frequent_codebook"].exists():
                icd10_freq_keys = {e.key for e in load_codebook(tc["tpaths"]["frequent_codebook"])}
                rare_keys = set(code_map.keys()) - icd10_freq_keys
                mf, pnf = evaluate_multilabel(
                    scores, [r.labels for r in tc["rows"]], code_map, threshold=tau_a, subset_keys=icd10_freq_keys
                )
                mr, pnr = evaluate_multilabel(
                    scores, [r.labels for r in tc["rows"]], code_map, threshold=tau_a, subset_keys=rare_keys
                )
                sweep_results[tag]["mimiciv-icd10-frequent"] = mf
                sweep_results[tag]["mimiciv-icd10-rare"] = mr
                per_note_by_step[n_steps]["mimiciv-icd10-frequent"] = pnf
                per_note_by_step[n_steps]["mimiciv-icd10-rare"] = pnr

    # Paired bootstrap falsification: (N>0 - N=0)
    falsification: dict[str, dict[str, dict[str, float]]] = {}
    if 0 in per_note_by_step:
        for n_steps in args.steps_grid:
            if n_steps == 0:
                continue
            falsification[f"N{n_steps}_minus_N0"] = {
                bench: asdict(paired_bootstrap_ci(per_note_by_step[0][bench], per_note_by_step[n_steps][bench]))
                for bench in per_note_by_step[0]
            }

    out_path = args.ocsf_dir / "flow_falsification.json"
    out_path.write_text(
        json.dumps({"sweep": sweep_results, "falsification_vs_N0": falsification}, indent=2), encoding="utf-8"
    )
    print(f"[flow] Saved falsification report -> {out_path}")
    for comp, benches in falsification.items():
        print(f"\n=== Falsification {comp} ===")
        for bench, ci in benches.items():
            sig = "PASS" if ci["ci_low"] > 0 else ("HURT" if ci["ci_high"] < 0 else "NULL")
            print(f"  {bench:<26} delta={ci['mean_delta']:+.4f} [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] p={ci['p_value']:.4f} -> {sig}")


if __name__ == "__main__":
    main()
