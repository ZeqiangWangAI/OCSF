"""Steps 1–3 OCSF Trainer & Zero-Training Multi-Ontology Evaluation Runner.

Trains StateFieldPredictor STRICTLY on Source ICD-9 train (checkpoint & threshold tau_A
selected STRICTLY on Source ICD-9 val; zero target exposure during training/selection).

Then compiles offline target operators and evaluates 3 ablations on the EXACT SAME checkpoint:
  1. B3_2D_dot:           2D State Matrix with naive dot-product readout (W_{B,r} = Z_{B,r})
  2. Step2_OCSF_deconv:   2D State Matrix + Self-Normalized Tikhonov Deconvolution (unaligned)
  3. Step3_OCSF_compiled: + Symmetric public-codebook centroid alignment + MS-DRG compositional
                          severity/surgical operator + CodiEsp OR-group max-pooling
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path
import numpy as np
import torch

from ocsf.compiler import (
    AlignmentStats,
    apply_or_groups,
    compile_drg_operator,
    compile_target_operator,
    debias_principal_state,
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
    SourceDeconvHead,
    StateFieldPredictor,
    build_batch_target_state,
    compute_ocsf_loss,
    predict_state_matrices,
)
from ocsf.state_matrix import (
    ChannelBank,
    DeconvolutionOperator,
    compile_deconvolution_operator,
    l2_normalize,
)


ALL_TARGETS = ("mimiciv-icd10", "mimiciv-ms-drg", "codiesp", "elcardiocc-2026", "chip-cdn")


def _build_naive_dot_operator(z_codes: np.ndarray, bank: ChannelBank) -> DeconvolutionOperator:
    z = l2_normalize(z_codes.astype(np.float32, copy=False))
    n_codes, dim = z.shape
    assign = bank.assign_hard(z)
    chan_indices: list[np.ndarray] = []
    chan_weights: list[np.ndarray] = []
    for r in range(bank.n_channels):
        idx = np.where(assign == r)[0].astype(np.int32)
        chan_indices.append(idx)
        chan_weights.append(z[idx] if idx.size > 0 else np.zeros((0, dim), dtype=np.float32))
    return DeconvolutionOperator(
        n_codes=n_codes,
        n_channels=bank.n_channels,
        dim=dim,
        assign=assign,
        chan_indices=tuple(chan_indices),
        chan_weights=tuple(chan_weights),
        diag_gains=np.ones(n_codes, dtype=np.float32),
    )


def evaluate_variant(
    variant_name: str,
    model: StateFieldPredictor,
    bank: ChannelBank,
    align: AlignmentStats,
    lam: float,
    cache_root: Path,
    data_root: Path,
    out_per_note_dir: Path,
    device: torch.device,
    use_deconv: bool,
    use_compiler: bool,
    targets: tuple[str, ...] = ALL_TARGETS,
    limit_val: int | None = None,
    limit_target: int | None = None,
) -> dict[str, dict[str, float]]:
    """Run zero-training evaluation across Source ICD-9 val + all frozen targets."""
    cb_dir = cache_root / "codebooks"
    sp_dir = cache_root / "splits"
    out_per_note_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}

    # 1. Source ICD-9 validation -> calibrate tau_A strictly here
    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_map = {e.key: i for i, e in enumerate(src_cb)}
    src_rows = list(iter_source_icd9("validation", data_root, limit=limit_val))
    z_src = np.load(cb_dir / "icd9-full.npy")
    src_op = (
        compile_deconvolution_operator(z_src, bank, lam=lam, self_normalize=True)
        if use_deconv
        else _build_naive_dot_operator(z_src, bank)
    )
    src_u = predict_state_matrices(
        model,
        np.load(sp_dir / "source_validation_chunks.npy", mmap_mode="r"),
        np.load(sp_dir / "source_validation_lengths.npy"),
        device=device,
    )
    src_scores = src_op.score_batch(src_u)
    src_gold_idx = [[src_map[k] for k in r.labels if k in src_map] for r in src_rows]
    tau_a, _ = calibrate_source_threshold(src_scores, src_gold_idx)
    src_metrics, src_pn = evaluate_multilabel(src_scores, [r.labels for r in src_rows], src_map, threshold=tau_a)
    src_metrics["tau_A"] = tau_a
    results["source_icd9_val"] = src_metrics
    np.save(out_per_note_dir / "source_icd9_val.npy", src_pn)

    icd9_freq_path = data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-frequent.json"
    if icd9_freq_path.exists():
        icd9_freq_keys = {e.key for e in load_codebook(icd9_freq_path)}
        m_f9, pn_f9 = evaluate_multilabel(
            src_scores, [r.labels for r in src_rows], src_map, threshold=tau_a, subset_keys=icd9_freq_keys
        )
        results["source_icd9_val_frequent"] = m_f9
        np.save(out_per_note_dir / "source_icd9_val_frequent.npy", pn_f9)

    source_mean_state = src_u.mean(axis=0).astype(np.float32)

    # 2. Frozen non-test targets
    source_subjects = load_source_subject_ids(data_root)
    drg_protos = dict(np.load(cb_dir / "drg_prototypes.npz")) if (cb_dir / "drg_prototypes.npz").exists() else {}

    for tgt in targets:
        tpaths = target_paths(tgt, data_root)
        cb = load_codebook(tpaths["codebook"])
        code_map = {e.key: i for i, e in enumerate(cb)}
        rows = list(iter_target_eval(tgt, data_root, source_subjects=source_subjects, limit=limit_target))
        z_t = np.load(cb_dir / f"{tgt}.npy")

        t_compile = time.perf_counter()
        anchors_override = None
        if not use_deconv:
            op = _build_naive_dot_operator(z_t, bank)
            drg_op = None
        elif not use_compiler:
            op = compile_deconvolution_operator(z_t, bank, lam=lam, self_normalize=True)
            drg_op = None
        else:
            anchors_override = align.align_anchors_for_target(bank.anchors, z_t, centroid_shift_min_codes=1000)
            if tgt == "mimiciv-ms-drg" and drg_protos:
                drg_op = compile_drg_operator(
                    cb,
                    z_t,
                    drg_protos,
                    bank,
                    cms_dir=tpaths["cms_dir"],
                    alignment=align,
                    lam=lam,
                )
                op = None
            else:
                op = compile_target_operator(z_t, bank, alignment=align, lam=lam, self_normalize=True)
                drg_op = None
        compile_sec = time.perf_counter() - t_compile

        chunks = np.load(sp_dir / f"target_{tgt}_chunks.npy", mmap_mode="r")
        lengths = np.load(sp_dir / f"target_{tgt}_lengths.npy")
        t_infer = time.perf_counter()
        u_hat = predict_state_matrices(model, chunks, lengths, device=device, anchors_override=anchors_override)
        if use_compiler and tgt in {"mimiciv-ms-drg", "elcardiocc-2026"}:
            u_hat = debias_principal_state(u_hat, source_mean_state, gamma=0.5)
        scores = drg_op.score_batch(u_hat) if drg_op is not None else op.score_batch(u_hat)
        infer_ms = (time.perf_counter() - t_infer) * 1000.0 / max(len(rows), 1)

        if tgt == "codiesp" and use_compiler and tpaths["groups"].exists():
            scores, code_map = apply_or_groups(scores, code_map, tpaths["groups"])

        if tgt in {"mimiciv-ms-drg", "elcardiocc-2026"}:
            metrics, pn = evaluate_answer_groups(scores, [r.label_groups for r in rows], code_map)
        else:
            metrics, pn = evaluate_multilabel(scores, [r.labels for r in rows], code_map, threshold=tau_a)

        metrics["compile_sec"] = compile_sec
        metrics["infer_ms_per_note"] = infer_ms
        results[tgt] = metrics
        np.save(out_per_note_dir / f"{tgt}.npy", pn)

        if tgt == "mimiciv-icd10" and tpaths["frequent_codebook"].exists():
            icd10_freq_keys = {e.key for e in load_codebook(tpaths["frequent_codebook"])}
            rare_keys = set(code_map.keys()) - icd10_freq_keys
            m_freq, pn_freq = evaluate_multilabel(
                scores, [r.labels for r in rows], code_map, threshold=tau_a, subset_keys=icd10_freq_keys
            )
            m_rare, pn_rare = evaluate_multilabel(
                scores, [r.labels for r in rows], code_map, threshold=tau_a, subset_keys=rare_keys
            )
            results["mimiciv-icd10-frequent"] = m_freq
            results["mimiciv-icd10-rare"] = m_rare
            np.save(out_per_note_dir / "mimiciv-icd10-frequent.npy", pn_freq)
            np.save(out_per_note_dir / "mimiciv-icd10-rare.npy", pn_rare)

    print(f"[eval] Finished {variant_name}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-channels", type=int, default=256)
    parser.add_argument("--n-channel-layers", type=int, default=3)
    parser.add_argument("--lam", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-code", type=float, default=1.0)
    parser.add_argument("--alpha-state", type=float, default=0.25)
    parser.add_argument("--beta-anchor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="+", default=list(ALL_TARGETS))
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
    out_root = args.out_dir / args.backbone / f"ocsf_R{args.n_channels}_lam{args.lam}"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Fit frozen ChannelBank + AlignmentStats + SourceDeconvHead STRICTLY on Source ICD-9 codebook
    src_cb = load_codebook(args.data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_map = {e.key: i for i, e in enumerate(src_cb)}
    z_src = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))
    bank = ChannelBank.fit(z_src, n_channels=args.n_channels, seed=args.seed)
    align = AlignmentStats.fit(z_src, use_zca=False)
    src_op = compile_deconvolution_operator(z_src, bank, lam=args.lam, self_normalize=True)
    np.savez(out_root / "channel_bank.npz", anchors=bank.anchors, source_mean=align.source_mean)
    print(f"[ocsf] Fit ChannelBank R={args.n_channels}, dim={z_src.shape[1]}, src_op mean self-gain={src_op.diag_gains.mean():.3f}")

    # 2. Load Source ICD-9 train + val
    train_rows = list(iter_source_icd9("train", args.data_root, limit=args.limit_train))
    val_rows = list(iter_source_icd9("validation", args.data_root, limit=args.limit_val))
    train_labels = [[src_map[k] for k in r.labels if k in src_map] for r in train_rows]
    val_labels = [r.labels for r in val_rows]

    chunks_tr = np.load(sp_dir / "source_train_chunks.npy", mmap_mode="r")
    lengths_tr = np.load(sp_dir / "source_train_lengths.npy")
    chunks_va = np.load(sp_dir / "source_validation_chunks.npy", mmap_mode="r")
    lengths_va = np.load(sp_dir / "source_validation_lengths.npy")

    model = StateFieldPredictor(
        torch.from_numpy(bank.anchors),
        n_channel_layers=args.n_channel_layers,
    ).to(dev)
    src_head = SourceDeconvHead(
        tuple(torch.from_numpy(idx) for idx in src_op.chan_indices),
        tuple(torch.from_numpy(w) for w in src_op.chan_weights),
        n_codes=src_op.n_codes,
    ).to(dev)
    z_a_gpu = torch.from_numpy(z_src).to(dev)
    code_ch_gpu = torch.from_numpy(src_op.assign).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    use_bf16 = dev.type == "cuda"

    best_val_r = -1.0
    best_ckpt = out_root / "best_state_predictor.pt"
    rng = np.random.default_rng(args.seed)
    n_train = len(train_rows)

    for ep in range(1, args.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        order = rng.permutation(n_train)
        tot_loss = tot_state = tot_code = 0.0
        n_batches = 0

        for s in range(0, n_train, args.batch_size):
            idx = order[s : s + args.batch_size]
            ch = torch.from_numpy(np.array(chunks_tr[idx], dtype=np.float32)).to(dev)
            ln = torch.from_numpy(lengths_tr[idx]).to(dev)
            mask = torch.arange(ch.shape[1], device=dev)[None, :] < ln[:, None]
            u_tgt, y_multi = build_batch_target_state([train_labels[i] for i in idx], z_a_gpu, code_ch_gpu, args.n_channels)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                u_hat = model(ch, mask)
                code_scores = src_head(u_hat.float())
                loss, parts = compute_ocsf_loss(
                    u_hat.float(),
                    u_tgt,
                    code_scores,
                    y_multi,
                    alpha_code=args.alpha_code,
                    alpha_state=args.alpha_state,
                    delta=model._last_delta.float(),
                    beta_anchor=args.beta_anchor,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += parts["loss_total"]
            tot_state += parts["loss_state"]
            tot_code += parts["loss_code"]
            n_batches += 1

        sched.step()
        u_va = predict_state_matrices(model, chunks_va, lengths_va, device=dev, batch_size=args.batch_size)
        va_scores = src_op.score_batch(u_va)
        va_metrics, _ = evaluate_multilabel(va_scores, val_labels, src_map, threshold=0.5)
        va_r = va_metrics["r@|y|"]
        va_f1 = va_metrics.get("micro_f1@tau_A", 0.0)
        marker = ""
        if va_r > best_val_r:
            best_val_r = va_r
            torch.save(
                {"model_state": model.state_dict(), "epoch": ep, "val_r_at_y": va_r},
                best_ckpt,
            )
            marker = " *best*"
        print(
            f"[ocsf] ep={ep:02d}/{args.epochs} loss={tot_loss/n_batches:.4f} "
            f"(state={tot_state/n_batches:.4f}, code={tot_code/n_batches:.4f}) "
            f"src_val R@|y|={va_r:.4f} F1@0.5={va_f1:.4f} R@15={va_metrics['r@15']:.4f} "
            f"({time.perf_counter() - t_ep:.1f}s){marker}"
        )

    # 3. Reload best source-val checkpoint and evaluate all 3 zero-training variants
    ckpt = torch.load(best_ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    summary: dict[str, dict[str, dict[str, float]]] = {}
    variants = [
        ("B3_2D_dot", False, False),
        ("Step2_OCSF_deconv", True, False),
        ("Step3_OCSF_compiled", True, True),
    ]
    for v_name, use_deconv, use_compiler in variants:
        summary[v_name] = evaluate_variant(
            v_name,
            model,
            bank,
            align,
            lam=args.lam,
            cache_root=cache_root,
            data_root=args.data_root,
            out_per_note_dir=out_root / "per_note" / v_name,
            device=dev,
            use_deconv=use_deconv,
            use_compiler=use_compiler,
            targets=tuple(args.targets),
            limit_val=args.limit_val,
            limit_target=args.limit_target,
        )

    # 4. Paired bootstrap significance: (B - A) where first arg is baseline, second arg is candidate
    bootstrap_report: dict[str, dict[str, dict[str, float]]] = {}
    comparisons = [
        ("Step2_minus_B3_dot", out_root / "per_note/B3_2D_dot", out_root / "per_note/Step2_OCSF_deconv"),
        ("Step3_minus_Step2", out_root / "per_note/Step2_OCSF_deconv", out_root / "per_note/Step3_OCSF_compiled"),
    ]
    b1_dir = args.out_dir / args.backbone / "per_note/B1_CAST_1D"
    if b1_dir.exists():
        comparisons.append(("Step3_minus_B1_CAST", b1_dir, out_root / "per_note/Step3_OCSF_compiled"))
    b1s_dir = args.out_dir / args.backbone / "per_note/B1_1D_State"
    if b1s_dir.exists():
        comparisons.append(("Step3_minus_B1_1D_State", b1s_dir, out_root / "per_note/Step3_OCSF_compiled"))

    for comp_name, base_dir, cand_dir in comparisons:
        bootstrap_report[comp_name] = {}
        for p_cand in sorted(cand_dir.glob("*.npy")):
            p_base = base_dir / p_cand.name
            if not p_base.exists():
                continue
            bootstrap_report[comp_name][p_cand.stem] = asdict(paired_bootstrap_ci(np.load(p_base), np.load(p_cand)))

    (out_root / "ocsf_summary.json").write_text(
        json.dumps({"results": summary, "bootstrap": bootstrap_report}, indent=2), encoding="utf-8"
    )
    print(f"[ocsf] Saved summary -> {out_root / 'ocsf_summary.json'}")
    for v_name, res in summary.items():
        print(f"\n=== {args.backbone} | {v_name} ===")
        for bench, m in res.items():
            key_metric = (
                f"acc@1={m['acc@1']:.4f} acc@5={m['acc@5']:.4f}"
                if "acc@1" in m
                else f"R@|y|={m['r@|y|']:.4f} R@15={m['r@15']:.4f} F1@tauA={m.get('micro_f1@tau_A', 0.0):.4f}"
            )
            print(f"  {bench:<26} {key_metric}  MRR={m['mrr']:.4f}")


if __name__ == "__main__":
    main()
