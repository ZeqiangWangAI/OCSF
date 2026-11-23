"""Zero-Retrain Post-Hoc Diagnostic & Compiler Ablation Evaluator for OCSF-E2E Checkpoints.

Loads an already-trained `best_e2e_trainable.pt` + `channel_bank.npz` (zero retraining) and runs
fast single-pass GPU streaming ablations across Source ICD-9 val and all 5 frozen targets:
  - Self-normalized vs unnormalized Tikhonov deconvolution vs naive 2D dot product
  - Regularization lambda sweep (--lams 0.05 0.15 0.30)
  - Inference-time LoRA scaling (--lora-scale 1.0 or 0.5) to test cross-lingual drift without retraining
  - Principal-state background debiasing (--debias-gamma) and hierarchical CMS DRG / CodiEsp OR compilation
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ocsf.compiler import (
    AlignmentStats,
    DRGCompositionalOperator,
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
from ocsf.e2e import (
    LoRALinear,
    StreamingGPUDeconvScorer,
    TokenChannelStateDecoder,
    TokenLevelOCSFModel,
    inject_upper_layer_lora,
    make_length_bucketed_batches,
)
from ocsf.encode import BACKBONE_SPECS
from ocsf.metrics import (
    calibrate_source_threshold,
    evaluate_answer_groups,
    evaluate_multilabel,
    paired_bootstrap_ci,
)
from ocsf.state_matrix import ChannelBank, compile_deconvolution_operator, l2_normalize

from run_train_e2e_ocsf import (
    ALL_TARGETS,
    _build_naive_dot_operator,
    _collate_ids,
    _combine_drg_scores,
    _tokenize_texts,
)


def _scale_lora_modules(model: torch.nn.Module, factor: float) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.scale *= float(factor)


@torch.inference_mode()
def _eval_split_multi_ops(
    model: TokenLevelOCSFModel,
    token_list: list[np.ndarray],
    pad_id: int,
    device: torch.device,
    max_eval_tokens: int,
    ops: dict[str, object],
    anchors_by_variant: dict[str, np.ndarray | None] | None = None,
    mean_state_by_variant: dict[str, tuple[np.ndarray | None, float]] | None = None,
    return_mean_state: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    model.eval()
    n = len(token_list)
    use_bf16 = device.type == "cuda"
    base_anc = model.decoder.anchors.detach().cpu().numpy()

    scorers: dict[str, tuple[StreamingGPUDeconvScorer, StreamingGPUDeconvScorer | None, object]] = {}
    outputs: dict[str, np.ndarray] = {}
    proto_outs: dict[str, np.ndarray] = {}
    for name, op in ops.items():
        if isinstance(op, DRGCompositionalOperator):
            scorers[name] = (
                StreamingGPUDeconvScorer(op.direct_op, device),
                StreamingGPUDeconvScorer(op.prototype_op, device),
                op,
            )
            outputs[name] = np.empty((n, op.direct_op.n_codes), dtype=np.float32)
            proto_outs[name] = np.empty((n, len(op.prototype_names)), dtype=np.float32)
        else:
            scorers[name] = (StreamingGPUDeconvScorer(op, device), None, op)
            outputs[name] = np.empty((n, op.n_codes), dtype=np.float32)

    alt_anc_t: dict[str, torch.Tensor | None] = {}
    for name in ops:
        ov = None if anchors_by_variant is None else anchors_by_variant.get(name)
        if ov is None or np.allclose(ov, base_anc, atol=1e-5):
            alt_anc_t[name] = None
        else:
            alt_anc_t[name] = torch.from_numpy(ov).to(device)

    mean_t: dict[str, tuple[torch.Tensor | None, float]] = {}
    for name in ops:
        if mean_state_by_variant and name in mean_state_by_variant and mean_state_by_variant[name][0] is not None:
            ms, g = mean_state_by_variant[name]
            mean_t[name] = (torch.from_numpy(ms).to(device), float(g))
        else:
            mean_t[name] = (None, 0.0)

    state_sum = (
        torch.zeros((model.decoder.n_channels, model.decoder.dim), device=device, dtype=torch.float64)
        if return_mean_state
        else None
    )
    batches = make_length_bucketed_batches([len(s) for s in token_list], max_tokens=max_eval_tokens, max_batch_size=48)
    for batch in batches:
        ids, mask = _collate_ids(token_list, batch, pad_id, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            h = model.encode_tokens(ids, mask)
            u_base = model.decoder(h, mask)
            if state_sum is not None:
                state_sum += u_base.double().sum(dim=0)
            u_alt_cache: dict[int, torch.Tensor] = {}

        for name, (direct_sc, proto_sc, _) in scorers.items():
            ov_t = alt_anc_t[name]
            if ov_t is None:
                u = u_base
            else:
                key = id(ov_t)
                if key not in u_alt_cache:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        u_alt_cache[key] = model.decoder(h, mask, anchors_override=ov_t)
                u = u_alt_cache[key]
            ms_t, g = mean_t[name]
            if ms_t is not None and g > 0.0:
                u = u - g * ms_t.unsqueeze(0)

            sc = direct_sc.score_tensor(u)
            psc = proto_sc.score_tensor(u) if proto_sc is not None else None
            for k, idx in enumerate(batch):
                outputs[name][idx] = sc[k]
                if psc is not None:
                    proto_outs[name][idx] = psc[k]

    for name, (_, _, op) in scorers.items():
        if isinstance(op, DRGCompositionalOperator):
            outputs[name] = _combine_drg_scores(outputs[name], proto_outs[name], op)

    mean_state = (state_sum / max(n, 1)).cpu().numpy().astype(np.float32) if state_sum is not None else None
    return outputs, mean_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=sorted(BACKBONE_SPECS.keys()))
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing best_e2e_trainable.pt and channel_bank.npz")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--freeze-ratio", type=float, default=0.5)
    parser.add_argument("--n-channel-layers", type=int, default=2)
    parser.add_argument("--lora-scale", type=float, default=1.0, help="Post-hoc LoRA multiplier (1.0 = nominal)")
    parser.add_argument("--lam", type=float, default=0.15)
    parser.add_argument("--debias-gamma", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--max-eval-tokens", type=int, default=65536)
    parser.add_argument("--targets", nargs="+", default=list(ALL_TARGETS))
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-target", type=int, default=None)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = BACKBONE_SPECS[args.backbone]
    cb_dir = args.cache_dir / args.backbone / "codebooks"

    saved_bank = np.load(args.run_dir / "channel_bank.npz")
    bank = ChannelBank(anchors=saved_bank["anchors"])
    align = AlignmentStats(source_mean=saved_bank["source_mean"], zca_matrix=None)

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    backbone = AutoModel.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    inject_upper_layer_lora(backbone, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0, freeze_ratio=args.freeze_ratio)
    decoder = TokenChannelStateDecoder(torch.from_numpy(bank.anchors), n_heads=8, n_channel_layers=args.n_channel_layers, dropout=0.0)
    model = TokenLevelOCSFModel(backbone, decoder, apply_tanh=(spec.family == "pplx_diffusion")).to(dev)
    ckpt = torch.load(args.run_dir / "best_e2e_trainable.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["trainable_state"], strict=False)
    if args.lora_scale != 1.0:
        _scale_lora_modules(model, args.lora_scale)
    model.eval()
    print(f"[eval-ckpt] Loaded ep={ckpt.get('epoch')} lora_scale={args.lora_scale} lam={args.lam}")

    # Evaluate Source ICD-9 val
    src_cb = load_codebook(args.data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_map = {e.key: i for i, e in enumerate(src_cb)}
    icd9_freq_path = args.data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-frequent.json"
    icd9_freq_keys = {e.key for e in load_codebook(icd9_freq_path)} if icd9_freq_path.exists() else None
    val_rows = list(iter_source_icd9("validation", args.data_root, limit=args.limit_val))
    val_labels = [r.labels for r in val_rows]
    val_gold_idx = [[src_map[k] for k in r.labels if k in src_map] for r in val_rows]
    val_tokens = _tokenize_texts(tokenizer, [r.text for r in val_rows], args.max_tokens)
    z_src = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))

    src_ops = {
        "B3_2D_dot": _build_naive_dot_operator(z_src, bank),
        "Step2_unnorm_deconv": compile_deconvolution_operator(z_src, bank, lam=args.lam, self_normalize=False),
        "Step2_selfnorm_deconv": compile_deconvolution_operator(z_src, bank, lam=args.lam, self_normalize=True),
    }
    src_scores, source_mean_state = _eval_split_multi_ops(
        model, val_tokens, pad_id, dev, args.max_eval_tokens, src_ops, return_mean_state=True
    )
    src_scores["Step3_OCSF_compiled"] = src_scores["Step2_selfnorm_deconv"]

    variants = ["B3_2D_dot", "Step2_unnorm_deconv", "Step2_selfnorm_deconv", "Step3_OCSF_compiled"]
    summary: dict[str, dict[str, dict[str, float]]] = {v: {} for v in variants}
    per_note: dict[str, dict[str, np.ndarray]] = {v: {} for v in variants}
    tau_by_var: dict[str, float] = {}

    for v in variants:
        tau_a, _ = calibrate_source_threshold(src_scores[v], val_gold_idx)
        tau_by_var[v] = tau_a
        m, pn = evaluate_multilabel(src_scores[v], val_labels, src_map, threshold=tau_a)
        m["tau_A"] = tau_a
        summary[v]["source_icd9_val"] = m
        per_note[v]["source_icd9_val"] = pn
        if icd9_freq_keys is not None:
            mf, pnf = evaluate_multilabel(src_scores[v], val_labels, src_map, threshold=tau_a, subset_keys=icd9_freq_keys)
            summary[v]["source_icd9_val_frequent"] = mf
            per_note[v]["source_icd9_val_frequent"] = pnf

    # Evaluate all 5 frozen targets
    source_subjects = load_source_subject_ids(args.data_root)
    drg_protos = dict(np.load(cb_dir / "drg_prototypes.npz")) if (cb_dir / "drg_prototypes.npz").exists() else {}
    for tgt in args.targets:
        tpaths = target_paths(tgt, args.data_root)
        cb = load_codebook(tpaths["codebook"])
        code_map = {e.key: i for i, e in enumerate(cb)}
        rows = list(iter_target_eval(tgt, args.data_root, source_subjects=source_subjects, limit=args.limit_target))
        z_t = np.load(cb_dir / f"{tgt}.npy")
        tgt_tokens = _tokenize_texts(tokenizer, [r.text for r in rows], args.max_tokens)

        t_c = time.perf_counter()
        if tgt == "mimiciv-ms-drg" and drg_protos:
            comp_op: object = compile_drg_operator(
                cb, z_t, drg_protos, bank, cms_dir=tpaths["cms_dir"], alignment=None, lam=args.lam
            )
        else:
            comp_op = compile_deconvolution_operator(z_t, bank, lam=args.lam, self_normalize=True)
        compile_sec = time.perf_counter() - t_c

        tgt_ops = {
            "B3_2D_dot": _build_naive_dot_operator(z_t, bank),
            "Step2_unnorm_deconv": compile_deconvolution_operator(z_t, bank, lam=args.lam, self_normalize=False),
            "Step2_selfnorm_deconv": compile_deconvolution_operator(z_t, bank, lam=args.lam, self_normalize=True),
            "Step3_OCSF_compiled": comp_op,
        }
        is_principal = tgt in {"mimiciv-ms-drg", "elcardiocc-2026"}
        sc_map, _ = _eval_split_multi_ops(
            model,
            tgt_tokens,
            pad_id,
            dev,
            args.max_eval_tokens,
            tgt_ops,
            anchors_by_variant=None,
            mean_state_by_variant={"Step3_OCSF_compiled": (source_mean_state if is_principal else None, args.debias_gamma)},
        )
        for v in variants:
            sc = sc_map[v]
            cmap_v = code_map
            if tgt == "codiesp" and v == "Step3_OCSF_compiled" and tpaths["groups"].exists():
                sc, cmap_v = apply_or_groups(sc, code_map, tpaths["groups"])
            if is_principal:
                m, pn = evaluate_answer_groups(sc, [r.label_groups for r in rows], cmap_v)
            else:
                m, pn = evaluate_multilabel(sc, [r.labels for r in rows], cmap_v, threshold=tau_by_var[v])
            m["compile_sec"] = compile_sec
            summary[v][tgt] = m
            per_note[v][tgt] = pn

    bootstrap = {}
    for label, b_v, c_v in [
        ("SelfNorm_minus_Unnorm", "Step2_unnorm_deconv", "Step2_selfnorm_deconv"),
        ("SelfNorm_minus_Dot", "B3_2D_dot", "Step2_selfnorm_deconv"),
        ("Step3_minus_SelfNorm", "Step2_selfnorm_deconv", "Step3_OCSF_compiled"),
    ]:
        bootstrap[label] = {
            bench: asdict(paired_bootstrap_ci(per_note[b_v][bench], per_note[c_v][bench]))
            for bench in per_note[c_v]
            if bench in per_note[b_v]
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "config": {"backbone": args.backbone, "lora_scale": args.lora_scale, "lam": args.lam, "debias_gamma": args.debias_gamma},
                "results": summary,
                "bootstrap": bootstrap,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[eval-ckpt] Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
