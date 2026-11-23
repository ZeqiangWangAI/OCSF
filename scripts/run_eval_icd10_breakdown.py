#!/usr/bin/env python3
"""Diagnostic breakdown of zero-training MIMIC ICD-10 transfer (Diagnoses vs 7-char PCS Procedures,
Frequent vs Rare) plus exact offline Woodbury compile latency across all 6 ontologies (115 to 185,458 codes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ocsf.data import (
    DEFAULT_DATA_ROOT,
    iter_target_eval,
    load_codebook,
    load_source_subject_ids,
    target_paths,
)
from ocsf.e2e import (
    StreamingGPUDeconvScorer,
    TokenChannelStateDecoder,
    TokenLevelOCSFModel,
    inject_upper_layer_lora,
)
from ocsf.encode import BACKBONE_SPECS
from ocsf.metrics import evaluate_multilabel
from ocsf.state_matrix import ChannelBank, compile_deconvolution_operator, l2_normalize
from run_eval_e2e_checkpoint import _eval_split_multi_ops
from run_train_e2e_ocsf import _build_naive_dot_operator, _tokenize_texts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=list(BACKBONE_SPECS.keys()))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument("--freeze-ratio", type=float, default=0.5)
    p.add_argument("--n-channel-layers", type=int, default=2)
    p.add_argument("--lam", type=float, default=0.15)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--max-eval-tokens", type=int, default=65536)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = BACKBONE_SPECS[args.backbone]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)

    cb_dir = args.cache_dir / args.backbone / "codebooks"
    saved_bank = np.load(args.run_dir / "channel_bank.npz")
    bank = ChannelBank(anchors=saved_bank["anchors"])

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
    model.eval()
    summary_p = args.run_dir / "e2e_ocsf_summary.json"
    s_json = json.loads(summary_p.read_text(encoding="utf-8"))
    tau_a = float(s_json["results"]["Step2_OCSF_deconv"]["source_icd9_val"]["tau_A"])

    # 1. Exact offline Woodbury compile latency benchmark across all 6 ontologies
    compile_bench: dict[str, dict[str, float]] = {}
    for name, fname in (
        ("elcardiocc-2026", "elcardiocc-2026.npy"),
        ("mimiciv-ms-drg", "mimiciv-ms-drg.npy"),
        ("source_icd9", "icd9-full.npy"),
        ("mimiciv-icd10", "mimiciv-icd10.npy"),
        ("chip-cdn", "chip-cdn.npy"),
        ("codiesp", "codiesp.npy"),
    ):
        z_t = l2_normalize(np.load(cb_dir / fname).astype(np.float32))
        t0 = time.perf_counter()
        op = compile_deconvolution_operator(z_t, bank, lam=args.lam, self_normalize=True)
        c_sec = time.perf_counter() - t0
        # Measure pure GPU readout latency (ms / note) over 256 synthetic state matrices
        scorer = StreamingGPUDeconvScorer(op, dev, dtype=torch.float16)
        u_syn = torch.randn((256, bank.n_channels, bank.dim), device=dev, dtype=torch.float16)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(5):
            _ = scorer.score_tensor(u_syn)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        readout_ms = 1000.0 * (time.perf_counter() - t1) / (5 * 256)
        compile_bench[name] = {
            "n_codes": int(z_t.shape[0]),
            "dim": int(z_t.shape[1]),
            "n_channels": int(bank.n_channels),
            "compile_sec": float(c_sec),
            "gpu_readout_ms_per_note": float(readout_ms),
        }
        del scorer

    # 2. MIMIC ICD-10 detailed slice breakdown (All / Diagnoses CM / Procedures PCS / Frequent / Rare)
    src_subjs = load_source_subject_ids(args.data_root)
    tp = target_paths("mimiciv-icd10", args.data_root)
    cb = load_codebook(tp["codebook"])
    key_to_idx = {e.key: i for i, e in enumerate(cb)}
    diag_idx = np.array([i for i, e in enumerate(cb) if e.system == "icd10_diag"], dtype=np.int64)
    proc_idx = np.array([i for i, e in enumerate(cb) if e.system == "icd10_proc"], dtype=np.int64)
    freq_keys = {e.key for e in load_codebook(tp["frequent_codebook"])}
    freq_idx = np.array([i for i, e in enumerate(cb) if e.key in freq_keys], dtype=np.int64)
    rare_idx = np.array([i for i, e in enumerate(cb) if e.key not in freq_keys], dtype=np.int64)

    exs = list(iter_target_eval("mimiciv-icd10", args.data_root, source_subjects=src_subjs))
    toks = _tokenize_texts(tokenizer, [e.text for e in exs], args.max_tokens)
    z_10 = l2_normalize(np.load(cb_dir / "mimiciv-icd10.npy").astype(np.float32))

    ops = {
        "B3_2D_dot": _build_naive_dot_operator(z_10, bank),
        "Step2_OCSF_deconv": compile_deconvolution_operator(z_10, bank, lam=args.lam, self_normalize=True),
    }
    sc_map, _ = _eval_split_multi_ops(model, toks, pad_id, dev, args.max_eval_tokens, ops)

    slices = {
        "all_26085": (None, [e.labels for e in exs], key_to_idx),
        "icd10_cm_diag_16146": (diag_idx, [[k for k in e.labels if k.startswith("icd10_diag:")] for e in exs], None),
        "icd10_pcs_proc_9939": (proc_idx, [[k for k in e.labels if k.startswith("icd10_proc:")] for e in exs], None),
        "frequent_7942": (freq_idx, [[k for k in e.labels if k in freq_keys] for e in exs], None),
        "rare_18143": (rare_idx, [[k for k in e.labels if k not in freq_keys] for e in exs], None),
    }

    breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for var_name, scores in sc_map.items():
        breakdown[var_name] = {}
        for s_name, (col_idx, gold_lists, k2i) in slices.items():
            if col_idx is None:
                sub_scores = scores
                sub_map = k2i
            else:
                sub_scores = scores[:, col_idx]
                sub_map = {cb[int(orig_i)].key: j for j, orig_i in enumerate(col_idx)}
            m, _ = evaluate_multilabel(sub_scores, gold_lists, sub_map, threshold=tau_a)
            breakdown[var_name][s_name] = m

    out = {
        "backbone": args.backbone,
        "run_dir": str(args.run_dir),
        "tau_a": tau_a,
        "compile_and_readout_latency": compile_bench,
        "icd10_breakdown": breakdown,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[icd10-diag] Saved -> {args.out_json}")
    for vname, smap in breakdown.items():
        print(f"=== {args.backbone} | {vname} ===")
        for sn, m in smap.items():
            print(
                f"  {sn:<22} n_valid={int(m['n_notes_valid']):<5} "
                f"R@|y|={m['r@|y|']:.4f} F1@tauA={m['micro_f1@tau_A']:.4f} "
                f"P@8={m['p@8']:.4f} R@50={m['r@50']:.4f} MRR={m['mrr']:.4f}"
            )


if __name__ == "__main__":
    main()
