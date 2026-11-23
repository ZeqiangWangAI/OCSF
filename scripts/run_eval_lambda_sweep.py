"""Single-pass zero-retrain shrinkage sensitivity sweep (lam in {0.05, 0.10, 0.15, 0.30, 0.50}).

Runs the backbone ONCE per split and simultaneously scores all 5 Tikhonov shrinkage settings
via StreamingGPUDeconvScorer (<6 min total on 1 GH200 across all 6 benchmarks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ocsf.data import iter_source_icd9, iter_target_eval, load_codebook, load_source_subject_ids, target_paths
from ocsf.e2e import TokenChannelStateDecoder, TokenLevelOCSFModel, inject_upper_layer_lora
from ocsf.encode import BACKBONE_SPECS
from ocsf.metrics import calibrate_source_threshold, evaluate_answer_groups, evaluate_multilabel
from ocsf.state_matrix import ChannelBank, compile_deconvolution_operator, l2_normalize
from run_eval_e2e_checkpoint import _eval_split_multi_ops
from run_train_e2e_ocsf import _tokenize_texts

DEFAULT_TARGETS = ("mimiciv-icd10", "mimiciv-ms-drg", "codiesp", "elcardiocc-2026", "chip-cdn")


def main() -> None:
    p = argparse.ArgumentParser(description="Single-pass Tikhonov shrinkage lambda sensitivity sweep.")
    p.add_argument("--backbone", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.30, 0.50])
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64.0)
    p.add_argument("--freeze-ratio", type=float, default=0.5)
    p.add_argument("--n-channel-layers", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--max-eval-tokens", type=int, default=49152)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    data_root = Path(args.data_root)
    cb_dir = Path(args.cache_dir) / args.backbone / "codebooks"
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_bank = np.load(run_dir / "channel_bank.npz")
    anchors = saved_bank["anchors"]
    bank = ChannelBank(anchors=anchors)

    spec = BACKBONE_SPECS[args.backbone]
    tok = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)
    backbone = AutoModel.from_pretrained(
        spec.hf_id,
        trust_remote_code=spec.trust_remote_code,
        dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    inject_upper_layer_lora(
        backbone, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0, freeze_ratio=args.freeze_ratio
    )
    decoder = TokenChannelStateDecoder(torch.from_numpy(anchors), n_heads=8, n_channel_layers=args.n_channel_layers, dropout=0.0)
    model = TokenLevelOCSFModel(backbone, decoder, apply_tanh=(spec.family == "pplx_diffusion")).to(dev)
    ckpt = torch.load(run_dir / "best_e2e_trainable.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["trainable_state"], strict=False)
    model.eval()

    lam_tags = [f"lam_{lam:.2f}" for lam in args.lambdas]
    results: dict[str, dict[str, dict[str, float]]] = {tag: {} for tag in lam_tags}
    taus: dict[str, float] = {}

    # 1. Source ICD-9 val
    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_k2i = {e.key: i for i, e in enumerate(src_cb)}
    z_src = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))
    val_ex = list(iter_source_icd9("validation", data_root))
    val_tok = _tokenize_texts(tok, [e.text for e in val_ex], args.max_tokens)
    src_ops = {
        tag: compile_deconvolution_operator(z_src, bank, lam=lam, self_normalize=True)
        for tag, lam in zip(lam_tags, args.lambdas)
    }
    t0 = time.perf_counter()
    sc_val, _ = _eval_split_multi_ops(model, val_tok, pad_id, dev, args.max_eval_tokens, src_ops)
    print(f"[lam-sweep] Source val (N={len(val_ex)}) done in {time.perf_counter()-t0:.1f}s")
    val_labels = [e.labels for e in val_ex]
    val_gold_idx = [[src_k2i[k] for k in ys if k in src_k2i] for ys in val_labels]
    for tag in lam_tags:
        tau, _ = calibrate_source_threshold(sc_val[tag], val_gold_idx)
        taus[tag] = float(tau)
        m, _ = evaluate_multilabel(sc_val[tag], val_labels, src_k2i, threshold=tau)
        results[tag]["source_icd9_val"] = m

    # 2. All 5 frozen targets
    src_subjects = load_source_subject_ids(data_root)
    for tgt in DEFAULT_TARGETS:
        tpaths = target_paths(tgt, data_root)
        cb = load_codebook(tpaths["codebook"])
        k2i = {e.key: i for i, e in enumerate(cb)}
        z_t = l2_normalize(np.load(cb_dir / f"{tgt}.npy").astype(np.float32))
        ex = list(iter_target_eval(tgt, data_root, src_subjects))
        tok_t = _tokenize_texts(tok, [e.text for e in ex], args.max_tokens)
        ops = {
            tag: compile_deconvolution_operator(z_t, bank, lam=lam, self_normalize=True)
            for tag, lam in zip(lam_tags, args.lambdas)
        }
        t1 = time.perf_counter()
        sc_t, _ = _eval_split_multi_ops(model, tok_t, pad_id, dev, args.max_eval_tokens, ops)
        print(f"[lam-sweep] {tgt} (N={len(ex)}) done in {time.perf_counter()-t1:.1f}s")
        for tag in lam_tags:
            if tgt in {"mimiciv-ms-drg", "elcardiocc-2026"}:
                m, _ = evaluate_answer_groups(sc_t[tag], [e.label_groups for e in ex], k2i)
            else:
                m, _ = evaluate_multilabel(sc_t[tag], [e.labels for e in ex], k2i, threshold=taus[tag])
            results[tag][tgt] = m

    out_path.write_text(json.dumps({"backbone": args.backbone, "lambdas": args.lambdas, "results": results}, indent=2))
    print(f"[lam-sweep] Saved -> {out_path}")
    for tag in lam_tags:
        row = []
        for b in ("source_icd9_val", *DEFAULT_TARGETS):
            m = results[tag][b]
            v = m["acc@1"] if "acc@1" in m else m["r@|y|"]
            row.append(f"{b}={v:.4f}")
        print(f"  {tag}: " + " | ".join(row))


if __name__ == "__main__":
    main()
