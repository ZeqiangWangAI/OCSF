"""Streaming zero-shot B0 evaluation (B0_doc + B0_chunk_max) for any backbone (incl. 4B).

Uses pre-encoded public codebooks in `cache_dir/<backbone>/codebooks/` and streams non-test
evaluation splits directly through `ClinicalEncoder` with zero disk bloat.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from ocsf.compiler import apply_or_groups
from ocsf.data import iter_source_icd9, iter_target_eval, load_codebook, load_source_subject_ids, target_paths
from ocsf.encode import ClinicalEncoder
from ocsf.metrics import calibrate_source_threshold, evaluate_answer_groups, evaluate_multilabel
from ocsf.state_matrix import l2_normalize

DEFAULT_TARGETS = ("mimiciv-icd10", "mimiciv-ms-drg", "codiesp", "elcardiocc-2026", "chip-cdn")


@torch.inference_mode()
def _score_texts_streaming(
    enc: ClinicalEncoder,
    texts: list[str],
    z_cb: np.ndarray,
    device: torch.device,
    max_tokens: int = 4096,
    batch_docs: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (doc_scores, chunk_max_scores) of shape (N, |B|) in float32."""
    n = len(texts)
    n_codes = z_cb.shape[0]
    z_t = torch.from_numpy(l2_normalize(z_cb.astype(np.float32, copy=False))).to(device=device, dtype=torch.float16)
    s_doc = np.empty((n, n_codes), dtype=np.float32)
    s_cmax = np.empty((n, n_codes), dtype=np.float32)

    for start in range(0, n, batch_docs):
        end = min(start + batch_docs, n)
        chunks_np, lens_np, doc_np = enc.encode_documents_late_chunks(
            texts[start:end], batch_size=16, max_tokens=max_tokens
        )
        d_t = torch.from_numpy(doc_np).to(device=device, dtype=torch.float16)
        s_doc[start:end] = (d_t @ z_t.T).float().cpu().numpy()
        for k in range(end - start):
            n_ch = max(int(lens_np[k]), 1)
            c_t = torch.from_numpy(chunks_np[k, :n_ch]).to(device=device, dtype=torch.float16)
            s_cmax[start + k] = (c_t @ z_t.T).float().max(dim=0).values.cpu().numpy()
    return s_doc, s_cmax


def main() -> None:
    p = argparse.ArgumentParser(description="Streaming zero-shot B0 (doc + chunk_max) evaluation.")
    p.add_argument("--backbone", required=True, choices=["pplx-0.6b", "qwen-0.6b", "pplx-4b", "qwen-4b"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--batch-docs", type=int, default=128)
    args = p.parse_args()

    data_root = Path(args.data_root)
    cb_dir = Path(args.cache_dir) / args.backbone / "codebooks"
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = ClinicalEncoder(args.backbone, device=str(dev))

    # 1. Source ICD-9 val (full + Edin frequent 6,150)
    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_key_to_idx = {e.key: i for i, e in enumerate(src_cb)}
    z_src = np.load(cb_dir / "icd9-full.npy")
    val_ex = list(iter_source_icd9("validation", data_root))
    val_labels = [e.labels for e in val_ex]

    freq_path = data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-frequent.json"
    freq_set = {e.key for e in load_codebook(freq_path)} if freq_path.exists() else set()
    freq_indices = np.array([src_key_to_idx[k] for k in freq_set if k in src_key_to_idx], dtype=np.int64)
    freq_k2i = {src_cb[int(i)].key: j for j, i in enumerate(freq_indices)}

    val_gold_idx = [[src_key_to_idx[k] for k in ys if k in src_key_to_idx] for ys in val_labels]
    t0 = time.perf_counter()
    s_doc_val, s_cmax_val = _score_texts_streaming(
        enc, [e.text for e in val_ex], z_src, dev, max_tokens=args.max_tokens, batch_docs=args.batch_docs
    )
    print(f"[b0-{args.backbone}] Scored source_icd9_val (N={len(val_ex)}) in {time.perf_counter()-t0:.1f}s")

    results: dict[str, dict[str, dict[str, float]]] = {"B0_doc": {}, "B0_chunk_max": {}}
    taus: dict[str, float] = {}
    for vname, sc in (("B0_doc", s_doc_val), ("B0_chunk_max", s_cmax_val)):
        tau, _ = calibrate_source_threshold(sc, val_gold_idx)
        taus[vname] = float(tau)
        m_full, _ = evaluate_multilabel(sc, val_labels, src_key_to_idx, threshold=tau)
        m_full["tau_A"] = float(tau)
        results[vname]["source_icd9_val"] = m_full
        if len(freq_indices) > 0:
            sc_f = sc[:, freq_indices]
            lbl_f = [tuple(k for k in ys if k in freq_k2i) for ys in val_labels]
            gold_f = [[freq_k2i[k] for k in ys] for ys in lbl_f]
            tau_f, _ = calibrate_source_threshold(sc_f, gold_f)
            m_f, _ = evaluate_multilabel(sc_f, lbl_f, freq_k2i, threshold=tau_f)
            results[vname]["source_icd9_val_frequent"] = m_f

    # 2. All 5 frozen non-test targets
    source_subjects = load_source_subject_ids(data_root)
    for tgt in args.targets:
        tpaths = target_paths(tgt, data_root)
        cb = load_codebook(tpaths["codebook"])
        k2i = {e.key: i for i, e in enumerate(cb)}
        z_t = np.load(cb_dir / f"{tgt}.npy")
        ex = list(iter_target_eval(tgt, data_root, source_subjects))
        t1 = time.perf_counter()
        s_doc, s_cmax = _score_texts_streaming(
            enc, [e.text for e in ex], z_t, dev, max_tokens=args.max_tokens, batch_docs=args.batch_docs
        )
        print(f"[b0-{args.backbone}] Scored {tgt} (N={len(ex)}) in {time.perf_counter()-t1:.1f}s")

        groups_json = tpaths.get("groups")
        for vname, sc in (("B0_doc", s_doc), ("B0_chunk_max", s_cmax)):
            if groups_json is not None and groups_json.exists():
                sc_eval, k2i_eval = apply_or_groups(sc.copy(), k2i, groups_json)
            else:
                sc_eval, k2i_eval = sc, k2i
            if tgt in {"mimiciv-ms-drg", "elcardiocc-2026"}:
                m, _ = evaluate_answer_groups(sc_eval, [e.label_groups for e in ex], k2i_eval)
            else:
                m, _ = evaluate_multilabel(sc_eval, [e.labels for e in ex], k2i_eval, threshold=taus[vname])
            results[vname][tgt] = m

            if tgt == "mimiciv-icd10" and tpaths.get("frequent_codebook") and tpaths["frequent_codebook"].exists():
                fset = {e.key for e in load_codebook(tpaths["frequent_codebook"])}
                m_f, _ = evaluate_multilabel(
                    sc_eval, [tuple(k for k in e.labels if k in fset) for e in ex], k2i_eval, threshold=taus[vname]
                )
                m_r, _ = evaluate_multilabel(
                    sc_eval, [tuple(k for k in e.labels if k not in fset) for e in ex], k2i_eval, threshold=taus[vname]
                )
                results[vname]["mimiciv-icd10-frequent"] = m_f
                results[vname]["mimiciv-icd10-rare"] = m_r

    payload = {"backbone": args.backbone, "max_tokens": args.max_tokens, "taus": taus, "results": results}
    out_path.write_text(json.dumps(payload, indent=2))
    for vname, bmap in results.items():
        print(f"\n=== {args.backbone} | {vname} ===")
        for b, m in bmap.items():
            if "acc@1" in m:
                print(f"  {b:<26} acc@1={m['acc@1']:.4f} acc@5={m['acc@5']:.4f} MRR={m['mrr']:.4f}")
            else:
                print(f"  {b:<26} R@|y|={m['r@|y|']:.4f} F1@tauA={m['micro_f1@tau_A']:.4f} P@8={m['p@8']:.4f} MRR={m['mrr']:.4f}")


if __name__ == "__main__":
    main()
