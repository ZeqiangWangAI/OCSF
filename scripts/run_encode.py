"""Step 0 Encoder Runner: encode codebooks + non-test source/target splits once with late chunking.

Stores clean mmap-ready .npy files under cache_dir/{backbone}/:
  - codebooks/{name}.npy               (N_codes, d) float32 unit vectors
  - codebooks/drg_prototypes.npz       named DRG public severity/surgical prototype vectors
  - splits/{split}_{chunks,lengths,doc_mean}.npy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np

from ocsf.compiler import DRG_PROTOTYPE_PROMPTS
from ocsf.data import (
    DEFAULT_DATA_ROOT,
    iter_source_icd9,
    iter_target_eval,
    load_codebook,
    load_source_subject_ids,
    target_paths,
)
from ocsf.encode import BACKBONE_SPECS, ClinicalEncoder


ALL_TARGETS = ("mimiciv-icd10", "mimiciv-ms-drg", "codiesp", "elcardiocc-2026", "chip-cdn")


def _save_split(out_dir: Path, name: str, chunks: np.ndarray, lengths: np.ndarray, doc_mean: np.ndarray, doc_ids: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}_chunks.npy", chunks)
    np.save(out_dir / f"{name}_lengths.npy", lengths)
    np.save(out_dir / f"{name}_doc_mean.npy", doc_mean)
    (out_dir / f"{name}_ids.json").write_text(json.dumps(doc_ids), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=sorted(BACKBONE_SPECS), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--doc-batch-size", type=int, default=8)
    parser.add_argument("--code-batch-size", type=int, default=256)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-chunks", type=int, default=32)
    parser.add_argument("--limit-train", type=int, default=None, help="Cap source train rows for smoke tests")
    parser.add_argument("--limit-val", type=int, default=None, help="Cap source val rows for smoke tests")
    parser.add_argument("--limit-target", type=int, default=None, help="Cap each target eval split for smoke tests")
    parser.add_argument("--skip-train", action="store_true", help="Encode codebooks + val + targets only (Step 0)")
    parser.add_argument("--codebooks-only", action="store_true", help="Encode public codebooks + DRG prototypes only")
    args = parser.parse_args()

    out_root = args.cache_dir / args.backbone
    cb_dir = out_root / "codebooks"
    sp_dir = out_root / "splits"
    cb_dir.mkdir(parents=True, exist_ok=True)
    sp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    encoder = ClinicalEncoder(args.backbone, device=args.device)
    print(f"[encode] Loaded {args.backbone} ({encoder.spec.hf_id}) in {time.perf_counter() - t0:.1f}s")

    # 1. Encode all public codebooks (source ICD-9 + 5 targets)
    cb_jobs: dict[str, Path] = {
        "icd9-full": args.data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json",
    }
    for tgt in ALL_TARGETS:
        cb_jobs[tgt] = target_paths(tgt, args.data_root)["codebook"]

    for name, cb_path in cb_jobs.items():
        dst = cb_dir / f"{name}.npy"
        if dst.exists():
            print(f"[encode] Codebook {name} cached at {dst}, skipping")
            continue
        entries = load_codebook(cb_path)
        prompts = [e.prompt_text for e in entries]
        t1 = time.perf_counter()
        vecs = encoder.encode_codebook_texts(prompts, batch_size=args.code_batch_size)
        np.save(dst, vecs)
        print(f"[encode] Codebook {name}: {vecs.shape} in {time.perf_counter() - t1:.1f}s")

    proto_dst = cb_dir / "drg_prototypes.npz"
    if not proto_dst.exists():
        p_names = list(DRG_PROTOTYPE_PROMPTS.keys())
        p_vecs = encoder.encode_codebook_texts([DRG_PROTOTYPE_PROMPTS[k] for k in p_names], batch_size=32)
        np.savez(proto_dst, **{k: p_vecs[i] for i, k in enumerate(p_names)})
        print(f"[encode] Saved {len(p_names)} DRG public prototypes -> {proto_dst}")

    if args.codebooks_only:
        print(f"[encode] Done codebooks-only for {args.backbone} in {time.perf_counter() - t0:.1f}s")
        return

    # 2. Encode source ICD-9 validation + (optionally) train
    for split, lim in (("validation", args.limit_val), ("train", args.limit_train)):
        if split == "train" and args.skip_train:
            continue
        sp_name = f"source_{split}"
        if (sp_dir / f"{sp_name}_chunks.npy").exists():
            print(f"[encode] Split {sp_name} cached, skipping")
            continue
        rows = list(iter_source_icd9(split, args.data_root, limit=lim))
        t1 = time.perf_counter()
        chunks, lengths, doc_mean = encoder.encode_documents_late_chunks(
            [r.text for r in rows],
            batch_size=args.doc_batch_size,
            chunk_tokens=args.chunk_tokens,
            max_tokens=args.max_tokens,
            max_chunks=args.max_chunks,
        )
        _save_split(sp_dir, sp_name, chunks, lengths, doc_mean, [r.doc_id for r in rows])
        print(f"[encode] Split {sp_name}: N={len(rows)} chunks={chunks.shape} in {time.perf_counter() - t1:.1f}s")

    # 3. Encode all 5 frozen non-test transfer targets (patient-disjoint enforced automatically)
    source_subjects = load_source_subject_ids(args.data_root)
    for tgt in ALL_TARGETS:
        sp_name = f"target_{tgt}"
        if (sp_dir / f"{sp_name}_chunks.npy").exists():
            print(f"[encode] Target {sp_name} cached, skipping")
            continue
        rows = list(iter_target_eval(tgt, args.data_root, source_subjects=source_subjects, limit=args.limit_target))
        t1 = time.perf_counter()
        chunks, lengths, doc_mean = encoder.encode_documents_late_chunks(
            [r.text for r in rows],
            batch_size=args.doc_batch_size,
            chunk_tokens=args.chunk_tokens,
            max_tokens=args.max_tokens,
            max_chunks=args.max_chunks,
        )
        _save_split(sp_dir, sp_name, chunks, lengths, doc_mean, [r.doc_id for r in rows])
        print(f"[encode] Target {sp_name}: N={len(rows)} chunks={chunks.shape} in {time.perf_counter() - t1:.1f}s")

    print(f"[encode] Done all encoding for {args.backbone} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
