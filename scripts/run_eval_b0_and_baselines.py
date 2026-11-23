"""Step 0 (B0 Zero-Shot Retrieval) and B1/B2 (1D Flat CAST Ridge Baseline) Evaluation Runner.

Evaluates on Source ICD-9 val + all 5 frozen non-test transfer targets with strict zero leakage:
  - B0_doc:       Full-document cosine retrieval (doc_mean @ Z_B^T)
  - B0_chunk_max: Late-chunk max-pooled zero-shot retrieval (max_m chunks[m] @ Z_B^T)
  - B1/B2_cast:   1D Flat Classifier trained ONLY on Source ICD-9 train -> closed-form CAST
                  ridge projection P = W_A^T Z_A (Z_A^T Z_A + lam I)^{-1} to all target tables.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ocsf.baselines import CastProjector, score_b0_zeroshot_retrieval
from ocsf.compiler import apply_or_groups
from ocsf.data import (
    DEFAULT_DATA_ROOT,
    iter_source_icd9,
    iter_target_eval,
    load_codebook,
    load_source_subject_ids,
    target_paths,
)
from ocsf.metrics import calibrate_source_threshold, evaluate_answer_groups, evaluate_multilabel
from ocsf.models import Flat1DClassifier, Flat1DStatePredictor, zlpr_loss
from ocsf.state_matrix import l2_normalize


ALL_TARGETS = ("mimiciv-icd10", "mimiciv-ms-drg", "codiesp", "elcardiocc-2026", "chip-cdn")


def _score_chunk_max(chunks: np.ndarray, lengths: np.ndarray, z_cb: np.ndarray, batch_size: int = 128) -> np.ndarray:
    z = l2_normalize(z_cb.astype(np.float32, copy=False))
    n = chunks.shape[0]
    out = np.empty((n, z.shape[0]), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for i in range(start, end):
            m = max(int(lengths[i]), 1)
            c = chunks[i, :m].astype(np.float32)
            out[i] = (c @ z.T).max(axis=0)
    return out


def evaluate_all_benchmarks(
    score_fn,
    cache_root: Path,
    data_root: Path,
    out_per_note_dir: Path,
    targets: tuple[str, ...] = ALL_TARGETS,
    limit_val: int | None = None,
    limit_target: int | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate any scoring function across Source ICD-9 val + all frozen transfer targets."""
    cb_dir = cache_root / "codebooks"
    sp_dir = cache_root / "splits"
    out_per_note_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}

    # 1. Source ICD-9 validation (used strictly to calibrate tau_A)
    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_map = {e.key: i for i, e in enumerate(src_cb)}
    src_rows = list(iter_source_icd9("validation", data_root, limit=limit_val))
    z_src = np.load(cb_dir / "icd9-full.npy")
    src_scores = score_fn("source_validation", "icd9-full", z_src, src_cb, sp_dir)
    src_gold_idx = [[src_map[k] for k in r.labels if k in src_map] for r in src_rows]
    tau_a, _ = calibrate_source_threshold(src_scores, src_gold_idx)
    src_metrics, src_pn = evaluate_multilabel(
        src_scores, [r.labels for r in src_rows], src_map, threshold=tau_a
    )
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

    # 2. Frozen non-test targets
    source_subjects = load_source_subject_ids(data_root)

    for tgt in targets:
        tpaths = target_paths(tgt, data_root)
        cb = load_codebook(tpaths["codebook"])
        code_map = {e.key: i for i, e in enumerate(cb)}
        rows = list(iter_target_eval(tgt, data_root, source_subjects=source_subjects, limit=limit_target))
        z_t = np.load(cb_dir / f"{tgt}.npy")
        t0 = time.perf_counter()
        scores = score_fn(f"target_{tgt}", tgt, z_t, cb, sp_dir)
        infer_ms = (time.perf_counter() - t0) * 1000.0 / max(len(rows), 1)

        if tgt == "codiesp" and tpaths["groups"].exists():
            scores, code_map = apply_or_groups(scores, code_map, tpaths["groups"])

        if tgt in {"mimiciv-ms-drg", "elcardiocc-2026"}:
            metrics, pn = evaluate_answer_groups(scores, [r.label_groups for r in rows], code_map)
        else:
            metrics, pn = evaluate_multilabel(scores, [r.labels for r in rows], code_map, threshold=tau_a)

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

    return results


def train_b1_cast(
    cache_root: Path,
    data_root: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    cast_lam: float,
    limit_train: int | None = None,
    limit_val: int | None = None,
) -> tuple[Flat1DClassifier, CastProjector]:
    """Train B1/B2 1D Flat Classifier strictly on Source ICD-9 train and fit closed-form CAST projector."""
    sp_dir = cache_root / "splits"
    cb_dir = cache_root / "codebooks"

    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_map = {e.key: i for i, e in enumerate(src_cb)}
    n_codes = len(src_cb)

    train_rows = list(iter_source_icd9("train", data_root, limit=limit_train))
    train_labels = [[src_map[k] for k in r.labels if k in src_map] for r in train_rows]
    chunks_tr = np.load(sp_dir / "source_train_chunks.npy", mmap_mode="r")
    lengths_tr = np.load(sp_dir / "source_train_lengths.npy")
    n_train = len(train_rows)
    dim = chunks_tr.shape[2]

    dev = torch.device(device)
    z_source = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))
    model = Flat1DClassifier(
        dim=dim, n_source_codes=n_codes, z_source_init=torch.from_numpy(z_source)
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    seen_mask = np.zeros(n_codes, dtype=bool)
    for lbls in train_labels:
        for c in lbls:
            seen_mask[c] = True

    rng = np.random.default_rng(42)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        running_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            ch = torch.from_numpy(np.array(chunks_tr[idx], dtype=np.float32)).to(dev)
            ln = torch.from_numpy(lengths_tr[idx]).to(dev)
            mask = torch.arange(ch.shape[1], device=dev)[None, :] < ln[:, None]
            y = torch.zeros((len(idx), n_codes), device=dev)
            for b_i, row_i in enumerate(idx):
                if train_labels[row_i]:
                    y[b_i, train_labels[row_i]] = 1.0

            opt.zero_grad(set_to_none=True)
            logits, _ = model(ch, mask)
            loss = F.binary_cross_entropy_with_logits(logits, y) * 200.0 + 0.15 * zlpr_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += float(loss.detach())
            n_batches += 1
        sched.step()
        print(f"[B1_CAST_1D] epoch={ep}/{epochs} loss={running_loss / max(n_batches, 1):.4f}")

    model.eval()
    w_source = model.normalized_weights().detach().cpu().numpy()  # (|A|, d) unit-norm
    cast = CastProjector.fit_from_classifier_weights(w_source, z_source, lam=cast_lam, active_mask=seen_mask)
    return model, cast


def train_b1_1d_state(
    cache_root: Path,
    data_root: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    limit_train: int | None = None,
) -> Flat1DStatePredictor:
    """Train B1_1D_State (exact R=1 special case of OCSF) strictly on Source ICD-9 train."""
    sp_dir = cache_root / "splits"
    cb_dir = cache_root / "codebooks"

    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    src_map = {e.key: i for i, e in enumerate(src_cb)}
    n_codes = len(src_cb)
    train_rows = list(iter_source_icd9("train", data_root, limit=limit_train))
    train_labels = [[src_map[k] for k in r.labels if k in src_map] for r in train_rows]

    chunks_tr = np.load(sp_dir / "source_train_chunks.npy", mmap_mode="r")
    lengths_tr = np.load(sp_dir / "source_train_lengths.npy")
    n_train = len(train_rows)
    dim = chunks_tr.shape[2]

    dev = torch.device(device)
    z_a = torch.from_numpy(l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))).to(dev)
    model = Flat1DStatePredictor(dim=dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    rng = np.random.default_rng(42)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        running_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            ch = torch.from_numpy(np.array(chunks_tr[idx], dtype=np.float32)).to(dev)
            ln = torch.from_numpy(lengths_tr[idx]).to(dev)
            mask = torch.arange(ch.shape[1], device=dev)[None, :] < ln[:, None]
            y = torch.zeros((len(idx), n_codes), device=dev)
            for b_i, row_i in enumerate(idx):
                if train_labels[row_i]:
                    y[b_i, train_labels[row_i]] = 1.0

            opt.zero_grad(set_to_none=True)
            u_1d = model(ch, mask)
            scores = u_1d @ z_a.T
            logits = 10.0 * (scores - 0.50)
            loss = F.binary_cross_entropy_with_logits(logits, y) * 200.0 + 0.15 * zlpr_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += float(loss.detach())
            n_batches += 1
        sched.step()
        print(f"[B1_1D_State] epoch={ep}/{epochs} loss={running_loss / max(n_batches, 1):.4f}")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-cast", action="store_true", help="Also train B1/B2 1D CAST baseline on source train")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--cast-lam", type=float, default=0.15)
    parser.add_argument("--targets", nargs="+", default=list(ALL_TARGETS))
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-target", type=int, default=None)
    args = parser.parse_args()
    targets = tuple(args.targets)

    cache_root = args.cache_dir / args.backbone
    out_root = args.out_dir / args.backbone
    out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, dict[str, float]]] = {}

    # B0_doc: Full-document zero-shot cosine retrieval
    def _b0_doc(sp_name: str, _tgt: str, z_cb: np.ndarray, _cb, sp_dir: Path) -> np.ndarray:
        doc_mean = np.load(sp_dir / f"{sp_name}_doc_mean.npy")
        return score_b0_zeroshot_retrieval(doc_mean, z_cb)

    summary["B0_doc"] = evaluate_all_benchmarks(
        _b0_doc, cache_root, args.data_root, out_root / "per_note" / "B0_doc", targets, args.limit_val, args.limit_target
    )

    # B0_chunk_max: Late-chunk max-pooled zero-shot retrieval
    def _b0_chunk(sp_name: str, _tgt: str, z_cb: np.ndarray, _cb, sp_dir: Path) -> np.ndarray:
        chunks = np.load(sp_dir / f"{sp_name}_chunks.npy", mmap_mode="r")
        lengths = np.load(sp_dir / f"{sp_name}_lengths.npy")
        return _score_chunk_max(chunks, lengths, z_cb)

    summary["B0_chunk_max"] = evaluate_all_benchmarks(
        _b0_chunk, cache_root, args.data_root, out_root / "per_note" / "B0_chunk_max", targets, args.limit_val, args.limit_target
    )

    if args.run_cast:
        model, cast = train_b1_cast(
            cache_root,
            args.data_root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            cast_lam=args.cast_lam,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
        )
        dev = torch.device(args.device)

        @torch.inference_mode()
        def _b1_cast(sp_name: str, _tgt: str, z_cb: np.ndarray, _cb, sp_dir: Path) -> np.ndarray:
            chunks = np.load(sp_dir / f"{sp_name}_chunks.npy", mmap_mode="r")
            lengths = np.load(sp_dir / f"{sp_name}_lengths.npy")
            n = chunks.shape[0]
            doc_vecs = np.empty((n, chunks.shape[2]), dtype=np.float32)
            for s in range(0, n, 256):
                e = min(s + 256, n)
                ch = torch.from_numpy(np.array(chunks[s:e], dtype=np.float32)).to(dev)
                ln = torch.from_numpy(lengths[s:e]).to(dev)
                mask = torch.arange(ch.shape[1], device=dev)[None, :] < ln[:, None]
                doc_vecs[s:e] = model.encode_doc(ch, mask).cpu().numpy()
            return cast.score_target(doc_vecs, z_cb)

        summary["B1_CAST_1D"] = evaluate_all_benchmarks(
            _b1_cast, cache_root, args.data_root, out_root / "per_note" / "B1_CAST_1D", targets, args.limit_val, args.limit_target
        )

        m_1d = train_b1_1d_state(
            cache_root,
            args.data_root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            limit_train=args.limit_train,
        )

        @torch.inference_mode()
        def _b1_1d_state(sp_name: str, _tgt: str, z_cb: np.ndarray, _cb, sp_dir: Path) -> np.ndarray:
            chunks = np.load(sp_dir / f"{sp_name}_chunks.npy", mmap_mode="r")
            lengths = np.load(sp_dir / f"{sp_name}_lengths.npy")
            n = chunks.shape[0]
            u_vecs = np.empty((n, chunks.shape[2]), dtype=np.float32)
            for s in range(0, n, 256):
                e = min(s + 256, n)
                ch = torch.from_numpy(np.array(chunks[s:e], dtype=np.float32)).to(dev)
                ln = torch.from_numpy(lengths[s:e]).to(dev)
                mask = torch.arange(ch.shape[1], device=dev)[None, :] < ln[:, None]
                u_vecs[s:e] = m_1d(ch, mask).cpu().numpy()
            z_b = l2_normalize(z_cb.astype(np.float32, copy=False))
            return (u_vecs @ z_b.T).astype(np.float32)

        summary["B1_1D_State"] = evaluate_all_benchmarks(
            _b1_1d_state, cache_root, args.data_root, out_root / "per_note" / "B1_1D_State", targets, args.limit_val, args.limit_target
        )

    out_json = out_root / "baselines_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[baselines] Wrote results -> {out_json}")
    for method, res in summary.items():
        print(f"\n=== {args.backbone} | {method} ===")
        for bench, m in res.items():
            key_metric = f"acc@1={m['acc@1']:.4f} acc@5={m['acc@5']:.4f}" if "acc@1" in m else f"R@|y|={m['r@|y|']:.4f} R@15={m['r@15']:.4f} F1@tauA={m.get('micro_f1@tau_A', 0.0):.4f}"
            print(f"  {bench:<26} {key_metric}  MRR={m['mrr']:.4f}")


if __name__ == "__main__":
    main()
