"""Verify theoretical Oracle & noisy deconvolution capacity of the 2D Clinical State Matrix U in R^{R x K}.

Tests the Forward-Inverse Duality formulation:
  Forward (ontology-invariant):  u_r = sum_{a in S+_r} z_a = Z_r^T y_r
  Inverse Readout (zero-training, pure matrix ops compiled from codebook Z_r):
    1. dot            : y_hat = Z_r u_r                                (naive matched filter)
    2. tikhonov       : y_hat = Z_r (Z_r^T Z_r + lambda I)^{-1} u_r    (1 matrix multiply!)
    3. nn_fista       : argmin_{y>=0} 0.5 ||Z_r^T y - u_r||^2 + lam||y||_1  (sparse non-negative unmixing)

Uses ONLY source MIMIC-IV ICD-9 validation ground-truth labels (non-test) and public
ICD-10 codebook descriptions (zero target dev/test label access).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_ROOT = Path(os.environ.get("OCSF_DATA_ROOT", "./data"))
ICD9_CODEBOOK = DATA_ROOT / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json"
ICD10_CODEBOOK = DATA_ROOT / "benchmarks/mimiciv-2.2/codebooks/icd10-full.json"
SOURCE_VAL = DATA_ROOT / "source-text/validation.jsonl"


def load_codebook(path: Path) -> tuple[list[str], list[str]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    keys = [e["key"] for e in entries]
    texts = [f"{e['system']} {e['code']} {e['description']}" for e in entries]
    return keys, texts


def build_whitened_embeddings(texts: list[str], dim: int, seed: int = 42) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60000, sublinear_tf=True)
    tfidf = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    z = svd.fit_transform(tfidf).astype(np.float32)
    z -= z.mean(axis=0, keepdims=True)
    z /= np.sqrt(z.var(axis=0, keepdims=True) + 1e-4)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    return z


def load_source_val_labels(key_to_idx: dict[str, int], max_notes: int = 2000) -> list[np.ndarray]:
    label_sets: list[np.ndarray] = []
    with SOURCE_VAL.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= max_notes:
                break
            row = json.loads(line)
            idxs = []
            for c in row["icd9_diag"]:
                k = f"icd9_diag:{c}"
                if k in key_to_idx:
                    idxs.append(key_to_idx[k])
            for c in row["icd9_proc"]:
                k = f"icd9_proc:{c}"
                if k in key_to_idx:
                    idxs.append(key_to_idx[k])
            if idxs:
                label_sets.append(np.array(sorted(set(idxs)), dtype=np.int32))
    return label_sets


def make_synthetic_sibling_draws(
    z: np.ndarray, n_notes: int = 1000, avg_codes: int = 14, seed: int = 42
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    km = MiniBatchKMeans(n_clusters=400, random_state=seed, batch_size=2048, n_init=3)
    fam = km.fit_predict(z)
    fam_members = [np.where(fam == f)[0] for f in range(400)]
    fam_members = [m for m in fam_members if len(m) > 0]

    label_sets: list[np.ndarray] = []
    for _ in range(n_notes):
        n_fams = int(rng.integers(4, 9))
        chosen_fams = rng.choice(len(fam_members), size=n_fams, replace=False)
        picks = []
        for f in chosen_fams:
            members = fam_members[f]
            k_take = int(min(len(members), rng.integers(1, 4)))
            picks.extend(rng.choice(members, size=k_take, replace=False).tolist())
        picks = sorted(set(picks))[: max(avg_codes + 6, 4)]
        label_sets.append(np.array(picks, dtype=np.int32))
    return label_sets


def fit_channels(z: np.ndarray, n_channels: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if n_channels == 1:
        c = z.mean(axis=0, keepdims=True)
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-8)
        return c, np.zeros(z.shape[0], dtype=np.int32)
    km = MiniBatchKMeans(n_clusters=n_channels, random_state=seed, batch_size=2048, n_init=5)
    km.fit(z)
    centers = km.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    assign = np.argmax(z @ centers.T, axis=1).astype(np.int32)
    return centers, assign


def compile_channel_operators(
    z: np.ndarray,
    assign: np.ndarray,
    n_channels: int,
    lam: float = 0.05,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
    """Precompile channel code indices, raw Z_r, Tikhonov dual W_r, and Lipschitz L_r."""
    dim = z.shape[1]
    eye = np.eye(dim, dtype=np.float32)
    chan_indices: list[np.ndarray] = []
    chan_z: list[np.ndarray] = []
    chan_w: list[np.ndarray] = []
    chan_lip: list[float] = []

    for r in range(n_channels):
        idx = np.where(assign == r)[0]
        zr = z[idx]  # (m_r, K)
        chan_indices.append(idx)
        chan_z.append(zr)
        if len(idx) == 0:
            chan_w.append(np.zeros((0, dim), dtype=np.float32))
            chan_lip.append(1.0)
            continue
        # Precompile Tikhonov deconvolution operator W_r = Z_r (Z_r^T Z_r + lam I)^{-1}
        cov = zr.T @ zr + lam * eye
        wr = np.linalg.solve(cov, zr.T).T.astype(np.float32)  # (m_r, K)
        # Diagonal self-gain normalization: guarantees w_i^T z_i == 1.0 for every code i
        # regardless of local sibling cluster density (100% closed-form from public codebook)
        diag_gain = np.sum(wr * zr, axis=1, keepdims=True)
        wr_norm = wr / np.maximum(diag_gain, 1e-4)
        chan_w.append(wr_norm)
        # Largest eigenvalue of Z_r^T Z_r for FISTA step size
        s_max = float(np.linalg.norm(zr, ord=2)) ** 2
        chan_lip.append(max(s_max, 1.0))
    return chan_indices, chan_z, chan_w, chan_lip


def nn_fista_batch(
    zr: np.ndarray,
    u_batch: np.ndarray,
    lip: float,
    l1: float = 0.08,
    n_steps: int = 12,
) -> np.ndarray:
    """Solve min_{y>=0} 0.5 ||Z_r^T y - u||^2 + l1 ||y||_1 for a batch of active notes in channel r."""
    # zr: (m, K), u_batch: (B, K) -> returns y: (B, m)
    step = 1.0 / lip
    gram = zr @ zr.T  # (m, m)
    zt_u = u_batch @ zr.T  # (B, m)
    y = np.zeros_like(zt_u)
    z_var = np.zeros_like(zt_u)
    t = 1.0
    thresh = l1 * step
    for _ in range(n_steps):
        grad = z_var @ gram - zt_u
        y_next = np.maximum(z_var - step * grad - thresh, 0.0)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z_var = y_next + ((t - 1.0) / t_next) * (y_next - y)
        y = y_next
        t = t_next
    return y


def evaluate_duality(
    z: np.ndarray,
    label_sets: list[np.ndarray],
    n_channels: int,
    method: str,
    lam: float = 0.05,
    l1: float = 0.08,
    fista_steps: int = 12,
    noise_std: float = 0.0,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    t_comp = time.perf_counter()
    _, assign = fit_channels(z, n_channels, seed=seed)
    chan_indices, chan_z, chan_w, chan_lip = compile_channel_operators(z, assign, n_channels, lam=lam)
    compile_ms = (time.perf_counter() - t_comp) * 1000.0

    n_notes = len(label_sets)
    dim = z.shape[1]

    # Group notes per active channel for vectorized batched decoding
    chan_note_ids: list[list[int]] = [[] for _ in range(n_channels)]
    chan_u_rows: list[list[np.ndarray]] = [[] for _ in range(n_channels)]

    for note_i, y in enumerate(label_sets):
        by_chan: dict[int, list[int]] = {}
        for idx in y:
            by_chan.setdefault(int(assign[idx]), []).append(int(idx))
        for r, members in by_chan.items():
            u_r = z[members].sum(axis=0)
            if noise_std > 0.0:
                u_r = u_r + rng.normal(0.0, noise_std / np.sqrt(dim), size=dim).astype(np.float32) * np.linalg.norm(u_r)
            chan_note_ids[r].append(note_i)
            chan_u_rows[r].append(u_r)

    t_inf = time.perf_counter()
    note_cands: list[list[tuple[float, int]]] = [[] for _ in range(n_notes)]

    for r in range(n_channels):
        nids = chan_note_ids[r]
        if not nids:
            continue
        u_mat = np.stack(chan_u_rows[r], axis=0)  # (B_r, K)
        if method == "dot":
            scores = u_mat @ chan_z[r].T
        elif method == "tikhonov":
            scores = u_mat @ chan_w[r].T
        elif method == "nn_fista":
            scores = nn_fista_batch(chan_z[r], u_mat, chan_lip[r], l1=l1, n_steps=fista_steps)
        else:
            raise ValueError(method)

        c_idx = chan_indices[r]
        # Keep candidates > 0.25 to compute exact F1@0.5 and R@|y|
        rows, cols = np.where(scores > 0.25)
        for bi, cj in zip(rows.tolist(), cols.tolist()):
            note_cands[nids[bi]].append((float(scores[bi, cj]), int(c_idx[cj])))

    inf_ms_per_note = (time.perf_counter() - t_inf) * 1000.0 / n_notes

    tp = fp = fn = 0
    total_pos = 0
    r_at_k_hits = 0
    for y, cands in zip(label_sets, note_cands):
        y_set = set(y.tolist())
        total_pos += len(y_set)
        preds = {cid for sc, cid in cands if sc >= 0.5}
        hit = len(preds & y_set)
        tp += hit
        fp += len(preds - y_set)
        fn += len(y_set - preds)

        cands.sort(key=lambda x: x[0], reverse=True)
        topk = {cid for _, cid in cands[: len(y_set)]}
        r_at_k_hits += len(topk & y_set)

    micro_f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    return {
        "micro_f1@0.5": micro_f1,
        "prec@0.5": tp / max(tp + fp, 1),
        "rec@0.5": tp / max(tp + fn, 1),
        "r@|y|": r_at_k_hits / max(total_pos, 1),
        "compile_ms": compile_ms,
        "ms/note": inf_ms_per_note,
    }


def run_suite(title: str, z: np.ndarray, label_sets: list[np.ndarray], configs: list[tuple[str, int, float]]) -> None:
    print(f"\n=== {title} (|C|={z.shape[0]}, K={z.shape[1]}, N={len(label_sets)}) ===")
    print(f"{'Method':<10} {'R':>4} {'Noise':>6} {'MicroF1':>8} {'Prec':>7} {'Rec':>7} {'R@|y|':>8} {'CompMs':>8} {'ms/note':>8}")
    for method, r, noise in configs:
        res = evaluate_duality(z, label_sets, n_channels=r, method=method, noise_std=noise)
        print(
            f"{method:<10} {r:>4d} {noise:>6.2f} "
            f"{res['micro_f1@0.5']:>8.4f} {res['prec@0.5']:>7.4f} {res['rec@0.5']:>7.4f} "
            f"{res['r@|y|']:>8.4f} {res['compile_ms']:>8.1f} {res['ms/note']:>8.3f}"
        )


def main() -> None:
    t0 = time.perf_counter()
    icd9_keys, icd9_texts = load_codebook(ICD9_CODEBOOK)
    key_to_idx = {k: i for i, k in enumerate(icd9_keys)}
    val_sets = load_source_val_labels(key_to_idx, max_notes=2000)

    z9 = build_whitened_embeddings(icd9_texts, dim=512)
    run_suite(
        "Real MIMIC-IV ICD-9 Validation Notes",
        z9,
        val_sets,
        [
            ("dot", 1, 0.00),
            ("dot", 64, 0.00),
            ("tikhonov", 1, 0.00),
            ("tikhonov", 32, 0.00),
            ("tikhonov", 64, 0.00),
            ("tikhonov", 128, 0.00),
            ("nn_fista", 64, 0.00),
            ("tikhonov", 64, 0.25),
            ("nn_fista", 64, 0.25),
            ("tikhonov", 64, 0.50),
            ("nn_fista", 64, 0.50),
        ],
    )

    _, icd10_texts = load_codebook(ICD10_CODEBOOK)
    z10 = build_whitened_embeddings(icd10_texts, dim=512)
    syn10 = make_synthetic_sibling_draws(z10, n_notes=1000)
    run_suite(
        "Public ICD-10 Codebook (Synthetic Sibling-Heavy Draws)",
        z10,
        syn10,
        [
            ("dot", 1, 0.00),
            ("dot", 64, 0.00),
            ("tikhonov", 1, 0.00),
            ("tikhonov", 64, 0.00),
            ("tikhonov", 128, 0.00),
            ("nn_fista", 64, 0.00),
            ("nn_fista", 128, 0.00),
            ("tikhonov", 64, 0.25),
            ("nn_fista", 64, 0.25),
            ("tikhonov", 128, 0.25),
            ("nn_fista", 128, 0.25),
        ],
    )
    print(f"\nCompleted in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
