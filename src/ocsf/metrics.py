"""Exact evaluation metrics for multi-label, answer-group, rare/frequent, and paired-bootstrap tests.

All metrics preserve OOV ground-truth labels in the recall denominator (critical for CHIP-CDN)
and enforce strict zero-leakage thresholds calibrated ONLY on source validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float
    cohens_dz: float = 0.0
    cohens_h: float = 0.0


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def calibrate_source_threshold(
    scores: np.ndarray,
    gold_index_lists: list[list[int]],
    grid_size: int = 41,
) -> tuple[float, float]:
    """Select global threshold tau_A strictly on source validation to maximize Micro-F1."""
    pos_scores = np.concatenate([scores[i, g] for i, g in enumerate(gold_index_lists) if g])
    lo = float(np.quantile(pos_scores, 0.05))
    hi = float(np.quantile(pos_scores, 0.95))
    best_tau, best_f1 = 0.5, -1.0
    for tau in np.linspace(lo, hi, grid_size):
        f1 = compute_threshold_micro_f1(scores, gold_index_lists, float(tau), total_gold_count=None)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau, best_f1


def compute_threshold_micro_f1(
    scores: np.ndarray,
    gold_index_lists: list[list[int]],
    threshold: float,
    total_gold_count: int | None = None,
) -> float:
    tp = 0
    pred_total = 0
    gold_total = 0
    for i, gold in enumerate(gold_index_lists):
        pred = np.flatnonzero(scores[i] >= threshold)
        pred_total += int(pred.size)
        gold_total += len(gold)
        if gold and pred.size:
            tp += len(set(gold).intersection(pred.tolist()))
    denom_gold = gold_total if total_gold_count is None else total_gold_count
    if pred_total + denom_gold == 0:
        return 0.0
    return float(2 * tp / (pred_total + denom_gold))


def evaluate_multilabel(
    scores: np.ndarray,
    gold_keys_list: list[tuple[str, ...]],
    code_to_idx: dict[str, int],
    threshold: float | None = None,
    subset_keys: set[str] | None = None,
    ks: tuple[int, ...] = (5, 8, 15, 50),
) -> tuple[dict[str, float], np.ndarray]:
    """Compute ranking + thresholded metrics for multi-label benchmarks.

    Returns (summary_dict, per_note_r_at_y) where per_note_r_at_y enables paired bootstrap tests.
    Preserves OOV ground-truth labels in every recall denominator.
    """
    n_notes = scores.shape[0]
    if subset_keys is not None:
        active_indices = np.array([code_to_idx[k] for k in subset_keys if k in code_to_idx], dtype=np.int32)
        sub_scores = scores[:, active_indices]
        sub_map = {k: pos for pos, k in enumerate(k for k in subset_keys if k in code_to_idx)}
    else:
        sub_scores = scores
        sub_map = code_to_idx

    max_k = max(max(ks), 1)
    gold_in_vocab: list[list[int]] = []
    gold_total_counts: list[int] = []
    for gkeys in gold_keys_list:
        filtered = [k for k in gkeys if subset_keys is None or k in subset_keys]
        gold_total_counts.append(len(filtered))
        gold_in_vocab.append([sub_map[k] for k in filtered if k in sub_map])

    max_gold = max(gold_total_counts) if gold_total_counts else 1
    top_n = min(max(max_k, max_gold), sub_scores.shape[1])
    ranked = _topk_indices(sub_scores, top_n)

    per_note_r_at_y = np.zeros(n_notes, dtype=np.float64)
    per_note_valid = np.zeros(n_notes, dtype=bool)
    recall_hits = {k: 0 for k in ks}
    prec_hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    ndcg15_sum = 0.0
    total_gold = 0
    valid_notes = 0

    for i in range(n_notes):
        n_gold = gold_total_counts[i]
        if n_gold == 0:
            continue
        valid_notes += 1
        per_note_valid[i] = True
        total_gold += n_gold
        gold_set = set(gold_in_vocab[i])
        row_rank = ranked[i]

        k_y = min(n_gold, top_n)
        hits_y = sum(1 for c in row_rank[:k_y] if int(c) in gold_set)
        per_note_r_at_y[i] = hits_y / n_gold

        for k in ks:
            kk = min(k, top_n)
            hits_k = sum(1 for c in row_rank[:kk] if int(c) in gold_set)
            recall_hits[k] += hits_k
            prec_hits[k] += hits_k

        rr = 0.0
        for pos, c in enumerate(row_rank):
            if int(c) in gold_set:
                rr = 1.0 / (pos + 1)
                break
        mrr_sum += rr

        k15 = min(15, top_n)
        dcg = sum(1.0 / np.log2(pos + 2) for pos, c in enumerate(row_rank[:k15]) if int(c) in gold_set)
        idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(n_gold, k15)))
        ndcg15_sum += dcg / idcg if idcg > 0 else 0.0

    metrics: dict[str, float] = {
        "n_notes_valid": float(valid_notes),
        "n_gold_total": float(total_gold),
        "r@|y|": float(per_note_r_at_y[per_note_valid].mean()) if valid_notes else 0.0,
        "mrr": float(mrr_sum / valid_notes) if valid_notes else 0.0,
        "ndcg@15": float(ndcg15_sum / valid_notes) if valid_notes else 0.0,
    }
    for k in ks:
        metrics[f"r@{k}"] = float(recall_hits[k] / total_gold) if total_gold else 0.0
        metrics[f"p@{k}"] = float(prec_hits[k] / (valid_notes * k)) if valid_notes else 0.0

    if threshold is not None:
        metrics["micro_f1@tau_A"] = compute_threshold_micro_f1(
            sub_scores, gold_in_vocab, threshold, total_gold_count=total_gold
        )

    return metrics, per_note_r_at_y[per_note_valid]


def evaluate_answer_groups(
    scores: np.ndarray,
    label_groups_list: list[tuple[tuple[str, ...], ...]],
    code_to_idx: dict[str, int],
    ks: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate MS-DRG (single answer group of acceptable aliases) or ElCardioCC (1+ answer groups)."""
    n_notes = scores.shape[0]
    max_k = max(max(ks), 20)
    ranked = _topk_indices(scores, max_k)

    top1_hits = np.zeros(n_notes, dtype=np.float64)
    acc_hits = {k: 0 for k in ks}
    mrr_sum = 0.0
    group_r_at_g = np.zeros(n_notes, dtype=np.float64)
    valid = np.zeros(n_notes, dtype=bool)

    for i, groups in enumerate(label_groups_list):
        if not groups:
            continue
        valid[i] = True
        idx_groups = [set(code_to_idx[k] for k in grp if k in code_to_idx) for grp in groups]
        union_idx = set().union(*idx_groups) if idx_groups else set()
        row_rank = ranked[i]

        if int(row_rank[0]) in union_idx:
            top1_hits[i] = 1.0
        for k in ks:
            if any(int(c) in union_idx for c in row_rank[: min(k, ranked.shape[1])]):
                acc_hits[k] += 1

        rr = 0.0
        for pos, c in enumerate(row_rank):
            if int(c) in union_idx:
                rr = 1.0 / (pos + 1)
                break
        mrr_sum += rr

        # Fraction of distinct answer groups satisfied within top-|groups| predictions
        k_g = min(len(idx_groups), ranked.shape[1])
        top_set = set(int(c) for c in row_rank[:k_g])
        matched_groups = sum(1 for grp in idx_groups if grp and not grp.isdisjoint(top_set))
        group_r_at_g[i] = matched_groups / len(idx_groups)

    n_valid = int(valid.sum())
    metrics: dict[str, float] = {
        "n_notes_valid": float(n_valid),
        "mrr": float(mrr_sum / n_valid) if n_valid else 0.0,
        "group_r@|g|": float(group_r_at_g[valid].mean()) if n_valid else 0.0,
    }
    for k in ks:
        metrics[f"acc@{k}"] = float(acc_hits[k] / n_valid) if n_valid else 0.0
    return metrics, top1_hits[valid]


def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap CI, two-sided p-value, and standardized effect sizes (Cohen's d_z and h) for (B - A)."""
    if scores_a.shape != scores_b.shape:
        raise ValueError(f"Mismatched lengths for paired bootstrap: {scores_a.shape} vs {scores_b.shape}")
    a64 = scores_a.astype(np.float64)
    b64 = scores_b.astype(np.float64)
    diff = b64 - a64
    n = diff.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff[idx].mean(axis=1)
    mean_delta = float(diff.mean())
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    p_pos = float(np.mean(boot_means <= 0.0))
    p_neg = float(np.mean(boot_means >= 0.0))
    p_val = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    sd_diff = float(diff.std(ddof=1)) if n > 1 else 0.0
    cohens_dz = float(mean_delta / sd_diff) if sd_diff > 1e-12 else 0.0
    pa = float(np.clip(a64.mean(), 0.0, 1.0))
    pb = float(np.clip(b64.mean(), 0.0, 1.0))
    cohens_h = float(2.0 * np.arcsin(np.sqrt(pb)) - 2.0 * np.arcsin(np.sqrt(pa)))
    return BootstrapResult(
        mean_delta=mean_delta,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p_val,
        cohens_dz=cohens_dz,
        cohens_h=cohens_h,
    )
