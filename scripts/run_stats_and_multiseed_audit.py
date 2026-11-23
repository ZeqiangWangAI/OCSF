"""Compute exact paired effect sizes (Cohen's d_z, Cohen's h) and 5-seed channel-partitioning / noise ablations.

Reads saved per-note evaluation arrays and non-test codebook embeddings to produce
`results/effect_sizes_and_multiseed.json` with zero access to raw test texts or labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ocsf.data import DEFAULT_DATA_ROOT, iter_source_icd9, load_codebook, target_paths
from ocsf.metrics import evaluate_multilabel, paired_bootstrap_ci
from ocsf.state_matrix import (
    ChannelBank,
    compile_deconvolution_operator,
    encode_ground_truth_state,
    l2_normalize,
)


def _boot_dict(a: np.ndarray, b: np.ndarray, n_resamples: int = 2000, seed: int = 42) -> dict:
    res = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
    return {
        "n_notes": int(a.shape[0]),
        "mean_a": round(float(a.mean()), 6),
        "mean_b": round(float(b.mean()), 6),
        "mean_delta": round(res.mean_delta, 6),
        "ci_low": round(res.ci_low, 6),
        "ci_high": round(res.ci_high, 6),
        "cohens_dz": round(res.cohens_dz, 4),
        "cohens_h": round(res.cohens_h, 4),
        "p_value": round(res.p_value, 6),
    }


def compute_sealed_effect_sizes(sealed_dir: Path) -> dict:
    """Compute exact paired bootstrap CIs, p-values, Cohen's d_z, and Cohen's h across sealed-test slices."""
    def sl(tag: str, prefix: str, bench: str) -> np.ndarray:
        return np.load(sealed_dir / tag / f"{prefix}__{bench}.npy")

    slices_all = [
        "source_icd9_test",
        "source_icd9_test_frequent",
        "mimiciv-icd10_test",
        "mimiciv-icd10_test_cm",
        "mimiciv-icd10_test_pcs",
        "mimiciv-icd10_test_frequent",
        "mimiciv-icd10_test_rare",
        "mimiciv-ms-drg_test",
        "codiesp_test",
    ]

    def vprefix(bench: str) -> str:
        return "Step3_OCSF_compiled" if bench in {"mimiciv-ms-drg_test", "codiesp_test"} else "Step2_OCSF_deconv"

    out: dict[str, dict] = {}

    # 1. In-domain vs Closed-Vocab LAAT
    for tag_ocsf, key_name in [
        ("pplx06_frz50", "pplx06_frz50_vs_closed_laat"),
        ("pplx06_frz0", "pplx06_frz0_vs_closed_laat"),
    ]:
        out[key_name] = {}
        for b in ("source_icd9_test", "source_icd9_test_frequent"):
            out[key_name][b] = _boot_dict(
                sl("closed_laat_pplx06", "closed", b),
                sl(tag_ocsf, "Step2_OCSF_deconv", b),
            )

    # 2. Cross-model pairs across all 9 slices
    pair_specs = [
        ("pplx06_frz50_vs_qwen06_frz50", "qwen06_frz50", "pplx06_frz50"),
        ("pplx06_frz0_vs_pplx06_frz50", "pplx06_frz50", "pplx06_frz0"),
        ("pplx06_r512_vs_r1", "pplx06_r1", "pplx06_frz50"),
        ("pplx06_r512_vs_r128", "pplx06_r128", "pplx06_frz50"),
        ("pplx4b_6ep_vs_qwen4b_6ep", "qwen4b_6ep", "pplx4b_6ep"),
        ("pplx4b_smooth_vs_qwen4b_6ep", "qwen4b_6ep", "pplx4b_smooth"),
        ("pplx4b_smooth_vs_pplx06_frz50", "pplx06_frz50", "pplx4b_smooth"),
    ]
    for name, tag_a, tag_b in pair_specs:
        out[name] = {}
        for b in slices_all:
            vp = vprefix(b)
            out[name][b] = _boot_dict(sl(tag_a, vp, b), sl(tag_b, vp, b))

    # 3. Within-checkpoint operator ablations (SelfNorm vs Dot, SelfNorm vs Unnorm, Step3 vs Step2)
    for tag in ("pplx06_frz50", "pplx4b_smooth"):
        out[f"{tag}_selfnorm_vs_dot"] = {}
        out[f"{tag}_selfnorm_vs_unnorm"] = {}
        for b in slices_all:
            out[f"{tag}_selfnorm_vs_dot"][b] = _boot_dict(
                sl(tag, "B3_2D_dot", b),
                sl(tag, "Step2_OCSF_deconv", b),
            )
            unnorm_path = sealed_dir / tag / f"Step2_unnorm_deconv__{b}.npy"
            if unnorm_path.exists():
                out[f"{tag}_selfnorm_vs_unnorm"][b] = _boot_dict(
                    np.load(unnorm_path),
                    sl(tag, "Step2_OCSF_deconv", b),
                )
        out[f"{tag}_compiled_vs_deconv"] = {}
        for b in ("mimiciv-ms-drg_test", "codiesp_test"):
            out[f"{tag}_compiled_vs_deconv"][b] = _boot_dict(
                sl(tag, "Step2_OCSF_deconv", b),
                sl(tag, "Step3_OCSF_compiled", b),
            )

    return out


def compute_dev_effect_sizes(dev_dir: Path) -> dict:
    """Compute exact paired effect sizes on non-test dev benchmarks (including ElCardioCC & CHIP-CDN)."""
    def dl(tag: str, variant: str, bench: str) -> np.ndarray:
        return np.load(dev_dir / tag / variant / f"{bench}.npy")

    dev_benches = [
        "source_icd9_val",
        "source_icd9_val_frequent",
        "mimiciv-icd10",
        "mimiciv-icd10-rare",
        "mimiciv-ms-drg",
        "codiesp",
        "elcardiocc-2026",
        "chip-cdn",
    ]
    out: dict[str, dict] = {}
    for name, tag_a, tag_b in [
        ("pplx06_vs_qwen06_compiled", "qwen06_r512", "pplx06_r512"),
        ("pplx4b_6ep_vs_qwen4b_6ep_compiled", "qwen4b_6ep", "pplx4b_6ep"),
        ("pplx4b_smooth_vs_qwen4b_6ep_compiled", "qwen4b_6ep", "pplx4b_smooth"),
        ("pplx06_r512_vs_r1_compiled", "pplx06_r1", "pplx06_r512"),
    ]:
        out[name] = {}
        for b in dev_benches:
            pa = dev_dir / tag_a / "Step3_OCSF_compiled" / f"{b}.npy"
            pb = dev_dir / tag_b / "Step3_OCSF_compiled" / f"{b}.npy"
            if pa.exists() and pb.exists():
                out[name][b] = _boot_dict(np.load(pa), np.load(pb))

    # Step3 vs Step2 and SelfNorm vs Dot on dev
    out["pplx06_compiled_vs_deconv"] = {}
    out["pplx06_selfnorm_vs_dot"] = {}
    for b in dev_benches:
        p_dot = dev_dir / "pplx06_r512" / "B3_2D_dot" / f"{b}.npy"
        p_s2 = dev_dir / "pplx06_r512" / "Step2_OCSF_deconv" / f"{b}.npy"
        p_s3 = dev_dir / "pplx06_r512" / "Step3_OCSF_compiled" / f"{b}.npy"
        if p_dot.exists() and p_s2.exists():
            out["pplx06_selfnorm_vs_dot"][b] = _boot_dict(np.load(p_dot), np.load(p_s2))
        if p_s2.exists() and p_s3.exists():
            out["pplx06_compiled_vs_deconv"][b] = _boot_dict(np.load(p_s2), np.load(p_s3))
    return out


def run_multiseed_partition_and_noise(
    cb_dir: Path,
    data_root: Path,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    n_channels: int = 512,
) -> dict:
    """Run 5-seed ablation for Spherical k-Means, Hierarchical Ward, Random Anchors, and codebook noise."""
    t0 = time.perf_counter()
    z_a = l2_normalize(np.load(cb_dir / "icd9-full.npy").astype(np.float32))
    z_10 = l2_normalize(np.load(cb_dir / "mimiciv-icd10.npy").astype(np.float32))

    src_val = list(iter_source_icd9("validation", data_root, limit=1000))
    src_cb = load_codebook(data_root / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    s_k2i = {e.key: i for i, e in enumerate(src_cb)}
    val_gold_arr = [np.array([s_k2i[k] for k in ex.labels if k in s_k2i], dtype=np.int32) for ex in src_val]
    val_gold_keys = [ex.labels for ex in src_val]

    cb10 = load_codebook(target_paths("mimiciv-icd10", data_root)["codebook"])
    k2i10 = {e.key: i for i, e in enumerate(cb10)}

    # Hierarchical Ward clustering is deterministic on z_a; fit once and evaluate across 5 noise/sample seeds
    ward = AgglomerativeClustering(n_clusters=n_channels, linkage="ward")
    w_lbl = ward.fit_predict(z_a)
    w_cent = np.zeros((n_channels, z_a.shape[1]), dtype=np.float32)
    for r in range(n_channels):
        idx = np.where(w_lbl == r)[0]
        w_cent[r] = z_a[idx].mean(axis=0) if len(idx) else z_a[r]
    ward_bank = ChannelBank(anchors=l2_normalize(w_cent))

    per_strategy: dict[str, list[dict]] = {
        "spherical_kmeans": [],
        "hierarchical_ward": [],
        "random_anchors": [],
    }
    noise_runs: dict[str, dict[str, list[float]]] = {
        "sigma_0.10": {"dot": [], "unnorm": [], "selfnorm": []},
        "sigma_0.20": {"dot": [], "unnorm": [], "selfnorm": []},
    }

    for sd in seeds:
        rng = np.random.default_rng(sd)
        icd10_gold_arr = [rng.choice(len(z_10), size=13, replace=False).astype(np.int32) for _ in range(500)]
        icd10_gold_keys = [tuple(cb10[i].key for i in g) for g in icd10_gold_arr]

        skm_bank = ChannelBank.fit(z_a, n_channels=n_channels, seed=sd)
        rand_idx = rng.choice(len(z_a), size=n_channels, replace=False)
        rnd_bank = ChannelBank(anchors=l2_normalize(z_a[rand_idx].copy()))

        for bname, b_obj in [
            ("spherical_kmeans", skm_bank),
            ("hierarchical_ward", ward_bank),
            ("random_anchors", rnd_bank),
        ]:
            noise_rng = np.random.default_rng(sd + 1000)
            op_a = compile_deconvolution_operator(z_a, b_obj, lam=0.15, self_normalize=True)
            u_a = encode_ground_truth_state(val_gold_arr, z_a, op_a.assign, n_channels)
            u_a_noisy = u_a + noise_rng.normal(0.0, 0.15, size=u_a.shape).astype(np.float32)
            m_a, _ = evaluate_multilabel(op_a.score_batch(u_a_noisy), val_gold_keys, s_k2i, threshold=0.5)

            op_10 = compile_deconvolution_operator(z_10, b_obj, lam=0.15, self_normalize=True)
            u_10 = encode_ground_truth_state(icd10_gold_arr, z_10, op_10.assign, n_channels)
            u_10_noisy = u_10 + noise_rng.normal(0.0, 0.15, size=u_10.shape).astype(np.float32)
            m_10, _ = evaluate_multilabel(op_10.score_batch(u_10_noisy), icd10_gold_keys, k2i10, threshold=0.5)

            counts = np.bincount(op_10.assign, minlength=n_channels)
            per_strategy[bname].append({
                "seed": sd,
                "icd9_r@|y|": float(m_a["r@|y|"]),
                "icd10_r@|y|": float(m_10["r@|y|"]),
                "icd10_max_channel_size": int(counts.max()),
                "icd10_empty_channels": int(np.sum(counts == 0)),
            })

            if bname == "spherical_kmeans":
                # 5-seed target codebook embedding perturbation stress test on real ICD-9 val oracle states
                for sigma, skey in ((0.10, "sigma_0.10"), (0.20, "sigma_0.20")):
                    z_pert = l2_normalize(z_a + noise_rng.normal(0.0, sigma, size=z_a.shape).astype(np.float32))
                    asg_p = skm_bank.assign_hard(z_pert)
                    # Dot readout
                    dot_scores = np.empty((len(val_gold_arr), len(z_pert)), dtype=np.float32)
                    for r_id in range(n_channels):
                        c_idx = np.where(asg_p == r_id)[0]
                        if c_idx.size:
                            dot_scores[:, c_idx] = u_a_noisy[:, r_id, :] @ z_pert[c_idx].T
                    op_un = compile_deconvolution_operator(z_pert, skm_bank, lam=0.15, self_normalize=False)
                    op_sn = compile_deconvolution_operator(z_pert, skm_bank, lam=0.15, self_normalize=True)
                    m_dot, _ = evaluate_multilabel(dot_scores, val_gold_keys, s_k2i, threshold=0.5)
                    m_un, _ = evaluate_multilabel(op_un.score_batch(u_a_noisy), val_gold_keys, s_k2i, threshold=0.5)
                    m_sn, _ = evaluate_multilabel(op_sn.score_batch(u_a_noisy), val_gold_keys, s_k2i, threshold=0.5)
                    noise_runs[skey]["dot"].append(float(m_dot["r@|y|"]))
                    noise_runs[skey]["unnorm"].append(float(m_un["r@|y|"]))
                    noise_runs[skey]["selfnorm"].append(float(m_sn["r@|y|"]))

    def _agg(rows: list[dict]) -> dict:
        icd9_v = np.array([r["icd9_r@|y|"] for r in rows])
        icd10_v = np.array([r["icd10_r@|y|"] for r in rows])
        max_c = np.array([r["icd10_max_channel_size"] for r in rows], dtype=np.float64)
        emp_c = np.array([r["icd10_empty_channels"] for r in rows], dtype=np.float64)
        return {
            "n_seeds": len(rows),
            "seeds": [r["seed"] for r in rows],
            "icd9_r@|y|_mean": round(float(icd9_v.mean()), 4),
            "icd9_r@|y|_sd": round(float(icd9_v.std(ddof=1)), 4),
            "icd10_r@|y|_mean": round(float(icd10_v.mean()), 4),
            "icd10_r@|y|_sd": round(float(icd10_v.std(ddof=1)), 4),
            "icd10_max_channel_mean": round(float(max_c.mean()), 1),
            "icd10_max_channel_sd": round(float(max_c.std(ddof=1)), 1),
            "icd10_empty_channels_mean": round(float(emp_c.mean()), 1),
            "per_seed": rows,
        }

    noise_summary = {
        skey: {
            op_k: {
                "mean": round(float(np.mean(vals)), 4),
                "sd": round(float(np.std(vals, ddof=1)), 4),
            }
            for op_k, vals in op_dict.items()
        }
        for skey, op_dict in noise_runs.items()
    }

    return {
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "partition_5seed": {k: _agg(v) for k, v in per_strategy.items()},
        "perturbation_5seed_icd9_oracle": noise_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-per-note", type=Path, required=True)
    parser.add_argument("--dev-per-note", type=Path, required=True)
    parser.add_argument("--cb-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    print("[1/3] Computing exact paired effect sizes on sealed test splits...", flush=True)
    sealed_fx = compute_sealed_effect_sizes(args.sealed_per_note)
    print("[2/3] Computing exact paired effect sizes on development splits...", flush=True)
    dev_fx = compute_dev_effect_sizes(args.dev_per_note)
    print("[3/3] Running 5-seed channel-partitioning & noise ablations...", flush=True)
    multiseed = run_multiseed_partition_and_noise(args.cb_dir, args.data_root)

    payload = {
        "sealed_test_effect_sizes": sealed_fx,
        "dev_effect_sizes": dev_fx,
        "multiseed_ablations": multiseed,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
