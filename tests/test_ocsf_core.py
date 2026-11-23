"""Unit tests for OCSF data guards, Self-Normalized Tikhonov Deconvolution, compiler, and metrics."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from ocsf.baselines import CastProjector, score_b0_zeroshot_retrieval
from ocsf.compiler import (
    AlignmentStats,
    apply_or_groups,
    compile_drg_operator,
    compile_target_operator,
    parse_drg_description,
)
from ocsf.data import CodebookEntry, assert_non_test_path, load_codebook, target_paths
from ocsf.metrics import (
    calibrate_source_threshold,
    evaluate_answer_groups,
    evaluate_multilabel,
    paired_bootstrap_ci,
)
from ocsf.state_matrix import ChannelBank, compile_deconvolution_operator, encode_ground_truth_state


def test_sealed_test_path_blocked() -> None:
    with pytest.raises(RuntimeError):
        assert_non_test_path(Path("/data/clinical-benchmarks/sealed-clinical-tests/icd10/test.jsonl"))
    with pytest.raises(RuntimeError):
        assert_non_test_path(Path("/tmp/some/test.jsonl"))
    assert_non_test_path(Path("/data/clinical-benchmarks/source-text/validation.jsonl"))


def test_self_normalized_tikhonov_exact_unit_self_response_and_deconvolution() -> None:
    rng = np.random.default_rng(7)
    n_codes, dim, n_channels = 240, 64, 8
    z = rng.normal(size=(n_codes, dim)).astype(np.float32)
    # Inject dense sibling cluster to stress-test cross-talk suppression
    base = rng.normal(size=(1, dim)).astype(np.float32)
    z[:12] = base + 0.15 * rng.normal(size=(12, dim)).astype(np.float32)

    bank = ChannelBank.fit(z, n_channels=n_channels, seed=0)
    op = compile_deconvolution_operator(z, bank=bank, lam=1e-3, self_normalize=True)

    # Every single code must have exact unit self-response w_i^T z_i == 1.0
    z_unit = z / np.linalg.norm(z, axis=1, keepdims=True)
    for r in range(n_channels):
        idx = op.chan_indices[r]
        if idx.size == 0:
            continue
        self_resp = np.sum(op.chan_weights[r] * z_unit[idx], axis=1)
        np.testing.assert_allclose(self_resp, 1.0, atol=1e-4)

    # Multi-label state recovery test (active codes must score ~1.0, siblings near ~0.0)
    gold_sets = [np.array([0, 25, 110, 200]), np.array([1, 2, 77, 150])]
    u0 = encode_ground_truth_state(gold_sets, z, op.assign, n_channels)
    scores = op.score_batch(u0)
    for i, g in enumerate(gold_sets):
        top = set(np.argsort(-scores[i])[: len(g)].tolist())
        assert top == set(g.tolist())


def test_alignment_preserves_narrow_chapter_and_shifts_broad_codebook() -> None:
    rng = np.random.default_rng(11)
    z_en = rng.normal(loc=0.5, scale=0.3, size=(1200, 32)).astype(np.float32)
    stats = AlignmentStats.fit(z_en, use_zca=False)

    # Broad Spanish-like shifted codebook (>=1000) gets mean aligned back to source_mean
    z_es = z_en + np.full((1, 32), -2.0, dtype=np.float32)
    aligned_broad = stats.align_codebook(z_es, centroid_shift_min_codes=1000)
    assert np.linalg.norm(aligned_broad.mean(axis=0) - stats.source_mean) < 0.25

    # Narrow ElCardioCC-like codebook (<1000) is NOT mean-subtracted
    z_narrow = np.ones((115, 32), dtype=np.float32) + 0.05 * rng.normal(size=(115, 32)).astype(np.float32)
    aligned_narrow = stats.align_codebook(z_narrow, centroid_shift_min_codes=1000)
    assert float(aligned_narrow.mean(axis=0).mean()) > 0.15


def test_drg_parser_and_compositional_operator() -> None:
    base, tier = parse_drg_description("HEART FAILURE AND SHOCK WITH MCC")
    assert base == "HEART FAILURE AND SHOCK"
    assert tier == "WITH_MCC"
    base2, tier2 = parse_drg_description("SIMPLE PNEUMONIA WITHOUT CC/MCC")
    assert tier2 == "WITHOUT_CC_MCC"

    rng = np.random.default_rng(3)
    entries = [
        CodebookEntry("ms_drg_semantic:291:v37", "ms_drg", "291", "HEART FAILURE AND SHOCK WITH MCC"),
        CodebookEntry("ms_drg_semantic:293:v37", "ms_drg", "293", "HEART FAILURE AND SHOCK WITHOUT CC/MCC"),
    ]
    z = rng.normal(size=(2, 32)).astype(np.float32)
    protos = {
        "WITH_MCC": rng.normal(size=(32,)).astype(np.float32),
        "WITHOUT_CC_MCC": rng.normal(size=(32,)).astype(np.float32),
        "MEDICAL_M": rng.normal(size=(32,)).astype(np.float32),
    }
    bank = ChannelBank.fit(z, n_channels=2, seed=0)
    op = compile_drg_operator(entries, z, protos, bank=bank, lam=0.05)
    states = rng.normal(size=(4, 2, 32)).astype(np.float32)
    scores = op.score_batch(states)
    assert scores.shape == (4, 2)
    assert np.isfinite(scores).all()


def test_metrics_oov_preservation_answer_groups_and_bootstrap() -> None:
    scores = np.array([[0.9, 0.2, 0.1], [0.1, 0.8, 0.7]], dtype=np.float32)
    code_to_idx = {"A": 0, "B": 1, "C": 2}
    # Second note has an OOV label "OOV_CODE" -> recall denominator must be 2, not 1!
    gold = [("A",), ("B", "OOV_CODE")]
    tau, _ = calibrate_source_threshold(scores, [[0], [1]])
    metrics, per_note = evaluate_multilabel(scores, gold, code_to_idx, threshold=tau, ks=(1, 2))
    assert metrics["n_gold_total"] == 3.0
    assert pytest.approx(metrics["r@|y|"], abs=1e-6) == 0.75  # (1.0 + 0.5) / 2
    assert per_note.shape == (2,)

    grp_metrics, hits = evaluate_answer_groups(
        scores,
        label_groups_list=[(("A", "ALIAS_A"),), (("C",),)],
        code_to_idx=code_to_idx,
        ks=(1, 2),
    )
    assert grp_metrics["acc@1"] == 0.5
    assert grp_metrics["acc@2"] == 1.0
    assert hits.tolist() == [1.0, 0.0]

    rng_b = np.random.default_rng(9)
    a_arr = np.clip(0.3 + 0.1 * rng_b.normal(size=80), 0.0, 1.0)
    b_arr = np.clip(a_arr + 0.2 + 0.05 * rng_b.normal(size=80), 0.0, 1.0)
    boot = paired_bootstrap_ci(a_arr, b_arr, n_resamples=200, seed=0)
    assert boot.mean_delta > 0.15
    assert boot.cohens_dz > 2.0
    assert boot.cohens_h > 0.3
    assert boot.p_value < 0.01


def test_cast_and_b0_baselines_shapes() -> None:
    rng = np.random.default_rng(5)
    w_a = rng.normal(size=(50, 16)).astype(np.float32)
    z_a = rng.normal(size=(50, 24)).astype(np.float32)
    z_b = rng.normal(size=(80, 24)).astype(np.float32)
    docs = rng.normal(size=(10, 16)).astype(np.float32)

    cast = CastProjector.fit_from_classifier_weights(w_a, z_a, lam=0.1)
    s_cast = cast.score_target(docs, z_b)
    assert s_cast.shape == (10, 80)

    s_b0 = score_b0_zeroshot_retrieval(rng.normal(size=(10, 24)).astype(np.float32), z_b)
    assert s_b0.shape == (10, 80)


def test_real_non_test_codebooks_and_or_groups_readable() -> None:
    paths = target_paths("codiesp")
    if not paths["groups"].exists():
        pytest.skip("External read-only data volume not mounted")
    cb = load_codebook(paths["codebook"])
    assert len(cb) == 185458
    code_to_idx = {e.key: i for i, e in enumerate(cb)}
    dummy = np.zeros((2, len(cb)), dtype=np.float32)
    # Give one leaf member a high score and confirm its OR parent inherits it in-place
    import json as _json
    first_grp = _json.loads(paths["groups"].read_text(encoding="utf-8"))[0]
    member_col = code_to_idx[first_grp["members"][0]]
    parent_col = code_to_idx[first_grp["key"]]
    dummy[0, member_col] = 0.85
    expanded, new_map = apply_or_groups(dummy, code_to_idx, paths["groups"])
    assert expanded.shape == (2, len(cb))
    assert len(new_map) == len(cb)
    assert pytest.approx(float(expanded[0, parent_col]), abs=1e-6) == 0.85
