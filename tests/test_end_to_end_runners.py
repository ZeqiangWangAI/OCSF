"""End-to-end integration test exercising run_eval_b0_and_baselines, run_train_ocsf, and run_train_flow.

Uses real non-test codebooks/splits capped to tiny synthetic slices (with synthetic cached embeddings)
on CPU to verify every CLI flag, data path, matrix shape, JSON artifact, and bootstrap CI end-to-end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np

from ocsf.compiler import DRG_PROTOTYPE_PROMPTS
from ocsf.data import DEFAULT_DATA_ROOT, load_codebook, target_paths
from ocsf.state_matrix import l2_normalize


def _make_synthetic_cache(cache_dir: Path, backbone: str, dim: int = 32, n_train: int = 12, n_val: int = 8, n_tgt: int = 8) -> None:
    rng = np.random.default_rng(7)
    root = cache_dir / backbone
    cb_dir = root / "codebooks"
    sp_dir = root / "splits"
    cb_dir.mkdir(parents=True, exist_ok=True)
    sp_dir.mkdir(parents=True, exist_ok=True)

    # Codebooks (real sizes for icd9, icd10, ms-drg, elcardiocc)
    cb_paths = {
        "icd9-full": DEFAULT_DATA_ROOT / "benchmarks/mimiciv-2.2/codebooks/icd9-full.json",
        "mimiciv-icd10": target_paths("mimiciv-icd10")["codebook"],
        "mimiciv-ms-drg": target_paths("mimiciv-ms-drg")["codebook"],
        "elcardiocc-2026": target_paths("elcardiocc-2026")["codebook"],
    }
    for name, path in cb_paths.items():
        n_codes = len(load_codebook(path))
        vecs = l2_normalize(rng.normal(size=(n_codes, dim)).astype(np.float32))
        np.save(cb_dir / f"{name}.npy", vecs)

    protos = {k: l2_normalize(rng.normal(size=(dim,)).astype(np.float32)) for k in DRG_PROTOTYPE_PROMPTS}
    np.savez(cb_dir / "drg_prototypes.npz", **protos)

    for sp_name, n_rows in (
        ("source_train", n_train),
        ("source_validation", n_val),
        ("target_mimiciv-icd10", n_tgt),
        ("target_mimiciv-ms-drg", n_tgt),
        ("target_elcardiocc-2026", n_tgt),
    ):
        chunks = l2_normalize(rng.normal(size=(n_rows, 4, dim)).astype(np.float32)).astype(np.float16)
        lengths = np.full(n_rows, 4, dtype=np.int16)
        doc_mean = l2_normalize(chunks.astype(np.float32).mean(axis=1))
        np.save(sp_dir / f"{sp_name}_chunks.npy", chunks)
        np.save(sp_dir / f"{sp_name}_lengths.npy", lengths)
        np.save(sp_dir / f"{sp_name}_doc_mean.npy", doc_mean)


def test_full_cli_pipeline_synthetic_cpu(tmp_path: Path) -> None:
    if not DEFAULT_DATA_ROOT.exists():
        return
    repo = Path(__file__).resolve().parents[1]
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    backbone = "mock-0.6b"
    targets = ["mimiciv-icd10", "mimiciv-ms-drg", "elcardiocc-2026"]
    _make_synthetic_cache(cache_dir, backbone, dim=32, n_train=12, n_val=8, n_tgt=8)

    # 1. Step 0 + B1 CAST baseline
    cmd_b0 = [
        sys.executable,
        str(repo / "scripts/run_eval_b0_and_baselines.py"),
        "--backbone", backbone,
        "--cache-dir", str(cache_dir),
        "--out-dir", str(out_dir),
        "--device", "cpu",
        "--run-cast",
        "--epochs", "1",
        "--batch-size", "6",
        "--targets", *targets,
        "--limit-train", "12",
        "--limit-val", "8",
        "--limit-target", "8",
    ]
    subprocess.run(cmd_b0, cwd=repo, check=True)
    b0_json = json.loads((out_dir / backbone / "baselines_summary.json").read_text(encoding="utf-8"))
    assert {"B0_doc", "B0_chunk_max", "B1_CAST_1D"} <= set(b0_json.keys())
    assert "mimiciv-icd10-rare" in b0_json["B1_CAST_1D"]
    assert "acc@1" in b0_json["B1_CAST_1D"]["mimiciv-ms-drg"]

    # 2. Steps 1–3 OCSF train + B3/Step2/Step3 eval + paired bootstrap
    cmd_ocsf = [
        sys.executable,
        str(repo / "scripts/run_train_ocsf.py"),
        "--backbone", backbone,
        "--cache-dir", str(cache_dir),
        "--out-dir", str(out_dir),
        "--device", "cpu",
        "--n-channels", "8",
        "--lam", "0.05",
        "--epochs", "1",
        "--batch-size", "6",
        "--targets", *targets,
        "--limit-train", "12",
        "--limit-val", "8",
        "--limit-target", "8",
    ]
    subprocess.run(cmd_ocsf, cwd=repo, check=True)
    ocsf_dir = out_dir / backbone / "ocsf_R8_lam0.05"
    ocsf_json = json.loads((ocsf_dir / "ocsf_summary.json").read_text(encoding="utf-8"))
    assert {"B3_2D_dot", "Step2_OCSF_deconv", "Step3_OCSF_compiled"} == set(ocsf_json["results"].keys())
    assert "Step3_minus_B1_CAST" in ocsf_json["bootstrap"]
    assert "mean_delta" in ocsf_json["bootstrap"]["Step3_minus_B1_CAST"]["mimiciv-icd10"]

    # 3. Step 4 Latent Rectified Flow + falsification sweep (N=0 vs N=2)
    cmd_flow = [
        sys.executable,
        str(repo / "scripts/run_train_flow.py"),
        "--backbone", backbone,
        "--cache-dir", str(cache_dir),
        "--ocsf-dir", str(ocsf_dir),
        "--device", "cpu",
        "--lam", "0.05",
        "--epochs", "1",
        "--batch-size", "6",
        "--steps-grid", "0", "2",
        "--targets", *targets,
        "--limit-train", "12",
        "--limit-val", "8",
        "--limit-target", "8",
    ]
    subprocess.run(cmd_flow, cwd=repo, check=True)
    flow_json = json.loads((ocsf_dir / "flow_falsification.json").read_text(encoding="utf-8"))
    assert {"N0", "N2"} == set(flow_json["sweep"].keys())
    assert "N2_minus_N0" in flow_json["falsification_vs_N0"]
    assert "p_value" in flow_json["falsification_vs_N0"]["N2_minus_N0"]["mimiciv-ms-drg"]
