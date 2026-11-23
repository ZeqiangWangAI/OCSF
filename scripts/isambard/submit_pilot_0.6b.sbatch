#!/bin/bash
#SBATCH --job-name=ocsf-pilot-0.6b
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
set -eu
umask 077

# Full 0.6B controlled experiment (all 154,542 Source ICD-9 train + 21,980 val + all 5 frozen non-test targets).
# Run twice with identical settings: BACKBONE=qwen-0.6b (causal control) vs BACKBONE=pplx-0.6b (diffusion bidirectional).
DATA_ROOT="${1:?Supply read-only data root (shared non-test raw/prepared text + codebooks)}"
BACKBONE="${2:?Supply backbone: pplx-0.6b or qwen-0.6b}"

CODE_DIR="${SLURM_SUBMIT_DIR}"
PYTHON="${OCSF_PYTHON:-${PROJECTDIR}/${USER}/envs/ocsf-env/bin/python}"
CACHE_DIR="${PROJECTDIR}/${USER}/cache/OCSF"
OUT_DIR="${PROJECTDIR}/${USER}/runs/OCSF/pilot-${BACKBONE}-${SLURM_JOB_ID}"

export PYTHONPATH="${CODE_DIR}/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${PROJECTDIR}/${USER}/cache/hf"
export HF_HUB_OFFLINE=1
export TMPDIR="${SCRATCHDIR}/ocsf-pilot-${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

mkdir -p "${CACHE_DIR}" "${OUT_DIR}" "${TMPDIR}"
cd "${CODE_DIR}"

"${PYTHON}" - <<'PY'
import importlib, transformers
assert hasattr(transformers, "Qwen3Model"), f"transformers={transformers.__version__} lacks Qwen3Model (>=4.51 required)"
for pkg in ("torch", "sentence_transformers", "sklearn", "scipy", "numpy"):
    importlib.import_module(pkg)
import torch
assert torch.cuda.is_available(), "CUDA GPU not available"
print(f"[preflight] OK | gpu={torch.cuda.get_device_name(0)} transformers={transformers.__version__}")
PY

srun --ntasks=1 "${PYTHON}" scripts/run_encode.py \
  --backbone "${BACKBONE}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --device cuda \
  --doc-batch-size 32 \
  --code-batch-size 512

srun --ntasks=1 "${PYTHON}" scripts/run_eval_b0_and_baselines.py \
  --backbone "${BACKBONE}" \
  --cache-dir "${CACHE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --device cuda \
  --run-cast \
  --epochs 10 \
  --batch-size 256

srun --ntasks=1 "${PYTHON}" scripts/run_train_ocsf.py \
  --backbone "${BACKBONE}" \
  --cache-dir "${CACHE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --device cuda \
  --n-channels 128 \
  --lam 0.15 \
  --epochs 12 \
  --batch-size 128

srun --ntasks=1 "${PYTHON}" scripts/run_train_flow.py \
  --backbone "${BACKBONE}" \
  --cache-dir "${CACHE_DIR}" \
  --ocsf-dir "${OUT_DIR}/${BACKBONE}/ocsf_R128_lam0.15" \
  --data-root "${DATA_ROOT}" \
  --device cuda \
  --lam 0.15 \
  --epochs 12 \
  --batch-size 128
