#!/bin/bash
#SBATCH --job-name=ocsf-v3-cached
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -eu
umask 077

# Track 1 (V3 Cached-Chunk Convergence + Fair Strong 1D Baselines):
# Reuses already-encoded 154k-note caches in $CACHE_DIR (0 encoding cost!).
DATA_ROOT="${1:?Supply read-only data root}"
BACKBONE="${2:?Supply backbone: pplx-0.6b or qwen-0.6b}"
N_CH="${3:-256}"
EPOCHS="${4:-25}"

CODE_DIR="${SLURM_SUBMIT_DIR}"
PYTHON="${OCSF_PYTHON:-${PROJECTDIR}/${USER}/envs/ocsf-env/bin/python}"
CACHE_DIR="${PROJECTDIR}/${USER}/cache/OCSF"
OUT_DIR="${PROJECTDIR}/${USER}/runs/OCSF/v3-cached-${BACKBONE}-R${N_CH}-${SLURM_JOB_ID}"

export PYTHONPATH="${CODE_DIR}/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TMPDIR="${SCRATCHDIR}/ocsf-v3-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=8

mkdir -p "${OUT_DIR}" "${TMPDIR}"
cd "${CODE_DIR}"

srun --ntasks=1 "${PYTHON}" scripts/run_eval_b0_and_baselines.py \
  --backbone "${BACKBONE}" \
  --cache-dir "${CACHE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --device cuda \
  --run-cast \
  --epochs 15 \
  --batch-size 256 \
  --lr 2e-3

srun --ntasks=1 "${PYTHON}" scripts/run_train_ocsf.py \
  --backbone "${BACKBONE}" \
  --cache-dir "${CACHE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --data-root "${DATA_ROOT}" \
  --device cuda \
  --n-channels "${N_CH}" \
  --n-channel-layers 3 \
  --lam 0.15 \
  --epochs "${EPOCHS}" \
  --batch-size 128 \
  --lr 1e-3 \
  --alpha-code 1.0 \
  --alpha-state 0.25 \
  --beta-anchor 0.01
