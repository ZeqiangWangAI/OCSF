#!/bin/bash
#SBATCH --job-name=ocsf-e2e-4gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -eu
umask 077

# End-to-End Token-Level Bidirectional Diffusion OCSF on 4x NVIDIA GH200 (1 Isambard node).
# Usage:
#   sbatch scripts/isambard/submit_e2e_4gpu.sbatch <DATA_ROOT> <BACKBONE> [EXTRA_ARGS...]
# Example (full OCSF-E2E):
#   sbatch scripts/isambard/submit_e2e_4gpu.sbatch "$DATA_ROOT" pplx-0.6b --n-channels 512 --epochs 8
# Example (closed-vocab PLM-ICD/LAAT reference):
#   sbatch scripts/isambard/submit_e2e_4gpu.sbatch "$DATA_ROOT" pplx-0.6b --closed-vocab-head --epochs 8
DATA_ROOT="${1:?Supply read-only data root}"
BACKBONE="${2:?Supply backbone: pplx-0.6b or qwen-0.6b}"
shift 2

CODE_DIR="${SLURM_SUBMIT_DIR}"
PYTHON="${OCSF_PYTHON:-${PROJECTDIR}/${USER}/envs/ocsf-env/bin/python}"
TORCHRUN="${OCSF_TORCHRUN:-${PROJECTDIR}/${USER}/envs/ocsf-env/bin/torchrun}"
CACHE_DIR="${PROJECTDIR}/${USER}/cache/OCSF"
OUT_DIR="${PROJECTDIR}/${USER}/runs/OCSF/e2e-${BACKBONE}-${SLURM_JOB_ID}"

export PYTHONPATH="${CODE_DIR}/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${PROJECTDIR}/${USER}/cache/hf"
export HF_HUB_OFFLINE=1
export TMPDIR="${SCRATCHDIR}/ocsf-e2e-${SLURM_JOB_ID}"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

mkdir -p "${OUT_DIR}" "${TMPDIR}"
cd "${CODE_DIR}"

NPROC="${SLURM_GPUS_ON_NODE:-4}"
MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))

"${TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  scripts/run_train_e2e_ocsf.py \
  --backbone "${BACKBONE}" \
  --cache-dir "${CACHE_DIR}" \
  --out-dir "${OUT_DIR}" \
  --data-root "${DATA_ROOT}" \
  "$@"
