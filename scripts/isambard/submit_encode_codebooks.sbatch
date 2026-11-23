#!/bin/bash
#SBATCH --job-name=ocsf-cb-enc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -eu
umask 077

# Fast 1-GPU public codebook encoder (all 6 codebooks + 7 DRG prototypes, zero train/test docs).
DATA_ROOT="${1:?Supply read-only data root}"
BACKBONE="${2:?Supply backbone: pplx-4b or qwen-4b}"
CODE_BS="${3:-256}"

CODE_DIR="${SLURM_SUBMIT_DIR}"
PYTHON="${OCSF_PYTHON:-${PROJECTDIR}/${USER}/envs/ocsf-env/bin/python}"
CACHE_DIR="${PROJECTDIR}/${USER}/cache/OCSF"

export PYTHONPATH="${CODE_DIR}/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${PROJECTDIR}/${USER}/cache/hf"
export HF_HUB_OFFLINE=1
export TMPDIR="${SCRATCHDIR}/ocsf-cb-${SLURM_JOB_ID}"

mkdir -p "${CACHE_DIR}" "${TMPDIR}"
cd "${CODE_DIR}"

"${PYTHON}" scripts/run_encode.py \
  --backbone "${BACKBONE}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --device cuda \
  --code-batch-size "${CODE_BS}" \
  --codebooks-only
