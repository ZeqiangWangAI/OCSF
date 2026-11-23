#!/bin/bash
#SBATCH --job-name=ocsf-eval-abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -eu
umask 077

# Fast 1-GPU zero-retrain compiler & LoRA-scaling ablation on an existing OCSF-E2E checkpoint.
DATA_ROOT="${1:?Supply data root}"
BACKBONE="${2:?Supply backbone}"
RUN_DIR="${3:?Supply run dir containing best_e2e_trainable.pt}"
OUT_DIR="${4:?Supply output dir}"

CODE_DIR="${SLURM_SUBMIT_DIR}"
PYTHON="${OCSF_PYTHON:-${PROJECTDIR}/${USER}/envs/ocsf-env/bin/python}"
CACHE_DIR="${PROJECTDIR}/${USER}/cache/OCSF"

export PYTHONPATH="${CODE_DIR}/src:${CODE_DIR}/scripts"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${PROJECTDIR}/${USER}/cache/hf"
export HF_HUB_OFFLINE=1

mkdir -p "${OUT_DIR}"
cd "${CODE_DIR}"

for LSCALE in 1.0 0.5; do
  "${PYTHON}" scripts/run_eval_e2e_checkpoint.py \
    --backbone "${BACKBONE}" \
    --run-dir "${RUN_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --data-root "${DATA_ROOT}" \
    --lora-scale "${LSCALE}" \
    --out-json "${OUT_DIR}/ablation_${BACKBONE}_lscale${LSCALE}.json"
done
