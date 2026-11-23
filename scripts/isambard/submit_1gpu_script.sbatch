#!/bin/bash
#SBATCH --job-name=ocsf-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
set -eu
umask 077

SCRIPT="${1:?Supply script path}"
shift

CODE_DIR="${SLURM_SUBMIT_DIR}"
PYTHON="${OCSF_PYTHON:-${PROJECTDIR}/${USER}/envs/ocsf-env/bin/python}"

export PYTHONPATH="${CODE_DIR}/src:${CODE_DIR}/scripts"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${PROJECTDIR}/${USER}/cache/hf"
export HF_HUB_OFFLINE=1

cd "${CODE_DIR}"
"${PYTHON}" "${SCRIPT}" "$@"
