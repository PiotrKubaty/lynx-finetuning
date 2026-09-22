#!/bin/bash
#SBATCH --job-name=czechlynx-loma-cache
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=c11
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --output=logs/czechlynx-loma-cache/czechlynx-loma-cache-%j.out
#SBATCH --error=logs/czechlynx-loma-cache/czechlynx-loma-cache-%j.err

set -euo pipefail
source "${LYNX_FINETUNING_ROOT:-$PWD}/env.sh"
activate_conda_env "${CONDA_ENV_LOMA}"

dataset_root=${CZECHLYNX_ROOT:-${CZECHLYNX_VIEW_ROOT}}
cache_dir=${CZECHLYNX_LOMA_CACHE:-${CHECKPOINTS_ROOT}/czechlynx-time-closed/loma-b-cache}
loma_weights=${LOMA_WEIGHTS}
benchmark_root=${RDD_BENCHMARK_ROOT}

mkdir -p logs
cd "${benchmark_root}"
python -m scripts.lynx_build_loma_cache \
  --dataset_root "${dataset_root}" \
  --cache_dir "${cache_dir}" \
  --weights "${loma_weights}" \
  --variant loma-b \
  --all_frames \
  --splits train val test \
  --resize_max 512 \
  --num_keypoints 512 \
  --batch_size "${LOMA_CACHE_BATCH_SIZE:-4}" \
  --num_workers 16 \
  --resume
