#!/bin/bash -l
#SBATCH --job-name=lynx-rdd-wandb
#SBATCH --partition=dgxh100
#SBATCH --qos=big
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --exclude=c11
#SBATCH --export=ALL
#SBATCH --output=logs/lynx-rdd-wandb/lynx-rdd-wandb-%j.out
#SBATCH --error=logs/lynx-rdd-wandb/lynx-rdd-wandb-%j.err

set -euo pipefail

source "${LYNX_FINETUNING_ROOT:-$PWD}/env.sh"
activate_conda_env "${CONDA_ENV_RDD}"

# Original (confidential) lynx dataset run. The mined index is not part of this repository:
# point LYNX_TRAIN_INDEX / LYNX_VAL_INDEX at the combined JSON files produced by
# rdd-parallel-benchmark's spawn_find_strong_matches.sh.
dataset_root=${LYNX_DATASET_ROOT:-/shared/sets/datasets/confidential/lynx/processed_frames/segmented/lynx-ds-Jul-20}
train_index=${LYNX_TRAIN_INDEX:?set LYNX_TRAIN_INDEX to the mined train index}
val_index=${LYNX_VAL_INDEX:?set LYNX_VAL_INDEX to the mined test index}
rdd_weights=${RDD_WEIGHTS}
lg_weights=${LG_WEIGHTS}
output_dir=${RDD_OUTPUT_DIR:-/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/matches-lg-wandb}
rdd_cache=${RDD_KEYPOINT_CACHE:-/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/rdd-keypoint-cache-resize512-topk512}
cache_batch_size=${RDD_CACHE_BATCH_SIZE:-32}

# RDD is frozen in this run, so its keypoints/descriptors can be computed once
# and reused for every optimizer step and evaluation pass. Build the complete
# train+test cache on the first invocation; later runs reuse it.
if [[ ! -f "${rdd_cache}/manifest.json" ]]; then
    echo "RDD keypoint cache not found; building ${rdd_cache}"
    python -m contrastive_finetuning.build_keypoint_cache \
        --data_root "${dataset_root}" \
        --cache_root "${rdd_cache}" \
        --rdd_weights "${rdd_weights}" \
        --splits train test \
        --index "${train_index}" "${val_index}" \
        --resize 512 \
        --top_k 512 \
        --batch_size "${cache_batch_size}" \
        --num_workers 16 \
        --resume
else
    echo "Using existing RDD keypoint cache: ${rdd_cache}"
fi

args=(
    --train_index "${train_index}"
    --val_index "${val_index}"
    --data_root "${dataset_root}"
    --rdd_weights "${rdd_weights}"
    --lg_weights "${lg_weights}"
    --output_dir "${output_dir}"
    --project lynx-contrastive
    --run_name matches-lg-rdd-dataset
    --trained_model lg
    --epochs 300
    --batch_size 8
    --lr 1e-5
    --weight_decay 1e-4
    --num_workers 8
    --lg_margin 0.5
    --random_negative_prob 0.3
    --resize 512
    --top_k 512
    --keypoint_cache "${rdd_cache}"
    --seed 0
)

mkdir -p logs
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m contrastive_finetuning.train_by_lg_matches \
    "${args[@]}"
