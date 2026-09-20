#!/bin/bash -l
#SBATCH --job-name=lynx-loma-wandb
#SBATCH --gres=gpu:2
#SBATCH --mem=125G
#SBATCH --cpus-per-task=32
#SBATCH --time=23:59:00
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --exclude=c11,c15
#SBATCH --output=logs/lynx-loma-wandb/lynx-loma-wandb-%j.out
#SBATCH --error=logs/lynx-loma-wandb/lynx-loma-wandb-%j.err

set -euo pipefail

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma

dataset_root=/shared/sets/datasets/confidential/lynx/processed_frames/segmented/lynx-ds-Jul-20
train_index=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/reports/strong_matches-big-512-5-10-512-20/top_k=5_top_m=10_train_combined.json
val_index=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/reports/strong_matches-big-512-5-10-512-20/top_k=5_top_m=10_test_combined.json
loma_weights=/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt
train_component=${LOMA_TRAIN_COMPONENT:-matcher}
loma_cache=${LOMA_KEYPOINT_CACHE:-/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/loma-b-cache-resize512-kp512}
keypoint_cache=${LOMA_DESCRIPTOR_KEYPOINT_CACHE:-/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/loma-b-keypoints-resize512-kp512}
cache_batch_size=${LOMA_CACHE_BATCH_SIZE:-4}
benchmark_root=/home/kargin/Projects/repositories/rdd-parallel-benchmark
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"
if [[ "${train_component}" == descriptor ]]; then
    default_run_name=matches-loma-descriptor-dataset
else
    default_run_name=matches-loma-dataset
fi
run_name=${LOMA_RUN_NAME:-${default_run_name}}

if [[ -n "${LOMA_OUTPUT_DIR:-}" ]]; then
    output_dir=${LOMA_OUTPUT_DIR}
elif [[ "${train_component}" == descriptor ]]; then
    output_dir=/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/loma-b-descriptor-wandb
else
    output_dir=/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/loma-b-wandb
fi

if [[ "${train_component}" == matcher ]]; then
    cache_args=(
        --dataset_root "${dataset_root}"
        --cache_dir "${loma_cache}"
        --weights "${loma_weights}"
        --variant loma-b --all_frames --splits train test
        --index "${train_index}" "${val_index}"
        --resize_max 512 --num_keypoints 512
    )
    if [[ -f "${loma_cache}/manifest.json" ]]; then
        set +e
        (cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
            "${cache_args[@]}" --verify)
        verify_status=$?
        set -e
        if [[ ${verify_status} -eq 0 ]]; then
            echo "Using existing LoMa feature cache: ${loma_cache}"
        elif [[ ${verify_status} -eq 2 ]]; then
            echo "LoMa feature-cache metadata is incompatible; rebuilding it"
            (cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
                "${cache_args[@]}" --batch_size "${cache_batch_size}" \
                --num_workers 16 --overwrite)
        else
            echo "LoMa feature cache is incomplete; resuming construction"
            (cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
                "${cache_args[@]}" --batch_size "${cache_batch_size}" \
                --num_workers 16 --resume)
        fi
    else
        echo "LoMa feature cache not found; building ${loma_cache}"
        (cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
            "${cache_args[@]}" --batch_size "${cache_batch_size}" \
            --num_workers 16 --resume)
    fi
    (cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
        "${cache_args[@]}" --verify)
elif [[ "${train_component}" == descriptor ]]; then
    python -m contrastive_finetuning.build_loma_keypoint_cache \
        --data_root "${dataset_root}" --cache_dir "${keypoint_cache}" \
        --weights "${loma_weights}" --variant loma-b --splits train test \
        --resize 512 --num_keypoints 512 --batch_size "${cache_batch_size}"
else
    echo "LOMA_TRAIN_COMPONENT must be matcher or descriptor, got ${train_component}" >&2
    exit 2
fi

args=(
    --trained_model loma
    --loma_train_component "${train_component}"
    --train_index "${train_index}"
    --val_index "${val_index}"
    --data_root "${dataset_root}"
    --output_dir "${output_dir}"
    --project lynx-loma-finetuning
    --run_name "${run_name}"
    --wandb_mode online
    --loma_variant loma-b
    --epochs 300
    --batch_size 8
    --lr 1e-5
    --weight_decay 1e-4
    --margin 0.5
    --random_negative_prob 0.3
    --num_workers 10
    --eval_every_epochs 10
    --seed 0
    --resize 512
    --num_keypoints 512
    --descriptor_microbatch_size "${LOMA_DESCRIPTOR_MICROBATCH_SIZE:-1}"
)
if [[ "${train_component}" == matcher ]]; then
    args+=(--loma_cache "${loma_cache}")
else
    args+=(--loma_keypoint_cache "${keypoint_cache}")
fi
if [[ -n "${loma_weights}" ]]; then
    args+=(--loma_weights "${loma_weights}")
fi

mkdir -p logs
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m contrastive_finetuning.train_loma_matches \
    "${args[@]}"
