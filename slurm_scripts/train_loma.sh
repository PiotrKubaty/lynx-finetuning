#!/usr/bin/env bash -l
#SBATCH --job-name=lynx-loma
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --output=logs/loma-%j.out
#SBATCH --error=logs/loma-%j.err

set -euo pipefail

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma

dataset_root=/shared/sets/datasets/confidential/lynx/processed_frames/segmented/lynx-ds-Jul-20
train_index=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/reports/strong_matches-big-512-5-10-512-20/top_k=5_top_m=10_train_combined.json
val_index=${LOMA_VAL_INDEX:?Set LOMA_VAL_INDEX to a validation index; keep the test index untouched}
loma_weights=${LOMA_WEIGHTS:-}
output_dir=${LOMA_OUTPUT_DIR:-/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/loma-b-wandb}

args=(
    --trained_model loma
    --train_index "${train_index}"
    --val_index "${val_index}"
    --data_root "${dataset_root}"
    --output_dir "${output_dir}"
    --project lynx-loma-finetuning
    --wandb_mode online
    --loma_variant loma-b
    --epochs 10
    --batch_size 2
    --resize 512
    --num_keypoints 512
)
if [[ -n "${loma_weights}" ]]; then
    args+=(--loma_weights "${loma_weights}")
fi

mkdir -p logs
python -m contrastive_finetuning.train_loma_matches "${args[@]}"
