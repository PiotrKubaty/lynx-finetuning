#!/bin/bash -l
#SBATCH --job-name=lynx-loma-wandb
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --output=logs/lynx-loma-wandb-%j.out
#SBATCH --error=logs/lynx-loma-wandb-%j.err

set -euo pipefail

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma

dataset_root=/shared/sets/datasets/confidential/lynx/processed_frames/segmented/lynx-ds-Jul-20
train_index=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/reports/strong_matches-big-512-5-10-512-20/top_k=5_top_m=10_train_combined.json
val_index=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/reports/strong_matches-big-512-5-10-512-20/top_k=5_top_m=10_test_combined.json
loma_weights=/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt
output_dir=${LOMA_OUTPUT_DIR:-/shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/loma-b-wandb}

args=(
    --trained_model loma
    --train_index "${train_index}"
    --val_index "${val_index}"
    --data_root "${dataset_root}"
    --output_dir "${output_dir}"
    --project lynx-loma-finetuning
    --run_name matches-loma-dataset
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
)
if [[ -n "${loma_weights}" ]]; then
    args+=(--loma_weights "${loma_weights}")
fi

mkdir -p logs
python -m contrastive_finetuning.train_loma_matches "${args[@]}"
