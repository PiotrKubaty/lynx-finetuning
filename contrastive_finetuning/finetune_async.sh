#!/bin/bash
#SBATCH --job-name=lx
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task=16
#SBATCH --partition=rtx4090_batch
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%j.out

batch_size=2

accelerate launch --num_processes ${SLURM_GPUS_ON_NODE:-1} -m finetuning.train \
    --train_data /shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/train/ \
    --val_data /shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-June-2026-merged/lynx/test/ \
    --rdd_weights rdd/weights/RDD-v2.pth \
    --lg_weights rdd/weights/RDD_lg-v2.pth \
    --output_dir /shared/sets/datasets/confidential/lynx/checkpoints/contrastive-finetuning/1706/batch_size=${batch_size} \
    --project lynx-contrastive \
    --epochs 10 \
    --batch_size ${batch_size} \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --num_workers 16 \
