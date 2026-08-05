#!/bin/bash -l
#SBATCH --job-name=lxft-lg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=plgittossl2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

source helios_scripts/_activate.sh

seed=$1

batch_size=32
lg_margin=0.5
epochs=300
random_negative_prob=0.3
trained_model=lg

python -m contrastive_finetuning.train_by_lg_matches \
    --train_index index-rgb/top_k=5_top_m=10_train_combined.json \
    --val_index index-rgb/top_k=5_top_m=10_test_combined.json \
    --data_root /net/scratch/hscra/plgrid/plgkubaty/lynx-ds-Jul-20 \
    --rdd_weights rdd/weights/RDD-v2.pth \
    --lg_weights rdd/weights/RDD_lg-v2.pth \
    --output_dir /net/scratch/hscra/plgrid/plgkubaty/lxft-results-lg-random-neg/batch_size=${batch_size},lg_margin=${lg_margin},random_negative_prob=${random_negative_prob},epochs=${epochs} \
    --project lynx-contrastive-v3 \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --lg_margin ${lg_margin} \
    --random_negative_prob ${random_negative_prob} \
    --trained_model ${trained_model} \
    --wandb_tags "${trained_model}-tune,random_neg" \
    --seed ${seed}
