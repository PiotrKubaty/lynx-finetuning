#!/bin/bash -l
#SBATCH --job-name=deit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120000M
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=plgflame-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs-base/job-%j.out
#SBATCH --error=logs-base/job-%j.err

conda deactivate
conda deactivate

ml ML-bundle/25.10
ml GCCcore/14.3.0
ml Python/3.13.5

cur_dir=$(pwd)
cd $SCRATCH
env_name=lynx-finetuning
source $env_name/bin/activate
cd $cur_dir

