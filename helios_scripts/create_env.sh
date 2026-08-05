#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=plgittossl2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err
 
# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/25.10
ml GCCcore/14.3.0
ml Python/3.13.5
 
cd $SCRATCH
 
# create and activate the virtual environment
env_name=lynx-finetuning

python -m venv $env_name/

cd -