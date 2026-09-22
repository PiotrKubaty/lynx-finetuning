#!/usr/bin/env bash
# TEMPLATE: copy to env.sh (gitignored) and edit for the cluster you are on. The values
# below are the GMUM ones this work ran with; see MIGRATION.md in rdd-parallel-benchmark.
# Site configuration of lynx-finetuning: every absolute path this repository uses lives here
# and nowhere else. Copy it to env.sh (gitignored) and edit it for the cluster you are on;
# the values below are the GMUM ones this work was run with. See MIGRATION.md in
# rdd-parallel-benchmark for the full move.
# Site configuration for every script under slurm_scripts/.
#
# Scripts are run from this repository root (`sbatch slurm_scripts/<name>.sh`) and load
# this file with `source "${LYNX_FINETUNING_ROOT:-$PWD}/env.sh"`. Each value below is a
# default: export the variable before calling sbatch to override it for one run.
#
# Cross-repository paths are explicit here — nothing assumes that the other repositories
# are checked out next to (or inside) this one.

export LYNX_FINETUNING_ROOT="${LYNX_FINETUNING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export RDD_BENCHMARK_ROOT="${RDD_BENCHMARK_ROOT:-/home/kubaty/rdd-parallel-benchmark}"
export EXREID_ROOT="${EXREID_ROOT:-/home/kubaty/explainable_individual_reidentification}"

# Conda (GMUM): scripts call `activate_conda_env "$CONDA_ENV_RDD"` (defined below).
export CONDA_SH="${CONDA_SH:-/shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV_RDD="${CONDA_ENV_RDD:-rdd}"
export CONDA_ENV_LOMA="${CONDA_ENV_LOMA:-loma}"

# Activate a conda environment and make sure its interpreter is the one on PATH — a
# submitting shell (e.g. a VS Code terminal) may carry another env's bin directory in
# front of PATH, and Slurm forwards that environment to the job (--export=ALL).
activate_conda_env() {
  source "${CONDA_SH}"
  conda activate "$1"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
}

# Pretrained checkpoints.
export RDD_WEIGHTS="${RDD_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/rdd/weights/RDD-v2.pth}"
export LG_WEIGHTS="${LG_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/rdd/weights/RDD_lg-v2.pth}"
export LOMA_WEIGHTS="${LOMA_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt}"

# Data. The processed views and the feature caches/checkpoints under CHECKPOINTS_ROOT were
# built by kargin and are read-only for us; everything we produce goes under FEWSHOT_ROOT.
export CZECHLYNX_DATA_ROOT="${CZECHLYNX_DATA_ROOT:-/shared/sets/datasets/vision/czechlynx}"
export WILDLIFE_SOURCE_ROOT="${WILDLIFE_SOURCE_ROOT:-${CZECHLYNX_DATA_ROOT}/WildlifeReID-10k}"
export WILDLIFE_PROCESSED_ROOT="${WILDLIFE_PROCESSED_ROOT:-${CZECHLYNX_DATA_ROOT}/wildlife_processed}"
export CZECHLYNX_VIEW_ROOT="${CZECHLYNX_VIEW_ROOT:-${CZECHLYNX_DATA_ROOT}/CzechLynx_processed_time_closed}"
export CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${CZECHLYNX_DATA_ROOT}/checkpoints}"
export FEWSHOT_ROOT="${FEWSHOT_ROOT:-${CZECHLYNX_DATA_ROOT}/fewshot}"

export WANDB_MODE="${WANDB_MODE:-online}"

# Pseudo-accuracy evaluation interval (epochs) of the wildlife trainers. The original runs
# used 10; each evaluation pass costs about as much as several training epochs and only
# affects the W&B curve (checkpoints are written every 50 epochs and epoch_299 is used).
export WILDLIFE_EVAL_EVERY="${WILDLIFE_EVAL_EVERY:-50}"
