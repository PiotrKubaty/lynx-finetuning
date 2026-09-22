#!/usr/bin/env bash
#SBATCH --job-name=wildlife-rdd-ft
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --exclude=c11
#SBATCH --output=logs/wildlife-rdd-ft/wildlife-rdd-ft-%j.out
#SBATCH --error=logs/wildlife-rdd-ft/wildlife-rdd-ft-%j.err

# The whole script is a function so that bash parses it completely before running it:
# this wrapper runs for hours inside few-shot jobs and must not be affected by edits
# to the file in the meantime.
main() {
set -euo pipefail
source "${LYNX_FINETUNING_ROOT:-$PWD}/env.sh"
activate_conda_env "${CONDA_ENV_RDD}"

benchmark_root=${WILDLIFE_BENCHMARK_ROOT:-${RDD_BENCHMARK_ROOT}}
config=${WILDLIFE_CONFIG:-${benchmark_root}/configs/wildlife/BelugaID.json}
protocol=${WILDLIFE_PROTOCOL:-strict}
eval "$(cd "${benchmark_root}" && python -m scripts.wildlife_config --config "${config}" --shell)"
dataset_root=${WILDLIFE_VIEW_ROOT:-${WILDLIFE_PROCESSED_ROOT}/${WILDLIFE_DATASET_ID}/${protocol}}
index_base=${WILDLIFE_INDEX_ROOT:-${benchmark_root}/outputs/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/indices}
if [[ -n "${WILDLIFE_INDEX_ROOT:-}" ]]; then
  index_root=${index_base}
elif [[ -f "${index_base}/rdd/strong-matches_train_combined.json" ]]; then
  index_root="${index_base}/rdd"
else
  # Fall back to the original shared path for existing experiments.
  index_root=${index_base}
fi
train_index=${WILDLIFE_TRAIN_INDEX:-${index_root}/strong-matches_train_combined.json}
if [[ "${protocol}" == legacy ]]; then
  default_val_index=${index_root}/strong-matches_test_combined.json
else
  default_val_index=${index_root}/strong-matches_val_combined.json
fi
val_index=${WILDLIFE_VAL_INDEX:-${default_val_index}}
rdd_weights=${RDD_WEIGHTS}
lg_weights=${LG_WEIGHTS}
cache_root=${WILDLIFE_RDD_CACHE:-${CHECKPOINTS_ROOT}/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-cache}
output_dir=${WILDLIFE_RDD_OUTPUT:-${CHECKPOINTS_ROOT}/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-finetuned/${protocol}}
run_name=${WILDLIFE_RDD_RUN_NAME:-${WILDLIFE_DATASET_ID}-rdd-${protocol}-finetuned}
wandb_project=${WILDLIFE_WANDB_PROJECT:-wildlife-reid-rdd-${WILDLIFE_DATASET_ID}-${protocol}}

echo "dataset=${WILDLIFE_DATASET_ID} protocol=${protocol}"
echo "training index=${train_index}"
echo "validation index=${val_index}"
echo "output directory=${output_dir}"
[[ -f "${train_index}" && -f "${val_index}" ]] || { echo "missing training or validation index" >&2; exit 1; }
[[ -f "${cache_root}/manifest.json" ]] || bash "${LYNX_FINETUNING_ROOT}/slurm_scripts/build_wildlife_rdd_cache.sh"

mkdir -p "${output_dir}"
cat > "${output_dir}/wildlife_protocol.json" <<EOF
{"dataset": "${WILDLIFE_DATASET_ID}", "protocol": "${protocol}", "train_index": "${train_index}", "validation_index": "${val_index}", "final_evaluation_split": "test"}
EOF

# Several trainings may share a node (few-shot jobs): give each its own rendezvous port.
accelerate launch --num_processes "${WILDLIFE_NUM_PROCESSES:-4}" --num_machines 1 \
  --main_process_port "${WILDLIFE_MAIN_PROCESS_PORT:-$(( 20000 + ${SLURM_JOB_ID:-0} % 10000 ))}" \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_by_lg_matches \
  --train_index "${train_index}" --val_index "${val_index}" \
  --data_root "${dataset_root}" --rdd_weights "${rdd_weights}" \
  --lg_weights "${lg_weights}" --output_dir "${output_dir}" \
  --project "${wandb_project}" \
  --run_name "${run_name}" --split_protocol "${protocol}" --trained_model lg \
  --epochs "${WILDLIFE_EPOCHS:-300}" --batch_size "${WILDLIFE_BATCH_SIZE:-8}" \
  --lr 1e-5 --weight_decay 1e-4 --num_workers "${WILDLIFE_NUM_WORKERS:-8}" \
  --lg_margin 0.5 --random_negative_prob 0.3 --resize 512 --top_k 512 \
  --keypoint_cache "${cache_root}" --eval_every_epochs "${WILDLIFE_EVAL_EVERY:-10}" --seed 0
}

main "$@"
