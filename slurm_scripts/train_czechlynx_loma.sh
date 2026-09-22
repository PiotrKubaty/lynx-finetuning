#!/bin/bash
#SBATCH --job-name=czechlynx-loma-ft
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclude=c11
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --output=logs/czechlynx-loma-ft/czechlynx-loma-ft-%j.out
#SBATCH --error=logs/czechlynx-loma-ft/czechlynx-loma-ft-%j.err

# The whole script is a function so that bash parses it completely before running it
# (it runs for hours inside few-shot jobs).
main() {
set -euo pipefail
source "${LYNX_FINETUNING_ROOT:-$PWD}/env.sh"
activate_conda_env "${CONDA_ENV_LOMA}"
source "${LYNX_FINETUNING_ROOT}/slurm_scripts/czechlynx_protocol.sh"
czechlynx_resolve_protocol

dataset_root=${CZECHLYNX_ROOT:-${CZECHLYNX_VIEW_ROOT}}
train_index=${CZECHLYNX_RESOLVED_TRAIN_INDEX}
val_index=${CZECHLYNX_RESOLVED_VAL_INDEX}
loma_weights=${LOMA_WEIGHTS}
cache_dir=${CZECHLYNX_LOMA_CACHE:-${CHECKPOINTS_ROOT}/czechlynx-time-closed/loma-b-cache}
# CZECHLYNX_LOMA_TRAIN_COMPONENT selects what is fine-tuned (train_loma_matches
# --loma_train_component):
#   matcher             the LoMa transformer on cached DaD keypoints + DeDoDe descriptors
#                       (default, the paper's setting; needs the feature cache)
#   descriptor          DeDoDe's descriptor through the frozen matcher
#   descriptor+matcher  both jointly
# The descriptor modes recompute descriptors every step, so only the DaD keypoints are
# cached (CZECHLYNX_LOMA_KEYPOINT_CACHE, built here when missing; the DaD detector is
# frozen in every mode, so the cache stays valid).
component=${CZECHLYNX_LOMA_TRAIN_COMPONENT:-matcher}
case "${component}" in
  matcher|descriptor|descriptor+matcher) ;;
  *) echo "CZECHLYNX_LOMA_TRAIN_COMPONENT must be matcher, descriptor or descriptor+matcher (got ${component})" >&2; exit 2 ;;
esac
component_tag=${component//+/-}
keypoint_cache=${CZECHLYNX_LOMA_KEYPOINT_CACHE:-${CHECKPOINTS_ROOT}/czechlynx-time-closed/loma-b-keypoint-cache}
if [[ -n "${CZECHLYNX_LOMA_OUTPUT:-}" ]]; then output_dir=${CZECHLYNX_LOMA_OUTPUT}
elif [[ "${component}" == matcher ]]; then output_dir=${CHECKPOINTS_ROOT}/czechlynx-time-closed/loma-b-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
else output_dir=${CHECKPOINTS_ROOT}/czechlynx-time-closed/loma-b-${component_tag}-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}; fi
if [[ -n "${CZECHLYNX_LOMA_RUN_NAME:-}" ]]; then run_name=${CZECHLYNX_LOMA_RUN_NAME}
elif [[ "${component}" == matcher ]]; then run_name=czechlynx-time-closed-loma-${CZECHLYNX_RESOLVED_PROTOCOL}
else run_name=czechlynx-time-closed-loma-${component_tag}-${CZECHLYNX_RESOLVED_PROTOCOL}; fi
benchmark_root=${RDD_BENCHMARK_ROOT}

echo "CzechLynx split protocol: ${CZECHLYNX_RESOLVED_PROTOCOL}"
echo "LoMa training component: ${component}"
echo "training index: ${train_index}"
echo "validation index: ${val_index}"
echo "output directory: ${output_dir}"

mkdir -p "${output_dir}"
cat > "${output_dir}/czechlynx_protocol.json" <<EOF
{
  "protocol": "${CZECHLYNX_RESOLVED_PROTOCOL}",
  "train_index": "${train_index}",
  "validation_index": "${val_index}",
  "final_evaluation_split": "test",
  "loma_train_component": "${component}"
}
EOF

mkdir -p logs
if [[ "${component}" == matcher ]]; then
  if [[ ! -f "${cache_dir}/manifest.json" ]]; then
    (cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
      --dataset_root "${dataset_root}" --cache_dir "${cache_dir}" \
      --weights "${loma_weights}" --variant loma-b --all_frames \
      --splits train val test --resize_max 512 --num_keypoints 512 \
      --batch_size "${LOMA_CACHE_BATCH_SIZE:-4}" --num_workers 16 --resume)
  fi
elif ! python -c "import json,sys; sys.exit(0 if json.load(open('${keypoint_cache}/manifest.json')).get('complete') is True else 1)" 2>/dev/null; then
  # a complete cache (e.g. a shared read-only one) is left alone: the trainer validates its
  # detector fingerprint and settings when it opens it
  python -m contrastive_finetuning.build_loma_keypoint_cache \
    --data_root "${dataset_root}" --cache_dir "${keypoint_cache}" \
    --weights "${loma_weights}" --variant "${CZECHLYNX_LOMA_VARIANT:-loma-b}" --splits train val test \
    --resize 512 --num_keypoints 512 --batch_size "${LOMA_CACHE_BATCH_SIZE:-4}"
fi

# Reference run: 4 processes x batch 8 (global batch 32). CZECHLYNX_NUM_PROCESSES /
# CZECHLYNX_BATCH_SIZE keep that global batch when the GPU count changes; several trainings
# may share a node (few-shot jobs), so each gets its own rendezvous port.
args=(
  --trained_model loma --loma_train_component "${component}"
  --train_index "${train_index}" --val_index "${val_index}" --data_root "${dataset_root}"
  --loma_weights "${loma_weights}"
  --output_dir "${output_dir}" --project "${CZECHLYNX_WANDB_PROJECT-lynx-czechlynx-loma}"
  --run_name "${run_name}" --split_protocol "${CZECHLYNX_RESOLVED_PROTOCOL}" --wandb_mode "${CZECHLYNX_WANDB_MODE:-online}"
  --loma_variant "${CZECHLYNX_LOMA_VARIANT:-loma-b}" --epochs "${CZECHLYNX_EPOCHS:-300}" --batch_size "${CZECHLYNX_BATCH_SIZE:-8}" --lr 1e-5
  --weight_decay 1e-4 --margin 0.5 --random_negative_prob 0.3
  --num_workers "${CZECHLYNX_NUM_WORKERS:-10}" --eval_every_epochs "${CZECHLYNX_EVAL_EVERY:-${WILDLIFE_EVAL_EVERY:-10}}" --seed 0 --resize 512
  --num_keypoints 512
)
if [[ "${component}" == matcher ]]; then
  args+=(--loma_cache "${cache_dir}")
else
  # descriptors are recomputed with gradient: triplets per descriptor forward/backward
  # (CZECHLYNX_LOMA_DESCRIPTOR_MICROBATCH, gradients accumulate over the batch)
  args+=(--loma_keypoint_cache "${keypoint_cache}"
         --descriptor_microbatch_size "${CZECHLYNX_LOMA_DESCRIPTOR_MICROBATCH:-1}")
fi
# CZECHLYNX_RESUME_FROM=<output_dir>/epoch_NNN continues an interrupted run (wall-time chains)
[[ -n "${CZECHLYNX_RESUME_FROM:-}" ]] && args+=(--resume "${CZECHLYNX_RESUME_FROM}")
accelerate launch --num_processes "${CZECHLYNX_NUM_PROCESSES:-4}" --num_machines 1 \
  --main_process_port "${CZECHLYNX_MAIN_PROCESS_PORT:-$(( 20000 + ${SLURM_JOB_ID:-0} % 10000 ))}" \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_loma_matches "${args[@]}"
}

main "$@"
