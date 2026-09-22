#!/bin/bash
#SBATCH --job-name=czechlynx-rdd-ft
#SBATCH --partition=dgxh100
#SBATCH --qos=big
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=125G
#SBATCH --time=48:00:00
#SBATCH --exclude=c11
#SBATCH --output=logs/czechlynx-rdd-ft/czechlynx-rdd-ft-%j.out
#SBATCH --error=logs/czechlynx-rdd-ft/czechlynx-rdd-ft-%j.err

# The whole script is a function so that bash parses it completely before running it
# (it runs for hours inside few-shot jobs).
main() {
set -euo pipefail
source "${LYNX_FINETUNING_ROOT:-$PWD}/env.sh"
activate_conda_env "${CONDA_ENV_RDD}"
source "${LYNX_FINETUNING_ROOT}/slurm_scripts/czechlynx_protocol.sh"
czechlynx_resolve_protocol

dataset_root=${CZECHLYNX_ROOT:-${CZECHLYNX_VIEW_ROOT}}
train_index=${CZECHLYNX_RESOLVED_TRAIN_INDEX}
val_index=${CZECHLYNX_RESOLVED_VAL_INDEX}
rdd_weights=${RDD_WEIGHTS}
lg_weights=${LG_WEIGHTS}
cache_root=${CZECHLYNX_RDD_CACHE:-${CHECKPOINTS_ROOT}/czechlynx-time-closed/rdd-cache}
# CZECHLYNX_RDD_TRAIN_COMPONENT selects what is fine-tuned (train_by_lg_matches
# --trained_model / --rdd_train_component):
#   lg             LightGlue on cached RDD features (default, the paper's setting)
#   descriptor     RDD's descriptor network (backbone + deformable transformer) through the
#                  frozen LightGlue; the RDD detector stays frozen
#   lg+descriptor  LightGlue and RDD's descriptor jointly
#   rdd, lg+rdd    as above but with the RDD detector unfrozen too (its NMS keypoints get no
#                  gradient, so this only adds BatchNorm drift — kept for completeness)
# Everything but `lg` recomputes RDD features every step, so the keypoint cache is unusable
# and the run is several times slower.
component=${CZECHLYNX_RDD_TRAIN_COMPONENT:-lg}
case "${component}" in
  lg) trained_model=lg; rdd_component=all ;;
  descriptor) trained_model=rdd; rdd_component=descriptor ;;
  lg+descriptor) trained_model=lg+rdd; rdd_component=descriptor ;;
  rdd) trained_model=rdd; rdd_component=all ;;
  lg+rdd) trained_model=lg+rdd; rdd_component=all ;;
  *) echo "CZECHLYNX_RDD_TRAIN_COMPONENT must be lg, descriptor, lg+descriptor, rdd or lg+rdd (got ${component})" >&2; exit 2 ;;
esac
component_tag=${component//+/-}
if [[ -n "${CZECHLYNX_RDD_OUTPUT:-}" ]]; then output_dir=${CZECHLYNX_RDD_OUTPUT}
elif [[ "${component}" == lg ]]; then output_dir=${CHECKPOINTS_ROOT}/czechlynx-time-closed/rdd-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
else output_dir=${CHECKPOINTS_ROOT}/czechlynx-time-closed/rdd-${component_tag}-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}; fi
if [[ -n "${CZECHLYNX_RDD_RUN_NAME:-}" ]]; then run_name=${CZECHLYNX_RDD_RUN_NAME}
elif [[ "${component}" == lg ]]; then run_name=czechlynx-time-closed-rdd-${CZECHLYNX_RESOLVED_PROTOCOL}
else run_name=czechlynx-time-closed-rdd-${component_tag}-${CZECHLYNX_RESOLVED_PROTOCOL}; fi

echo "CzechLynx split protocol: ${CZECHLYNX_RESOLVED_PROTOCOL}"
echo "RDD training component: ${component} (--trained_model ${trained_model} --rdd_train_component ${rdd_component})"
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
  "rdd_train_component": "${component}"
}
EOF

mkdir -p logs
if [[ "${trained_model}" == lg && ! -f "${cache_root}/manifest.json" ]]; then
  python -m contrastive_finetuning.build_keypoint_cache \
    --data_root "${dataset_root}" --cache_root "${cache_root}" \
    --rdd_weights "${rdd_weights}" --splits train val test \
    --resize 512 --top_k 512 \
    --batch_size "${RDD_CACHE_BATCH_SIZE:-32}" --num_workers 16 --resume
fi

# Reference run: 2 processes x batch 8 (global batch 16). CZECHLYNX_NUM_PROCESSES /
# CZECHLYNX_BATCH_SIZE keep that global batch when the GPU count changes; several trainings
# may share a node (few-shot jobs), so each gets its own rendezvous port.
args=(
  --train_index "${train_index}" --val_index "${val_index}"
  --data_root "${dataset_root}" --rdd_weights "${rdd_weights}"
  --lg_weights "${lg_weights}" --output_dir "${output_dir}"
  --project "${CZECHLYNX_WANDB_PROJECT-lynx-czechlynx-rdd}" --run_name "${run_name}"
  --split_protocol "${CZECHLYNX_RESOLVED_PROTOCOL}"
  --trained_model "${trained_model}" --rdd_train_component "${rdd_component}"
  --epochs "${CZECHLYNX_EPOCHS:-300}" --batch_size "${CZECHLYNX_BATCH_SIZE:-8}" --lr 1e-5
  --weight_decay 1e-4 --num_workers "${CZECHLYNX_NUM_WORKERS:-8}" --lg_margin 0.5
  --random_negative_prob 0.3 --resize 512 --top_k 512
  --eval_every_epochs "${CZECHLYNX_EVAL_EVERY:-${WILDLIFE_EVAL_EVERY:-10}}" --seed 0
)
# the cache holds one fixed feature set per frame: only valid while RDD itself is frozen
[[ "${trained_model}" == lg ]] && args+=(--keypoint_cache "${cache_root}")
# CZECHLYNX_RESUME_FROM=<output_dir>/epoch_NN continues an interrupted run (wall-time chains)
[[ -n "${CZECHLYNX_RESUME_FROM:-}" ]] && args+=(--resume "${CZECHLYNX_RESUME_FROM}")
accelerate launch --num_processes "${CZECHLYNX_NUM_PROCESSES:-2}" --num_machines 1 \
  --main_process_port "${CZECHLYNX_MAIN_PROCESS_PORT:-$(( 20000 + ${SLURM_JOB_ID:-0} % 10000 ))}" \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_by_lg_matches "${args[@]}"
}

main "$@"
