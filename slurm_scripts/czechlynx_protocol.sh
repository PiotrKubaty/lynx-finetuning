#!/usr/bin/env bash
# Shared CzechLynx split-protocol resolver, sourced by train_czechlynx_rdd.sh and
# train_czechlynx_loma.sh (see CZECHLYNX.md, "Split protocols").
#
#   legacy (default): train -> fine-tuning; test -> validation and final reporting
#   strict:           train -> fine-tuning; val  -> validation; test -> final reporting
#
# Inputs (all optional):
#   CZECHLYNX_SPLIT_PROTOCOL  legacy | strict
#   CZECHLYNX_INDEX_ROOT      directory holding strong-matches_{train,val,test}_combined.json
#                             (default: <benchmark>/outputs/czechlynx-time-closed/<protocol>)
#   CZECHLYNX_TRAIN_INDEX     explicit training index (e.g. the optional train+val file)
#   CZECHLYNX_VAL_INDEX       explicit validation index; wins over the protocol default
# Outputs:
#   CZECHLYNX_RESOLVED_PROTOCOL, CZECHLYNX_RESOLVED_TRAIN_INDEX,
#   CZECHLYNX_RESOLVED_VAL_INDEX, CZECHLYNX_RESOLVED_OUTPUT_SUFFIX

czechlynx_resolve_protocol() {
  local protocol=${CZECHLYNX_SPLIT_PROTOCOL:-legacy}
  case "${protocol}" in
    legacy|strict) ;;
    *)
      echo "CZECHLYNX_SPLIT_PROTOCOL must be 'legacy' or 'strict' (got '${protocol}')" >&2
      return 1
      ;;
  esac
  local benchmark_root=${RDD_BENCHMARK_ROOT:-${WILDLIFE_BENCHMARK_ROOT:-${PWD}}}
  local index_root=${CZECHLYNX_INDEX_ROOT:-${benchmark_root}/outputs/czechlynx-time-closed/${protocol}}
  local default_val=${index_root}/strong-matches_test_combined.json
  if [[ "${protocol}" == strict ]]; then
    default_val=${index_root}/strong-matches_val_combined.json
  fi
  export CZECHLYNX_RESOLVED_PROTOCOL=${protocol}
  export CZECHLYNX_RESOLVED_TRAIN_INDEX=${CZECHLYNX_TRAIN_INDEX:-${index_root}/strong-matches_train_combined.json}
  export CZECHLYNX_RESOLVED_VAL_INDEX=${CZECHLYNX_VAL_INDEX:-${default_val}}
  export CZECHLYNX_RESOLVED_OUTPUT_SUFFIX=${protocol}
}
