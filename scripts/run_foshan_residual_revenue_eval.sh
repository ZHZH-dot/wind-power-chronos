#!/usr/bin/env bash
set -euo pipefail

SITE_WORKBOOK="${1:?Usage: $0 SITE_WORKBOOK STORAGE_WORKBOOK MAY_DISPATCH_CSV RESIDUAL_FORECAST_DIR}"
STORAGE_WORKBOOK="${2:?Usage: $0 SITE_WORKBOOK STORAGE_WORKBOOK MAY_DISPATCH_CSV RESIDUAL_FORECAST_DIR}"
DISPATCH_INPUT="${3:?Usage: $0 SITE_WORKBOOK STORAGE_WORKBOOK MAY_DISPATCH_CSV RESIDUAL_FORECAST_DIR}"
FORECAST_DIR="${4:?Usage: $0 SITE_WORKBOOK STORAGE_WORKBOOK MAY_DISPATCH_CSV RESIDUAL_FORECAST_DIR}"
HF_HOME="${HF_HOME:-${MODEL_CACHE:-$HOME/.cache/huggingface}}"
REVISION="29ec3766d36d6f73f0696f85560a422f50e8498c"
SNAPSHOT="${CHRONOS_MODEL_PATH:-${HF_HOME}/hub/models--amazon--chronos-2/snapshots/${REVISION}}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_DIR:-results/revenue_ablation/foshan_residual_controller_v5/${RUN_ID}}"
RESIDUAL_DATA="${RESIDUAL_DATA:-${FORECAST_DIR%/forecast}/data/signed_residual_15min.parquet}"
RESIDUAL_PREDICTIONS="${RESIDUAL_PREDICTIONS:-${FORECAST_DIR}/predictions_long.csv}"
APRIL_PV_PREDICTIONS="${APRIL_PV_PREDICTIONS:-${FORECAST_DIR}/frozen_pv_april_predictions.csv}"
MAY_PV_PREDICTIONS="${MAY_PV_PREDICTIONS:-results/zero_shot/foshan_chronos2/predictions_long.csv}"
MAY_PV_SELECTION="${MAY_PV_SELECTION:-results/zero_shot/foshan_chronos2/selected_configuration.json}"
GROSS_LOAD_PREDICTIONS="${GROSS_LOAD_PREDICTIONS:-results/load_forecast_ablation/foshan_april_select_may_controller_v5/load_predictions_long.csv}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/foshan_chronos2_residual.json}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

missing=0
required_files=(
  "${SITE_WORKBOOK}"
  "${STORAGE_WORKBOOK}"
  "${DISPATCH_INPUT}"
  "${RESIDUAL_DATA}"
  "${RESIDUAL_PREDICTIONS}"
  "${APRIL_PV_PREDICTIONS}"
  "${MAY_PV_PREDICTIONS}"
  "${MAY_PV_SELECTION}"
  "${GROSS_LOAD_PREDICTIONS}"
  "${TRAINING_CONFIG}"
  "${SNAPSHOT}/config.json"
  "${SNAPSHOT}/model.safetensors"
)
for path in "${required_files[@]}"; do
  if [[ -f "${path}" ]]; then
    echo "INPUT_OK: ${path}"
  else
    echo "INPUT_MISSING: ${path}"
    missing=1
  fi
done

TRAINED_ARGS=()
if [[ -n "${TRAINED_RUN_DIRS:-}" ]]; then
  for candidate_dir in ${TRAINED_RUN_DIRS}; do
    for path in \
      "${candidate_dir}/training_manifest.json" \
      "${candidate_dir}/april_predictions.csv"; do
      if [[ -f "${path}" ]]; then
        echo "INPUT_OK: ${path}"
      else
        echo "INPUT_MISSING: ${path}"
        missing=1
      fi
    done
    if [[ -d "${candidate_dir}/predictor" ]]; then
      echo "INPUT_OK: ${candidate_dir}/predictor"
    else
      echo "INPUT_MISSING: ${candidate_dir}/predictor"
      missing=1
    fi
    TRAINED_ARGS+=(--trained-run-dir "${candidate_dir}")
  done
fi
[[ "${missing}" == "0" ]] || exit 1

RESUME_ARGS=()
if [[ -e "${OUTPUT_ROOT}" && "${RESUME:-0}" != "1" ]]; then
  echo "Refusing to overwrite ${OUTPUT_ROOT}; set RESUME=1 only for a compatible run."
  exit 1
fi
[[ "${RESUME:-0}" == "1" ]] && RESUME_ARGS+=(--resume)

python -u -m src.evaluation.foshan_residual_revenue \
  --site-workbook "${SITE_WORKBOOK}" \
  --storage-workbook "${STORAGE_WORKBOOK}" \
  --dispatch-input "${DISPATCH_INPUT}" \
  --residual-data "${RESIDUAL_DATA}" \
  --residual-predictions "${RESIDUAL_PREDICTIONS}" \
  --april-pv-predictions "${APRIL_PV_PREDICTIONS}" \
  --may-pv-predictions "${MAY_PV_PREDICTIONS}" \
  --may-pv-selection "${MAY_PV_SELECTION}" \
  --gross-load-predictions "${GROSS_LOAD_PREDICTIONS}" \
  --training-config "${TRAINING_CONFIG}" \
  --model-path "${SNAPSHOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  "${RESUME_ARGS[@]}" \
  "${TRAINED_ARGS[@]}"

echo "Revenue comparison: ${OUTPUT_ROOT}/may_revenue_comparison.csv"
