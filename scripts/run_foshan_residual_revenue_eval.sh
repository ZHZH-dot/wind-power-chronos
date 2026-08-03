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
export CUDA_VISIBLE_DEVICES=0
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TRAINED_ARGS=()
if [[ -n "${TRAINED_RUN_DIRS:-}" ]]; then
  for candidate_dir in ${TRAINED_RUN_DIRS}; do
    TRAINED_ARGS+=(--trained-run-dir "${candidate_dir}")
  done
fi

python -m src.evaluation.foshan_residual_revenue \
  --site-workbook "${SITE_WORKBOOK}" \
  --storage-workbook "${STORAGE_WORKBOOK}" \
  --dispatch-input "${DISPATCH_INPUT}" \
  --residual-data "${RESIDUAL_DATA}" \
  --residual-predictions "${FORECAST_DIR}/predictions_long.csv" \
  --april-pv-predictions "${FORECAST_DIR}/frozen_pv_april_predictions.csv" \
  --model-path "${SNAPSHOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  "${TRAINED_ARGS[@]}"

echo "Revenue comparison: ${OUTPUT_ROOT}/may_revenue_comparison.csv"
