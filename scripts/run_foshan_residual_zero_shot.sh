#!/usr/bin/env bash
set -euo pipefail

SITE_WORKBOOK="${1:?Usage: $0 SITE_WORKBOOK STORAGE_WORKBOOK MAY_DISPATCH_CSV}"
STORAGE_WORKBOOK="${2:?Usage: $0 SITE_WORKBOOK STORAGE_WORKBOOK MAY_DISPATCH_CSV}"
DISPATCH_INPUT="${3:?Usage: $0 SITE_WORKBOOK STORAGE_WORKBOOK MAY_DISPATCH_CSV}"
CONFIG="${CONFIG:-configs/foshan_chronos2_residual.json}"
HF_HOME="${HF_HOME:-${MODEL_CACHE:-$HOME/.cache/huggingface}}"
REVISION="29ec3766d36d6f73f0696f85560a422f50e8498c"
SNAPSHOT="${CHRONOS_MODEL_PATH:-${HF_HOME}/hub/models--amazon--chronos-2/snapshots/${REVISION}}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_DIR:-results/residual_forecast/foshan_chronos2/${RUN_ID}}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ ! -f "${SITE_WORKBOOK}" || ! -f "${STORAGE_WORKBOOK}" || ! -f "${DISPATCH_INPUT}" ]]; then
  echo "A source workbook or dispatch CSV does not exist."
  exit 1
fi
if [[ ! -f "${SNAPSHOT}/config.json" || ! -f "${SNAPSHOT}/model.safetensors" ]]; then
  echo "Pinned Chronos-2 snapshot is incomplete: ${SNAPSHOT}"
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite residual run: ${OUTPUT_ROOT}"
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

python -m pytest tests
python -m src.data.reconstruct_foshan_residual \
  --site-workbook "${SITE_WORKBOOK}" \
  --storage-workbook "${STORAGE_WORKBOOK}" \
  --dispatch-input "${DISPATCH_INPUT}" \
  --output-dir "${OUTPUT_ROOT}/data"

# One independent April issue first. No issue times are batched together.
python -m src.models.foshan_residual_zero_shot \
  --config "${CONFIG}" \
  --input "${OUTPUT_ROOT}/data/signed_residual_15min.parquet" \
  --model-path "${SNAPSHOT}" \
  --output-dir "${OUTPUT_ROOT}/smoke" \
  --stage smoke \
  --splits april \
  --candidates chronos2_residual_hourly_ctx672 \
  --skip-frozen-april-pv

# Full April-selection and May-test forecast generation.
python -m src.models.foshan_residual_zero_shot \
  --config "${CONFIG}" \
  --input "${OUTPUT_ROOT}/data/signed_residual_15min.parquet" \
  --processed-foshan-input "${PROCESSED_FOSHAN_INPUT:-results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet}" \
  --pv-selection "${PV_SELECTION:-results/foshan_chronos2/selected_configuration.json}" \
  --model-path "${SNAPSHOT}" \
  --output-dir "${OUTPUT_ROOT}/forecast" \
  --stage forecast \
  --splits april,may

echo "Residual data: ${OUTPUT_ROOT}/data/signed_residual_15min.parquet"
echo "Residual forecasts: ${OUTPUT_ROOT}/forecast/predictions_long.csv"
