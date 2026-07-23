#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet}"
CONFIG="${CONFIG:-configs/foshan_chronos2_lora.json}"
ZERO_SHOT_DIR="${ZERO_SHOT_DIR:-results/zero_shot/foshan_chronos2}"
RUN_NAME="${RUN_NAME:-foshan_chronos2_lora_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/fine_tune/${RUN_NAME}}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export CUDA_VISIBLE_DEVICES=0

if [[ ! -f "${INPUT}" ]]; then
  echo "Processed Foshan input does not exist: ${INPUT}"
  exit 1
fi
if [[ ! -f "${ZERO_SHOT_DIR}/predictions_long.csv" ]] \
  || [[ ! -f "${ZERO_SHOT_DIR}/selected_configuration.json" ]]; then
  echo "Completed frozen zero-shot outputs are missing under ${ZERO_SHOT_DIR}"
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite LoRA output directory: ${OUTPUT_DIR}"
  exit 1
fi

MODEL_ARGS=(--model-id amazon/chronos-2)
if [[ -n "${CHRONOS_MODEL_PATH:-}" ]]; then
  MODEL_ARGS=(--model-path "${CHRONOS_MODEL_PATH}")
fi

nvidia-smi
python scripts/preflight_finetune_4090.py
python -m pytest tests

COMMON_ARGS=(
  --input "${INPUT}"
  --config "${CONFIG}"
  --zero-shot-dir "${ZERO_SHOT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  "${MODEL_ARGS[@]}"
)

python -m src.training.foshan_chronos_finetune \
  "${COMMON_ARGS[@]}" \
  --stage dry-run

# One gradient step with batch size one, followed by one May forecast origin.
python -m src.training.foshan_chronos_finetune \
  "${COMMON_ARGS[@]}" \
  --stage smoke

# Train all configured March-April LoRA candidates, select on May, and evaluate
# the frozen winner exactly once on June.
python -m src.training.foshan_chronos_finetune \
  "${COMMON_ARGS[@]}" \
  --stage search
