#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet}"
CONFIG="${CONFIG:-configs/foshan_chronos2_full_finetune.json}"
ZERO_SHOT_DIR="${ZERO_SHOT_DIR:-results/zero_shot/foshan_chronos2}"
LORA_RUN_DIR="${LORA_RUN_DIR:-results/fine_tune/foshan_chronos2_lora_20260723T225913Z}"
RUN_NAME="${RUN_NAME:-foshan_chronos2_full_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/full_fine_tune/${RUN_NAME}}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export CUDA_VISIBLE_DEVICES=0

if [[ ! -f "${INPUT}" ]]; then
  echo "Processed Foshan input does not exist: ${INPUT}"
  exit 1
fi
if [[ ! -f "${ZERO_SHOT_DIR}/predictions_long.csv" ]] \
  || [[ ! -f "${ZERO_SHOT_DIR}/selected_configuration.json" ]]; then
  echo "Frozen zero-shot artifacts are incomplete under ${ZERO_SHOT_DIR}"
  exit 1
fi
if [[ ! -f "${LORA_RUN_DIR}/search/june_predictions.csv" ]] \
  || [[ ! -f "${LORA_RUN_DIR}/search/selected_configuration.json" ]]; then
  echo "Frozen LoRA artifacts are incomplete under ${LORA_RUN_DIR}"
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
  --lora-run-dir "${LORA_RUN_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  "${MODEL_ARGS[@]}"
)

python -m src.training.foshan_chronos_full_finetune \
  "${COMMON_ARGS[@]}" \
  --stage dry-run

# One BF16 full-tuning gradient step and one May forecast origin.
python -m src.training.foshan_chronos_full_finetune \
  "${COMMON_ARGS[@]}" \
  --stage smoke

# Four isolated candidates. Completed candidates are reused when OUTPUT_DIR is
# supplied again; OOM retries change only batch size, in the order 4, 2, 1.
python -m src.training.foshan_chronos_full_finetune \
  "${COMMON_ARGS[@]}" \
  --stage search
