#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?Usage: $0 SIGNED_RESIDUAL_15MIN_PARQUET}"
CONFIG="${CONFIG:-configs/foshan_chronos2_residual.json}"
HF_HOME="${HF_HOME:-${MODEL_CACHE:-$HOME/.cache/huggingface}}"
REVISION="29ec3766d36d6f73f0696f85560a422f50e8498c"
SNAPSHOT="${CHRONOS_MODEL_PATH:-${HF_HOME}/hub/models--amazon--chronos-2/snapshots/${REVISION}}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_DIR:-results/fine_tune/foshan_chronos2_residual/lora_${RUN_ID}}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

[[ -f "${INPUT}" ]] || { echo "Residual input does not exist: ${INPUT}"; exit 1; }
[[ -f "${SNAPSHOT}/model.safetensors" ]] || { echo "Pinned snapshot is missing: ${SNAPSHOT}"; exit 1; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing to overwrite ${OUTPUT_ROOT}"; exit 1; }
mkdir -p "${OUTPUT_ROOT}"

nvidia-smi
python scripts/preflight_finetune_4090.py
python -m pytest tests
python -m src.training.foshan_residual_finetune \
  --input "${INPUT}" --config "${CONFIG}" --model-path "${SNAPSHOT}" \
  --fine-tune-mode lora --stage dry-run --output-dir "${OUTPUT_ROOT}/dry_run"
python -m src.training.foshan_residual_finetune \
  --input "${INPUT}" --config "${CONFIG}" --model-path "${SNAPSHOT}" \
  --fine-tune-mode lora --stage smoke --output-dir "${OUTPUT_ROOT}/smoke" \
  --dataloader-num-workers 0
python -m src.training.foshan_residual_finetune \
  --input "${INPUT}" --config "${CONFIG}" --model-path "${SNAPSHOT}" \
  --fine-tune-mode lora --stage search --output-dir "${OUTPUT_ROOT}/search" \
  --dataloader-num-workers 0

echo "LoRA candidate manifests: ${OUTPUT_ROOT}/search/*/training_manifest.json"
