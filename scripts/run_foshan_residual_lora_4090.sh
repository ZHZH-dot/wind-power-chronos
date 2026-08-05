#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?Usage: $0 {dry-run|smoke|search} SIGNED_RESIDUAL_15MIN_PARQUET}"
INPUT="${2:?Usage: $0 {dry-run|smoke|search} SIGNED_RESIDUAL_15MIN_PARQUET}"
case "${STAGE}" in
  dry-run|smoke|search) ;;
  *) echo "Stage must be dry-run, smoke, or search: ${STAGE}"; exit 2 ;;
esac
CONFIG="${CONFIG:-configs/foshan_chronos2_residual.json}"
HF_HOME="${HF_HOME:-${MODEL_CACHE:-$HOME/.cache/huggingface}}"
REVISION="29ec3766d36d6f73f0696f85560a422f50e8498c"
SNAPSHOT="${CHRONOS_MODEL_PATH:-${HF_HOME}/hub/models--amazon--chronos-2/snapshots/${REVISION}}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_DIR:-results/fine_tune/foshan_chronos2_residual/lora_${STAGE}_${RUN_ID}}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

[[ -f "${INPUT}" ]] || { echo "Residual input does not exist: ${INPUT}"; exit 1; }
for required in config.json model.safetensors; do
  [[ -f "${SNAPSHOT}/${required}" ]] || {
    echo "Pinned snapshot is missing: ${SNAPSHOT}/${required}"
    exit 1
  }
done
if [[ -e "${OUTPUT_ROOT}" && "${RESUME:-0}" != "1" ]]; then
  echo "Refusing to overwrite ${OUTPUT_ROOT}; set RESUME=1 only for a compatible run."
  exit 1
fi

python -u -m pytest tests
if [[ "${STAGE}" != "dry-run" ]]; then
  nvidia-smi
  python -u scripts/preflight_finetune_4090.py
fi

EXTRA_ARGS=()
[[ "${RESUME:-0}" == "1" ]] && EXTRA_ARGS+=(--resume)
[[ -n "${BATCH_SIZE_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--batch-size-override "${BATCH_SIZE_OVERRIDE}")
[[ -n "${MAX_CANDIDATES:-}" ]] && EXTRA_ARGS+=(--max-candidates "${MAX_CANDIDATES}")

python -u -m src.training.foshan_residual_finetune \
  --input "${INPUT}" --config "${CONFIG}" --model-path "${SNAPSHOT}" \
  --fine-tune-mode lora --stage "${STAGE}" --output-dir "${OUTPUT_ROOT}" \
  --dataloader-num-workers 0 "${EXTRA_ARGS[@]}"

echo "Residual LoRA ${STAGE} output: ${OUTPUT_ROOT}"
