#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_foshan_zero_shot_autodl.sh /path/to/pv_grid.xlsx [/path/to/storage.xlsx]"
  echo "The primary workbook path is required and is never inferred from a fixed filename."
  echo "Optional env vars: ENV_NAME, OUTPUT_DIR, RUN_ID, CHRONOS_MODEL_PATH, HF_HOME"
  exit 1
fi

SOURCE_WORKBOOK="$1"
STORAGE_WORKBOOK="${2:-}"
ENV_NAME="${ENV_NAME:-foshan-chronos2}"
OUTPUT_DIR="${OUTPUT_DIR:-results/zero_shot/foshan_chronos2}"
PROCESSED="${OUTPUT_DIR}/processed_foshan_15min.parquet"
CONFIG="${CONFIG:-configs/foshan_chronos2_zero_shot.json}"
RUN_ID="${RUN_ID:-foshan_$(date -u +%Y%m%dT%H%M%SZ)}"

export CUDA_VISIBLE_DEVICES=0

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -r requirements-foshan-zero-shot.txt

nvidia-smi
python - <<'PY'
import importlib.metadata as metadata
import platform
import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("chronos-forecasting:", metadata.version("chronos-forecasting"))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", props.total_memory / 1024**3)
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

python -m pytest tests

PREPARE_ARGS=(
  --source-workbook "${SOURCE_WORKBOOK}"
  --output "${PROCESSED}"
  --audit-dir "${OUTPUT_DIR}"
)
if [[ -n "${STORAGE_WORKBOOK}" ]]; then
  PREPARE_ARGS+=(--storage-workbook "${STORAGE_WORKBOOK}")
fi

# Stage 1: deterministic data preparation and data/storage audit.
python -m src.data.prepare_foshan "${PREPARE_ARGS[@]}"

# Stage 2: complete May and June causal baselines, without loading Chronos.
python -m src.models.foshan_chronos_zero_shot \
  --config "${CONFIG}" \
  --processed-input "${PROCESSED}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --stage baselines

MODEL_ARGS=(--model-id amazon/chronos-2)
if [[ -n "${CHRONOS_MODEL_PATH:-}" ]]; then
  MODEL_ARGS=(--model-path "${CHRONOS_MODEL_PATH}")
fi

# Stage 3: one-origin smoke, complete May selection, then frozen June test.
# The pipeline is loaded once and reused throughout this process.
python -m src.models.foshan_chronos_zero_shot \
  --config "${CONFIG}" \
  --processed-input "${PROCESSED}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --device-map cuda \
  --stage chronos \
  "${MODEL_ARGS[@]}"
