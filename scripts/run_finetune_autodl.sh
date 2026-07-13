#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_finetune_autodl.sh /path/to/sdwpf_hourly_regularized.parquet"
  echo "Optional env vars: ENV_NAME, MODEL_ID, MODE, COVARIATES, STEPS, LEARNING_RATE, BATCH_SIZE, INFERENCE_BATCH_SIZE, MAX_TURBINES, OUTPUT_DIR, SPLIT_MANIFEST, DRY_RUN_ONLY"
  exit 1
fi

INPUT="$1"
ENV_NAME="${ENV_NAME:-wind-chronos-ft}"
MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
MODE="${MODE:-multivariate}"
COVARIATES="${COVARIATES:-Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv}"
SPLIT_CONFIG="${SPLIT_CONFIG:-configs/splits/sdwpf_70_10_20.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-data/processed/sdwpf_split_manifest.json}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-72}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-168}"
STEPS="${STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
MAX_TURBINES="${MAX_TURBINES:-}"
SEED="${SEED:-42}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
RUN_NAME="chronos2_lora_${MODE}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-results/fine_tune/${RUN_NAME}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${INPUT}" ]]; then
  echo "Processed SDWPF input does not exist: ${INPUT}"
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output directory: ${OUTPUT_DIR}"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -r requirements-finetune-autodl.txt
python -m pytest tests

COMMON_ARGS=(
  --input "${INPUT}"
  --split-config "${SPLIT_CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --model-id "${MODEL_ID}"
  --mode "${MODE}"
  --covariates "${COVARIATES}"
  --prediction-length "${PREDICTION_LENGTH}"
  --context-length "${CONTEXT_LENGTH}"
  --steps "${STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --batch-size "${BATCH_SIZE}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"
  --seed "${SEED}"
)

if [[ -f "${SPLIT_MANIFEST}" ]]; then
  COMMON_ARGS+=(--split-manifest "${SPLIT_MANIFEST}")
fi
if [[ -n "${MAX_TURBINES}" ]]; then
  COMMON_ARGS+=(--max-turbines "${MAX_TURBINES}")
fi

python -m src.training.chronos_finetune "${COMMON_ARGS[@]}" --dry-run
if [[ "${DRY_RUN_ONLY}" == "1" ]]; then
  echo "Dry run complete; skipping GPU fine-tuning."
  exit 0
fi

python -m src.training.chronos_finetune "${COMMON_ARGS[@]}"
