#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_zero_shot_autodl.sh /path/to/raw_sdwpf.csv"
  echo "Optional env vars: ENV_NAME, COVARIATES, RESULT_DIR, CONTEXT_LENGTH, STRIDE"
  exit 1
fi

RAW_CSV="$1"
ENV_NAME="${ENV_NAME:-wind-chronos}"
PROCESSED="${PROCESSED:-data/processed/sdwpf_hourly.parquet}"
COVARIATES="${COVARIATES:-Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv}"
RESULT_DIR="${RESULT_DIR:-results/zero_shot}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-168}"
STRIDE="${STRIDE:-24}"
HORIZONS=(1 6 24 72)

mkdir -p data/processed "${RESULT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install -r requirements-autodl.txt

python -m src.data.prepare_sdwpf \
  --input "${RAW_CSV}" \
  --output "${PROCESSED}" \
  --id-column TurbID \
  --day-column Day \
  --time-column Tmstamp \
  --target-column Patv \
  --covariates "${COVARIATES}" \
  --freq 1h

python -m pytest tests

python -m src.models.chronos_zero_shot \
  --input "${PROCESSED}" \
  --output "${RESULT_DIR}/smoke_univariate.csv" \
  --mode univariate \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons "${HORIZONS[@]}" \
  --context-length "${CONTEXT_LENGTH}" \
  --stride "${STRIDE}" \
  --max_turbines 1 \
  --max-windows-per-turbine 1

python -m src.models.chronos_zero_shot \
  --input "${PROCESSED}" \
  --output "${RESULT_DIR}/smoke_multivariate.csv" \
  --mode multivariate \
  --covariates "${COVARIATES}" \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons "${HORIZONS[@]}" \
  --context-length "${CONTEXT_LENGTH}" \
  --stride "${STRIDE}" \
  --max_turbines 1 \
  --max-windows-per-turbine 1

python -m src.evaluation.evaluate \
  --predictions \
  "${RESULT_DIR}/smoke_univariate.csv" \
  "${RESULT_DIR}/smoke_multivariate.csv" \
  --ground-truth "${PROCESSED}" \
  --output "${RESULT_DIR}/smoke_metrics.csv"

python -m src.models.chronos_zero_shot \
  --input "${PROCESSED}" \
  --output "${RESULT_DIR}/predictions_univariate.csv" \
  --mode univariate \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons "${HORIZONS[@]}" \
  --context-length "${CONTEXT_LENGTH}" \
  --stride "${STRIDE}"

python -m src.models.chronos_zero_shot \
  --input "${PROCESSED}" \
  --output "${RESULT_DIR}/predictions_multivariate.csv" \
  --mode multivariate \
  --covariates "${COVARIATES}" \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons "${HORIZONS[@]}" \
  --context-length "${CONTEXT_LENGTH}" \
  --stride "${STRIDE}"

python -m src.evaluation.evaluate \
  --predictions \
  "${RESULT_DIR}/predictions_univariate.csv" \
  "${RESULT_DIR}/predictions_multivariate.csv" \
  --ground-truth "${PROCESSED}" \
  --output "${RESULT_DIR}/metrics.csv"
