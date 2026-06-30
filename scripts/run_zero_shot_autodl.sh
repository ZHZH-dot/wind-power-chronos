#!/usr/bin/env bash
set -euo pipefail

RAW_CSV="${RAW_CSV:-data/raw/sdwpf.csv}"
PROCESSED="${PROCESSED:-data/processed/sdwpf_hourly.parquet}"
COVARIATES="${COVARIATES:-Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv}"
RESULT_DIR="${RESULT_DIR:-results/zero_shot}"
LIMIT_ARGS=()

if [[ -n "${LIMIT_TURBINES:-}" ]]; then
  LIMIT_ARGS+=(--limit-turbines "${LIMIT_TURBINES}")
fi

python -m pip install --upgrade pip
python -m pip install "chronos-forecasting" "pandas[pyarrow]" numpy

python -m src.data.prepare_sdwpf \
  --input "${RAW_CSV}" \
  --output "${PROCESSED}" \
  --id-column TurbID \
  --day-column Day \
  --time-column Tmstamp \
  --target-column Patv \
  --covariates "${COVARIATES}" \
  --freq 1h

python -m src.models.chronos_zero_shot \
  --input "${PROCESSED}" \
  --output "${RESULT_DIR}/predictions_univariate.csv" \
  --mode univariate \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1,6,24,72 \
  --context-length 168 \
  --stride 24 \
  "${LIMIT_ARGS[@]}"

python -m src.models.chronos_zero_shot \
  --input "${PROCESSED}" \
  --output "${RESULT_DIR}/predictions_multivariate.csv" \
  --mode multivariate \
  --covariates "${COVARIATES}" \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1,6,24,72 \
  --context-length 168 \
  --stride 24 \
  "${LIMIT_ARGS[@]}"

python -m src.evaluation.evaluate \
  --predictions \
  "${RESULT_DIR}/predictions_univariate.csv" \
  "${RESULT_DIR}/predictions_multivariate.csv" \
  --ground-truth "${PROCESSED}" \
  --output "${RESULT_DIR}/metrics.csv"
