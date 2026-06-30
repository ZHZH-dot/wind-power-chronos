# Wind Power Chronos MVP

Production-oriented SDWPF wind power forecasting MVP using Chronos-2 zero-shot inference.

## Environment Setup

Use Python 3.10+ on AutoDL or a similar CUDA machine.

```bash
python -m pip install --upgrade pip
python -m pip install "chronos-forecasting" "pandas[pyarrow]" numpy pytest
```

The inference script loads the model with:

```python
from chronos import Chronos2Pipeline

Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
```

Local tests are CPU-only and do not download the model.

## Data Layout

Place the raw SDWPF CSV at:

```text
data/raw/sdwpf.csv
```

Raw files under `data/raw/` are treated as read-only. Prepared data is written to `data/processed/`, and predictions/metrics are written to `results/`.

The processed table uses:

```text
id, timestamp, target, optional covariates...
```

Default SDWPF covariates:

```text
Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv
```

## Prepare SDWPF

```bash
python -m src.data.prepare_sdwpf \
  --input data/raw/sdwpf.csv \
  --output data/processed/sdwpf_hourly.parquet \
  --id-column TurbID \
  --day-column Day \
  --time-column Tmstamp \
  --target-column Patv \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --freq 1h
```

For a raw file that already has a datetime column, use `--timestamp-column` instead of `--day-column` and `--time-column`.

## Smoke Test

Run one turbine and one rolling window first:

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly.parquet \
  --output results/smoke_multivariate.csv \
  --mode multivariate \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1,6,24,72 \
  --context-length 168 \
  --stride 24 \
  --limit-turbines 1 \
  --max-windows-per-turbine 1
```

## Univariate Example

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly.parquet \
  --output results/predictions_univariate.csv \
  --mode univariate \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1,6,24,72 \
  --context-length 168 \
  --stride 24
```

## Multivariate Example

Measured future covariates are not used by default. The multivariate run passes past covariates only, which avoids leakage for SDWPF zero-shot evaluation.

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly.parquet \
  --output results/predictions_multivariate.csv \
  --mode multivariate \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --model_id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1,6,24,72 \
  --context-length 168 \
  --stride 24
```

Only use `--allow-future-covariates` when those future covariates are realistically available at prediction time.

## Evaluate

```bash
python -m src.evaluation.evaluate \
  --predictions results/predictions_univariate.csv results/predictions_multivariate.csv \
  --ground-truth data/processed/sdwpf_hourly.parquet \
  --output results/zero_shot_metrics.csv
```

The result table reports MAE, RMSE, NMAE, and NRMSE by mode and horizon.

## Full AutoDL Run

```bash
bash scripts/run_zero_shot_autodl.sh
```

Useful overrides:

```bash
RAW_CSV=/root/autodl-tmp/sdwpf.csv LIMIT_TURBINES=1 bash scripts/run_zero_shot_autodl.sh
```

Remove `LIMIT_TURBINES` for the full run.

## Tests

```bash
python -m pytest tests
```
