# Wind Power Chronos MVP

Production-oriented SDWPF wind power forecasting MVP using Chronos-2 zero-shot inference and LoRA fine-tuning.

Zero-shot remains the benchmark baseline. Fine-tuning uses AutoGluon TimeSeries with the Chronos-2 model only.

## Model Constraint

Use Chronos-2 only:

```python
from chronos import Chronos2Pipeline

Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
```

LoRA fine-tuning uses:

```python
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
```

The default `--model-id` is `amazon/chronos-2`. The same argument can also be a local model directory such as `/data/GDUT_stu/models/chronos-2`. The legacy spelling `--model_id` is still accepted. The excluded model families are not part of this repository.

## Data Layout

Put the raw SDWPF CSV anywhere readable on the machine, commonly:

```text
data/raw/<your-sdwpf-file>.csv
```

Raw files under `data/raw/` are read-only inputs. Prepared data is written to `data/processed/`; predictions and metrics are written to `results/`.

The regularized processed table schema is:

```text
id, timestamp, target, optional covariates..., is_imputed_target
```

Default SDWPF covariates:

```text
Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv
```

## Local CPU Test

Local tests do not load Chronos-2 and do not require a GPU.

```bash
python -m pip install -r requirements.txt
python -m pytest tests
```

## Prepare SDWPF

For the standard SDWPF `Day` plus `Tmstamp` format:

```bash
python -m src.data.prepare_sdwpf \
  --input data/raw/<your-sdwpf-file>.csv \
  --output data/processed/sdwpf_hourly.parquet \
  --id-column TurbID \
  --day-column Day \
  --time-column Tmstamp \
  --target-column Patv \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --freq 1h
```

If the raw file already has a datetime column, use `--timestamp-column` instead of `--day-column` and `--time-column`.

SDWPF can still have missing hourly rows after resampling. Chronos-2 requires regular hourly frequency, so zero-shot benchmark runs must prepare the parquet with `--regularize-hourly`. Each turbine is regularized only between its own first and last timestamp. Inserted or originally missing targets are interpolated for model context and marked by `is_imputed_target`; they are not scored by default.

```bash
python -m src.data.prepare_sdwpf \
  --input data/raw/<your-sdwpf-file>.csv \
  --output data/processed/sdwpf_hourly_regularized.parquet \
  --id-column TurbID \
  --day-column Day \
  --time-column Tmstamp \
  --target-column Patv \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --freq 1h \
  --regularize-hourly
```

Create the exact split manifest once for this processed dataset:

```bash
python -m src.evaluation.splits \
  --input data/processed/sdwpf_hourly_regularized.parquet \
  --config configs/sdwpf_benchmark.json \
  --output data/processed/sdwpf_split_manifest.json
```

## Benchmark Protocol

`configs/sdwpf_benchmark.json` defines the reusable SDWPF benchmark. Sorted unique timestamps across the complete wind farm are divided chronologically: the first 70% are train, the next 10% are validation, and the final 20% are test. `data/processed/sdwpf_split_manifest.json` stores the resulting exact global boundaries so Chronos zero-shot, future statistical baselines, and future fine-tuning can use the same periods.

Chronos zero-shot may use historical context before the test boundary, but it emits predictions only for test targets. A window is used only when its complete maximum forecast horizon remains in the test period. The benchmark horizons remain 1, 6, 24, and 72 hours.

## AutoDL GPU Run

Clone the repo on AutoDL, place or mount the raw SDWPF CSV, then run:

```bash
bash scripts/run_zero_shot_autodl.sh /root/autodl-tmp/<your-sdwpf-file>.csv
```

The script creates a conda environment named `wind-chronos`, installs `requirements-autodl.txt`, prepares and regularizes SDWPF, persists the benchmark split, runs CPU-only tests, runs one-turbine smoke tests, then runs full univariate and multivariate zero-shot evaluation.

`requirements-autodl.txt` installs Chronos from the official GitHub repository. If installing manually, use:

```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

Quick diagnostics:

```bash
python -c "from chronos import Chronos2Pipeline; print('Chronos2Pipeline import OK')"
python -c "from chronos import Chronos2Pipeline; p=Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cuda'); print('Chronos-2 loaded OK')"
```

Override defaults when needed:

```bash
ENV_NAME=wind-chronos \
COVARIATES=Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
RESULT_DIR=results/zero_shot \
bash scripts/run_zero_shot_autodl.sh /root/autodl-tmp/<your-sdwpf-file>.csv
```

A100 is not required for the zero-shot one-turbine smoke test. A single GPU with sufficient memory is recommended for LoRA fine-tuning; the AutoDL script restricts training to one visible GPU.

## First Smoke Test

Run one turbine and one rolling window before the full evaluation:

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly_regularized.parquet \
  --output results/zero_shot/smoke_multivariate.csv \
  --mode multivariate \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --model-id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1 6 24 72 \
  --context-length 168 \
  --stride 24 \
  --benchmark-config configs/sdwpf_benchmark.json \
  --split-manifest data/processed/sdwpf_split_manifest.json \
  --max_turbines 1 \
  --max-windows-per-turbine 1
```

## Full SDWPF Zero-Shot Run

Univariate baseline:

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly_regularized.parquet \
  --output results/zero_shot/predictions_univariate.csv \
  --mode univariate \
  --model-id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1 6 24 72 \
  --context-length 168 \
  --stride 24 \
  --benchmark-config configs/sdwpf_benchmark.json \
  --split-manifest data/processed/sdwpf_split_manifest.json
```

Multivariate covariate-informed run:

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly_regularized.parquet \
  --output results/zero_shot/predictions_multivariate.csv \
  --mode multivariate \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --model-id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1 6 24 72 \
  --context-length 168 \
  --stride 24 \
  --benchmark-config configs/sdwpf_benchmark.json \
  --split-manifest data/processed/sdwpf_split_manifest.json
```

Measured future covariates are not used by default. Only pass `--allow-future-covariates` if those values would realistically be available at prediction time.

Evaluate both modes:

```bash
python -m src.evaluation.evaluate \
  --predictions \
  results/zero_shot/predictions_univariate.csv \
  results/zero_shot/predictions_multivariate.csv \
  --ground-truth data/processed/sdwpf_hourly_regularized.parquet \
  --rated-capacity-kw 1500 \
  --output results/zero_shot/metrics.csv
```

Evaluation excludes `is_imputed_target == True` rows unless `--include-imputed-targets` is explicitly supplied for diagnostics. It reports `n_scored`, `n_excluded_imputed`, MAE, RMSE, mean bias, `nmae_capacity`, and `nrmse_capacity`. A `rated_capacity_kw` column is used when present; otherwise SDWPF defaults to 1500 kW through `--rated-capacity-kw`.

Prediction files preserve Chronos-2 `p10`, `p50`, and `p90`; `y_pred` remains an alias for `p50`. Point metrics use `p50`. Probabilistic metrics include pinball loss at each quantile, mean pinball loss, P10-P90 interval coverage, and mean interval width.

## Chronos-2 LoRA Fine-Tuning

Fine-tuning reuses the same global chronological benchmark. `configs/splits/sdwpf_70_10_20.json` defines the 70% train, 10% validation, and 20% test ratios plus the 72-hour prediction length, 168-hour context, benchmark horizons, hourly frequency, and seed 42.

Only timestamps through the train boundary are passed as `train_data`. AutoGluon receives cumulative data through the validation boundary as `tuning_data`, which provides historical context for validation without exposing test targets. The resolved boundaries are copied to `<output-dir>/resolved_split_manifest.json`. If `--split-manifest` is supplied, it must match the input data and is reused exactly.

Measured SDWPF covariates are treated as past-only covariates. Regularized rows remain present, but targets marked `is_imputed_target` are masked as missing supervision. The training script never produces or scores test forecasts; subsequent model comparison must use the existing test-window and evaluation functions.

Install the AutoDL fine-tuning environment:

```bash
python -m pip install -r requirements-finetune-autodl.txt
```

Run a CPU-only data and leakage check. This does not import AutoGluon or load the model:

```bash
python -m src.training.chronos_finetune \
  --input data/processed/sdwpf_hourly_regularized.parquet \
  --split-config configs/splits/sdwpf_70_10_20.json \
  --split-manifest data/processed/sdwpf_split_manifest.json \
  --output-dir results/fine_tune/dry_run \
  --model-id amazon/chronos-2 \
  --mode multivariate \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --prediction-length 72 \
  --context-length 168 \
  --max-turbines 1 \
  --dry-run
```

First GPU smoke fine-tune with one turbine and ten update steps:

```bash
MAX_TURBINES=1 STEPS=10 \
bash scripts/run_finetune_autodl.sh \
  data/processed/sdwpf_hourly_regularized.parquet
```

Full multivariate LoRA run with the default 1000 steps:

```bash
bash scripts/run_finetune_autodl.sh \
  data/processed/sdwpf_hourly_regularized.parquet
```

Use a local Chronos-2 directory when Hugging Face access is unavailable:

```bash
MODEL_ID=/data/GDUT_stu/models/chronos-2 \
OUTPUT_DIR=results/fine_tune/chronos2_lora_multivariate_local \
bash scripts/run_finetune_autodl.sh \
  data/processed/sdwpf_hourly_regularized.parquet
```

Univariate LoRA uses the same split and simply omits past covariates:

```bash
MODE=univariate OUTPUT_DIR=results/fine_tune/chronos2_lora_univariate \
bash scripts/run_finetune_autodl.sh \
  data/processed/sdwpf_hourly_regularized.parquet
```

Each run writes `run_config.json`, `resolved_split_manifest.json`, and the AutoGluon predictor under its output directory. The script refuses to overwrite an existing output directory.

## If Hugging Face Is Unavailable

Do not use `https://huggingface.co/amazon/chronos-2/resolve/main` directly. That URL is not a valid model source in this pipeline. The code should load models only through `Chronos2Pipeline.from_pretrained(...)`.

Option A: install Chronos code from GitHub.

```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

Option B: pre-download the model on a machine that can access Hugging Face, then copy the local model folder to the server. Pass that folder as `--model-id`:

```bash
python src/models/chronos_zero_shot.py \
  --input data/processed/sdwpf_hourly_regularized.parquet \
  --output results/zero_shot/chronos2_smoke_univariate.csv \
  --model-id /data/GDUT_stu/models/chronos-2 \
  --device-map cuda \
  --mode univariate \
  --horizons 24 \
  --context-length 168 \
  --max-turbines 1 \
  --max-windows-per-turbine 3
```

Option C: set Hugging Face mirror or cache environment variables if they are available in the server environment, then keep using `--model-id amazon/chronos-2`.
