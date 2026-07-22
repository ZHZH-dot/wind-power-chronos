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

A100 is not required. The fine-tuning launcher targets one NVIDIA RTX 4090 with 24 GB VRAM and restricts training to `CUDA_VISIBLE_DEVICES=0`; it does not use multi-GPU training or `device_map="auto"`.

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

## Foshan PV-Storage Zero-Shot Benchmark

This route is a separate, zero-shot-only Chronos-2 benchmark. It does not alter
the SDWPF split or fine-tuning workflow. The primary workbook supplies two
signals:

- `光伏` is mapped to `pv_kw`.
- `负荷` is mapped to `net_grid_kw`, provisionally classified as bidirectional
  grid exchange. It is not confirmed gross factory load. Negative values are
  preserved and the field is never clipped or renamed `gross_load_kw`.

The storage workbook is audit-only. Its five-minute total active power is
aggregated by arithmetic mean into left-closed 15-minute intervals. The
diagnostic `gross_load_proxy_kw = net_grid_kw + pv_kw + pcs_kw` is not a
forecast target.

Install the isolated zero-shot environment dependencies. The exact Chronos
release is pinned so this route does not force a dependency change in the
AutoGluon fine-tuning environment:

```bash
python -m pip install -r requirements-foshan-zero-shot.txt
```

### Foshan Data Audit And Baselines

Pass the workbook paths explicitly; the code never writes under `data/raw/`.

```bash
python -m src.data.prepare_foshan \
  --source-workbook 'data/raw/光伏与负荷数据_202603-06.xlsx' \
  --storage-workbook 'data/raw/储能数据20260707230737.xlsx' \
  --output results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet \
  --audit-dir results/zero_shot/foshan_chronos2

python -m src.models.foshan_chronos_zero_shot \
  --config configs/foshan_chronos2_zero_shot.json \
  --processed-input results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet \
  --output-dir results/zero_shot/foshan_chronos2 \
  --stage baselines
```

Preparation sorts the reverse-chronological source, localizes timestamps to
`Asia/Shanghai`, and creates an exact 15-minute grid without globally filling
targets. `pv_kw_raw` is retained while the physical target is clipped to
`[0, 1700]` kW. Every negative PV reading is written to
`pv_negative_readings.csv`. Signed `net_grid_kw` is unchanged.

At each origin, context ends at 23:45 and contains timestamps strictly before
the 00:00 issue time. Missing context values may only be forward-filled for two
15-minute intervals; no backward-fill or interpolation can cross an issue
boundary. Origins with unresolved context gaps are skipped and reported.

The causal baselines use the same May and June origins: PV zero, last-value
persistence, previous-day slot, previous-week slot, and the mean of the same
slot over the preceding four weeks. Deterministic baselines leave P10/P90
missing rather than inventing uncertainty.

### Foshan Selection And Test Protocol

`configs/foshan_chronos2_zero_shot.json` fixes the benchmark contract:

- 15-minute frequency and all 96 next-day steps;
- P10/P50/P90 and context candidates 672, 1344, and 2688 points;
- May 2026 for configuration/context selection only;
- June 2026 as the untouched final test;
- calendar covariates only as known-future inputs;
- univariate PV, calendar-informed PV, provisional calendar-informed grid, and
  official multi-target joint PV/grid configurations.

The joint configuration passes `target=["pv_kw", "net_grid_kw"]` and maps
Chronos output through `target_name`. Future dataframes contain only numeric
calendar columns. Optional real weather forecasts can later be supplied with
`--weather-covariates ghi,dni,dhi,cloud_cover,temperature` after those columns
are added to the processed table; no weather is fabricated or downloaded.

May selection uses postprocessed PV WAPE, then PV-active MAE (`y_true > 1 kW`),
then model name and context length as deterministic tie-breaks. June never
participates in selection. Grid selection is secondary and remains provisional.

Metrics include MAE, RMSE, WAPE, causal seasonal MASE, bias, P10/P50/P90
pinball loss, mean pinball loss, P10-P90 coverage, interval width, and PV-active
MAE/RMSE/WAPE. They are saved both in aggregate and for every horizon step
1-96. Missing target rows are not scored. MAPE is intentionally omitted because
PV is zero at night and grid exchange crosses zero.

### Foshan One-Origin Smoke

On one CUDA GPU, run the safety smoke against an already prepared table:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.models.foshan_chronos_zero_shot \
  --config configs/foshan_chronos2_zero_shot.json \
  --processed-input results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet \
  --output-dir results/zero_shot/foshan_chronos2 \
  --model-id amazon/chronos-2 \
  --device-map cuda \
  --stage smoke
```

For a restricted server, set `CHRONOS_MODEL_PATH` or pass `--model-path` to a
complete local `amazon/chronos-2` directory. Loading still uses
`Chronos2Pipeline.from_pretrained(...)`; the repository contains no copied
Chronos source or direct Hugging Face download URLs.

### Foshan Full AutoDL Run

The launcher creates a separate conda environment, pins
`chronos-forecasting==2.3.1`, exposes only GPU 0, prints the CUDA/BF16 preflight,
runs tests, prepares and audits the workbooks, completes all baselines, runs a
one-origin Chronos smoke, selects on May, and evaluates the frozen choice on
June. One loaded pipeline is reused for the smoke, selection configurations,
and final test.

```bash
HF_HOME=/data/GDUT_stu/.cache/huggingface \
CHRONOS_MODEL_PATH=/data/GDUT_stu/models/chronos-2 \
bash scripts/run_foshan_zero_shot_autodl.sh \
  '/path/to/光伏与负荷数据_202603-06.xlsx' \
  '/path/to/储能数据20260707230737.xlsx'
```

Without a local model path, omit `CHRONOS_MODEL_PATH` and the launcher uses
`amazon/chronos-2`. The required outputs are written under
`results/zero_shot/foshan_chronos2/`, including audits, the processed parquet,
long-form predictions, May/June metrics, per-horizon metrics, metadata,
environment freeze, report, and representative high-output, variable-output,
and median-output PV-day plots. Plot labels make no unsupported clear/cloudy
claim.

The equivalent single-process command below computes baselines first, runs the
one-origin smoke, then reuses that pipeline for May selection and frozen June
evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.models.foshan_chronos_zero_shot \
  --config configs/foshan_chronos2_zero_shot.json \
  --source-workbook '/path/to/光伏与负荷数据_202603-06.xlsx' \
  --storage-workbook '/path/to/储能数据20260707230737.xlsx' \
  --output-dir results/zero_shot/foshan_chronos2 \
  --model-id amazon/chronos-2 \
  --device-map cuda \
  --stage all
```

## Chronos-2 LoRA Fine-Tuning

Fine-tuning reuses the same global chronological benchmark. `configs/splits/sdwpf_70_10_20.json` defines the 70% train, 10% validation, and 20% test ratios plus the 72-hour prediction length, 168-hour context, benchmark horizons, hourly frequency, and seed 42.

Only timestamps through the train boundary are passed as `train_data`. AutoGluon receives cumulative data through the validation boundary as `tuning_data`, which provides historical context for validation without exposing test targets. The resolved boundaries are copied to `<output-dir>/resolved_split_manifest.json`. If `--split-manifest` is supplied, it must match the input data and is reused exactly.

Measured SDWPF covariates are treated as past-only covariates. Regularized rows remain present, but targets marked `is_imputed_target` are masked as missing supervision. The training script never produces or scores test forecasts; subsequent model comparison must use the existing test-window and evaluation functions.

Install the AutoDL fine-tuning environment:

```bash
python -m pip install -r requirements-finetune-autodl.txt
```

The fine-tuning requirements use `autogluon.timeseries>=1.5,<1.6` with `chronos-forecasting>=2.2.2,<2.4`. This is AutoGluon 1.5's compatible Chronos range and includes the Chronos 2.1.0 fix for past-only covariates.

Run the model-free 4090 preflight before loading Chronos-2. It prints `nvidia-smi`, Python, PyTorch, CUDA, AutoGluon, and Chronos versions, CUDA availability, the visible GPU and VRAM, and BF16 support:

```bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python scripts/preflight_finetune_4090.py
```

The launcher selects BF16 when `torch.cuda.is_bf16_supported()` is true and otherwise uses FP16. The initial full-run batch size is 16; `BATCH_SIZE`, `INFERENCE_BATCH_SIZE`, and `DATALOADER_NUM_WORKERS` can override the launcher defaults.

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

Validate the complete processed dataset without downloading or training the model:

```bash
CUDA_VISIBLE_DEVICES=0 DRY_RUN_ONLY=1 \
OUTPUT_DIR=results/fine_tune/rtx4090_dry_run \
bash scripts/run_finetune_autodl.sh \
  data/processed/sdwpf_hourly_regularized.parquet
```

The dry-run report includes the source and selected turbine counts, hourly-frequency check, exact train/validation/test boundaries, past-covariate columns, imputation-mask counts, leakage check, and output paths.

Next GPU smoke fine-tune with five turbines, 100 update steps, batch size 8, and no data-loader subprocesses:

```bash
CUDA_VISIBLE_DEVICES=0 MAX_TURBINES=5 STEPS=100 BATCH_SIZE=8 \
DATALOADER_NUM_WORKERS=0 \
OUTPUT_DIR=results/fine_tune/chronos2_lora_multivariate_5t_100s \
bash scripts/run_finetune_autodl.sh \
  data/processed/sdwpf_hourly_regularized.parquet
```

If batch size 8 causes CUDA OOM, rerun with `BATCH_SIZE=4` and a new `OUTPUT_DIR`.

Full multivariate LoRA run with the default 1000 steps and initial batch size 16:

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=16 DATALOADER_NUM_WORKERS=0 \
OUTPUT_DIR=results/fine_tune/chronos2_lora_multivariate_full \
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

### Fine-Tuned Test Evaluation

Evaluate a saved LoRA predictor with one leak-free test window per turbine before evaluating all test windows. The evaluator reads the expected turbine count from the saved `run_config.json`, so the same command supports both five-turbine smoke predictors and the 134-turbine full predictor. If `--max-turbines` is supplied, it must equal the saved count. The evaluator reloads `Chronos2LoRA`, verifies the adapter checkpoint and saved configuration, reuses the run's resolved split manifest, and writes zero-shot-compatible predictions plus the existing metric table. Measured SDWPF variables are passed only as historical covariates, and imputed targets are excluded from scoring by default.

```bash
HF_HOME=/data/GDUT_stu/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
python -m src.evaluation.chronos_finetune_predict \
  --predictor-path results/fine_tune/chronos2_lora_multivariate_134t_1000s/predictor \
  --input src/data/processed/sdwpf_hourly_regularized.parquet \
  --split-config configs/splits/sdwpf_70_10_20.json \
  --split-manifest results/fine_tune/chronos2_lora_multivariate_134t_1000s/resolved_split_manifest.json \
  --output results/fine_tune/chronos2_lora_multivariate_134t_1000s/test_predictions_one_window.csv \
  --metrics-output results/fine_tune/chronos2_lora_multivariate_134t_1000s/test_metrics_one_window.csv \
  --metadata-output results/fine_tune/chronos2_lora_multivariate_134t_1000s/test_metadata_one_window.json \
  --mode multivariate \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --prediction-length 72 \
  --context-length 168 \
  --horizons 1 6 24 72 \
  --inference-batch-size 64 \
  --stride 24 \
  --max-windows-per-turbine 1
```

The metadata JSON records the saved expected turbine count, all deterministically selected turbine IDs, predictor model name, adapter checkpoint, split boundaries, covariate policy, and output paths. Existing five-turbine runs may optionally pass `--max-turbines 5`; the 134-turbine run may pass `--max-turbines 134`. Omitting the option is preferred because the saved run configuration remains authoritative. Remove only `--max-windows-per-turbine 1` after the one-window evaluation succeeds and its outputs have been inspected.

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
