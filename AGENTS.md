# AGENTS.md

## Project goal

We are building a production-oriented wind power forecasting MVP using Chronos-2 on SDWPF first.

The project is mainly multivariate / covariate-informed wind power forecasting.

The Forecasting Agent supports Chronos-2 zero-shot inference and Chronos-2 LoRA fine-tuning through AutoGluon TimeSeries. Do not add other agents or forecasting model families unless explicitly asked.

## Model rule

Use Chronos-2 only for this stage.

Required for zero-shot inference:
- package: chronos-forecasting
- import: `from chronos import Chronos2Pipeline`
- model id: `amazon/chronos-2`
- loading style: `Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")`

Required for fine-tuning:
- package: `autogluon.timeseries`
- imports: `TimeSeriesDataFrame`, `TimeSeriesPredictor`
- AutoGluon model key: `Chronos2`
- base model: `amazon/chronos-2` or a user-provided local Chronos-2 directory
- fine-tuning mode: LoRA

Do not use:
- original Chronos T5 models
- Chronos-Bolt models
- custom copied Chronos source code
- Mamba
- PatchTST
- 2DxFormer

## Repository structure

Follow this structure:

- `configs/`: YAML config files
- `data/raw/`: raw data, never modified by code
- `data/processed/`: processed parquet/csv data
- `data/external/`: optional external data such as NWP
- `src/data/`: dataset loading and preprocessing
- `src/models/`: model wrappers and inference code
- `src/training/`: Chronos-2 LoRA fine-tuning logic
- `src/evaluation/`: metrics and evaluation scripts
- `src/utils/`: shared utilities
- `notebooks/`: exploratory analysis only
- `experiments/`: experiment logs
- `scripts/`: runnable shell scripts
- `tests/`: unit tests
- `docker/`: Docker-related files
- `deployment/`: API/deployment files
- `docs/`: documentation

## Coding rules

- Keep code simple and production-readable.
- Use type hints.
- Use argparse for runnable scripts.
- Do not hardcode local absolute paths.
- Never modify files under `data/raw/`.
- Save generated outputs under `data/processed/`, `results/`, or `experiments/`.
- Add small tests where possible.
- Add README instructions for AutoDL.
- Do not assume GPU is available in the Codex environment. Tests should be CPU-only and tiny.

## Dataset format rule

The processed SDWPF dataset should support both univariate and multivariate modes.

Required columns:
- `id`: turbine identifier
- `timestamp`: datetime
- `target`: active power to forecast

Optional covariate columns:
- wind speed
- wind direction
- temperature
- nacelle direction
- pitch angle
- reactive power
- other available SDWPF features

The preprocessing script should allow covariate column names to be specified by argparse or YAML config.

## Multivariate / covariate rule

Chronos-2 should be run in two modes:

1. Univariate baseline mode:
   - input columns: `id`, `timestamp`, `target`

2. Multivariate covariate-informed mode:
   - input columns: `id`, `timestamp`, `target`, plus selected covariates

Avoid data leakage:
- Past covariates are allowed.
- Future covariates should only be used if they would realistically be available at prediction time.
- For SDWPF zero-shot evaluation, default to no future measured covariates unless explicitly enabled.

## Zero-shot experiment rules

The first experiment should:

1. Read raw SDWPF data.
2. Convert it into a standard forecasting table with:
   - `id`
   - `timestamp`
   - `target`
   - optional covariates
3. Resample each turbine to hourly frequency by default.
4. Preserve selected covariates during resampling.
5. Run rolling-window Chronos-2 zero-shot inference.
6. Support horizons:
   - 1
   - 6
   - 24
   - 72
7. Support both:
   - univariate zero-shot
   - multivariate/covariate-informed zero-shot
8. Save predictions.
9. Compute point, capacity-normalized, bias, pinball, and interval metrics with imputed targets excluded by default.
10. Produce a simple result table comparing univariate vs multivariate mode.

## Fine-tuning rules

- Reuse the global chronological timestamp split and resolved manifest from evaluation.
- Train only on the first 70% of global timestamps.
- Use cumulative data through the validation boundary as validation context.
- Never pass test targets to AutoGluon `fit`, tuning, checkpoint selection, or refit.
- Preserve hourly rows and mask `is_imputed_target` rows as missing supervision.
- Treat measured SDWPF covariates as past-only covariates.
- Use only the AutoGluon `Chronos2` model with `fine_tune_mode="lora"` and ensembles disabled.
- Dry runs and unit tests must not import AutoGluon, load Chronos-2, or require a GPU.
- Save fine-tuning outputs under `results/fine_tune/` and never overwrite an existing predictor.

## Acceptance criteria

The task is finished only if:

- `src/data/prepare_sdwpf.py` exists.
- `src/models/chronos_zero_shot.py` exists.
- `src/evaluation/metrics.py` exists.
- `src/evaluation/evaluate.py` exists.
- `scripts/run_zero_shot_autodl.sh` exists.
- `README.md` explains how to run on AutoDL.
- The code supports `--model_id amazon/chronos-2`.
- The default model is Chronos-2, not Chronos-Bolt or original Chronos.
- The code supports a `--covariates` argument or YAML config field.
- The README includes both univariate and multivariate examples.
- `src/training/chronos_finetune.py` supports CPU-only `--dry-run` validation.
- `configs/splits/sdwpf_70_10_20.json` defines the shared fine-tuning split contract.
- `scripts/run_finetune_autodl.sh` runs tests and a dry run before GPU fine-tuning.
