# AGENTS.md

## Project goal

We are building a production-oriented wind power forecasting MVP using Chronos-2 on SDWPF first.

The project is mainly multivariate / covariate-informed wind power forecasting.

The current task is zero-shot inference only. Do not implement fine-tuning unless explicitly asked.

## Model rule

Use Chronos-2 only for this stage.

Required:
- package: chronos-forecasting
- import: `from chronos import Chronos2Pipeline`
- model id: `amazon/chronos-2`
- loading style: `Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")`

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
- `src/training/`: training or fine-tuning logic; leave mostly empty for zero-shot
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
9. Compute:
   - MAE
   - RMSE
   - NMAE
   - NRMSE
10. Produce a simple result table comparing univariate vs multivariate mode.

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