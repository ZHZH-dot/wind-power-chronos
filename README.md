# Wind Power Chronos MVP

Production-oriented SDWPF wind power forecasting MVP using Chronos-2 zero-shot inference.

This stage is inference only. There is no fine-tuning or training code in the first pipeline.

## Model Constraint

Use Chronos-2 only:

```python
from chronos import Chronos2Pipeline

Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
```

The default `--model-id` is `amazon/chronos-2`. The same argument can also be a local model directory such as `/data/GDUT_stu/models/chronos-2`. The legacy spelling `--model_id` is still accepted. The excluded model families are not part of this repository.

## Data Layout

Put the raw SDWPF CSV anywhere readable on the machine, commonly:

```text
data/raw/<your-sdwpf-file>.csv
```

Raw files under `data/raw/` are read-only inputs. Prepared data is written to `data/processed/`; predictions and metrics are written to `results/`.

The processed table schema is:

```text
id, timestamp, target, optional covariates...
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

## AutoDL GPU Run

Clone the repo on AutoDL, place or mount the raw SDWPF CSV, then run:

```bash
bash scripts/run_zero_shot_autodl.sh /root/autodl-tmp/<your-sdwpf-file>.csv
```

The script creates a conda environment named `wind-chronos`, installs `requirements-autodl.txt`, prepares SDWPF, runs CPU-only tests, runs one-turbine smoke tests, then runs full univariate and multivariate zero-shot evaluation.

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

A100 is not required for the one-turbine smoke test. It is useful for full SDWPF evaluation and later fine-tuning work, but fine-tuning is not implemented in this stage.

## First Smoke Test

Run one turbine and one rolling window before the full evaluation:

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly.parquet \
  --output results/zero_shot/smoke_multivariate.csv \
  --mode multivariate \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --model-id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1 6 24 72 \
  --context-length 168 \
  --stride 24 \
  --max_turbines 1 \
  --max-windows-per-turbine 1
```

## Full SDWPF Zero-Shot Run

Univariate baseline:

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly.parquet \
  --output results/zero_shot/predictions_univariate.csv \
  --mode univariate \
  --model-id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1 6 24 72 \
  --context-length 168 \
  --stride 24
```

Multivariate covariate-informed run:

```bash
python -m src.models.chronos_zero_shot \
  --input data/processed/sdwpf_hourly.parquet \
  --output results/zero_shot/predictions_multivariate.csv \
  --mode multivariate \
  --covariates Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv \
  --model-id amazon/chronos-2 \
  --device-map cuda \
  --horizons 1 6 24 72 \
  --context-length 168 \
  --stride 24
```

Measured future covariates are not used by default. Only pass `--allow-future-covariates` if those values would realistically be available at prediction time.

## If Hugging Face Is Unavailable

Do not use `https://huggingface.co/amazon/chronos-2/resolve/main` directly. That URL is not a valid model source in this pipeline. The code should load models only through `Chronos2Pipeline.from_pretrained(...)`.

Option A: install Chronos code from GitHub.

```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

Option B: pre-download the model on a machine that can access Hugging Face, then copy the local model folder to the server. Pass that folder as `--model-id`:

```bash
python src/models/chronos_zero_shot.py \
  --input data/processed/sdwpf_hourly.parquet \
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

Evaluate both modes:

```bash
python -m src.evaluation.evaluate \
  --predictions \
  results/zero_shot/predictions_univariate.csv \
  results/zero_shot/predictions_multivariate.csv \
  --ground-truth data/processed/sdwpf_hourly.parquet \
  --output results/zero_shot/metrics.csv
```

The metrics CSV reports MAE, RMSE, NMAE, and NRMSE by mode and horizon.
