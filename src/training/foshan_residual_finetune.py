"""Chronos-2 LoRA/full challengers for the Foshan signed residual target."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.reconstruct_foshan_residual import (
    CALENDAR_COLUMNS,
    FREQUENCY,
    TARGET_COLUMN,
    TARGET_LABEL,
    TIMEZONE,
    sha256_file,
)
from src.evaluation.foshan_benchmark import normalize_chronos_quantiles
from src.models.foshan_residual_zero_shot import (
    MODEL_ID,
    MODEL_REVISION,
    PREDICTION_COLUMNS,
    PREDICTION_LENGTH,
    QUANTILES,
    build_inference_frames,
    forecast_issue_times,
    load_residual_config,
    package_version,
    resolve_pinned_snapshot,
)
from src.training.chronos_finetune import build_chronos2_hyperparameters
from src.utils.runtime import git_commit, git_is_dirty


@dataclass(frozen=True)
class ResidualTrainingFrames:
    train: pd.DataFrame
    tuning: pd.DataFrame
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    selection_start: pd.Timestamp
    selection_end_exclusive: pd.Timestamp
    n_missing_train_targets: int
    n_missing_tuning_targets: int


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _local_naive(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if isinstance(timestamps, pd.DatetimeIndex):
        series = pd.Series(timestamps)
    else:
        series = timestamps
    if series.dt.tz is None:
        return series.astype("datetime64[ns]")
    return series.dt.tz_convert(TIMEZONE).dt.tz_localize(None).astype("datetime64[ns]")


def _site_aware(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if isinstance(timestamps, pd.DatetimeIndex):
        series = pd.Series(timestamps)
    else:
        series = timestamps
    if series.dt.tz is None:
        return series.dt.tz_localize(TIMEZONE)
    return series.dt.tz_convert(TIMEZONE)


def prepare_residual_training_frames(
    residual_table: pd.DataFrame,
    config: dict[str, Any],
) -> ResidualTrainingFrames:
    """Use March targets for fitting and March-April only for validation context."""
    table = residual_table.copy()
    table["timestamp"] = _site_aware(table["timestamp"]).to_numpy()
    required = {"timestamp", TARGET_COLUMN, *CALENDAR_COLUMNS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Residual training data is missing columns: {missing}")
    train_start = pd.Timestamp(config["train_period"]["start"])
    train_end = pd.Timestamp(config["train_period"]["end_exclusive"])
    selection_start = pd.Timestamp(config["selection_period"]["start"])
    selection_end = pd.Timestamp(config["selection_period"]["end_exclusive"])
    if train_end != selection_start:
        raise ValueError("March training must end exactly when April selection starts.")
    if selection_end != pd.Timestamp(config["test_period"]["start"]):
        raise ValueError("April selection must end exactly when May test starts.")
    train_source = table.loc[
        (table["timestamp"] >= train_start) & (table["timestamp"] < train_end)
    ].copy()
    tuning_source = table.loc[
        (table["timestamp"] >= train_start) & (table["timestamp"] < selection_end)
    ].copy()
    if train_source.empty or tuning_source.empty:
        raise ValueError("March training or cumulative April validation context is empty.")

    def boundary_frame(source: pd.DataFrame) -> pd.DataFrame:
        result = source[["timestamp", TARGET_COLUMN, *CALENDAR_COLUMNS]].copy()
        result.insert(0, "id", "foshan_signed_residual")
        result = result.rename(columns={TARGET_COLUMN: "target"})
        result["timestamp"] = _local_naive(result["timestamp"]).to_numpy()
        return result.sort_values(["id", "timestamp"]).reset_index(drop=True)

    train = boundary_frame(train_source)
    tuning = boundary_frame(tuning_source)
    if train["timestamp"].max() >= train_end.tz_localize(None):
        raise AssertionError("April targets entered train_data.")
    if tuning["timestamp"].max() >= selection_end.tz_localize(None):
        raise AssertionError("May targets entered tuning_data or model selection.")
    expected_train = pd.date_range(
        train_start.tz_localize(None), train_end.tz_localize(None), freq=FREQUENCY, inclusive="left"
    )
    if train["timestamp"].tolist() != expected_train.tolist():
        raise ValueError("March residual training timestamps are not a complete 15-minute grid.")
    return ResidualTrainingFrames(
        train=train,
        tuning=tuning,
        train_start=train_start,
        train_end_exclusive=train_end,
        selection_start=selection_start,
        selection_end_exclusive=selection_end,
        n_missing_train_targets=int(train["target"].isna().sum()),
        n_missing_tuning_targets=int(tuning["target"].isna().sum()),
    )


def _float_name(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-0", "-")


def training_candidates(
    config: dict[str, Any],
    fine_tune_mode: str,
    *,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    if fine_tune_mode == "lora":
        search = config["lora_search"]
        if smoke:
            return [
                {
                    "name": "residual_lora_smoke_5s",
                    "rank": int(search["ranks"][0]),
                    "lora_alpha": int(search["alphas"][0]),
                    "learning_rate": float(search["learning_rates"][0]),
                    "steps": int(search["smoke_steps"]),
                    "batch_size": int(search["batch_size"]),
                    "seed": int(config["seed"]),
                }
            ]
        candidates = []
        for rank, alpha, learning_rate, steps in itertools.product(
            search["ranks"],
            search["alphas"],
            search["learning_rates"],
            search["steps"],
        ):
            candidates.append(
                {
                    "name": (
                        f"residual_lora_r{rank}_a{alpha}_lr{_float_name(float(learning_rate))}_"
                        f"s{steps}_seed{config['seed']}"
                    ),
                    "rank": int(rank),
                    "lora_alpha": int(alpha),
                    "learning_rate": float(learning_rate),
                    "steps": int(steps),
                    "batch_size": int(search["batch_size"]),
                    "seed": int(config["seed"]),
                }
            )
        return candidates
    if fine_tune_mode == "full":
        search = config["full_search"]
        steps_values = [int(search["smoke_steps"])] if smoke else [int(value) for value in search["steps"]]
        learning_rates = [float(search["learning_rates"][0])] if smoke else [
            float(value) for value in search["learning_rates"]
        ]
        return [
            {
                "name": (
                    "residual_full_smoke_5s"
                    if smoke
                    else f"residual_full_lr{_float_name(lr)}_s{steps}_seed{config['seed']}"
                ),
                "learning_rate": lr,
                "steps": steps,
                "batch_size": int(search["batch_size"]),
                "gradient_accumulation_steps": int(search["gradient_accumulation_steps"]),
                "effective_batch_size": int(search["effective_batch_size"]),
                "seed": int(config["seed"]),
            }
            for lr, steps in itertools.product(learning_rates, steps_values)
        ]
    raise ValueError("fine_tune_mode must be 'lora' or 'full'.")


def build_residual_hyperparameters(
    snapshot: Path,
    config: dict[str, Any],
    candidate: dict[str, Any],
    fine_tune_mode: str,
    *,
    dataloader_num_workers: int = 0,
) -> dict[str, dict[str, Any]]:
    hyperparameters = build_chronos2_hyperparameters(
        model_id=str(snapshot),
        mode="univariate",
        prediction_length=int(config["prediction_length"]),
        context_length=int(config["training_context_length"]),
        steps=int(candidate["steps"]),
        learning_rate=float(candidate["learning_rate"]),
        batch_size=int(candidate["batch_size"]),
        inference_batch_size=int(config["inference_batch_size"]),
        seed=int(candidate["seed"]),
        dataloader_num_workers=dataloader_num_workers,
        bf16=True,
        fp16=False,
        lora_rank=(int(candidate["rank"]) if fine_tune_mode == "lora" else None),
        lora_alpha=(
            int(candidate["lora_alpha"]) if fine_tune_mode == "lora" else None
        ),
        disable_known_covariates=False,
        disable_past_covariates=True,
        fine_tune_mode=fine_tune_mode,
        model_name_suffix="ResidualLoRA" if fine_tune_mode == "lora" else "ResidualFull",
    )
    if fine_tune_mode == "full":
        hyperparameters["Chronos2"]["fine_tune_trainer_kwargs"][
            "gradient_accumulation_steps"
        ] = int(candidate["gradient_accumulation_steps"])
    return hyperparameters


def gpu_preflight_4090() -> dict[str, Any]:
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 for residual fine-tuning.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Residual fine-tuning requires exactly one visible CUDA GPU.")
    name = torch.cuda.get_device_name(0)
    if "RTX 4090" not in name:
        raise RuntimeError(f"Residual fine-tuning requires RTX 4090; detected {name}.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The RTX 4090 runtime must support BF16.")
    return {
        "gpu_name": name,
        "gpu_total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "bf16_supported": True,
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _load_autogluon() -> tuple[Any, Any]:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    return TimeSeriesDataFrame, TimeSeriesPredictor


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def find_fine_tuned_checkpoint(
    predictor_path: Path,
    fine_tune_mode: str,
) -> tuple[Path, list[Path]]:
    checkpoints = sorted(
        path for path in predictor_path.rglob("fine-tuned-ckpt") if path.is_dir()
    )
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected one fine-tuned-ckpt under {predictor_path}, found {checkpoints}."
        )
    checkpoint = checkpoints[0]
    adapters = sorted(checkpoint.rglob("adapter_model.safetensors"))
    weights = sorted(checkpoint.rglob("*.safetensors")) + sorted(
        checkpoint.rglob("pytorch_model*.bin")
    )
    if fine_tune_mode == "lora" and not adapters:
        raise RuntimeError("LoRA checkpoint contains no adapter_model.safetensors.")
    if fine_tune_mode == "full" and (adapters or not weights):
        raise RuntimeError("Full checkpoint must contain full weights and no LoRA adapter.")
    return checkpoint, weights


def verify_loaded_predictor(
    predictor: Any,
    predictor_path: Path,
    model_name: str,
    fine_tune_mode: str,
    snapshot: Path,
    *,
    inspect_parameters: bool = True,
) -> dict[str, Any]:
    names = [str(value) for value in predictor.model_names()]
    if model_name not in names or len([value for value in names if value.startswith("Chronos2")]) != 1:
        raise RuntimeError(f"Expected one saved Chronos2 model after reload, found {names}.")
    trainer = getattr(predictor, "_trainer", None)
    if trainer is None or not hasattr(trainer, "load_model"):
        raise RuntimeError("Reloaded predictor does not expose its trained model.")
    ag_model = trainer.load_model(model_name)
    hyperparameters = ag_model.get_hyperparameters()
    if hyperparameters.get("fine_tune") is not True:
        raise RuntimeError("Reloaded Chronos2 model is not marked fine_tune=true.")
    if hyperparameters.get("fine_tune_mode") != fine_tune_mode:
        raise RuntimeError(
            f"Reloaded model mode is {hyperparameters.get('fine_tune_mode')!r}, "
            f"expected {fine_tune_mode!r}."
        )
    saved_source = Path(str(hyperparameters.get("model_path", ""))).expanduser().resolve()
    if saved_source != snapshot.resolve():
        raise RuntimeError(
            f"Reloaded model silently changed base checkpoint: {saved_source} != {snapshot}."
        )
    counts = {
        "trainable_parameters": None,
        "total_parameters": None,
        "frozen_parameters": None,
    }
    if inspect_parameters:
        ag_model.load_model_pipeline()
        pipeline = getattr(ag_model, "_model_pipeline", None)
        core = getattr(pipeline, "model", None)
        if core is None:
            core = getattr(pipeline, "_model", None)
        if core is None:
            raise RuntimeError("Could not inspect the reloaded Chronos-2 model parameters.")
        parameters = list(core.parameters())
        total = sum(int(value.numel()) for value in parameters)
        trainable = sum(int(value.numel()) for value in parameters if value.requires_grad)
        counts = {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "frozen_parameters": total - trainable,
        }
        if fine_tune_mode == "full" and trainable != total:
            raise RuntimeError(
                f"Full fine-tuning left parameters frozen: {trainable}/{total} trainable."
            )
        ag_model._model_pipeline = None
        del core, pipeline
        gc.collect()
    checkpoint, weight_files = find_fine_tuned_checkpoint(predictor_path, fine_tune_mode)
    return {
        **counts,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _directory_sha256(checkpoint),
        "checkpoint_size_bytes": sum(
            value.stat().st_size for value in checkpoint.rglob("*") if value.is_file()
        ),
        "checkpoint_weight_files": [str(value.resolve()) for value in weight_files],
        "base_model_source_verified": str(snapshot.resolve()),
        "fine_tune_mode_verified": fine_tune_mode,
    }


def fit_residual_candidate(
    frames: ResidualTrainingFrames,
    config: dict[str, Any],
    candidate: dict[str, Any],
    fine_tune_mode: str,
    snapshot: Path,
    candidate_dir: Path,
    *,
    dataloader_num_workers: int = 0,
    autogluon_classes: tuple[Any, Any] | None = None,
) -> tuple[Any, Any, str, dict[str, Any]]:
    predictor_path = candidate_dir / "predictor"
    if predictor_path.exists():
        raise FileExistsError(f"Refusing to overwrite predictor: {predictor_path}")
    dataframe_class, predictor_class = autogluon_classes or _load_autogluon()
    train_data = dataframe_class.from_data_frame(
        frames.train, id_column="id", timestamp_column="timestamp"
    )
    tuning_data = dataframe_class.from_data_frame(
        frames.tuning, id_column="id", timestamp_column="timestamp"
    )
    hyperparameters = build_residual_hyperparameters(
        snapshot,
        config,
        candidate,
        fine_tune_mode,
        dataloader_num_workers=dataloader_num_workers,
    )
    predictor = predictor_class(
        path=str(predictor_path),
        prediction_length=int(config["prediction_length"]),
        target="target",
        known_covariates_names=list(config["calendar_covariates"]),
        quantile_levels=[float(value) for value in config["quantile_levels"]],
        eval_metric="WQL",
        freq=str(config["frequency"]),
    )
    torch_module: Any | None = None
    if autogluon_classes is None:
        import torch

        torch_module = torch
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    predictor.fit(
        train_data=train_data,
        tuning_data=tuning_data,
        hyperparameters=hyperparameters,
        enable_ensemble=False,
        random_seed=int(candidate["seed"]),
        refit_full=False,
        skip_model_selection=False,
    )
    runtime = time.monotonic() - started
    names = [str(value) for value in predictor.model_names()]
    trained = [value for value in names if value.startswith("Chronos2")]
    if len(trained) != 1:
        raise RuntimeError(f"AutoGluon trained no usable Chronos2 model: {names}")
    reloaded = predictor_class.load(str(predictor_path))
    inspection = verify_loaded_predictor(
        reloaded,
        predictor_path,
        trained[0],
        fine_tune_mode,
        snapshot,
        inspect_parameters=autogluon_classes is None,
    )
    stats = {
        "training_runtime_seconds": runtime,
        "peak_allocated_gpu_bytes": (
            int(torch_module.cuda.max_memory_allocated(0)) if torch_module else None
        ),
        "peak_reserved_gpu_bytes": (
            int(torch_module.cuda.max_memory_reserved(0)) if torch_module else None
        ),
        "trained_model_name": trained[0],
        "predictor_path": str(predictor_path.resolve()),
        "hyperparameters": hyperparameters,
        **inspection,
    }
    return reloaded, dataframe_class, trained[0], stats


def _normalize_autogluon_forecast(forecast: Any) -> pd.DataFrame:
    frame = pd.DataFrame(forecast).reset_index()
    if "item_id" in frame and "id" not in frame:
        frame = frame.rename(columns={"item_id": "id"})
    return frame


def predict_saved_period(
    predictor: Any,
    dataframe_class: Any,
    model_name: str,
    residual_table: pd.DataFrame,
    issues: Iterable[pd.Timestamp],
    candidate_name: str,
    fine_tune_mode: str,
    checkpoint_path: Path,
    context_length: int,
    split: str,
    *,
    max_issues: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = list(issues)
    if max_issues is not None:
        selected = selected[:max_issues]
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for issue in selected:
        context, future, metadata = build_inference_frames(
            residual_table, issue, context_length
        )
        context_data = dataframe_class.from_data_frame(
            context, id_column="id", timestamp_column="timestamp"
        )
        future_data = dataframe_class.from_data_frame(
            future, id_column="id", timestamp_column="timestamp"
        )
        started = time.monotonic()
        forecast = predictor.predict(
            context_data,
            known_covariates=future_data,
            model=model_name,
        )
        runtime = time.monotonic() - started
        issue_ts = pd.Timestamp(metadata["issue_time"])
        target_times = pd.date_range(issue_ts, periods=PREDICTION_LENGTH, freq=FREQUENCY)
        normalized = normalize_chronos_quantiles(
            _normalize_autogluon_forecast(forecast),
            targets=["target"],
            expected_times=target_times,
            site_id="foshan_signed_residual",
        )
        ordered = np.sort(
            normalized[["p10", "p50", "p90"]].to_numpy(dtype=float), axis=1
        )
        truth = residual_table.copy()
        truth["timestamp"] = _site_aware(truth["timestamp"]).to_numpy()
        truth = truth.set_index("timestamp").reindex(target_times)
        call_id = f"{candidate_name}:{issue_ts.isoformat()}"
        rows = pd.DataFrame(
            {
                "split": split,
                "candidate": candidate_name,
                "candidate_kind": f"chronos2_{fine_tune_mode}_residual",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "context_length": context_length,
                "refresh_cadence_minutes": 60,
                "issue_time": issue_ts,
                "context_start": metadata["context_start"],
                "context_end": metadata["context_end"],
                "target_time": target_times,
                "horizon_step": np.arange(1, PREDICTION_LENGTH + 1),
                "p10": ordered[:, 0],
                "p50": ordered[:, 1],
                "p90": ordered[:, 2],
                "y_pred": ordered[:, 1],
                "y_true_kw": truth[TARGET_COLUMN].to_numpy(dtype=float),
                "is_missing_target": truth[TARGET_COLUMN].isna().to_numpy(dtype=bool),
                "target_label": TARGET_LABEL,
                "known_future_covariates": "|".join(CALENDAR_COLUMNS),
                "used_future_realized_data": False,
                "inference_call_id": call_id,
            },
            columns=PREDICTION_COLUMNS,
        )
        frames.append(rows)
        audits.append(
            {
                **metadata,
                "candidate": candidate_name,
                "fine_tune_mode": fine_tune_mode,
                "inference_call_id": call_id,
                "runtime_seconds": runtime,
                "forecast_rows": len(rows),
                "different_issue_times_batched": False,
                "used_future_realized_data": False,
            }
        )
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.groupby("inference_call_id")["issue_time"].nunique().gt(1).any():
        raise AssertionError("A trained Chronos inference call batched issue times.")
    return predictions, pd.DataFrame(audits)


def _manifest_base(
    config_path: Path,
    input_path: Path,
    snapshot: Path,
    candidate: dict[str, Any],
    fine_tune_mode: str,
    frames: ResidualTrainingFrames,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "configured",
        "target_label": TARGET_LABEL,
        "verified_gross_factory_load": False,
        "base_model_id": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "base_model_snapshot": str(snapshot.resolve()),
        "fine_tune_mode": fine_tune_mode,
        "candidate_name": candidate["name"],
        "hyperparameters": candidate,
        "seed": int(candidate["seed"]),
        "train_period": {
            "start": frames.train_start.isoformat(),
            "end_exclusive": frames.train_end_exclusive.isoformat(),
            "weights_updated_from_this_period_only": True,
        },
        "validation_period": {
            "start": frames.selection_start.isoformat(),
            "end_exclusive": frames.selection_end_exclusive.isoformat(),
            "used_for_model_selection_only": True,
        },
        "may_targets_passed_to_fit_or_selection": False,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "chronos_forecasting_version": package_version("chronos-forecasting"),
        "autogluon_version": package_version("autogluon.timeseries"),
        "python_version": sys.version,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "runtime_environment": preflight,
        "april_selection_results": None,
    }


def run_training(args: argparse.Namespace) -> Path:
    config = load_residual_config(args.config)
    residual = pd.read_parquet(args.input)
    frames = prepare_residual_training_frames(residual, config)
    snapshot = resolve_pinned_snapshot(
        model_path=args.model_path,
        hf_home=args.hf_home,
        allow_download=args.allow_download,
    )
    candidates = training_candidates(
        config,
        args.fine_tune_mode,
        smoke=args.stage == "smoke",
    )
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    dry_summary = {
        "status": "dry_run_validated",
        "fine_tune_mode": args.fine_tune_mode,
        "candidate_count": len(candidates),
        "candidate_grid": candidates,
        "train_rows": len(frames.train),
        "tuning_rows": len(frames.tuning),
        "train_max_timestamp": frames.train["timestamp"].max(),
        "tuning_max_timestamp": frames.tuning["timestamp"].max(),
        "may_targets_passed_to_fit_or_selection": False,
        "snapshot": str(snapshot.resolve()),
    }
    if args.stage == "dry-run":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(args.output_dir / "dry_run.json", dry_summary)
        print(json.dumps(dry_summary, indent=2, default=str))
        return args.output_dir

    preflight = gpu_preflight_4090()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataframe_class, predictor_class = _load_autogluon()
    for candidate in candidates:
        candidate_dir = args.output_dir / str(candidate["name"])
        candidate_dir.mkdir(parents=True, exist_ok=False)
        manifest = _manifest_base(
            args.config,
            args.input,
            snapshot,
            candidate,
            args.fine_tune_mode,
            frames,
            preflight,
        )
        _write_json(candidate_dir / "candidate_config.json", candidate)
        try:
            predictor, _, model_name, stats = fit_residual_candidate(
                frames,
                config,
                candidate,
                args.fine_tune_mode,
                snapshot,
                candidate_dir,
                dataloader_num_workers=args.dataloader_num_workers,
            )
            april_issues = forecast_issue_times(
                config["selection_period"]["start"],
                config["selection_period"]["end_exclusive"],
                int(config["refresh_cadence_minutes"]),
            )
            max_issues = 1 if args.stage == "smoke" else None
            predictions, inference_audit = predict_saved_period(
                predictor,
                dataframe_class,
                model_name,
                residual,
                april_issues,
                str(candidate["name"]),
                args.fine_tune_mode,
                Path(stats["checkpoint_path"]),
                int(config["training_context_length"]),
                str(config["selection_period"]["name"]),
                max_issues=max_issues,
            )
            predictions.to_csv(
                candidate_dir / "april_predictions.csv", index=False, float_format="%.15g"
            )
            inference_audit.to_csv(
                candidate_dir / "april_inference_audit.csv",
                index=False,
                float_format="%.15g",
            )
            manifest.update(
                {
                    "status": "trained_checkpoint_reloaded",
                    "trained_model_name": model_name,
                    "april_issue_count": int(predictions["issue_time"].nunique()),
                    "may_inference_completed": False,
                    **stats,
                }
            )
        except Exception as error:
            manifest.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            _write_json(candidate_dir / "training_manifest.json", manifest)
            raise
        _write_json(candidate_dir / "training_manifest.json", manifest)
    return args.output_dir


def run_selected_may_inference(args: argparse.Namespace) -> Path:
    if args.candidate_dir is None:
        raise ValueError("--candidate-dir is required for --stage may-predict.")
    config = load_residual_config(args.config)
    snapshot = resolve_pinned_snapshot(
        model_path=args.model_path,
        hf_home=args.hf_home,
        allow_download=args.allow_download,
    )
    gpu_preflight_4090()
    manifest_path = args.candidate_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "trained_checkpoint_reloaded":
        raise RuntimeError("Selected candidate has no verified trained checkpoint.")
    if manifest.get("base_model_revision") != MODEL_REVISION:
        raise RuntimeError("Selected candidate uses a different base revision.")
    if (args.candidate_dir / "may_predictions.csv").exists():
        raise FileExistsError("Refusing to overwrite existing May predictions.")
    dataframe_class, predictor_class = _load_autogluon()
    predictor_path = Path(manifest["predictor_path"])
    predictor = predictor_class.load(str(predictor_path))
    verify_loaded_predictor(
        predictor,
        predictor_path,
        str(manifest["trained_model_name"]),
        str(manifest["fine_tune_mode"]),
        snapshot,
    )
    residual = pd.read_parquet(args.input)
    issues = forecast_issue_times(
        config["test_period"]["start"],
        config["test_period"]["end_exclusive"],
        int(config["refresh_cadence_minutes"]),
    )
    predictions, audit = predict_saved_period(
        predictor,
        dataframe_class,
        str(manifest["trained_model_name"]),
        residual,
        issues,
        str(manifest["candidate_name"]),
        str(manifest["fine_tune_mode"]),
        Path(manifest["checkpoint_path"]),
        int(config["training_context_length"]),
        str(config["test_period"]["name"]),
    )
    predictions.to_csv(
        args.candidate_dir / "may_predictions.csv", index=False, float_format="%.15g"
    )
    audit.to_csv(
        args.candidate_dir / "may_inference_audit.csv", index=False, float_format="%.15g"
    )
    manifest["may_inference_completed"] = True
    manifest["may_issue_count"] = int(predictions["issue_time"].nunique())
    _write_json(manifest_path, manifest)
    return args.candidate_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--config", default=Path("configs/foshan_chronos2_residual.json"), type=Path
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fine-tune-mode", choices=("lora", "full"), required=True)
    parser.add_argument(
        "--stage", choices=("dry-run", "smoke", "search", "may-predict"), required=True
    )
    parser.add_argument("--candidate-dir", default=None, type=Path)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--hf-home", default=None, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dataloader-num-workers", default=0, type=int)
    parser.add_argument("--max-candidates", default=None, type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "may-predict":
        output = run_selected_may_inference(args)
    else:
        output = run_training(args)
    print(f"Residual {args.fine_tune_mode} stage {args.stage} completed at {output.resolve()}")


if __name__ == "__main__":
    main()
