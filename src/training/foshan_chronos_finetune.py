"""Leakage-safe Chronos-2 LoRA search for the Foshan PV benchmark."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.foshan_benchmark import (
    PREDICTION_COLUMNS,
    build_forecast_window,
    chronos_rows_for_window,
    evaluate_common_scored_timestamps,
    evaluate_foshan_predictions,
    period_origins,
    select_may_configurations,
    validate_processed_table,
)
from src.models.foshan_chronos_zero_shot import chronos_input_frame
from src.training.chronos_finetune import (
    build_chronos2_hyperparameters,
    select_training_precision,
)
from src.utils.runtime import git_commit, git_is_dirty


DEFAULT_CONFIG = Path("configs/foshan_chronos2_lora.json")
DEFAULT_INPUT = Path("results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet")
DEFAULT_ZERO_SHOT_DIR = Path("results/zero_shot/foshan_chronos2")
DEFAULT_OUTPUT_DIR = Path("results/fine_tune/foshan_chronos2_lora")


@dataclass(frozen=True)
class FoshanFineTuneFrames:
    train: pd.DataFrame
    tuning: pd.DataFrame
    targets: list[str]
    item_target_map: dict[str, str]
    n_masked_train: int
    n_masked_tuning: int


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_autogluon() -> tuple[Any, Any]:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    return TimeSeriesDataFrame, TimeSeriesPredictor


def resolve_model_source(model_id: str, model_path: Path | None) -> str:
    local = model_path or (
        Path(os.environ["CHRONOS_MODEL_PATH"])
        if os.environ.get("CHRONOS_MODEL_PATH")
        else None
    )
    if local is not None:
        resolved = local.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Chronos-2 model directory does not exist: {resolved}")
        return str(resolved)
    if model_id != "amazon/chronos-2":
        raise ValueError("--model-id must be amazon/chronos-2 unless --model-path is used.")
    return model_id


def load_lora_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "model_id",
        "site_id",
        "frequency",
        "timezone",
        "prediction_length",
        "context_length",
        "quantile_levels",
        "inference_batch_size",
        "causal_fill_limit",
        "pv_capacity_kw",
        "pv_active_threshold_kw",
        "calendar_covariates",
        "train_period",
        "selection_period",
        "test_period",
        "variants",
        "smoke_candidate",
        "search_candidates",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Foshan LoRA config is missing keys: {missing}")
    if int(config["prediction_length"]) != 96:
        raise ValueError("Foshan LoRA prediction_length must remain 96.")
    if int(config["context_length"]) != 672:
        raise ValueError("Foshan LoRA context_length must remain 672.")
    if [float(value) for value in config["quantile_levels"]] != [0.1, 0.5, 0.9]:
        raise ValueError("Foshan LoRA quantiles must remain P10/P50/P90.")
    if int(config["causal_fill_limit"]) != 3:
        raise ValueError("Foshan LoRA causal_fill_limit must remain 3.")
    return config


def _period_bounds(period: dict[str, str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(period["start"]), pd.Timestamp(period["end_exclusive"])


def _item_id(site_id: str, target: str, joint: bool) -> str:
    return f"{site_id}::{target}" if joint else site_id


def _long_target_frame(
    table: pd.DataFrame,
    targets: list[str],
    calendar_covariates: list[str],
    site_id: str,
) -> tuple[pd.DataFrame, dict[str, str], int]:
    joint = len(targets) > 1
    frames: list[pd.DataFrame] = []
    item_target_map: dict[str, str] = {}
    n_masked = 0
    for target in targets:
        item = _item_id(site_id, target, joint)
        item_target_map[item] = target
        frame = table[["timestamp", target, *calendar_covariates]].copy()
        frame.insert(0, "id", item)
        frame = frame.rename(columns={target: "target"})
        frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
        n_masked += int(frame["target"].isna().sum())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), item_target_map, n_masked


def prepare_foshan_finetune_frames(
    table: pd.DataFrame,
    config: dict[str, Any],
    variant: dict[str, Any],
) -> FoshanFineTuneFrames:
    """Build timezone-aware March-April train and cumulative May tuning frames."""
    targets = [str(target) for target in variant["targets"]]
    if not targets or not set(targets).issubset({"pv_kw", "net_grid_kw"}):
        raise ValueError(f"Unsupported Foshan LoRA targets: {targets}")
    if str(variant.get("selection_target")) != "pv_kw":
        raise ValueError("Every Foshan LoRA variant must be selected using PV only.")

    train_start, train_end = _period_bounds(config["train_period"])
    selection_start, selection_end = _period_bounds(config["selection_period"])
    test_start, _ = _period_bounds(config["test_period"])
    if train_end != selection_start or selection_end != test_start:
        raise ValueError("Train, May selection, and June test periods must be contiguous.")

    train_table = table[
        (table["timestamp"] >= train_start) & (table["timestamp"] < train_end)
    ].copy()
    tuning_table = table[
        (table["timestamp"] >= train_start) & (table["timestamp"] < selection_end)
    ].copy()
    if train_table.empty or tuning_table.empty:
        raise ValueError("Foshan LoRA training or May tuning data is empty.")

    calendar = [str(column) for column in config["calendar_covariates"]]
    train, item_target_map, n_masked_train = _long_target_frame(
        train_table, targets, calendar, str(config["site_id"])
    )
    tuning, tuning_map, n_masked_tuning = _long_target_frame(
        tuning_table, targets, calendar, str(config["site_id"])
    )
    if tuning_map != item_target_map:
        raise AssertionError("Training and tuning target mappings differ.")
    if train["timestamp"].max() >= selection_start:
        raise AssertionError("May targets entered the training frame.")
    if tuning["timestamp"].max() >= test_start:
        raise AssertionError("June targets entered training or model selection.")
    if str(table["timestamp"].dt.tz) != str(config["timezone"]):
        raise ValueError(
            f"Internal Foshan timestamps must remain timezone-aware {config['timezone']}."
        )

    return FoshanFineTuneFrames(
        train=train.sort_values(["id", "timestamp"]).reset_index(drop=True),
        tuning=tuning.sort_values(["id", "timestamp"]).reset_index(drop=True),
        targets=targets,
        item_target_map=item_target_map,
        n_masked_train=n_masked_train,
        n_masked_tuning=n_masked_tuning,
    )


def build_foshan_lora_hyperparameters(
    model_source: str,
    config: dict[str, Any],
    candidate: dict[str, Any],
    bf16: bool,
    fp16: bool,
    dataloader_num_workers: int,
) -> dict[str, dict[str, Any]]:
    return build_chronos2_hyperparameters(
        model_id=model_source,
        mode="univariate",
        prediction_length=int(config["prediction_length"]),
        context_length=int(config["context_length"]),
        steps=int(candidate["steps"]),
        learning_rate=float(candidate["learning_rate"]),
        batch_size=int(candidate["batch_size"]),
        inference_batch_size=int(config["inference_batch_size"]),
        seed=int(candidate["seed"]),
        dataloader_num_workers=dataloader_num_workers,
        bf16=bf16,
        fp16=fp16,
        lora_rank=int(candidate["rank"]),
        lora_alpha=int(candidate["lora_alpha"]),
        disable_known_covariates=False,
        disable_past_covariates=True,
    )


def gpu_preflight() -> dict[str, Any]:
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 before Foshan LoRA training.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to emulate LoRA training.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}."
        )
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 4090" not in gpu_name:
        raise RuntimeError(f"Foshan LoRA requires one RTX 4090; detected {gpu_name}.")
    properties = torch.cuda.get_device_properties(0)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    precision, bf16, fp16 = select_training_precision(True, bf16_supported)
    return {
        "gpu_name": gpu_name,
        "gpu_total_vram_bytes": int(properties.total_memory),
        "bf16_supported": bf16_supported,
        "precision": precision,
        "bf16": bf16,
        "fp16": fp16,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _to_timeseries_frame(dataframe_class: Any, frame: pd.DataFrame) -> Any:
    boundary_frame = chronos_input_frame(frame)
    assert boundary_frame is not None
    return dataframe_class.from_data_frame(
        boundary_frame,
        id_column="id",
        timestamp_column="timestamp",
    )


def fit_candidate(
    frames: FoshanFineTuneFrames,
    config: dict[str, Any],
    candidate: dict[str, Any],
    model_source: str,
    output_dir: Path,
    precision: dict[str, Any],
    dataloader_num_workers: int = 0,
    autogluon_classes: tuple[Any, Any] | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """Fit one isolated AutoGluon Chronos2 LoRA candidate."""
    predictor_path = output_dir / "predictor"
    if predictor_path.exists():
        raise FileExistsError(f"Refusing to overwrite predictor: {predictor_path}")
    dataframe_class, predictor_class = autogluon_classes or _load_autogluon()
    train_data = _to_timeseries_frame(dataframe_class, frames.train)
    tuning_data = _to_timeseries_frame(dataframe_class, frames.tuning)
    hyperparameters = build_foshan_lora_hyperparameters(
        model_source,
        config,
        candidate,
        bf16=bool(precision["bf16"]),
        fp16=bool(precision["fp16"]),
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

    model_names = [str(name) for name in predictor.model_names()]
    trained = [name for name in model_names if name.startswith("Chronos2")]
    if len(trained) != 1:
        raise RuntimeError(
            f"Expected exactly one trained Chronos2 LoRA model, found: {model_names}"
        )
    adapters = sorted(predictor_path.rglob("adapter_model.safetensors"))
    if autogluon_classes is None and not adapters:
        raise RuntimeError(
            f"Chronos2 LoRA training produced no adapter_model.safetensors under {predictor_path}."
        )
    stats = {
        "training_runtime_seconds": runtime,
        "peak_allocated_gpu_bytes": (
            int(torch_module.cuda.max_memory_allocated(0))
            if torch_module is not None
            else None
        ),
        "peak_reserved_gpu_bytes": (
            int(torch_module.cuda.max_memory_reserved(0))
            if torch_module is not None
            else None
        ),
        "adapter_paths": [str(path) for path in adapters],
        "trained_model_name": trained[0],
        "hyperparameters": hyperparameters,
    }
    return predictor, trained[0], stats


def _window_context_frame(
    window: Any,
    targets: list[str],
    calendar_covariates: list[str],
    site_id: str,
) -> pd.DataFrame:
    joint = len(targets) > 1
    frames: list[pd.DataFrame] = []
    for target in targets:
        frame = window.context_df[["timestamp", target, *calendar_covariates]].copy()
        frame.insert(0, "id", _item_id(site_id, target, joint))
        frame = frame.rename(columns={target: "target"})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _window_future_frame(
    window: Any,
    targets: list[str],
    calendar_covariates: list[str],
    site_id: str,
) -> pd.DataFrame:
    if window.future_df is None:
        raise ValueError("Foshan calendar LoRA requires a target-free future dataframe.")
    forbidden = {"pv_kw", "net_grid_kw", "pv_kw_raw", "net_grid_kw_raw", "target"}
    if forbidden.intersection(window.future_df.columns):
        raise AssertionError("Measured Foshan targets entered future known covariates.")
    joint = len(targets) > 1
    frames: list[pd.DataFrame] = []
    for target in targets:
        frame = window.future_df[["timestamp", *calendar_covariates]].copy()
        frame.insert(0, "id", _item_id(site_id, target, joint))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _normalize_autogluon_forecast(
    forecast: Any,
    item_target_map: dict[str, str],
    site_id: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(forecast).reset_index()
    item_column = "item_id" if "item_id" in frame.columns else "id"
    if item_column not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("AutoGluon forecast is missing item_id/id or timestamp.")
    frame["target_name"] = frame[item_column].astype(str).map(item_target_map)
    if frame["target_name"].isna().any():
        unknown = sorted(frame.loc[frame["target_name"].isna(), item_column].unique())
        raise ValueError(f"AutoGluon returned unknown target item IDs: {unknown}")
    frame["id"] = site_id
    return frame


def run_lora_origins(
    predictor: Any,
    dataframe_class: Any,
    model_name: str,
    table: pd.DataFrame,
    config: dict[str, Any],
    variant: dict[str, Any],
    candidate_name: str,
    origins: list[pd.Timestamp],
    split_name: str,
    run_id: str,
    model_source: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], float]:
    """Run rolling predictor inference while exposing only calendar future covariates."""
    targets = [str(target) for target in variant["targets"]]
    site_id = str(config["site_id"])
    calendar = [str(column) for column in config["calendar_covariates"]]
    joint = len(targets) > 1
    item_target_map = {
        _item_id(site_id, target, joint): target for target in targets
    }
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    started = time.monotonic()
    for issue_time in origins:
        window, reason = build_forecast_window(
            table=table,
            issue_time=issue_time,
            targets=targets,
            context_length=int(config["context_length"]),
            prediction_length=int(config["prediction_length"]),
            known_future_covariates=calendar,
            causal_fill_limit=int(config["causal_fill_limit"]),
            frequency=str(config["frequency"]),
        )
        if window is None:
            skipped.append(
                {
                    "split": split_name,
                    "variant": variant["name"],
                    "candidate": candidate_name,
                    "issue_time": issue_time.isoformat(),
                    "reason": reason,
                }
            )
            continue
        context = _window_context_frame(window, targets, calendar, site_id)
        future = _window_future_frame(window, targets, calendar, site_id)
        context_data = _to_timeseries_frame(dataframe_class, context)
        future_data = _to_timeseries_frame(dataframe_class, future)
        forecast = predictor.predict(
            context_data,
            known_covariates=future_data,
            model=model_name,
        )
        normalized = _normalize_autogluon_forecast(
            forecast, item_target_map=item_target_map, site_id=site_id
        )
        rows = chronos_rows_for_window(
            forecast_df=normalized,
            window=window,
            targets=targets,
            split_name=split_name,
            run_id=run_id,
            model_name=f"chronos2_lora_{variant['name']}_{candidate_name}",
            model_id=f"{model_source}+lora:{variant['name']}:{candidate_name}",
            context_length=int(config["context_length"]),
            known_future_covariates=calendar,
            pv_capacity_kw=float(config["pv_capacity_kw"]),
            site_id=site_id,
        )
        frames.append(rows[rows["target"] == "pv_kw"].copy())
    runtime = time.monotonic() - started
    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS), skipped, runtime
    return pd.concat(frames, ignore_index=True), skipped, runtime


def select_lora_candidate(
    may_metrics: pd.DataFrame,
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    selection = select_may_configurations(may_metrics)
    model_name = str(selection["targets"]["pv_kw"]["model_name"])
    matches = [record for record in candidate_records if record["model_name"] == model_name]
    if len(matches) != 1:
        raise RuntimeError(f"Could not map selected model to one candidate: {model_name}")
    selected = dict(matches[0])
    selected.update(selection["targets"]["pv_kw"])
    selected["selected_on"] = "may_2026_selection"
    selected["selection_metric"] = "postprocessed_pv_wape"
    selected["tie_break_metric"] = "pv_active_mae"
    return selected


def _frozen_zero_shot_comparison_rows(zero_shot_dir: Path) -> pd.DataFrame:
    predictions_path = zero_shot_dir / "predictions_long.csv"
    selection_path = zero_shot_dir / "selected_configuration.json"
    if not predictions_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(
            "Frozen zero-shot comparison requires predictions_long.csv and "
            f"selected_configuration.json under {zero_shot_dir}."
        )
    predictions = pd.read_csv(predictions_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pv = selection["targets"]["pv_kw"]
    base = predictions[
        (predictions["split"] == "june_2026_test")
        & (predictions["target"] == "pv_kw")
    ].copy()
    baselines = base[
        (base["model_id"] == "causal_baseline")
        & (base["postprocessing"] == "physical_target")
    ]
    zero_shot = base[
        (base["model_name"] == pv["model_name"])
        & (pd.to_numeric(base["context_length"]) == int(pv["context_length"]))
        & (base["postprocessing"] == "physical_clip_0_1700")
    ]
    if baselines.empty or zero_shot.empty:
        raise ValueError("Frozen June zero-shot or causal baseline rows are missing.")
    return pd.concat([zero_shot, baselines], ignore_index=True)[PREDICTION_COLUMNS]


def comparison_improvements(
    common_metrics: pd.DataFrame,
    selected_lora_model: str,
) -> dict[str, Any]:
    lora = common_metrics[common_metrics["model_name"] == selected_lora_model]
    zero = common_metrics[
        common_metrics["model_name"].astype(str).str.startswith("chronos2_")
        & ~common_metrics["model_name"].astype(str).str.startswith("chronos2_lora_")
    ]
    baselines = common_metrics[common_metrics["model_id"] == "causal_baseline"]
    if len(lora) != 1 or len(zero) != 1 or baselines.empty:
        raise ValueError("Common metrics do not contain one LoRA, one zero-shot, and baselines.")
    best_baseline = baselines.sort_values(
        ["wape", "pv_active_mae", "model_name"], kind="mergesort"
    ).iloc[0]
    lora_row = lora.iloc[0]
    zero_row = zero.iloc[0]
    metrics = ["mae", "rmse", "wape", "mase", "pv_active_mae", "pv_active_rmse", "pv_active_wape"]

    def improvement(reference: pd.Series, metric: str) -> float | None:
        value = float(reference[metric])
        if value == 0 or pd.isna(value):
            return None
        return 100.0 * (value - float(lora_row[metric])) / value

    return {
        "selected_lora_model": selected_lora_model,
        "frozen_zero_shot_model": str(zero_row["model_name"]),
        "best_baseline_model_by_wape": str(best_baseline["model_name"]),
        "percent_improvement_over_zero_shot": {
            metric: improvement(zero_row, metric) for metric in metrics
        },
        "percent_improvement_over_best_baseline": {
            metric: improvement(best_baseline, metric) for metric in metrics
        },
    }


def _runtime_metadata(model_source: str, config_path: Path) -> dict[str, Any]:
    return {
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "model_id_or_path": model_source,
        "config": str(config_path),
        "packages": {
            name: _package_version(name)
            for name in ("autogluon.timeseries", "chronos-forecasting", "torch", "pandas")
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _candidate_log_row(
    variant: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant["name"],
        "candidate": candidate["name"],
        "rank": int(candidate["rank"]),
        "lora_alpha": int(candidate["lora_alpha"]),
        "learning_rate": float(candidate["learning_rate"]),
        "steps": int(candidate["steps"]),
        "batch_size": int(candidate["batch_size"]),
        "seed": int(candidate["seed"]),
    }


def _variant_by_name(config: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [variant for variant in config["variants"] if variant["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate LoRA variant: {name}")
    return matches[0]


def _run_smoke(
    args: argparse.Namespace,
    table: pd.DataFrame,
    config: dict[str, Any],
    model_source: str,
    precision: dict[str, Any],
    autogluon_classes: tuple[Any, Any] | None,
) -> dict[str, Any]:
    smoke_dir = Path(args.output_dir) / "smoke"
    if smoke_dir.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {smoke_dir}")
    smoke_dir.mkdir(parents=True)
    variant = _variant_by_name(config, "pv_calendar")
    candidate = dict(config["smoke_candidate"])
    frames = prepare_foshan_finetune_frames(table, config, variant)
    predictor, model_name, stats = fit_candidate(
        frames,
        config,
        candidate,
        model_source,
        smoke_dir,
        precision,
        dataloader_num_workers=args.dataloader_num_workers,
        autogluon_classes=autogluon_classes,
    )
    dataframe_class = (
        autogluon_classes[0] if autogluon_classes is not None else _load_autogluon()[0]
    )
    origins = period_origins(
        table,
        config["selection_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=str(config["frequency"]),
        stride_steps=96,
        timezone=str(config["timezone"]),
    )[:1]
    predictions, skipped, inference_runtime = run_lora_origins(
        predictor,
        dataframe_class,
        model_name,
        table,
        config,
        variant,
        str(candidate["name"]),
        origins,
        "may_2026_selection",
        "foshan_lora_smoke",
        model_source,
    )
    if predictions.empty or skipped:
        raise RuntimeError(f"LoRA smoke prediction failed or skipped: {skipped}")
    predictions.to_csv(smoke_dir / "smoke_predictions.csv", index=False)
    result = {
        **_candidate_log_row(variant, candidate),
        **stats,
        "inference_runtime_seconds": inference_runtime,
        "prediction_rows": len(predictions),
        "status": "passed",
    }
    _write_json(smoke_dir / "smoke_metadata.json", result)
    return result


def _run_search(
    args: argparse.Namespace,
    table: pd.DataFrame,
    config: dict[str, Any],
    model_source: str,
    precision: dict[str, Any],
    autogluon_classes: tuple[Any, Any] | None,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    search_dir = output_dir / "search"
    if search_dir.exists():
        raise FileExistsError(f"Refusing to overwrite search output: {search_dir}")
    frozen_rows = _frozen_zero_shot_comparison_rows(Path(args.zero_shot_dir))
    search_dir.mkdir(parents=True)
    dataframe_class, predictor_class = autogluon_classes or _load_autogluon()
    may_origins = period_origins(
        table,
        config["selection_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=str(config["frequency"]),
        stride_steps=96,
        timezone=str(config["timezone"]),
    )
    june_origins = period_origins(
        table,
        config["test_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=str(config["frequency"]),
        stride_steps=96,
        timezone=str(config["timezone"]),
    )

    log_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    may_prediction_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for variant in config["variants"]:
        frames = prepare_foshan_finetune_frames(table, config, variant)
        for candidate in config["search_candidates"]:
            log = _candidate_log_row(variant, candidate)
            predictor: Any | None = None
            candidate_dir = search_dir / "candidates" / str(variant["name"]) / str(
                candidate["name"]
            )
            if candidate_dir.exists():
                raise FileExistsError(f"Refusing to overwrite candidate: {candidate_dir}")
            candidate_dir.mkdir(parents=True)
            _write_json(candidate_dir / "candidate_config.json", log)
            try:
                predictor, model_name, stats = fit_candidate(
                    frames,
                    config,
                    candidate,
                    model_source,
                    candidate_dir,
                    precision,
                    dataloader_num_workers=args.dataloader_num_workers,
                    autogluon_classes=autogluon_classes,
                )
                predictions, skipped, inference_runtime = run_lora_origins(
                    predictor,
                    dataframe_class,
                    model_name,
                    table,
                    config,
                    variant,
                    str(candidate["name"]),
                    may_origins,
                    "may_2026_selection",
                    "foshan_lora_search",
                    model_source,
                )
                if skipped or len(predictions["issue_time"].unique()) != len(may_origins):
                    raise RuntimeError(
                        f"Candidate did not cover every May origin. Skipped: {skipped}"
                    )
                predictions.to_csv(candidate_dir / "may_predictions.csv", index=False)
                metrics = evaluate_foshan_predictions(
                    predictions,
                    pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
                )
                metrics.to_csv(candidate_dir / "may_metrics.csv", index=False)
                physical = metrics[
                    metrics["postprocessing"] == "physical_clip_0_1700"
                ].iloc[0]
                model_label = str(physical["model_name"])
                log.update(
                    {
                        "status": "completed",
                        "model_name": model_label,
                        "predictor_path": str(candidate_dir / "predictor"),
                        "trained_model_name": model_name,
                        "adapter_paths": stats["adapter_paths"],
                        "training_runtime_seconds": stats["training_runtime_seconds"],
                        "may_inference_runtime_seconds": inference_runtime,
                        "peak_allocated_gpu_bytes": stats["peak_allocated_gpu_bytes"],
                        "peak_reserved_gpu_bytes": stats["peak_reserved_gpu_bytes"],
                        "may_wape": float(physical["wape"]),
                        "may_pv_active_mae": float(physical["pv_active_mae"]),
                    }
                )
                records.append(dict(log))
                may_prediction_frames.append(predictions)
            except Exception as error:
                log.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                failures.append(dict(log))
            finally:
                predictor = None
                if autogluon_classes is None:
                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
            log_rows.append(log)
    search_log = pd.DataFrame(log_rows)
    search_log.to_csv(search_dir / "search_log.csv", index=False)
    _write_json(search_dir / "failures.json", failures)
    if not may_prediction_frames:
        raise RuntimeError("Every Foshan LoRA search candidate failed.")

    may_predictions = pd.concat(may_prediction_frames, ignore_index=True)
    may_metrics = evaluate_foshan_predictions(
        may_predictions,
        pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
    )
    selection = select_lora_candidate(may_metrics, records)
    _write_json(search_dir / "selected_configuration.json", selection)
    may_predictions.to_csv(search_dir / "may_predictions.csv", index=False)
    may_metrics.to_csv(search_dir / "may_metrics.csv", index=False)

    selected_variant = _variant_by_name(config, str(selection["variant"]))
    predictor = predictor_class.load(str(selection["predictor_path"]))
    june_predictions, june_skipped, june_runtime = run_lora_origins(
        predictor,
        dataframe_class,
        str(selection["trained_model_name"]),
        table,
        config,
        selected_variant,
        str(selection["candidate"]),
        june_origins,
        "june_2026_test",
        "foshan_lora_frozen_june",
        model_source,
    )
    if june_skipped or len(june_predictions["issue_time"].unique()) != len(june_origins):
        raise RuntimeError(
            f"Frozen selected model did not cover every June origin: {june_skipped}"
        )
    june_metrics = evaluate_foshan_predictions(
        june_predictions,
        pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
    )
    june_predictions.to_csv(search_dir / "june_predictions.csv", index=False)
    june_metrics.to_csv(search_dir / "june_metrics.csv", index=False)

    selected_lora = june_predictions[
        june_predictions["postprocessing"] == "physical_clip_0_1700"
    ]
    comparison_input = pd.concat(
        [selected_lora[PREDICTION_COLUMNS], frozen_rows],
        ignore_index=True,
    )
    common_predictions, common_metrics = evaluate_common_scored_timestamps(
        comparison_input,
        target="pv_kw",
        split="june_2026_test",
        pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
    )
    common_predictions.to_csv(
        search_dir / "common_scored_predictions_june.csv", index=False
    )
    common_metrics.to_csv(search_dir / "common_scored_metrics_june.csv", index=False)
    improvements = comparison_improvements(
        common_metrics,
        selected_lora_model=str(selection["model_name"]),
    )
    _write_json(search_dir / "comparison_improvements.json", improvements)

    completed = search_log[search_log["status"] == "completed"]
    best_by_variant = (
        completed.sort_values(["may_wape", "may_pv_active_mae", "candidate"])
        .groupby("variant", sort=True)
        .first()
    )
    joint_improved: bool | None = None
    if {"pv_calendar", "joint_calendar"}.issubset(best_by_variant.index):
        joint_improved = bool(
            best_by_variant.loc["joint_calendar", "may_wape"]
            < best_by_variant.loc["pv_calendar", "may_wape"]
        )

    metric_columns = [
        "mae",
        "rmse",
        "wape",
        "mase",
        "bias",
        "pv_active_n_scored",
        "pv_active_mae",
        "pv_active_rmse",
        "pv_active_wape",
        "pinball_p10",
        "pinball_p50",
        "pinball_p90",
        "mean_pinball_loss",
        "p10_p90_coverage",
        "mean_interval_width",
        "n_scored",
        "n_excluded_missing",
    ]
    selected_may_row = may_metrics[
        (may_metrics["model_name"] == selection["model_name"])
        & (may_metrics["postprocessing"] == "physical_clip_0_1700")
    ].iloc[0]
    selected_june_row = june_metrics[
        june_metrics["postprocessing"] == "physical_clip_0_1700"
    ].iloc[0]
    completed_records = [
        record for record in records if record.get("status") == "completed"
    ]
    summary = {
        "selected_configuration": selection,
        "selected_adapter_paths": selection["adapter_paths"],
        "selected_may_metrics": {
            column: float(selected_may_row[column]) for column in metric_columns
        },
        "selected_june_metrics": {
            column: float(selected_june_row[column]) for column in metric_columns
        },
        "may_origin_count": len(may_origins),
        "june_origin_count": len(june_origins),
        "june_evaluation_count": 1,
        "june_inference_runtime_seconds": june_runtime,
        "total_candidate_training_runtime_seconds": sum(
            float(record["training_runtime_seconds"]) for record in completed_records
        ),
        "peak_training_allocated_gpu_bytes": max(
            (
                int(record["peak_allocated_gpu_bytes"])
                for record in completed_records
                if record["peak_allocated_gpu_bytes"] is not None
            ),
            default=None,
        ),
        "peak_training_reserved_gpu_bytes": max(
            (
                int(record["peak_reserved_gpu_bytes"])
                for record in completed_records
                if record["peak_reserved_gpu_bytes"] is not None
            ),
            default=None,
        ),
        "joint_history_improved_may_pv_wape": joint_improved,
        "failures": failures,
        "comparison": improvements,
    }
    _write_json(search_dir / "summary.json", summary)
    return summary


def run(
    args: argparse.Namespace,
    autogluon_classes: tuple[Any, Any] | None = None,
    precision_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_lora_config(Path(args.config))
    table = validate_processed_table(pd.read_parquet(args.input), config)
    model_source = resolve_model_source(args.model_id, args.model_path)
    output_dir = Path(args.output_dir)
    zero_shot_dir = Path(args.zero_shot_dir).resolve()
    if output_dir.resolve() == zero_shot_dir:
        raise ValueError("LoRA output directory must not overwrite zero-shot results.")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        str(variant["name"]): prepare_foshan_finetune_frames(table, config, variant)
        for variant in config["variants"]
    }
    metadata = _runtime_metadata(model_source, Path(args.config))
    metadata.update(
        {
            "input": str(Path(args.input).resolve()),
            "output_dir": str(output_dir.resolve()),
            "zero_shot_dir": str(zero_shot_dir),
            "stage": args.stage,
            "train_period": config["train_period"],
            "selection_period": config["selection_period"],
            "test_period": config["test_period"],
            "prediction_length": int(config["prediction_length"]),
            "context_length": int(config["context_length"]),
            "quantile_levels": config["quantile_levels"],
            "known_future_covariates": config["calendar_covariates"],
            "future_measured_target_covariates": [],
            "variants": {
                name: {
                    "targets": frames.targets,
                    "item_target_map": frames.item_target_map,
                    "n_train_rows": len(frames.train),
                    "n_tuning_rows": len(frames.tuning),
                    "n_masked_train": frames.n_masked_train,
                    "n_masked_tuning": frames.n_masked_tuning,
                }
                for name, frames in variants.items()
            },
            "june_targets_used_for_training_or_selection": False,
            "net_grid_kw_role": "provisional signed auxiliary target only",
        }
    )
    if args.stage == "dry-run":
        _write_json(output_dir / "dry_run.json", metadata)
        return metadata

    precision = precision_override or gpu_preflight()
    metadata["hardware"] = precision
    _write_json(output_dir / f"{args.stage}_run_metadata.json", metadata)
    if args.stage == "smoke":
        result = _run_smoke(
            args,
            table,
            config,
            model_source,
            precision,
            autogluon_classes,
        )
    else:
        result = _run_search(
            args,
            table,
            config,
            model_source,
            precision,
            autogluon_classes,
        )
    metadata["result"] = result
    _write_json(output_dir / f"{args.stage}_run_metadata.json", metadata)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--zero-shot-dir", type=Path, default=DEFAULT_ZERO_SHOT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--stage", choices=["dry-run", "smoke", "search"], required=True)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
