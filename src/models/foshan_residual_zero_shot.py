"""Revision-pinned Chronos-2 forecasts for the Foshan signed residual target."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
from src.models.foshan_chronos_zero_shot import run_chronos_configuration
from src.utils.runtime import git_commit, git_is_dirty


DEFAULT_CONFIG = Path("configs/foshan_chronos2_residual.json")
DEFAULT_DATA = Path(
    "results/residual_forecast/foshan_chronos2/data/signed_residual_15min.parquet"
)
DEFAULT_PROCESSED_FOSHAN = Path(
    "results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet"
)
DEFAULT_PV_SELECTION = Path("results/foshan_chronos2/selected_configuration.json")
DEFAULT_PV_CONFIG = Path("configs/foshan_chronos2_zero_shot.json")
MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
REQUIRED_CHRONOS_VERSION = "2.3.1"
PREDICTION_LENGTH = 96
QUANTILES = (0.1, 0.5, 0.9)
PREDICTION_COLUMNS = [
    "split",
    "candidate",
    "candidate_kind",
    "model_id",
    "model_revision",
    "checkpoint_path",
    "context_length",
    "refresh_cadence_minutes",
    "issue_time",
    "context_start",
    "context_end",
    "target_time",
    "horizon_step",
    "p10",
    "p50",
    "p90",
    "y_pred",
    "y_true_kw",
    "is_missing_target",
    "target_label",
    "known_future_covariates",
    "used_future_realized_data",
    "inference_call_id",
]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_residual_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_id",
        "model_revision",
        "chronos_forecasting_version",
        "prediction_length",
        "frequency",
        "timezone",
        "quantile_levels",
        "calendar_covariates",
        "zero_shot_candidates",
        "selection_period",
        "test_period",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Residual configuration is missing fields: {missing}")
    if config["model_id"] != MODEL_ID:
        raise ValueError(f"Residual benchmark requires model_id={MODEL_ID}.")
    revision = str(config["model_revision"])
    if revision != MODEL_REVISION or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Chronos-2 must use the configured immutable 40-character revision.")
    if config["chronos_forecasting_version"] != REQUIRED_CHRONOS_VERSION:
        raise ValueError(
            f"Residual benchmark requires chronos-forecasting={REQUIRED_CHRONOS_VERSION}."
        )
    if int(config["prediction_length"]) != PREDICTION_LENGTH:
        raise ValueError("Residual forecast horizon must remain 96 quarter hours.")
    if str(config["frequency"]) != FREQUENCY or str(config["timezone"]) != TIMEZONE:
        raise ValueError("Residual benchmark must remain 15-minute Asia/Shanghai data.")
    if [float(value) for value in config["quantile_levels"]] != list(QUANTILES):
        raise ValueError("Residual benchmark quantiles must be 0.1, 0.5, and 0.9.")
    if [str(value) for value in config["calendar_covariates"]] != CALENDAR_COLUMNS:
        raise ValueError("Residual calendar covariates differ from the frozen contract.")
    return config


def validate_snapshot(path: Path, revision: str = MODEL_REVISION) -> Path:
    snapshot = path.expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Chronos-2 snapshot does not exist: {snapshot}")
    if snapshot.name != revision:
        raise ValueError(
            "--model-path must be the exact revision-specific snapshot directory; "
            f"expected final path component {revision}, found {snapshot.name}."
        )
    required = [snapshot / "config.json", snapshot / "model.safetensors"]
    missing = [str(value) for value in required if not value.is_file()]
    if missing:
        raise FileNotFoundError(f"Chronos-2 snapshot is incomplete: {missing}")
    model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    architectures = [str(value) for value in model_config.get("architectures", [])]
    if "Chronos2Model" not in architectures:
        raise ValueError(f"Snapshot is not Chronos-2: architectures={architectures}")
    return snapshot


def resolve_pinned_snapshot(
    *,
    model_path: Path | None = None,
    hf_home: Path | None = None,
    allow_download: bool = False,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> Path:
    """Resolve only the exact immutable snapshot, optionally downloading that SHA."""
    if model_id != MODEL_ID or revision != MODEL_REVISION:
        raise ValueError("Mutable or alternate Chronos model identities are not permitted.")
    if model_path is not None:
        return validate_snapshot(model_path, revision)
    cache_root = Path(
        hf_home
        or os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ).expanduser()
    snapshot = (
        cache_root
        / "hub"
        / "models--amazon--chronos-2"
        / "snapshots"
        / revision
    )
    if snapshot.is_dir():
        return validate_snapshot(snapshot, revision)
    if not allow_download:
        raise FileNotFoundError(
            f"Pinned Chronos-2 snapshot is absent: {snapshot}. Set --model-path to "
            "that revision-specific directory or pass --allow-download on a connected host."
        )
    from huggingface_hub import snapshot_download

    downloaded = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_root),
        )
    )
    return validate_snapshot(downloaded, revision)


def collect_model_identity(
    snapshot: Path,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_snapshot_path": str(snapshot),
        "chronos_forecasting_version": package_version("chronos-forecasting"),
        "autogluon_version": package_version("autogluon.timeseries"),
        "python_version": sys.version,
        "pytorch_version": package_version("torch"),
        "cuda_version": None,
        "gpu_name": None,
        "gpu_total_vram_bytes": None,
        "dtype": None,
        "inference_batch_size": int(config["inference_batch_size"]),
        "seed": int(config["seed"]),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "configuration_path": str(config_path.resolve()),
        "configuration_sha256": sha256_file(config_path),
        "peak_allocated_gpu_bytes": None,
        "peak_reserved_gpu_bytes": None,
    }
    try:
        import torch

        identity["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            identity["gpu_name"] = torch.cuda.get_device_name(0)
            identity["gpu_total_vram_bytes"] = int(
                torch.cuda.get_device_properties(0).total_memory
            )
            identity["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except (ImportError, RuntimeError) as error:
        identity["torch_probe_error"] = repr(error)
    return identity


def load_pinned_pipeline(snapshot: Path, device_map: str, identity: dict[str, Any]) -> Any:
    installed = package_version("chronos-forecasting")
    if installed != REQUIRED_CHRONOS_VERSION:
        raise RuntimeError(
            f"Pinned inference requires chronos-forecasting=={REQUIRED_CHRONOS_VERSION}; "
            f"found {installed or 'not installed'}."
        )
    if device_map != "cuda":
        raise ValueError("Pinned Foshan inference requires one explicit CUDA device.")
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(str(snapshot), device_map="cuda")
    model = getattr(pipeline, "model", None)
    identity["dtype"] = str(getattr(model, "dtype", None))
    try:
        import torch

        torch.cuda.reset_peak_memory_stats(0)
    except (ImportError, RuntimeError):
        pass
    return pipeline


def _site_aware(values: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if timestamps.tz is None:
        return timestamps.tz_localize(TIMEZONE)
    return timestamps.tz_convert(TIMEZONE)


def _local_naive(values: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    timestamps = _site_aware(values)
    return timestamps.tz_localize(None).astype("datetime64[ns]")


def forecast_issue_times(
    start: pd.Timestamp | str,
    end_exclusive: pd.Timestamp | str,
    refresh_cadence_minutes: int,
) -> list[pd.Timestamp]:
    if refresh_cadence_minutes <= 0 or refresh_cadence_minutes % 15:
        raise ValueError("Refresh cadence must be a positive multiple of 15 minutes.")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end_exclusive)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(TIMEZONE)
    else:
        start_ts = start_ts.tz_convert(TIMEZONE)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(TIMEZONE)
    else:
        end_ts = end_ts.tz_convert(TIMEZONE)
    return list(
        pd.date_range(
            start_ts,
            end_ts,
            freq=f"{refresh_cadence_minutes}min",
            inclusive="left",
        )
    )


def build_inference_frames(
    residual_table: pd.DataFrame,
    issue_time: pd.Timestamp,
    context_length: int,
    calendar_columns: Sequence[str] = CALENDAR_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build one issue's local-naive boundary frames with causal target history."""
    issue = pd.Timestamp(issue_time)
    if issue.tzinfo is None:
        issue = issue.tz_localize(TIMEZONE)
    else:
        issue = issue.tz_convert(TIMEZONE)
    source = residual_table.copy()
    source["timestamp"] = _site_aware(source["timestamp"])
    missing_columns = sorted(
        {TARGET_COLUMN, *calendar_columns, "timestamp"} - set(source.columns)
    )
    if missing_columns:
        raise ValueError(f"Residual table is missing columns: {missing_columns}")
    source = source.set_index("timestamp").sort_index()
    context_times = pd.date_range(
        end=issue - pd.Timedelta(FREQUENCY),
        periods=context_length,
        freq=FREQUENCY,
    )
    future_times = pd.date_range(issue, periods=PREDICTION_LENGTH, freq=FREQUENCY)
    if context_times.max() >= issue:
        raise AssertionError("Residual context must end strictly before issue time.")
    context_values = source.reindex(context_times)
    future_values = source.reindex(future_times)
    if future_values[list(calendar_columns)].isna().any().any():
        raise ValueError(f"Known-future calendar coverage is incomplete at {issue}.")
    item_id = "foshan_signed_residual"
    context = context_values[[TARGET_COLUMN, *calendar_columns]].reset_index(
        names="timestamp"
    )
    context.insert(0, "id", item_id)
    context = context.rename(columns={TARGET_COLUMN: "target"})
    future = future_values[list(calendar_columns)].reset_index(names="timestamp")
    future.insert(0, "id", item_id)
    context["timestamp"] = _local_naive(context["timestamp"])
    future["timestamp"] = _local_naive(future["timestamp"])
    forbidden = {
        TARGET_COLUMN,
        "target",
        "pv_kw",
        "net_grid_kw",
        "pcs_kw",
        "provisional_load_kw",
    }
    if forbidden.intersection(future.columns):
        raise AssertionError("Future frame contains a realized target or site measurement.")
    metadata = {
        "issue_time": issue,
        "context_start": context_times.min(),
        "context_end": context_times.max(),
        "target_start": future_times.min(),
        "target_end": future_times.max(),
        "context_length": context_length,
        "prediction_length": PREDICTION_LENGTH,
        "missing_context_targets": int(context["target"].isna().sum()),
        "known_future_covariates": list(calendar_columns),
        "future_realized_columns": [],
    }
    return context, future, metadata


def predict_one_issue(
    pipeline: Any,
    residual_table: pd.DataFrame,
    issue_time: pd.Timestamp,
    *,
    candidate: str,
    split: str,
    context_length: int,
    refresh_cadence_minutes: int,
    snapshot: Path,
    inference_batch_size: int,
    calendar_columns: Sequence[str] = CALENDAR_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run exactly one Chronos call for one issue time."""
    context, future, metadata = build_inference_frames(
        residual_table, issue_time, context_length, calendar_columns
    )
    call_id = str(uuid.uuid4())
    started = time.monotonic()
    forecast = pipeline.predict_df(
        context,
        future_df=future,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=list(QUANTILES),
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        batch_size=inference_batch_size,
        context_length=context_length,
        freq=FREQUENCY,
    )
    runtime = time.monotonic() - started
    issue = pd.Timestamp(metadata["issue_time"])
    target_times = pd.date_range(issue, periods=PREDICTION_LENGTH, freq=FREQUENCY)
    normalized = normalize_chronos_quantiles(
        forecast,
        targets=["target"],
        expected_times=target_times,
        site_id="foshan_signed_residual",
    )
    quantiles = np.sort(
        normalized[["p10", "p50", "p90"]].to_numpy(dtype=float), axis=1
    )
    truth = (
        residual_table.assign(timestamp=_site_aware(residual_table["timestamp"]))
        .set_index("timestamp")
        .reindex(target_times)
    )
    rows = pd.DataFrame(
        {
            "split": split,
            "candidate": candidate,
            "candidate_kind": "zero_shot_residual",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_path": str(snapshot),
            "context_length": context_length,
            "refresh_cadence_minutes": refresh_cadence_minutes,
            "issue_time": issue,
            "context_start": metadata["context_start"],
            "context_end": metadata["context_end"],
            "target_time": target_times,
            "horizon_step": np.arange(1, PREDICTION_LENGTH + 1),
            "p10": quantiles[:, 0],
            "p50": quantiles[:, 1],
            "p90": quantiles[:, 2],
            "y_pred": quantiles[:, 1],
            "y_true_kw": truth[TARGET_COLUMN].to_numpy(dtype=float),
            "is_missing_target": truth[TARGET_COLUMN].isna().to_numpy(dtype=bool),
            "target_label": TARGET_LABEL,
            "known_future_covariates": "|".join(calendar_columns),
            "used_future_realized_data": False,
            "inference_call_id": call_id,
        },
        columns=PREDICTION_COLUMNS,
    )
    audit = {
        **metadata,
        "candidate": candidate,
        "split": split,
        "inference_call_id": call_id,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_path": str(snapshot),
        "forecast_rows": len(rows),
        "runtime_seconds": runtime,
        "different_issue_times_batched": False,
        "used_future_realized_data": False,
    }
    return rows, audit


def run_residual_candidate(
    pipeline: Any,
    residual_table: pd.DataFrame,
    issues: Iterable[pd.Timestamp],
    *,
    candidate: str,
    split: str,
    context_length: int,
    refresh_cadence_minutes: int,
    snapshot: Path,
    inference_batch_size: int,
    max_issues: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = list(issues)
    if max_issues is not None:
        if max_issues <= 0:
            raise ValueError("max_issues must be positive.")
        selected = selected[:max_issues]
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for issue in selected:
        rows, audit = predict_one_issue(
            pipeline,
            residual_table,
            issue,
            candidate=candidate,
            split=split,
            context_length=context_length,
            refresh_cadence_minutes=refresh_cadence_minutes,
            snapshot=snapshot,
            inference_batch_size=inference_batch_size,
        )
        frames.append(rows)
        audits.append(audit)
    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS), pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.groupby("inference_call_id")["issue_time"].nunique().gt(1).any():
        raise AssertionError("A Chronos call contains more than one issue time.")
    if predictions.groupby("issue_time").size().ne(PREDICTION_LENGTH).any():
        raise AssertionError("Every residual issue must return exactly 96 forecasts.")
    return predictions, pd.DataFrame(audits)


def residual_metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    scored = frame.loc[
        ~frame["is_missing_target"].astype(bool)
        & frame[["y_true_kw", "p50"]].notna().all(axis=1)
    ]
    if scored.empty:
        raise ValueError("Residual metric scope contains no scored rows.")
    actual = scored["y_true_kw"].to_numpy(dtype=float)
    predicted = scored["p50"].to_numpy(dtype=float)
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {
        "n_scored": len(scored),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
        "bias": float(np.mean(error)),
        "absolute_bias": float(abs(np.mean(error))),
    }


def evaluate_residual_predictions(
    predictions: pd.DataFrame,
    start: pd.Timestamp | str,
    end_exclusive: pd.Timestamp | str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end_exclusive)
    target_time = pd.to_datetime(predictions["target_time"], errors="raise", utc=True)
    start_utc = start_ts.tz_convert("UTC") if start_ts.tzinfo else start_ts.tz_localize(TIMEZONE).tz_convert("UTC")
    end_utc = end_ts.tz_convert("UTC") if end_ts.tzinfo else end_ts.tz_localize(TIMEZONE).tz_convert("UTC")
    selected = predictions.loc[(target_time >= start_utc) & (target_time < end_utc)].copy()
    rows: list[dict[str, Any]] = []
    for candidate, group in selected.groupby("candidate", sort=True):
        base = {
            "split": str(group["split"].iloc[0]),
            "candidate": candidate,
            "metric_scope": "overall",
            "horizon_step": np.nan,
            **residual_metric_values(group),
        }
        rows.append(base)
        for horizon, horizon_group in group.groupby("horizon_step", sort=True):
            rows.append(
                {
                    "split": str(group["split"].iloc[0]),
                    "candidate": candidate,
                    "metric_scope": "horizon",
                    "horizon_step": int(horizon),
                    **residual_metric_values(horizon_group),
                }
            )
    return pd.DataFrame(rows)


def generate_frozen_april_pv(
    pipeline: Any,
    processed_table_path: Path,
    pv_config_path: Path,
    pv_selection_path: Path,
    model_source: str,
    max_origins: int | None = None,
) -> pd.DataFrame:
    """Apply the already-selected frozen PV configuration to April origins."""
    table = pd.read_parquet(processed_table_path)
    table["timestamp"] = _site_aware(table["timestamp"])
    pv_config = json.loads(pv_config_path.read_text(encoding="utf-8"))
    selection = json.loads(pv_selection_path.read_text(encoding="utf-8"))
    selected = selection["targets"]["pv_kw"]
    configuration = next(
        item
        for item in pv_config["configurations"]
        if item["name"] == selected["model_name"]
    )
    context_length = int(selected["context_length"])
    start = pd.Timestamp("2026-04-01T00:00:00+08:00")
    end = pd.Timestamp("2026-05-01T00:00:00+08:00")
    origins = list(pd.date_range(start, end, freq="1D", inclusive="left"))
    rows, skipped, _ = run_chronos_configuration(
        pipeline,
        table,
        pv_config,
        configuration,
        [context_length],
        origins,
        "april_2026_selection",
        f"frozen_pv_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
        model_source,
        max_origins=max_origins,
    )
    rows = rows.loc[rows["target"].eq("pv_kw")].copy()
    if skipped:
        raise RuntimeError(f"Frozen April PV forecast skipped origins: {skipped}")
    expected = (max_origins if max_origins is not None else len(origins)) * PREDICTION_LENGTH
    if len(rows) != expected:
        raise RuntimeError(f"Frozen April PV forecast returned {len(rows)} rows, expected {expected}.")
    rows["model_revision"] = MODEL_REVISION
    rows["checkpoint_path"] = model_source
    rows["selection_was_frozen_before_residual_experiment"] = True
    return rows


def update_peak_gpu(identity: dict[str, Any]) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            identity["peak_allocated_gpu_bytes"] = int(torch.cuda.max_memory_allocated(0))
            identity["peak_reserved_gpu_bytes"] = int(torch.cuda.max_memory_reserved(0))
    except (ImportError, RuntimeError):
        pass


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _timestamped_output_root() -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results/residual_forecast/foshan_chronos2") / run_id


def run(args: argparse.Namespace, pipeline: Any | None = None) -> Path:
    config = load_residual_config(args.config)
    output_dir = args.output_dir or _timestamped_output_root()
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = resolve_pinned_snapshot(
        model_path=args.model_path,
        hf_home=args.hf_home,
        allow_download=args.allow_download,
    )
    identity = collect_model_identity(snapshot, args.config, config)
    active_pipeline = pipeline or load_pinned_pipeline(snapshot, args.device_map, identity)
    residual = pd.read_parquet(args.input)
    residual["timestamp"] = _site_aware(residual["timestamp"])

    candidates = config["zero_shot_candidates"]
    requested = set(args.candidates.split(",")) if args.candidates else None
    if requested is not None:
        candidates = [item for item in candidates if item["name"] in requested]
        missing = requested - {item["name"] for item in candidates}
        if missing:
            raise ValueError(f"Unknown residual candidates: {sorted(missing)}")
    splits = {
        "april": config["selection_period"],
        "may": config["test_period"],
    }
    requested_splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    unknown_splits = sorted(set(requested_splits) - set(splits))
    if unknown_splits:
        raise ValueError(f"Unknown splits: {unknown_splits}")

    prediction_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    max_issues = 1 if args.stage == "smoke" else args.max_issues
    for split_key in requested_splits:
        period = splits[split_key]
        for candidate in candidates:
            issues = forecast_issue_times(
                period["start"],
                period["end_exclusive"],
                int(candidate["refresh_cadence_minutes"]),
            )
            rows, audit = run_residual_candidate(
                active_pipeline,
                residual,
                issues,
                candidate=str(candidate["name"]),
                split=str(period["name"]),
                context_length=int(candidate["context_length"]),
                refresh_cadence_minutes=int(candidate["refresh_cadence_minutes"]),
                snapshot=snapshot,
                inference_batch_size=int(config["inference_batch_size"]),
                max_issues=max_issues,
            )
            prediction_parts.append(rows)
            audit_parts.append(audit)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    inference_audit = pd.concat(audit_parts, ignore_index=True)
    predictions.to_csv(output_dir / "predictions_long.csv", index=False, float_format="%.15g")
    inference_audit.to_csv(output_dir / "inference_audit.csv", index=False, float_format="%.15g")
    april = predictions.loc[predictions["split"].eq(config["selection_period"]["name"])]
    if not april.empty:
        metrics = evaluate_residual_predictions(
            april,
            config["selection_period"]["start"],
            config["selection_period"]["end_exclusive"],
        )
        metrics.to_csv(output_dir / "april_forecast_metrics.csv", index=False, float_format="%.15g")

    if not args.skip_frozen_april_pv:
        frozen_pv = generate_frozen_april_pv(
            active_pipeline,
            args.processed_foshan_input,
            args.pv_config,
            args.pv_selection,
            str(snapshot),
            max_origins=max_issues,
        )
        frozen_pv.to_csv(
            output_dir / "frozen_pv_april_predictions.csv",
            index=False,
            float_format="%.15g",
        )
    update_peak_gpu(identity)
    identity["run_stage"] = args.stage
    identity["output_dir"] = str(output_dir.resolve())
    identity["completed_issue_calls"] = int(len(inference_audit))
    _write_json(output_dir / "model_identity.json", identity)
    print(f"Saved {len(predictions):,} revision-pinned residual forecasts to {output_dir.resolve()}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--input", default=DEFAULT_DATA, type=Path)
    parser.add_argument("--processed-foshan-input", default=DEFAULT_PROCESSED_FOSHAN, type=Path)
    parser.add_argument("--pv-config", default=DEFAULT_PV_CONFIG, type=Path)
    parser.add_argument("--pv-selection", default=DEFAULT_PV_SELECTION, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--hf-home", default=None, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--stage", choices=("smoke", "forecast"), default="forecast")
    parser.add_argument("--splits", default="april,may")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--max-issues", default=None, type=int)
    parser.add_argument("--skip-frozen-april-pv", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
