"""Run the reproducible Foshan Chronos-2 zero-shot benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.prepare_foshan import (
    CALENDAR_COLUMNS,
    prepare_foshan_workbooks,
    sha256_file,
    write_audit_documents,
    write_prepared_outputs,
)
from src.evaluation.foshan_benchmark import (
    PREDICTION_COLUMNS,
    build_forecast_window,
    chronos_rows_for_window,
    configurations_for_frozen_test,
    evaluate_foshan_predictions,
    load_foshan_config,
    period_origins,
    run_causal_baselines,
    select_may_configurations,
    validate_processed_table,
)


DEFAULT_CONFIG = Path("configs/foshan_chronos2_zero_shot.json")
DEFAULT_OUTPUT_DIR = Path("results/zero_shot/foshan_chronos2")
REQUIRED_CHRONOS_VERSION = "2.3.1"


def parse_csv_list(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_known_future_covariates(
    configuration: dict[str, Any],
    config: dict[str, Any],
    weather_covariates: list[str],
    table: pd.DataFrame,
) -> list[str]:
    requested = configuration.get("known_future_covariates", [])
    columns: list[str] = []
    if "calendar" in requested:
        columns.extend(config["calendar_covariates"])
        columns.extend(weather_covariates)
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(
            "Requested known-future weather columns are not present in the processed table: "
            f"{missing}. Supply real observations/forecasts; the benchmark will not fabricate them."
        )
    return list(dict.fromkeys(columns))


def chronos_input_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Copy a frame and remove timezone metadata without shifting local clock time."""
    if frame is None:
        return None
    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_localize(None)
    result["timestamp"] = timestamps.astype("datetime64[ns]")
    return result


def run_chronos_configuration(
    pipeline: Any,
    table: pd.DataFrame,
    config: dict[str, Any],
    configuration: dict[str, Any],
    context_lengths: list[int],
    origins: list[pd.Timestamp],
    split_name: str,
    run_id: str,
    model_source: str,
    weather_covariates: list[str] | None = None,
    max_origins: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    """Run one Chronos schema over rolling origins without loading a model."""
    targets = [str(target) for target in configuration["targets"]]
    covariates = resolve_known_future_covariates(
        configuration,
        config,
        weather_covariates or [],
        table,
    )
    selected_origins = origins[:max_origins] if max_origins is not None else origins
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    runtimes: dict[str, float] = {}
    for context_length in context_lengths:
        runtime_key = f"{split_name}:{configuration['name']}:{context_length}"
        started = time.monotonic()
        for issue_time in selected_origins:
            window, reason = build_forecast_window(
                table=table,
                issue_time=issue_time,
                targets=targets,
                context_length=context_length,
                prediction_length=int(config["prediction_length"]),
                known_future_covariates=covariates,
                causal_fill_limit=int(config["causal_fill_limit"]),
                frequency=config["frequency"],
            )
            if window is None:
                skipped.append(
                    {
                        "split": split_name,
                        "model_name": configuration["name"],
                        "context_length": context_length,
                        "issue_time": issue_time.isoformat(),
                        "targets": targets,
                        "reason": reason,
                    }
                )
                continue
            target_argument: str | list[str] = targets[0] if len(targets) == 1 else targets
            context_df = chronos_input_frame(window.context_df)
            future_df = chronos_input_frame(window.future_df)
            forecast_df = pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=int(config["prediction_length"]),
                quantile_levels=[float(value) for value in config["quantile_levels"]],
                id_column="id",
                timestamp_column="timestamp",
                target=target_argument,
                batch_size=int(config["inference_batch_size"]),
                context_length=context_length,
                freq=config["frequency"],
            )
            frames.append(
                chronos_rows_for_window(
                    forecast_df=forecast_df,
                    window=window,
                    targets=targets,
                    split_name=split_name,
                    run_id=run_id,
                    model_name=configuration["name"],
                    model_id=model_source,
                    context_length=context_length,
                    known_future_covariates=covariates,
                    pv_capacity_kw=float(config["pv_capacity_kw"]),
                    site_id=config["site_id"],
                )
            )
        runtimes[runtime_key] = time.monotonic() - started
    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS), skipped, runtimes
    return pd.concat(frames, ignore_index=True), skipped, runtimes


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_environment_metadata(model_source: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": sys.version,
        "packages": {
            package: _package_version(package)
            for package in (
                "chronos-forecasting",
                "torch",
                "pandas",
                "numpy",
                "pyarrow",
                "openpyxl",
                "matplotlib",
            )
        },
        "model_id_or_path": model_source,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_total_vram_bytes": None,
        "bf16_supported": False,
        "inference_dtype": None,
        "peak_allocated_gpu_bytes": None,
        "peak_reserved_gpu_bytes": None,
    }
    try:
        import torch

        metadata["torch_cuda_version"] = torch.version.cuda
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
            metadata["gpu_total_vram_bytes"] = int(properties.total_memory)
            metadata["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except (ImportError, RuntimeError) as error:
        metadata["torch_probe_error"] = repr(error)
    return metadata


def load_chronos_pipeline(
    model_source: str,
    device_map: str,
    metadata: dict[str, Any],
) -> Any:
    installed = _package_version("chronos-forecasting")
    if installed != REQUIRED_CHRONOS_VERSION:
        raise RuntimeError(
            f"Foshan inference requires chronos-forecasting=={REQUIRED_CHRONOS_VERSION}; "
            f"found {installed or 'not installed'}."
        )
    if device_map == "auto":
        raise ValueError("device_map='auto' is not permitted; use one explicit CUDA device.")
    if device_map != "cuda":
        raise ValueError("The production Foshan benchmark requires --device-map cuda.")
    local_candidate = Path(model_source).expanduser()
    if local_candidate.is_absolute() and not local_candidate.is_dir():
        raise FileNotFoundError(f"Local Chronos-2 model directory does not exist: {local_candidate}")

    import torch
    from chronos import Chronos2Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to emulate the Chronos GPU benchmark.")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    metadata["inference_dtype"] = str(dtype).removeprefix("torch.")
    return Chronos2Pipeline.from_pretrained(
        model_source,
        device_map="cuda",
        torch_dtype=dtype,
    )


def update_peak_gpu_metadata(metadata: dict[str, Any]) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            metadata["peak_allocated_gpu_bytes"] = int(torch.cuda.max_memory_allocated(0))
            metadata["peak_reserved_gpu_bytes"] = int(torch.cuda.max_memory_reserved(0))
    except (ImportError, RuntimeError):
        return


def write_environment_freeze(path: Path, metadata: dict[str, Any]) -> None:
    distributions = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.lower,
    )
    lines = [
        "# Runtime metadata",
        json.dumps(metadata, indent=2, ensure_ascii=False),
        "",
        "# Installed Python distributions",
        *distributions,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_prediction_outputs(
    predictions: pd.DataFrame,
    output_dir: Path,
    pv_active_threshold_kw: float = 1.0,
) -> dict[str, pd.DataFrame]:
    predictions = predictions[PREDICTION_COLUMNS].sort_values(
        ["split", "target", "model_name", "context_length", "issue_time", "horizon_step", "postprocessing"]
    )
    predictions.to_csv(output_dir / "predictions_long.csv", index=False)
    metrics_by_model = evaluate_foshan_predictions(
        predictions,
        by_horizon=False,
        pv_active_threshold_kw=pv_active_threshold_kw,
    )
    metrics_by_horizon = evaluate_foshan_predictions(
        predictions,
        by_horizon=True,
        pv_active_threshold_kw=pv_active_threshold_kw,
    )
    selection_metrics = metrics_by_model[
        metrics_by_model["split"] == "may_2026_selection"
    ].reset_index(drop=True)
    test_metrics = metrics_by_model[
        metrics_by_model["split"] == "june_2026_test"
    ].reset_index(drop=True)
    selection_metrics.to_csv(output_dir / "selection_metrics_may.csv", index=False)
    test_metrics.to_csv(output_dir / "test_metrics_june.csv", index=False)
    metrics_by_model.to_csv(output_dir / "metrics_by_model.csv", index=False)
    metrics_by_horizon.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    return {
        "metrics_by_model": metrics_by_model,
        "metrics_by_horizon": metrics_by_horizon,
        "selection_metrics": selection_metrics,
        "test_metrics": test_metrics,
    }


def representative_pv_days(table: pd.DataFrame) -> dict[str, pd.Timestamp]:
    pv = table[(table["timestamp"] >= pd.Timestamp("2026-05-01", tz="Asia/Shanghai")) &
               (table["timestamp"] < pd.Timestamp("2026-07-01", tz="Asia/Shanghai"))].copy()
    pv["day"] = pv["timestamp"].dt.floor("D")
    daily = pv.groupby("day")["pv_kw"].agg(["sum", "median"])
    variability = pv.groupby("day")["pv_kw"].apply(
        lambda values: float(pd.to_numeric(values, errors="coerce").diff().abs().mean())
    )
    daily["variability"] = variability
    median_sum = daily["sum"].median()
    return {
        "high_output": pd.Timestamp(daily["sum"].idxmax()),
        "variable_output": pd.Timestamp(daily["variability"].idxmax()),
        "median_output": pd.Timestamp((daily["sum"] - median_sum).abs().idxmin()),
    }


def write_representative_plots(
    table: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    indexed = table.set_index("timestamp")
    for label, day in representative_pv_days(table).items():
        end = day + pd.Timedelta(days=1)
        actual = indexed.loc[(indexed.index >= day) & (indexed.index < end), "pv_kw"]
        figure, axis = plt.subplots(figsize=(11, 4))
        axis.plot(actual.index, actual.to_numpy(), label="actual pv_kw", color="black", linewidth=2)
        candidates = predictions[
            (predictions["target"] == "pv_kw")
            & (pd.to_datetime(predictions["issue_time"]) == day)
            & predictions["postprocessing"].isin(["physical_clip_0_1700", "physical_target"])
        ]
        model_names = list(dict.fromkeys(candidates["model_name"].tolist()))[:5]
        for model_name in model_names:
            model = candidates[candidates["model_name"] == model_name].sort_values("target_time")
            axis.plot(pd.to_datetime(model["target_time"]), model["p50"], label=model_name, alpha=0.8)
        axis.set_title(f"Foshan PV: {label.replace('_', ' ')} day ({day.date()})")
        axis.set_ylabel("kW")
        axis.set_xlabel("Asia/Shanghai time")
        axis.legend(loc="upper right", fontsize=8)
        figure.tight_layout()
        path = output_dir / f"pv_{label}_day.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


def write_report(
    output_dir: Path,
    audit: dict[str, Any],
    metrics: dict[str, pd.DataFrame] | None,
    selection: dict[str, Any] | None,
    skipped: list[dict[str, Any]],
    metadata: dict[str, Any],
    stage: str,
) -> None:
    lines = [
        "# Foshan Chronos-2 Zero-Shot Report",
        "",
        f"- Stage completed: `{stage}`",
        f"- Model source: `{metadata['model_id_or_path']}`",
        f"- Total runtime: {metadata.get('total_runtime_seconds', 0.0):.2f} seconds",
        f"- Configured Chronos windows skipped: {len(skipped)}",
        "",
        "> `net_grid_kw` remains provisional bidirectional grid exchange. It was never treated as gross load and its negative values were never clipped.",
        "",
        "## Data",
        "",
        f"- Site rows: {metadata.get('processed_rows')}",
        f"- Range: {metadata.get('processed_start')} through {metadata.get('processed_end')}",
        f"- PV missing intervals: {audit['targets']['pv_kw']['missing_intervals']}",
        f"- Grid missing intervals: {audit['targets']['net_grid_kw']['missing_intervals']}",
        f"- PV negative raw readings corrected: {audit['targets']['pv_kw']['corrected_low_count']}",
        "",
        "The MASE denominator is the mean absolute 96-step seasonal difference computed only from observations before each issue time. Missing target rows are excluded from scoring.",
        "",
    ]
    if selection:
        lines.extend(["## May-Only Selection", "", "```json", json.dumps(selection, indent=2), "```", ""])
    if metrics is not None and not metrics["test_metrics"].empty:
        display = metrics["test_metrics"][
            ["target", "model_name", "context_length", "postprocessing", "mae", "rmse", "wape", "mase", "bias"]
        ]
        lines.extend(["## June Metrics", "", "```text", display.to_string(index=False), "```", ""])
    if skipped:
        reasons = pd.DataFrame(skipped)["reason"].value_counts().to_dict()
        lines.extend(["## Skipped Origins", "", f"Reasons: `{reasons}`", ""])
    if metadata.get("chronos_status") != "completed":
        lines.extend(
            [
                "## Chronos Status",
                "",
                "Chronos inference did not run in this stage. No Chronos metrics are reported or inferred.",
                "",
            ]
        )
    (output_dir / "zero_shot_report.md").write_text("\n".join(lines), encoding="utf-8")


def _load_or_prepare_data(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame | None, pd.DataFrame]:
    if args.processed_input is not None:
        table = pd.read_parquet(args.processed_input)
        audit_path = args.output_dir / "data_audit.json"
        if not audit_path.is_file():
            raise FileNotFoundError(
                f"Processed input requires its existing audit at {audit_path}. Run --stage prepare first."
            )
        with audit_path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
        return table, audit, None, pd.DataFrame(columns=["timestamp", "pv_kw_raw"])
    if args.source_workbook is None:
        raise ValueError("--source-workbook is required unless --processed-input is provided.")
    return prepare_foshan_workbooks(
        source_workbook=args.source_workbook,
        storage_workbook=args.storage_workbook,
        pv_capacity_kw=float(config["pv_capacity_kw"]),
    )


def _model_source(args: argparse.Namespace, config: dict[str, Any]) -> str:
    local = args.model_path or os.environ.get("CHRONOS_MODEL_PATH")
    return (
        str(Path(local).expanduser().resolve())
        if local
        else str(args.model_id or config["model_id"])
    )


def collect_window_eligibility_skips(
    table: pd.DataFrame,
    config: dict[str, Any],
    split_origins: dict[str, list[pd.Timestamp]],
) -> list[dict[str, Any]]:
    """Audit all configured context windows without invoking Chronos."""
    skipped: list[dict[str, Any]] = []
    for split_name, origins in split_origins.items():
        for configuration in config["configurations"]:
            targets = [str(target) for target in configuration["targets"]]
            covariates = (
                list(config["calendar_covariates"])
                if "calendar" in configuration.get("known_future_covariates", [])
                else []
            )
            for context_length in (int(value) for value in config["context_lengths"]):
                for issue_time in origins:
                    window, reason = build_forecast_window(
                        table=table,
                        issue_time=issue_time,
                        targets=targets,
                        context_length=context_length,
                        prediction_length=int(config["prediction_length"]),
                        known_future_covariates=covariates,
                        causal_fill_limit=int(config["causal_fill_limit"]),
                        frequency=config["frequency"],
                    )
                    if window is None:
                        skipped.append(
                            {
                                "split": split_name,
                                "model_name": configuration["name"],
                                "context_length": context_length,
                                "issue_time": issue_time.isoformat(),
                                "targets": targets,
                                "reason": reason,
                            }
                        )
    return skipped


def apply_eligibility_audit(
    audit: dict[str, Any],
    skipped: list[dict[str, Any]],
) -> None:
    for target in ("pv_kw", "net_grid_kw"):
        target_skips = [item for item in skipped if target in item["targets"]]
        unique_origins = {
            (item["split"], item["issue_time"])
            for item in target_skips
        }
        audit["targets"][target]["skipped_forecast_origins"] = len(unique_origins)
        audit["targets"][target]["skipped_forecast_windows"] = len(target_skips)
        audit["targets"][target]["skipped_origin_reasons"] = (
            pd.DataFrame(target_skips)["reason"].value_counts().to_dict()
            if target_skips
            else {}
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--source-workbook", default=None, type=Path)
    parser.add_argument("--storage-workbook", default=None, type=Path)
    parser.add_argument("--processed-input", default=None, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument(
        "--stage",
        choices=["prepare", "baselines", "smoke", "chronos", "all"],
        default="all",
    )
    parser.add_argument(
        "--weather-covariates",
        default="",
        help="Comma-separated real known-future weather columns already present in the input.",
    )
    parser.add_argument("--max-origins", default=None, type=int)
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    started = time.monotonic()
    config = load_foshan_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_metadata: dict[str, Any] = {}
    metadata_path = args.output_dir / "run_metadata.json"
    if args.stage == "chronos" and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous_metadata = json.load(handle)
    run_id = (
        args.run_id
        or previous_metadata.get("run_id")
        or datetime.now(timezone.utc).strftime("foshan_%Y%m%dT%H%M%SZ")
    )
    model_source = _model_source(args, config)
    metadata = collect_environment_metadata(model_source)
    metadata.update({"run_id": run_id, "stage": args.stage, "config_sha256": sha256_file(args.config)})

    table, audit, storage_aligned, negative_readings = _load_or_prepare_data(args, config)
    table = validate_processed_table(table, config)
    processed_output = args.output_dir / "processed_foshan_15min.parquet"
    if args.processed_input is None:
        write_prepared_outputs(
            table,
            audit,
            negative_readings,
            processed_output,
            args.output_dir,
            storage_aligned,
        )
    metadata.update(
        {
            "processed_path": str(args.processed_input or processed_output),
            "processed_rows": int(len(table)),
            "processed_start": table["timestamp"].min().isoformat(),
            "processed_end": table["timestamp"].max().isoformat(),
            "input_hashes": {
                "source_workbook": audit.get("source_workbook_sha256"),
                "storage_workbook": audit.get("storage_workbook_sha256"),
            },
            "split": {
                "selection_period": config["selection_period"],
                "test_period": config["test_period"],
            },
            "per_configuration_runtime_seconds": {},
            "chronos_status": "not_run",
            "prior_stage_runtime_seconds": float(
                previous_metadata.get("total_runtime_seconds", 0.0)
            ),
        }
    )
    may_origins = period_origins(
        table,
        config["selection_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=config["frequency"],
        stride_steps=int(config["origin_stride_steps"]),
        timezone=config["timezone"],
    )
    june_origins = period_origins(
        table,
        config["test_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=config["frequency"],
        stride_steps=int(config["origin_stride_steps"]),
        timezone=config["timezone"],
    )
    metadata["origin_counts"] = {"may": len(may_origins), "june": len(june_origins)}
    eligibility_skips = collect_window_eligibility_skips(
        table,
        config,
        {
            "may_2026_selection": may_origins,
            "june_2026_test": june_origins,
        },
    )
    apply_eligibility_audit(audit, eligibility_skips)
    write_audit_documents(audit, args.output_dir)
    (args.output_dir / "window_eligibility_skips.json").write_text(
        json.dumps(eligibility_skips, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata["window_eligibility_skips"] = len(eligibility_skips)
    if args.stage == "prepare":
        metadata["total_runtime_seconds"] = time.monotonic() - started
        write_environment_freeze(args.output_dir / "environment_freeze.txt", metadata)
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_report(
            args.output_dir, audit, None, None, eligibility_skips, metadata, args.stage
        )
        print(f"Prepared and audited {len(table):,} Foshan rows in {args.output_dir}")
        return

    baseline_frames: list[pd.DataFrame] = []
    if args.stage in {"baselines", "all"}:
        baseline_started = time.monotonic()
        baseline_frames = [
            run_causal_baselines(
                table,
                may_origins,
                "may_2026_selection",
                run_id,
                prediction_length=int(config["prediction_length"]),
                causal_fill_limit=int(config["causal_fill_limit"]),
                frequency=config["frequency"],
            ),
            run_causal_baselines(
                table,
                june_origins,
                "june_2026_test",
                run_id,
                prediction_length=int(config["prediction_length"]),
                causal_fill_limit=int(config["causal_fill_limit"]),
                frequency=config["frequency"],
            ),
        ]
        metadata["baseline_runtime_seconds"] = time.monotonic() - baseline_started
    elif args.stage == "chronos":
        baseline_path = args.output_dir / "predictions_long.csv"
        if baseline_path.is_file():
            existing = pd.read_csv(baseline_path)
            for column in ("issue_time", "target_time"):
                existing[column] = pd.to_datetime(existing[column])
            baseline_frames = [existing[existing["model_id"] == "causal_baseline"]]

    if args.stage == "baselines":
        predictions = pd.concat(baseline_frames, ignore_index=True)
        metrics = save_prediction_outputs(
            predictions,
            args.output_dir,
            pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
        )
        write_representative_plots(table, predictions, args.output_dir)
        metadata["total_runtime_seconds"] = time.monotonic() - started
        write_environment_freeze(args.output_dir / "environment_freeze.txt", metadata)
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_report(
            args.output_dir,
            audit,
            metrics,
            None,
            eligibility_skips,
            metadata,
            args.stage,
        )
        print(f"Wrote full May/June causal baseline benchmark to {args.output_dir}")
        return

    try:
        pipeline = load_chronos_pipeline(model_source, args.device_map, metadata)
    except Exception as error:
        failure = dict(metadata)
        failure["chronos_status"] = "preflight_failed"
        failure["error_type"] = type(error).__name__
        failure["error"] = str(error)
        failure["current_stage_runtime_seconds"] = time.monotonic() - started
        failure["total_runtime_seconds"] = (
            failure["prior_stage_runtime_seconds"]
            + failure["current_stage_runtime_seconds"]
        )
        (args.output_dir / "chronos_preflight_failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise
    weather_covariates = parse_csv_list(args.weather_covariates)
    skipped: list[dict[str, Any]] = []
    chronos_frames: list[pd.DataFrame] = []
    runtimes: dict[str, float] = {}

    smoke_configuration = config["configurations"][0]
    smoke, smoke_skipped, smoke_runtime = run_chronos_configuration(
        pipeline,
        table,
        config,
        smoke_configuration,
        [int(config["context_lengths"][0])],
        may_origins,
        "may_2026_selection",
        run_id,
        model_source,
        weather_covariates=weather_covariates,
        max_origins=1,
    )
    smoke.to_csv(args.output_dir / "smoke_predictions.csv", index=False)
    skipped.extend(smoke_skipped)
    runtimes.update({f"smoke:{key}": value for key, value in smoke_runtime.items()})
    if args.stage == "smoke":
        metadata["chronos_status"] = "smoke_completed"
        metadata["per_configuration_runtime_seconds"] = runtimes
        update_peak_gpu_metadata(metadata)
        metadata["total_runtime_seconds"] = time.monotonic() - started
        write_environment_freeze(args.output_dir / "environment_freeze.txt", metadata)
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_report(args.output_dir, audit, None, None, skipped, metadata, args.stage)
        print(f"Chronos one-origin smoke completed: {len(smoke):,} prediction rows")
        return

    for configuration in config["configurations"]:
        frame, config_skipped, config_runtime = run_chronos_configuration(
            pipeline,
            table,
            config,
            configuration,
            [int(value) for value in config["context_lengths"]],
            may_origins,
            "may_2026_selection",
            run_id,
            model_source,
            weather_covariates=weather_covariates,
            max_origins=args.max_origins,
        )
        chronos_frames.append(frame)
        skipped.extend(config_skipped)
        runtimes.update(config_runtime)

    may_predictions = pd.concat([*baseline_frames, *chronos_frames], ignore_index=True)
    may_metrics = evaluate_foshan_predictions(
        may_predictions[may_predictions["split"] == "may_2026_selection"],
        pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
    )
    selection = select_may_configurations(may_metrics)
    (args.output_dir / "selected_configuration.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    june_frames: list[pd.DataFrame] = []
    for configuration, context_length in configurations_for_frozen_test(
        config["configurations"], selection
    ):
        frame, config_skipped, config_runtime = run_chronos_configuration(
            pipeline,
            table,
            config,
            configuration,
            [context_length],
            june_origins,
            "june_2026_test",
            run_id,
            model_source,
            weather_covariates=weather_covariates,
            max_origins=args.max_origins,
        )
        selected_targets = {
            target
            for target, target_selection in selection["targets"].items()
            if target_selection["model_name"] == configuration["name"]
            and int(target_selection["context_length"]) == context_length
        }
        frame = frame[frame["target"].isin(selected_targets)].reset_index(drop=True)
        june_frames.append(frame)
        skipped.extend(config_skipped)
        runtimes.update(config_runtime)

    predictions = pd.concat([may_predictions, *june_frames], ignore_index=True)
    metrics = save_prediction_outputs(
        predictions,
        args.output_dir,
        pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
    )
    plot_paths = write_representative_plots(table, predictions, args.output_dir)
    for target in ("pv_kw", "net_grid_kw"):
        target_skips = [item for item in skipped if target in item["targets"]]
        audit["targets"][target]["chronos_skipped_windows_executed"] = len(target_skips)
    if args.processed_input is None:
        write_prepared_outputs(
            table,
            audit,
            negative_readings,
            processed_output,
            args.output_dir,
            storage_aligned,
        )
    else:
        write_audit_documents(audit, args.output_dir)
    (args.output_dir / "skipped_origins.json").write_text(
        json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata["chronos_status"] = "completed"
    metadata["may_selection"] = selection
    metadata["per_configuration_runtime_seconds"] = runtimes
    metadata["plots"] = plot_paths
    update_peak_gpu_metadata(metadata)
    metadata["current_stage_runtime_seconds"] = time.monotonic() - started
    metadata["total_runtime_seconds"] = (
        metadata["prior_stage_runtime_seconds"] + metadata["current_stage_runtime_seconds"]
    )
    write_environment_freeze(args.output_dir / "environment_freeze.txt", metadata)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(args.output_dir, audit, metrics, selection, skipped, metadata, args.stage)
    print(f"Completed Foshan Chronos-2 benchmark in {metadata['total_runtime_seconds']:.2f}s")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
