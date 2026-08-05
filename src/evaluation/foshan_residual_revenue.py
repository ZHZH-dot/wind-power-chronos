"""Select signed-residual forecasts on April revenue and test once on May."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.reconstruct_foshan_provisional_load import (
    TIMEZONE,
    load_provisional_load_signals,
    reconstruct_provisional_load,
)
from src.data.reconstruct_foshan_residual import (
    TARGET_COLUMN,
    TARGET_LABEL,
    aggregate_signed_residual_15min,
    reconstruct_signed_residual,
    sha256_file,
)
from src.models.foshan_residual_zero_shot import (
    MODEL_ID,
    MODEL_REVISION,
    residual_metric_values,
)
from src.optimization.foshan_battery_milp import DispatchParameters
from src.optimization.foshan_controller_v5_final_benchmark import summarize_result
from src.optimization.foshan_forecast_backtest import (
    load_reference_dispatch,
    load_selected_chronos_p50,
    previous_day_forecast,
)
from src.optimization.foshan_residual_controller_eval import (
    ForecastBook,
    make_forecast_book,
    run_rolling_v5_evaluation,
    verify_frozen_controller_sources,
)
from src.utils.runtime import git_commit, git_is_dirty


APRIL_START = pd.Timestamp("2026-04-01 00:00:00")
MAY_START = pd.Timestamp("2026-05-01 00:00:00")
JUNE_START = pd.Timestamp("2026-06-01 00:00:00")
GROSS_BASELINE = "previous_day_gross_load"
GROSS_CHRONOS = "chronos2_univariate_ctx1344"
BLEND_CANDIDATE = "blend_best_hourly_residual_four_week_median"
HISTORICAL_REVENUE_YUAN = 112029.39
CURRENT_CTX1344_REVENUE_YUAN = 109408.85776975
OPTIMIZED_BASELINE_YUAN = 122211.25
EQUAL_CONDITION_ORACLE_YUAN = 123819.09


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _file_identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required revenue input does not exist: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def build_revenue_run_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Return the immutable inputs that make a revenue run resumable."""
    files = {
        name: _file_identity(Path(getattr(args, name)))
        for name in (
            "site_workbook",
            "storage_workbook",
            "dispatch_input",
            "residual_predictions",
            "april_pv_predictions",
            "may_pv_predictions",
            "may_pv_selection",
            "gross_load_predictions",
            "residual_data",
            "training_config",
        )
    }
    trained_candidates: list[dict[str, Any]] = []
    for run_dir_value in args.trained_run_dir:
        run_dir = Path(run_dir_value)
        manifest_path = run_dir / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trained_candidates.append(
            {
                "path": str(run_dir.resolve()),
                "candidate_name": manifest.get("candidate_name"),
                "fine_tune_mode": manifest.get("fine_tune_mode"),
                "input_sha256": manifest.get("input_sha256"),
                "config_sha256": manifest.get("config_sha256"),
                "base_model_revision": manifest.get("base_model_revision"),
                "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                "hyperparameters": manifest.get("hyperparameters"),
                "april_predictions": _file_identity(
                    run_dir / "april_predictions.csv"
                ),
            }
        )
    return {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "inputs": files,
        "trained_candidates": trained_candidates,
        "model_path": (
            str(Path(args.model_path).expanduser().resolve())
            if args.model_path is not None
            else None
        ),
        "mip_relative_gap": float(args.mip_relative_gap),
    }


def prepare_revenue_output_dir(
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Create a new run or reopen only an input-compatible existing run."""
    identity = build_revenue_run_identity(args)
    manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite revenue output: {output_dir}")
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Cannot resume revenue output without run_manifest.json: {output_dir}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_identity") != identity:
            raise RuntimeError(
                f"Cannot resume revenue output with incompatible inputs: {output_dir}"
            )
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["last_resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "running"
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "resume_count": 0,
            "run_identity": identity,
        }
    _write_json(manifest_path, manifest)
    return manifest


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


def _price_profile(dispatch_path: Path) -> dict[int, float]:
    raw = pd.read_csv(dispatch_path, low_memory=False)
    timestamp_column = "timestamp" if "timestamp" in raw else str(raw.columns[0])
    timestamps = pd.to_datetime(raw[timestamp_column], errors="raise")
    table = pd.DataFrame(
        {
            "minute": timestamps.dt.hour * 60 + timestamps.dt.minute,
            "price": pd.to_numeric(raw["price"], errors="raise"),
        }
    )
    grouped = table.groupby("minute")["price"]
    if grouped.nunique().gt(1).any():
        raise ValueError("Known tariff differs across days for one clock interval.")
    return {int(key): float(value) for key, value in grouped.first().items()}


def build_realized_inputs(
    site_workbook: Path,
    storage_workbook: Path,
    dispatch_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build March-May residual truth and April/May realized controller inputs."""
    signals = load_provisional_load_signals(site_workbook, storage_workbook)
    gross_five, gross_audit = reconstruct_provisional_load(
        signals,
        "2026-03-31T00:00:00+08:00",
        "2026-06-01T00:00:00+08:00",
    )
    residual_five, residual_audit = reconstruct_signed_residual(
        signals,
        "2026-03-01T00:00:00+08:00",
        "2026-06-01T00:00:00+08:00",
    )
    residual_15 = aggregate_signed_residual_15min(residual_five)
    profile = _price_profile(dispatch_path)

    raw_realized = gross_five[
        ["timestamp", "pv_kw", "provisional_load_kw"]
    ].rename(columns={"pv_kw": "pv", "provisional_load_kw": "load"})
    raw_realized["timestamp"] = _local_naive(raw_realized["timestamp"]).to_numpy()
    minute = raw_realized["timestamp"].dt.hour * 60 + raw_realized["timestamp"].dt.minute
    raw_realized["price"] = minute.map(profile)
    april_realized = raw_realized.loc[
        (raw_realized["timestamp"] >= APRIL_START - pd.Timedelta(minutes=5))
        & (raw_realized["timestamp"] < MAY_START)
    ].copy()
    may_dispatch = load_reference_dispatch(dispatch_path)[
        ["timestamp", "pv", "load", "price"]
    ]
    april30_history = raw_realized.loc[
        (raw_realized["timestamp"] >= MAY_START - pd.Timedelta(days=1))
        & (raw_realized["timestamp"] < MAY_START)
    ]
    may_realized = pd.concat([april30_history, may_dispatch], ignore_index=True).sort_values(
        "timestamp"
    )
    for label, frame in (
        ("April realized inputs", april_realized),
        ("May realized inputs plus April 30 history", may_realized),
    ):
        if frame[["timestamp", "pv", "load", "price"]].isna().any().any():
            raise ValueError(f"{label} contain missing values.")
    audit = {
        "target_label": TARGET_LABEL,
        "gross_load_is_provisional": True,
        "verified_gross_factory_load": False,
        "gross_reconstruction": gross_audit,
        "residual_reconstruction": residual_audit,
        "april_realized_rows_including_prior_interval": len(april_realized),
        "may_realized_rows_including_april30": len(may_realized),
    }
    return april_realized, may_realized, residual_15, audit


def expand_pv_prediction_rows(
    predictions: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    table = predictions.copy()
    if "target" in table:
        table = table.loc[table["target"].eq("pv_kw")].copy()
    table["issue_time"] = _local_naive(table["issue_time"]).to_numpy()
    table["target_time"] = _local_naive(table["target_time"]).to_numpy()
    table = table.loc[
        (table["issue_time"] >= start) & (table["issue_time"] < end_exclusive)
    ].copy()
    if table.duplicated(["issue_time", "target_time"]).any():
        raise ValueError("Frozen PV predictions contain duplicate issue/target rows.")
    parts: list[pd.DataFrame] = []
    for offset in (0, 5, 10):
        parts.append(
            table.assign(
                timestamp=table["target_time"] + pd.Timedelta(minutes=offset)
            )
        )
    expanded = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    expanded = expanded.rename(columns={"p50": "forecast_pv_kw"})[
        ["timestamp", "forecast_pv_kw", "issue_time"]
    ].rename(columns={"issue_time": "pv_forecast_issue_time"})
    expected = pd.date_range(start, end_exclusive, freq="5min", inclusive="left")
    if expanded["timestamp"].tolist() != expected.tolist():
        raise ValueError("Expanded frozen PV forecast does not cover the evaluation grid.")
    if not expanded["forecast_pv_kw"].between(0.0, 1700.0).all():
        raise ValueError("Frozen postprocessed PV forecast is outside [0, 1700] kW.")
    return expanded.reset_index(drop=True)


def previous_day_gross_book(
    realized_with_history: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> ForecastBook:
    forecast = previous_day_forecast(
        realized_with_history,
        "load",
        "forecast_load_kw",
        start,
        end_exclusive,
    )
    rows = pd.DataFrame(
        {
            "candidate": GROSS_BASELINE,
            "issue_time": forecast["timestamp"].dt.normalize(),
            "target_time": forecast["timestamp"],
            "p50": forecast["forecast_load_kw"],
            "context_end": forecast["forecast_load_kw_source_timestamp"],
        }
    )
    return make_forecast_book(rows, GROSS_BASELINE, "gross_load")


def load_gross_chronos_book(
    path: Path,
    split: str,
) -> ForecastBook:
    table = pd.read_csv(path, low_memory=False)
    table = table.loc[
        table["candidate"].eq(GROSS_CHRONOS) & table["split"].eq(split)
    ].copy()
    return make_forecast_book(table, GROSS_CHRONOS, "gross_load")


def load_residual_books(
    path: Path,
    split: str,
) -> dict[str, ForecastBook]:
    table = pd.read_csv(path, low_memory=False)
    table = table.loc[table["split"].eq(split)].copy()
    books: dict[str, ForecastBook] = {}
    for candidate, group in table.groupby("candidate", sort=True):
        books[str(candidate)] = make_forecast_book(
            group, str(candidate), "signed_residual"
        )
    if not books:
        raise ValueError(f"No residual forecasts found for split {split} in {path}.")
    return books


def _four_week_median_for_rows(
    rows: pd.DataFrame,
    residual_target_15min: pd.DataFrame,
) -> np.ndarray:
    target = residual_target_15min.copy()
    target["timestamp"] = _local_naive(target["timestamp"]).to_numpy()
    values = target.set_index("timestamp")[TARGET_COLUMN]
    medians: list[float] = []
    for row in rows.itertuples(index=False):
        issue = pd.Timestamp(row.issue_time)
        target_time = pd.Timestamp(row.target_time)
        source_times = [target_time - pd.Timedelta(days=value) for value in (7, 14, 21, 28)]
        if any(value >= issue for value in source_times):
            raise AssertionError("Four-week blend attempted to use a non-historical target.")
        history = pd.to_numeric(values.reindex(source_times), errors="coerce").dropna()
        if history.empty:
            raise ValueError(f"Four-week blend has no history for {target_time}.")
        medians.append(float(np.median(history.to_numpy(dtype=float))))
    return np.asarray(medians, dtype=float)


def build_blend_book(
    best_hourly_book: ForecastBook,
    residual_target_15min: pd.DataFrame,
) -> ForecastBook:
    if "hourly" not in best_hourly_book.candidate:
        raise ValueError("Blend requires an hourly residual Chronos candidate.")
    rows = best_hourly_book.predictions.copy()
    seasonal = _four_week_median_for_rows(rows, residual_target_15min)
    rows["p50"] = 0.5 * rows["p50"].to_numpy(dtype=float) + 0.5 * seasonal
    for quantile in ("p10", "p90"):
        if quantile in rows:
            rows[quantile] = 0.5 * rows[quantile].to_numpy(dtype=float) + 0.5 * seasonal
    rows["candidate"] = BLEND_CANDIDATE
    rows["blend_components"] = f"{best_hourly_book.candidate}|four_week_median"
    return make_forecast_book(rows, BLEND_CANDIDATE, "signed_residual")


def operational_residual_predictions(
    book: ForecastBook,
    pv_forecast: pd.DataFrame,
    residual_target_15min: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Return one forecast per common quarter hour using the newest causal issue."""
    targets = pd.date_range(start, end_exclusive, freq="15min", inclusive="left")
    pv = pv_forecast.loc[pv_forecast["timestamp"].dt.minute.mod(15).eq(0)].set_index(
        "timestamp"
    )["forecast_pv_kw"]
    truth_table = residual_target_15min.copy()
    truth_table["timestamp"] = _local_naive(truth_table["timestamp"]).to_numpy()
    truth = truth_table.set_index("timestamp")[TARGET_COLUMN]
    rows: list[dict[str, Any]] = []
    for target_time in targets:
        eligible = book.predictions.loc[
            (book.predictions["issue_time"] <= target_time)
            & (book.predictions["target_time"] == target_time)
        ]
        if book.target_frequency_minutes == 5:
            substeps = pd.date_range(target_time, periods=3, freq="5min")
            eligible_issue = book.predictions.loc[
                (book.predictions["issue_time"] <= target_time)
                & (book.predictions["target_time"].isin(substeps))
            ]
            if eligible_issue.empty:
                raise ValueError(f"{book.candidate} has no operational forecast at {target_time}.")
            issue = pd.Timestamp(eligible_issue["issue_time"].max())
            selected = eligible_issue.loc[eligible_issue["issue_time"].eq(issue)]
            if len(selected) != 3:
                raise ValueError(f"{book.candidate} lacks three five-minute substeps at {target_time}.")
            prediction = float(selected["p50"].mean())
        else:
            if eligible.empty:
                raise ValueError(f"{book.candidate} has no operational forecast at {target_time}.")
            issue = pd.Timestamp(eligible["issue_time"].max())
            selected = eligible.loc[eligible["issue_time"].eq(issue)]
            if len(selected) != 1:
                raise ValueError(f"{book.candidate} has ambiguous forecast at {target_time}.")
            prediction = float(selected["p50"].iloc[0])
        signed_prediction = (
            prediction
            if book.kind == "signed_residual"
            else prediction - float(pv.loc[target_time])
        )
        rows.append(
            {
                "split": "april_2026_selection",
                "candidate": book.candidate,
                "issue_time": issue,
                "target_time": target_time,
                "p50": signed_prediction,
                "y_true_kw": float(truth.loc[target_time]) if pd.notna(truth.loc[target_time]) else np.nan,
                "is_missing_target": bool(pd.isna(truth.loc[target_time])),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(targets) or result["target_time"].tolist() != targets.tolist():
        raise AssertionError("Operational residual comparison does not use common timestamps.")
    return result


def evaluate_operational_books(
    books: dict[str, ForecastBook],
    pv_forecast: pd.DataFrame,
    residual_target_15min: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    for candidate, book in books.items():
        rows = operational_residual_predictions(
            book, pv_forecast, residual_target_15min, start, end_exclusive
        )
        prediction_parts.append(rows)
        metrics.append(
            {
                "split": "april_2026_selection",
                "candidate": candidate,
                "metric_scope": "common_operational_timestamps",
                **residual_metric_values(rows),
            }
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    counts = predictions.groupby("candidate")["target_time"].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) != 30 * 96:
        raise AssertionError(f"Forecast candidates do not share April target coverage: {counts}")
    return pd.DataFrame(metrics), predictions


def _candidate_paths(output_dir: Path, stage: str, candidate: str) -> tuple[Path, Path, Path]:
    root = output_dir / "controller_runs" / stage
    root.mkdir(parents=True, exist_ok=True)
    safe = candidate.replace("/", "_")
    return (
        root / f"{safe}_replay.parquet",
        root / f"{safe}_daily.json",
        root / f"{safe}_replans.parquet",
    )


def run_book_controller(
    stage: str,
    book: ForecastBook,
    realized: pd.DataFrame,
    pv_forecast: pd.DataFrame,
    residual_target_15min: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    output_dir: Path,
    *,
    resume: bool,
    show_progress: bool,
    mip_relative_gap: float,
) -> tuple[Any, list[dict[str, Any]]]:
    replay_path, daily_path, replans_path = _candidate_paths(
        output_dir, stage, book.candidate
    )
    if resume and replay_path.is_file() and daily_path.is_file() and replans_path.is_file():
        from src.optimization.foshan_forecast_backtest import StrategyResult

        return (
            StrategyResult(
                pd.read_parquet(replay_path),
                json.loads(daily_path.read_text(encoding="utf-8")),
            ),
            pd.read_parquet(replans_path).to_dict(orient="records"),
        )
    result, replans = run_rolling_v5_evaluation(
        f"controller_v5_{book.candidate}",
        realized,
        pv_forecast,
        book,
        residual_target_15min,
        start,
        end_exclusive,
        output_dir / "solver_logs" / stage / book.candidate,
        mip_relative_gap=mip_relative_gap,
        show_progress=show_progress,
    )
    result.replay.to_parquet(replay_path, index=False)
    _write_json(daily_path, result.daily_runs)
    pd.DataFrame(replans).to_parquet(replans_path, index=False)
    return result, replans


def summarize_controller(
    candidate: str,
    result: Any,
    runtime_seconds: float,
) -> dict[str, Any]:
    parameters = DispatchParameters()
    solver_replans = sum(int(row["replan_count"]) for row in result.daily_runs)
    solver_failures = sum(int(row["solver_failure_count"]) for row in result.daily_runs)
    solver_runtime = sum(float(row["solver_runtime_seconds"]) for row in result.daily_runs)
    row, _, _ = summarize_result(
        f"controller_v5_{candidate}",
        result,
        parameters,
        runtime_seconds=solver_runtime,
        solver_replans=solver_replans,
        solver_failures=solver_failures,
        terminal_policy="band_895_905",
    )
    row["candidate"] = candidate
    row["controller_wall_clock_seconds"] = runtime_seconds
    row["physical_requirements_satisfied"] = bool(
        895.0 - 1e-7 <= float(row["final_soc_kwh"]) <= 905.0 + 1e-7
        and float(row["maximum_constraint_violation"]) <= 1e-6
        and int(row["solver_failures"]) == 0
        and float(row["revenue_recalculation_abs_error_yuan"]) <= 0.01
    )
    return row


def select_by_april_revenue(
    revenue: pd.DataFrame,
    metrics: pd.DataFrame,
    candidates: Iterable[str],
) -> dict[str, Any]:
    names = list(candidates)
    selected = revenue.loc[
        revenue["candidate"].isin(names) & revenue["physical_requirements_satisfied"]
    ].merge(
        metrics[["candidate", "wape", "absolute_bias"]],
        on="candidate",
        how="inner",
        validate="one_to_one",
    )
    if selected.empty:
        raise RuntimeError(f"No physically valid April controller candidate among {names}.")
    winner = selected.sort_values(
        ["raw_revenue_yuan", "wape", "absolute_bias", "candidate"],
        ascending=[False, True, True, True],
        kind="stable",
    ).iloc[0]
    return {
        "candidate": str(winner["candidate"]),
        "april_revenue_yuan": float(winner["raw_revenue_yuan"]),
        "april_residual_wape": float(winner["wape"]),
        "april_absolute_bias_kw": float(winner["absolute_bias"]),
        "selection_rule": (
            "April equal-SOC controller revenue descending; residual WAPE then "
            "absolute bias ascending"
        ),
        "may_targets_or_revenue_used": False,
    }


def _trained_books(
    run_dirs: Iterable[Path],
    split: str,
) -> tuple[dict[str, ForecastBook], dict[str, str], dict[str, Path]]:
    books: dict[str, ForecastBook] = {}
    modes: dict[str, str] = {}
    candidate_dirs: dict[str, Path] = {}
    filename = "april_predictions.csv" if "april" in split else "may_predictions.csv"
    for run_dir in run_dirs:
        manifest_path = run_dir / "training_manifest.json"
        predictions_path = run_dir / filename
        if not manifest_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"Trained residual run is incomplete: {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "trained_checkpoint_reloaded":
            raise RuntimeError(f"Trained run did not prove checkpoint reload: {run_dir}")
        if manifest.get("base_model_revision") != MODEL_REVISION:
            raise RuntimeError(f"Trained run uses the wrong base revision: {run_dir}")
        mode = str(manifest["fine_tune_mode"])
        if mode not in {"lora", "full"}:
            raise RuntimeError(f"Unsupported trained mode in {run_dir}: {mode}")
        table = pd.read_csv(predictions_path, low_memory=False)
        candidate = str(manifest["candidate_name"])
        checkpoint_values = {
            str(Path(value).expanduser().resolve())
            for value in table["checkpoint_path"].dropna().unique()
        }
        expected_checkpoint = str(Path(manifest["checkpoint_path"]).expanduser().resolve())
        if checkpoint_values != {expected_checkpoint}:
            raise RuntimeError(
                f"Trained predictions do not identify the verified checkpoint: "
                f"{checkpoint_values} != {expected_checkpoint}"
            )
        books[candidate] = make_forecast_book(table, candidate, "signed_residual")
        modes[candidate] = mode
        candidate_dirs[candidate] = run_dir
    return books, modes, candidate_dirs


def _record_training_april_selection(
    candidate_dirs: dict[str, Path],
    modes: dict[str, str],
    selections: dict[str, Any],
    april_revenue: pd.DataFrame,
    april_metrics: pd.DataFrame,
) -> None:
    revenue = april_revenue.set_index("candidate")
    metrics = april_metrics.set_index("candidate")
    for candidate, candidate_dir in candidate_dirs.items():
        manifest_path = candidate_dir / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mode = modes[candidate]
        manifest["april_selection_results"] = {
            "controller_revenue_yuan": float(revenue.loc[candidate, "raw_revenue_yuan"]),
            "final_soc_kwh": float(revenue.loc[candidate, "final_soc_kwh"]),
            "physical_requirements_satisfied": bool(
                revenue.loc[candidate, "physical_requirements_satisfied"]
            ),
            "residual_wape": float(metrics.loc[candidate, "wape"]),
            "residual_absolute_bias_kw": float(metrics.loc[candidate, "absolute_bias"]),
            "selected_within_mode": bool(
                mode in selections and selections[mode]["candidate"] == candidate
            ),
            "selection_period": "April 2026 only",
            "may_used_for_selection": False,
        }
        _write_json(manifest_path, manifest)


def run_pipeline(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results/revenue_ablation/foshan_residual_controller_v5") / datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
    run_manifest = prepare_revenue_output_dir(output_dir, args)
    frozen_hashes = verify_frozen_controller_sources()
    april_realized, may_realized, residual_target, data_audit = build_realized_inputs(
        args.site_workbook, args.storage_workbook, args.dispatch_input
    )

    april_pv_rows = pd.read_csv(args.april_pv_predictions, low_memory=False)
    april_pv = expand_pv_prediction_rows(april_pv_rows, APRIL_START, MAY_START)
    may_pv, may_pv_metadata = load_selected_chronos_p50(
        args.may_pv_predictions,
        args.may_pv_selection,
        MAY_START,
        JUNE_START,
    )
    april_books = load_residual_books(
        args.residual_predictions, "april_2026_selection"
    )
    may_books = load_residual_books(
        args.residual_predictions, "may_2026_economic_test"
    )
    april_books[GROSS_BASELINE] = previous_day_gross_book(
        april_realized, APRIL_START, MAY_START
    )
    may_books[GROSS_BASELINE] = previous_day_gross_book(
        may_realized, MAY_START, JUNE_START
    )
    april_books[GROSS_CHRONOS] = load_gross_chronos_book(
        args.gross_load_predictions, "april_2026_selection"
    )
    may_books[GROSS_CHRONOS] = load_gross_chronos_book(
        args.gross_load_predictions, "may_2026_controller_evaluation"
    )

    preliminary_metrics, _ = evaluate_operational_books(
        april_books, april_pv, residual_target, APRIL_START, MAY_START
    )
    rolling_names = [
        name for name in april_books if name.startswith("chronos2_residual_hourly_")
    ]
    if not rolling_names:
        raise ValueError("No hourly residual zero-shot candidates are available.")
    best_rolling = (
        preliminary_metrics.loc[preliminary_metrics["candidate"].isin(rolling_names)]
        .sort_values(["wape", "absolute_bias", "candidate"], kind="stable")
        .iloc[0]["candidate"]
    )
    april_books[BLEND_CANDIDATE] = build_blend_book(
        april_books[str(best_rolling)], residual_target
    )
    may_books[BLEND_CANDIDATE] = build_blend_book(
        may_books[str(best_rolling)], residual_target
    )
    trained_april, trained_modes, trained_candidate_dirs = _trained_books(
        args.trained_run_dir, "april_2026_selection"
    )
    april_books.update(trained_april)

    april_metrics, operational_predictions = evaluate_operational_books(
        april_books, april_pv, residual_target, APRIL_START, MAY_START
    )
    april_metrics.to_csv(
        output_dir / "april_forecast_metrics.csv", index=False, float_format="%.15g"
    )
    operational_predictions.to_csv(
        output_dir / "april_operational_predictions.csv",
        index=False,
        float_format="%.15g",
    )

    april_rows: list[dict[str, Any]] = []
    for candidate, book in april_books.items():
        started = time.perf_counter()
        result, _ = run_book_controller(
            "april",
            book,
            april_realized,
            april_pv,
            residual_target,
            APRIL_START,
            MAY_START,
            output_dir,
            resume=args.resume,
            show_progress=not args.quiet,
            mip_relative_gap=args.mip_relative_gap,
        )
        april_rows.append(
            summarize_controller(candidate, result, time.perf_counter() - started)
        )
    april_revenue = pd.DataFrame(april_rows)
    april_revenue.to_csv(
        output_dir / "april_revenue_selection.csv", index=False, float_format="%.15g"
    )
    zero_shot_names = [
        name
        for name in april_books
        if name not in {GROSS_BASELINE, GROSS_CHRONOS} and name not in trained_modes
    ]
    selections: dict[str, Any] = {
        "zero_shot": select_by_april_revenue(
            april_revenue, april_metrics, zero_shot_names
        )
    }
    for mode in ("lora", "full"):
        names = [name for name, value in trained_modes.items() if value == mode]
        if names:
            selections[mode] = select_by_april_revenue(
                april_revenue, april_metrics, names
            )

    _record_training_april_selection(
        trained_candidate_dirs,
        trained_modes,
        selections,
        april_revenue,
        april_metrics,
    )
    selected_trained_dirs: list[Path] = []
    for mode in ("lora", "full"):
        if mode not in selections:
            continue
        candidate = str(selections[mode]["candidate"])
        candidate_dir = trained_candidate_dirs[candidate]
        selected_trained_dirs.append(candidate_dir)
        if not (candidate_dir / "may_predictions.csv").is_file():
            from src.training.foshan_residual_finetune import run_selected_may_inference

            inference_args = argparse.Namespace(
                candidate_dir=candidate_dir,
                config=args.training_config,
                input=args.residual_data,
                model_path=args.model_path,
                hf_home=args.hf_home,
                allow_download=args.allow_download,
                output_dir=candidate_dir,
                fine_tune_mode=mode,
                stage="may-predict",
            )
            run_selected_may_inference(inference_args)
    if selected_trained_dirs:
        trained_may, _, _ = _trained_books(
            selected_trained_dirs, "may_2026_economic_test"
        )
        may_books.update(trained_may)

    may_candidates = [GROSS_CHRONOS, selections["zero_shot"]["candidate"]]
    may_candidates.extend(
        selections[mode]["candidate"] for mode in ("lora", "full") if mode in selections
    )
    may_rows: list[dict[str, Any]] = []
    for candidate in may_candidates:
        started = time.perf_counter()
        result, _ = run_book_controller(
            "may",
            may_books[candidate],
            may_realized,
            may_pv,
            residual_target,
            MAY_START,
            JUNE_START,
            output_dir,
            resume=args.resume,
            show_progress=not args.quiet,
            mip_relative_gap=args.mip_relative_gap,
        )
        row = summarize_controller(candidate, result, time.perf_counter() - started)
        row.update(
            {
                "difference_from_current_ctx1344_yuan": row["raw_revenue_yuan"]
                - CURRENT_CTX1344_REVENUE_YUAN,
                "difference_from_historical_yuan": row["raw_revenue_yuan"]
                - HISTORICAL_REVENUE_YUAN,
                "difference_from_optimized_baseline_yuan": row["raw_revenue_yuan"]
                - OPTIMIZED_BASELINE_YUAN,
                "oracle_attainment_percent": 100.0
                * row["raw_revenue_yuan"]
                / EQUAL_CONDITION_ORACLE_YUAN,
            }
        )
        may_rows.append(row)
    may_revenue = pd.DataFrame(may_rows)
    may_revenue.to_csv(
        output_dir / "may_revenue_comparison.csv", index=False, float_format="%.15g"
    )
    technical_columns = [
        "candidate",
        "initial_soc_kwh",
        "final_soc_kwh",
        "terminal_slack_kwh",
        "planned_charge_kwh",
        "planned_discharge_kwh",
        "executed_charge_kwh",
        "executed_discharge_kwh",
        "anti_export_clipped_intervals",
        "anti_export_clipped_kwh",
        "clipped_discharge_fraction",
        "maximum_constraint_violation",
        "anti_export_violation_kw",
        "solver_failures",
        "runtime_seconds",
        "controller_wall_clock_seconds",
        "physical_requirements_satisfied",
    ]
    may_revenue[technical_columns].to_csv(
        output_dir / "technical_metrics.csv", index=False, float_format="%.15g"
    )

    summary = {
        "status": "counterfactual provisional revenue benchmark",
        "target_label": TARGET_LABEL,
        "verified_gross_factory_load": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "selection_period": "April 2026 only",
        "final_economic_test": "May 2026 only",
        "may_used_for_selection": False,
        "selections": selections,
        "may_results": may_revenue.to_dict(orient="records"),
        "references_yuan": {
            "current_ctx1344": CURRENT_CTX1344_REVENUE_YUAN,
            "historical": HISTORICAL_REVENUE_YUAN,
            "optimized_baseline": OPTIMIZED_BASELINE_YUAN,
            "equal_condition_oracle": EQUAL_CONDITION_ORACLE_YUAN,
        },
        "frozen_controller_source_sha256": frozen_hashes,
        "data_audit": data_audit,
        "may_pv_forecast": may_pv_metadata,
        "provenance": {"git_commit": git_commit(), "git_dirty": git_is_dirty()},
    }
    _write_json(output_dir / "summary.json", summary)
    run_manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selections": selections,
        }
    )
    _write_json(output_dir / "run_manifest.json", run_manifest)
    report = [
        "# Foshan signed residual forecast revenue benchmark",
        "",
        "> All revenue is counterfactual and provisional. The reconstructed load is not verified gross factory load.",
        "",
        "April alone selected each model class by equal-SOC controller revenue. May was not used for selection.",
        "",
        f"Selected zero-shot candidate: {selections['zero_shot']['candidate']}.",
        "",
        "Controller_v5 and the direct HiGHS MILP remained source-hash identical.",
    ]
    (output_dir / "residual_forecast_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-workbook", required=True, type=Path)
    parser.add_argument("--storage-workbook", required=True, type=Path)
    parser.add_argument("--dispatch-input", required=True, type=Path)
    parser.add_argument("--residual-predictions", required=True, type=Path)
    parser.add_argument("--april-pv-predictions", required=True, type=Path)
    parser.add_argument(
        "--may-pv-predictions",
        default=Path("results/zero_shot/foshan_chronos2/predictions_long.csv"),
        type=Path,
    )
    parser.add_argument(
        "--may-pv-selection",
        default=Path(
            "results/zero_shot/foshan_chronos2/selected_configuration.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--gross-load-predictions",
        default=Path(
            "results/load_forecast_ablation/foshan_april_select_may_controller_v5/"
            "load_predictions_long.csv"
        ),
        type=Path,
    )
    parser.add_argument("--trained-run-dir", action="append", default=[], type=Path)
    parser.add_argument(
        "--residual-data",
        default=Path(
            "results/residual_forecast/foshan_chronos2/data/"
            "signed_residual_15min.parquet"
        ),
        type=Path,
    )
    parser.add_argument(
        "--training-config",
        default=Path("configs/foshan_chronos2_residual.json"),
        type=Path,
    )
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--hf-home", default=None, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--mip-relative-gap", default=1e-7, type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    output = run_pipeline(build_parser().parse_args())
    print(f"Saved Foshan residual revenue benchmark to {output.resolve()}")


if __name__ == "__main__":
    main()
