"""Leakage-safe May validation demo from Chronos-2 forecasts to HiGHS dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.optimization.foshan_battery_milp import (
    LOAD_DATA_NOTICE,
    DispatchParameters,
    DispatchSolution,
    recalculate_objective,
    solve_dispatch,
)
from src.utils.runtime import git_commit, git_is_dirty


TIMEZONE = "Asia/Shanghai"
START = pd.Timestamp("2026-05-02 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2026-06-01 00:00:00")
POSTPROCESSING = "physical_clip_0_1700"
OUTPUT_FILENAMES = (
    "replay_timeseries.csv",
    "strategy_summary.csv",
    "revenue_comparison.json",
    "constraint_audit.json",
    "report.md",
)
FORECAST_NOTICE = (
    "Forecast-driven revenue is a counterfactual, provisional reconstruction; "
    "it is not observed actual revenue."
)
VALIDATION_NOTICE = (
    "May 2026 is a validation-period demo because May was already used to select "
    "the Chronos-2 configuration."
)


Solver = Callable[..., DispatchSolution]


@dataclass
class StrategyResult:
    replay: pd.DataFrame
    daily_runs: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _timestamp_column(headers: list[str]) -> str:
    if "timestamp" in headers:
        return "timestamp"
    if headers and (not str(headers[0]).strip() or str(headers[0]).startswith("Unnamed:")):
        return headers[0]
    raise ValueError("Input must contain timestamp or an unnamed first timestamp column.")


def _validate_five_minute_grid(table: pd.DataFrame, label: str) -> None:
    if table.empty:
        raise ValueError(f"{label} is empty.")
    if not table["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{label} timestamps must be sorted.")
    if not table["timestamp"].is_unique:
        raise ValueError(f"{label} timestamps must be unique.")
    if len(table) > 1:
        differences = table["timestamp"].diff().iloc[1:]
        if not differences.eq(pd.Timedelta(minutes=5)).all():
            raise ValueError(f"{label} must use an exact five-minute grid.")


def load_reference_dispatch(path: Path) -> pd.DataFrame:
    """Load realized inputs and comparison-only historical/DP schedules."""
    headers = list(pd.read_csv(path, nrows=0).columns)
    timestamp_source = _timestamp_column(headers)
    columns = [
        timestamp_source,
        "pv",
        "load",
        "price",
        "p_actual",
        "p_opt",
        "soc_actual_est",
        "soc_opt",
    ]
    missing = [column for column in columns if column not in headers]
    if missing:
        raise ValueError(f"Reference dispatch is missing columns: {missing}")
    table = pd.read_csv(path, usecols=columns).rename(
        columns={timestamp_source: "timestamp"}
    )
    table["timestamp"] = pd.to_datetime(table["timestamp"], errors="raise")
    numeric = [column for column in columns if column != timestamp_source]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="raise")
    table = table.sort_values("timestamp").reset_index(drop=True)
    if table[["timestamp", *numeric]].isna().any().any():
        raise ValueError("Reference dispatch contains missing required values.")
    if (table[["pv", "load", "price"]] < 0).any().any():
        raise ValueError("Realized PV, provisional load, and tariff must be nonnegative.")
    _validate_five_minute_grid(table, "reference dispatch")
    return table


def _prediction_local_time(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise", utc=True)
    return parsed.dt.tz_convert(TIMEZONE).dt.tz_localize(None)


def load_selected_chronos_p50(
    predictions_path: Path,
    selection_path: Path,
    start: pd.Timestamp = START,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load selected postprocessed May P50 values and expand 15 to 5 minutes."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selected_on") != "may_2026_selection":
        raise ValueError("Chronos selection must be the existing May selection.")
    selected = selection.get("targets", {}).get("pv_kw")
    if not selected:
        raise ValueError("Selection metadata does not define a pv_kw model.")
    model_name = str(selected["model_name"])
    context_length = int(selected["context_length"])

    usecols = [
        "split",
        "issue_time",
        "target_time",
        "horizon_step",
        "target",
        "model_name",
        "context_length",
        "postprocessing",
        "p50",
    ]
    predictions = pd.read_csv(predictions_path, usecols=usecols, low_memory=False)
    predictions = predictions.loc[
        (predictions["split"] == "may_2026_selection")
        & (predictions["target"] == "pv_kw")
        & (predictions["model_name"] == model_name)
        & (predictions["context_length"] == context_length)
        & (predictions["postprocessing"] == POSTPROCESSING)
    ].copy()
    predictions["issue_time"] = _prediction_local_time(predictions["issue_time"])
    predictions["target_time"] = _prediction_local_time(predictions["target_time"])
    predictions = predictions.loc[
        (predictions["issue_time"] >= start)
        & (predictions["issue_time"] < end_exclusive)
    ].sort_values(["issue_time", "horizon_step"])

    expected_issues = pd.date_range(start, end_exclusive, freq="1D", inclusive="left")
    if predictions["issue_time"].drop_duplicates().tolist() != expected_issues.tolist():
        raise ValueError("Selected Chronos predictions do not cover every May 2-31 origin.")
    if predictions.duplicated(["issue_time", "target_time", "horizon_step"]).any():
        raise ValueError("Selected Chronos predictions contain duplicate forecast rows.")
    if predictions["p50"].isna().any() or not np.isfinite(predictions["p50"]).all():
        raise ValueError("Selected Chronos P50 contains missing or non-finite values.")
    if not predictions["p50"].between(0.0, 1700.0).all():
        raise ValueError("Postprocessed Chronos P50 must remain in [0, 1700] kW.")

    for issue_time, group in predictions.groupby("issue_time", sort=True):
        if group["horizon_step"].tolist() != list(range(1, 97)):
            raise ValueError(f"Chronos origin {issue_time} does not contain horizons 1-96.")
        expected_targets = pd.date_range(issue_time, periods=96, freq="15min")
        if group["target_time"].tolist() != expected_targets.tolist():
            raise ValueError(f"Chronos origin {issue_time} has misaligned target times.")

    expanded_parts = []
    for offset_minutes in (0, 5, 10):
        expanded_parts.append(
            predictions.assign(
                timestamp=predictions["target_time"]
                + pd.Timedelta(minutes=offset_minutes),
                five_minute_substep=offset_minutes // 5,
            )
        )
    expanded = pd.concat(expanded_parts, ignore_index=True).sort_values("timestamp")
    expanded = expanded.rename(
        columns={
            "p50": "forecast_pv_kw",
            "issue_time": "pv_forecast_issue_time",
        }
    )[
        [
            "timestamp",
            "forecast_pv_kw",
            "pv_forecast_issue_time",
            "horizon_step",
            "five_minute_substep",
        ]
    ].reset_index(drop=True)
    _validate_five_minute_grid(expanded, "expanded Chronos forecast")
    expected_grid = pd.date_range(start, end_exclusive, freq="5min", inclusive="left")
    if expanded["timestamp"].tolist() != expected_grid.tolist():
        raise ValueError("Expanded Chronos forecast does not match the backtest grid.")

    metadata = {
        "model_name": model_name,
        "context_length": context_length,
        "postprocessing": POSTPROCESSING,
        "quantile": "p50",
        "source_frequency": "15min",
        "control_frequency": "5min",
        "origin_count": len(expected_issues),
        "source_prediction_rows": len(predictions),
        "expanded_rows": len(expanded),
    }
    return expanded, metadata


def previous_day_forecast(
    realized: pd.DataFrame,
    value_column: str,
    output_column: str,
    start: pd.Timestamp = START,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
) -> pd.DataFrame:
    """Forecast each five-minute value from the same slot on the previous day."""
    target_times = pd.date_range(start, end_exclusive, freq="5min", inclusive="left")
    source_times = target_times - pd.Timedelta(days=1)
    lookup = realized.set_index("timestamp")[value_column]
    values = lookup.reindex(source_times)
    if values.isna().any():
        missing = source_times[values.isna()]
        raise ValueError(
            f"Previous-day {value_column} forecast is missing {len(missing)} source rows."
        )
    return pd.DataFrame(
        {
            "timestamp": target_times,
            output_column: values.to_numpy(dtype=float),
            f"{output_column}_source_timestamp": source_times,
        }
    )


def perfect_forecast(
    realized: pd.DataFrame,
    value_column: str,
    output_column: str,
    start: pd.Timestamp = START,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
) -> pd.DataFrame:
    window = realized.loc[
        (realized["timestamp"] >= start)
        & (realized["timestamp"] < end_exclusive),
        ["timestamp", value_column],
    ].copy()
    return window.rename(columns={value_column: output_column}).assign(
        **{f"{output_column}_source_timestamp": window["timestamp"].to_numpy()}
    )


def replay_day(
    schedule: pd.DataFrame,
    realized: pd.DataFrame,
    initial_soc_kwh: float,
    parameters: DispatchParameters,
) -> pd.DataFrame:
    """Replay frozen actions, reducing them only when required for physical safety."""
    merged = schedule[
        ["timestamp", "charge_kw", "discharge_kw", "soc_start_kwh", "soc_kwh"]
    ].merge(
        realized[["timestamp", "pv", "load", "price"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(schedule) or len(merged) != len(realized):
        raise ValueError("Schedule and realized day do not share identical timestamps.")

    rows: list[dict[str, Any]] = []
    soc = float(initial_soc_kwh)
    dt = parameters.interval_hours
    eta_charge = parameters.charge_efficiency
    eta_discharge = parameters.discharge_efficiency
    tolerance = 1e-7

    for row in merged.itertuples(index=False):
        scheduled_charge = max(0.0, float(row.charge_kw))
        scheduled_discharge = max(0.0, float(row.discharge_kw))
        if min(scheduled_charge, scheduled_discharge) > tolerance:
            raise ValueError("MILP schedule contains simultaneous charge and discharge.")

        charge_after_power = min(scheduled_charge, parameters.power_limit_kw)
        charge_soc_limit = max(
            0.0,
            (parameters.capacity_kwh - soc) / (eta_charge * dt),
        )
        applied_charge = min(charge_after_power, charge_soc_limit)

        discharge_after_power = min(scheduled_discharge, parameters.power_limit_kw)
        discharge_soc_limit = max(0.0, soc * eta_discharge / dt)
        discharge_after_soc = min(discharge_after_power, discharge_soc_limit)
        residual_load = max(float(row.load) - float(row.pv), 0.0)
        applied_discharge = min(discharge_after_soc, residual_load)

        soc_end = (
            soc
            + eta_charge * dt * applied_charge
            - dt / eta_discharge * applied_discharge
        )
        if abs(soc_end) < tolerance:
            soc_end = 0.0
        if abs(soc_end - parameters.capacity_kwh) < tolerance:
            soc_end = parameters.capacity_kwh
        if soc_end < -tolerance or soc_end > parameters.capacity_kwh + tolerance:
            raise AssertionError("Safety replay produced an out-of-bounds SOC.")

        net_grid = (
            float(row.load)
            - float(row.pv)
            - applied_discharge
            + applied_charge
        )
        grid_import = max(net_grid, 0.0)
        grid_export = max(-net_grid, 0.0)
        power_clip_kw = (scheduled_charge - charge_after_power) + (
            scheduled_discharge - discharge_after_power
        )
        soc_clip_kw = (charge_after_power - applied_charge) + (
            discharge_after_power - discharge_after_soc
        )
        anti_export_clip_kw = discharge_after_soc - applied_discharge
        total_clip_kw = power_clip_kw + soc_clip_kw + anti_export_clip_kw

        rows.append(
            {
                "timestamp": row.timestamp,
                "realized_pv_kw": float(row.pv),
                "realized_load_kw": float(row.load),
                "price_yuan_per_kwh": float(row.price),
                "scheduled_charge_kw": scheduled_charge,
                "scheduled_discharge_kw": scheduled_discharge,
                "applied_charge_kw": applied_charge,
                "applied_discharge_kw": applied_discharge,
                "scheduled_soc_start_kwh": float(row.soc_start_kwh),
                "scheduled_soc_end_kwh": float(row.soc_kwh),
                "realized_soc_start_kwh": soc,
                "realized_soc_end_kwh": soc_end,
                "grid_import_kw": grid_import,
                "grid_export_kw": grid_export,
                "power_clip_kw": power_clip_kw,
                "soc_clip_kw": soc_clip_kw,
                "anti_export_clip_kw": anti_export_clip_kw,
                "total_clip_kw": total_clip_kw,
                "was_clipped": total_clip_kw > tolerance,
            }
        )
        soc = soc_end
    return pd.DataFrame(rows)


def run_daily_strategy(
    name: str,
    realized: pd.DataFrame,
    pv_forecast: pd.DataFrame,
    load_forecast: pd.DataFrame,
    initial_soc_kwh: float,
    solver_log_dir: Path,
    *,
    solver: Solver = solve_dispatch,
    parameters: DispatchParameters = DispatchParameters(),
    start: pd.Timestamp = START,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
    mip_relative_gap: float = 1e-7,
) -> StrategyResult:
    """Solve each day on forecasts, then replay against actuals after scheduling."""
    forecast = pv_forecast.merge(
        load_forecast,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp").reset_index(drop=True)
    expected = pd.date_range(start, end_exclusive, freq="5min", inclusive="left")
    if forecast["timestamp"].tolist() != expected.tolist():
        raise ValueError(f"{name} forecasts do not match the complete backtest grid.")

    replay_parts = []
    daily_runs: list[dict[str, Any]] = []
    current_soc = float(initial_soc_kwh)
    days = pd.date_range(start, end_exclusive, freq="1D", inclusive="left")

    for day in days:
        day_end = day + pd.Timedelta(days=1)
        day_forecast = forecast.loc[
            (forecast["timestamp"] >= day) & (forecast["timestamp"] < day_end)
        ].copy()
        day_tariff = realized.loc[
            (realized["timestamp"] >= day) & (realized["timestamp"] < day_end),
            ["timestamp", "price"],
        ].copy()
        if len(day_forecast) != 288 or len(day_tariff) != 288:
            raise ValueError(f"{day.date()} must contain exactly 288 five-minute rows.")

        milp_input = day_forecast[
            ["timestamp", "forecast_pv_kw", "forecast_load_kw"]
        ].merge(
            day_tariff,
            on="timestamp",
            how="inner",
            validate="one_to_one",
        ).rename(
            columns={
                "forecast_pv_kw": "pv",
                "forecast_load_kw": "load",
            }
        )
        daily_parameters = replace(
            parameters,
            initial_soc_kwh=current_soc,
            terminal_soc_kwh=current_soc,
        )
        solved = solver(
            milp_input,
            solver_log_dir / f"{name}_{day:%Y%m%d}.log",
            daily_parameters,
            mip_relative_gap=mip_relative_gap,
            log_to_console=False,
        )
        if solved.solver_metadata.get("solver_status") != "Optimal":
            raise RuntimeError(f"{name} {day.date()} did not solve optimally.")

        # Realized PV/load are sliced only after the frozen schedule exists.
        day_realized = realized.loc[
            (realized["timestamp"] >= day) & (realized["timestamp"] < day_end),
            ["timestamp", "pv", "load", "price"],
        ].copy()
        replay = replay_day(
            solved.dispatch,
            day_realized,
            current_soc,
            daily_parameters,
        ).merge(day_forecast, on="timestamp", how="left", validate="one_to_one")
        replay.insert(0, "strategy", name)
        replay["issue_time"] = day
        replay["safety_filter_applied"] = True
        replay["counterfactual_provisional"] = True
        replay["validation_period_demo"] = True
        replay_parts.append(replay)

        realized_end_soc = float(replay["realized_soc_end_kwh"].iloc[-1])
        daily_runs.append(
            {
                "date": day.date().isoformat(),
                "initial_soc_kwh": current_soc,
                "scheduled_terminal_soc_kwh": current_soc,
                "realized_terminal_soc_kwh": realized_end_soc,
                "terminal_difference_kwh": realized_end_soc - current_soc,
                "clipped_intervals": int(replay["was_clipped"].sum()),
                "clipped_energy_kwh": float(
                    replay["total_clip_kw"].sum() * parameters.interval_hours
                ),
                "forecast_objective_yuan": solved.solver_objective_yuan,
                "solver_status": solved.solver_metadata.get("solver_status"),
                "solver_runtime_seconds": solved.solver_metadata.get(
                    "wall_clock_runtime_seconds"
                ),
                "solver_gap": solved.solver_metadata.get("optimality_gap"),
            }
        )
        current_soc = realized_end_soc

    return StrategyResult(pd.concat(replay_parts, ignore_index=True), daily_runs)


def reference_strategy(
    realized: pd.DataFrame,
    name: str,
    power_column: str,
    soc_column: str,
    start: pd.Timestamp = START,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
    parameters: DispatchParameters = DispatchParameters(),
) -> StrategyResult:
    """Build an unmodified historical or original-DP comparison series."""
    window = realized.loc[
        (realized["timestamp"] >= start)
        & (realized["timestamp"] < end_exclusive)
    ].copy()
    power = window[power_column].to_numpy(dtype=float)
    charge = np.maximum(-power, 0.0)
    discharge = np.maximum(power, 0.0)
    soc_start = window[soc_column].to_numpy(dtype=float)
    soc_end = (
        soc_start
        + parameters.charge_efficiency * parameters.interval_hours * charge
        - parameters.interval_hours / parameters.discharge_efficiency * discharge
    )
    net_grid = window["load"].to_numpy(dtype=float) - window["pv"].to_numpy(
        dtype=float
    ) - discharge + charge
    replay = pd.DataFrame(
        {
            "strategy": name,
            "timestamp": window["timestamp"].to_numpy(),
            "realized_pv_kw": window["pv"].to_numpy(dtype=float),
            "realized_load_kw": window["load"].to_numpy(dtype=float),
            "price_yuan_per_kwh": window["price"].to_numpy(dtype=float),
            "forecast_pv_kw": np.nan,
            "forecast_load_kw": np.nan,
            "scheduled_charge_kw": charge,
            "scheduled_discharge_kw": discharge,
            "applied_charge_kw": charge,
            "applied_discharge_kw": discharge,
            "scheduled_soc_start_kwh": soc_start,
            "scheduled_soc_end_kwh": soc_end,
            "realized_soc_start_kwh": soc_start,
            "realized_soc_end_kwh": soc_end,
            "grid_import_kw": np.maximum(net_grid, 0.0),
            "grid_export_kw": np.maximum(-net_grid, 0.0),
            "power_clip_kw": 0.0,
            "soc_clip_kw": 0.0,
            "anti_export_clip_kw": 0.0,
            "total_clip_kw": 0.0,
            "was_clipped": False,
            "issue_time": pd.NaT,
            "safety_filter_applied": False,
            "counterfactual_provisional": name != "historical_operation",
            "validation_period_demo": True,
        }
    )
    return StrategyResult(replay, [])


def validate_common_timestamps(
    results: dict[str, StrategyResult],
    start: pd.Timestamp = START,
    end_exclusive: pd.Timestamp = END_EXCLUSIVE,
) -> list[pd.Timestamp]:
    expected = pd.date_range(start, end_exclusive, freq="5min", inclusive="left").tolist()
    for name, result in results.items():
        timestamps = result.replay["timestamp"].tolist()
        if timestamps != expected or result.replay["timestamp"].duplicated().any():
            raise ValueError(f"Strategy {name} does not use the identical timestamp set.")
    return expected


def _accounting_frame(replay: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pv": replay["realized_pv_kw"],
            "load": replay["realized_load_kw"],
            "price": replay["price_yuan_per_kwh"],
            "charge_kw": replay["applied_charge_kw"],
            "discharge_kw": replay["applied_discharge_kw"],
            "grid_import_kw": replay["grid_import_kw"],
            "grid_export_kw": replay["grid_export_kw"],
        }
    )


def summarize_strategy(
    name: str,
    result: StrategyResult,
    parameters: DispatchParameters = DispatchParameters(),
) -> dict[str, Any]:
    replay = result.replay
    accounting = recalculate_objective(_accounting_frame(replay), parameters)
    initial_soc = float(replay["realized_soc_start_kwh"].iloc[0])
    final_soc = float(replay["realized_soc_end_kwh"].iloc[-1])
    final_target = (
        result.daily_runs[-1]["scheduled_terminal_soc_kwh"]
        if result.daily_runs
        else None
    )
    daily_terminal_differences = [
        abs(float(day["terminal_difference_kwh"])) for day in result.daily_runs
    ]
    return {
        "strategy": name,
        "revenue_status": (
            "reconstructed_provisional_reference"
            if name == "historical_operation"
            else "counterfactual_provisional"
        ),
        "validation_period_demo": True,
        "timestamp_count": len(replay),
        "initial_soc_kwh": initial_soc,
        "final_soc_kwh": final_soc,
        "final_soc_difference_from_initial_kwh": final_soc - initial_soc,
        "final_scheduled_terminal_soc_kwh": final_target,
        "final_terminal_difference_kwh": (
            final_soc - final_target
            if final_target is not None
            else None
        ),
        "days_with_terminal_difference": (
            sum(value > 1e-6 for value in daily_terminal_differences)
            if daily_terminal_differences
            else None
        ),
        "maximum_abs_daily_terminal_difference_kwh": (
            max(daily_terminal_differences) if daily_terminal_differences else None
        ),
        "sum_abs_daily_terminal_difference_kwh": (
            sum(daily_terminal_differences) if daily_terminal_differences else None
        ),
        "clipped_intervals": int(replay["was_clipped"].sum()),
        "clipped_energy_kwh": float(
            replay["total_clip_kw"].sum() * parameters.interval_hours
        ),
        **accounting,
    }


def audit_strategy(
    name: str,
    result: StrategyResult,
    parameters: DispatchParameters = DispatchParameters(),
) -> dict[str, Any]:
    replay = result.replay
    minimum_soc = min(
        float(replay["realized_soc_start_kwh"].min()),
        float(replay["realized_soc_end_kwh"].min()),
    )
    maximum_soc = max(
        float(replay["realized_soc_start_kwh"].max()),
        float(replay["realized_soc_end_kwh"].max()),
    )
    residual_load = np.maximum(
        replay["realized_load_kw"] - replay["realized_pv_kw"], 0.0
    )
    continuity = np.abs(
        replay["realized_soc_start_kwh"].iloc[1:].to_numpy(dtype=float)
        - replay["realized_soc_end_kwh"].iloc[:-1].to_numpy(dtype=float)
    )
    final_target = (
        result.daily_runs[-1]["scheduled_terminal_soc_kwh"]
        if result.daily_runs
        else None
    )
    initial_soc = float(replay["realized_soc_start_kwh"].iloc[0])
    final_soc = float(replay["realized_soc_end_kwh"].iloc[-1])
    daily_terminal_differences = [
        abs(float(day["terminal_difference_kwh"])) for day in result.daily_runs
    ]
    return {
        "strategy": name,
        "safety_filter_applied": bool(replay["safety_filter_applied"].iloc[0]),
        "minimum_soc_kwh": minimum_soc,
        "maximum_soc_kwh": maximum_soc,
        "maximum_charge_kw": float(replay["applied_charge_kw"].max()),
        "maximum_discharge_kw": float(replay["applied_discharge_kw"].max()),
        "soc_lower_violation_kwh": float(
            max(0.0, -minimum_soc)
        ),
        "soc_upper_violation_kwh": float(
            max(0.0, maximum_soc - parameters.capacity_kwh)
        ),
        "charge_power_violation_kw": float(
            max(0.0, replay["applied_charge_kw"].max() - parameters.power_limit_kw)
        ),
        "discharge_power_violation_kw": float(
            max(0.0, replay["applied_discharge_kw"].max() - parameters.power_limit_kw)
        ),
        "simultaneous_charge_discharge_kw": float(
            np.minimum(
                replay["applied_charge_kw"], replay["applied_discharge_kw"]
            ).max()
        ),
        "anti_export_violation_kw": float(
            np.maximum(replay["applied_discharge_kw"] - residual_load, 0.0).max()
        ),
        "pv_export_limit_violation_kw": float(
            np.maximum(
                replay["grid_export_kw"] - replay["realized_pv_kw"], 0.0
            ).max()
        ),
        "soc_continuity_violation_kwh": float(continuity.max()),
        "initial_soc_kwh": initial_soc,
        "final_soc_kwh": final_soc,
        "final_soc_difference_from_initial_kwh": final_soc - initial_soc,
        "final_scheduled_terminal_soc_kwh": final_target,
        "final_terminal_difference_kwh": (
            final_soc - final_target
            if final_target is not None
            else None
        ),
        "days_with_terminal_difference": (
            sum(value > 1e-6 for value in daily_terminal_differences)
            if daily_terminal_differences
            else None
        ),
        "maximum_abs_daily_terminal_difference_kwh": (
            max(daily_terminal_differences) if daily_terminal_differences else None
        ),
        "sum_abs_daily_terminal_difference_kwh": (
            sum(daily_terminal_differences) if daily_terminal_differences else None
        ),
        "clipped_intervals": int(replay["was_clipped"].sum()),
        "power_clipped_kwh": float(
            replay["power_clip_kw"].sum() * parameters.interval_hours
        ),
        "soc_clipped_kwh": float(
            replay["soc_clip_kw"].sum() * parameters.interval_hours
        ),
        "anti_export_clipped_kwh": float(
            replay["anti_export_clip_kw"].sum() * parameters.interval_hours
        ),
        "daily_solver_statuses": sorted(
            {str(day["solver_status"]) for day in result.daily_runs}
        ),
        "maximum_daily_solver_gap": max(
            (float(day["solver_gap"] or 0.0) for day in result.daily_runs),
            default=None,
        ),
    }


def _comparison(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    values = summary.set_index("strategy")["objective_yuan"].to_dict()
    chronos = float(values["chronos_zero_shot_pv_previous_day_load_milp"])
    comparisons = {}
    for baseline in (
        "historical_operation",
        "original_full_month_dp_perfect_foresight",
        "daily_perfect_foresight_milp",
        "previous_day_pv_previous_day_load_milp",
    ):
        baseline_value = float(values[baseline])
        difference = chronos - baseline_value
        comparisons[baseline] = {
            "difference_yuan": difference,
            "relative_difference_percent": 100.0 * difference / baseline_value,
        }
    return comparisons


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    audits: dict[str, Any],
    chronos_metadata: dict[str, Any],
) -> None:
    lines = [
        "# Foshan Chronos-2 to HiGHS Revenue Backtest",
        "",
        f"**Status:** {FORECAST_NOTICE}",
        "",
        f"**Evaluation warning:** {VALIDATION_NOTICE}",
        "",
        f"**Load caveat:** {LOAD_DATA_NOTICE}",
        "",
        "## Protocol",
        "",
        "- Window: May 2-31, 2026, with 8,640 common five-minute timestamps.",
        (
            "- Chronos input: selected postprocessed P50 from "
            f"`{chronos_metadata['model_name']}` at context "
            f"{chronos_metadata['context_length']}."
        ),
        "- Each 15-minute PV forecast is repeated across three five-minute controls.",
        "- Load uses the previous day's five-minute provisional-load profile.",
        "- Tariff prices are treated as known future inputs.",
        "- New daily strategies start at 900 kWh; each daily plan targets its propagated start SOC.",
        "- Frozen actions are reduced only for power, SOC, or anti-export safety.",
        "",
        "## Revenue Reconstruction",
        "",
        "| Strategy | Objective (yuan) | Final SOC (kWh) | Clipped intervals |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.objective_yuan:.2f} | "
            f"{row.final_soc_kwh:.3f} | {row.clipped_intervals} |"
        )
    lines.extend(
        [
            "",
            "## Safety And Terminal State",
            "",
        ]
    )
    for name in (
        "daily_perfect_foresight_milp",
        "previous_day_pv_previous_day_load_milp",
        "chronos_zero_shot_pv_previous_day_load_milp",
    ):
        audit = audits["strategies"][name]
        lines.append(
            f"- `{name}`: final SOC {audit['final_soc_kwh']:.3f} kWh; "
            f"difference from May 2 initial SOC "
            f"{audit['final_soc_difference_from_initial_kwh']:+.3f} kWh; "
            f"last-day terminal-target difference "
            f"{audit['final_terminal_difference_kwh']:.3f} kWh; "
            f"maximum daily terminal-target difference "
            f"{audit['maximum_abs_daily_terminal_difference_kwh']:.3f} kWh; "
            f"{audit['clipped_intervals']} clipped intervals."
        )
    lines.extend(
        [
            "",
            "A day's terminal target is its propagated replay SOC at the start of that day. "
            "Consequently, a zero last-day target difference does not erase cumulative SOC "
            "drift from the May 2 initial state. No terminal-energy adjustment is added to "
            "revenue, and both quantities are reported directly.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Backtest outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace this backtest only."
        )


def run_backtest(
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    output_dir: Path,
    *,
    initial_soc_kwh: float = 900.0,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not 0.0 <= initial_soc_kwh <= 2000.0:
        raise ValueError("initial_soc_kwh must be in [0, 2000].")
    _ensure_output_available(output_dir, overwrite)
    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)
    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path
    )
    previous_load = previous_day_forecast(
        realized, "load", "forecast_load_kw"
    )
    previous_pv = previous_day_forecast(realized, "pv", "forecast_pv_kw")
    perfect_pv = perfect_forecast(realized, "pv", "forecast_pv_kw")
    perfect_load = perfect_forecast(realized, "load", "forecast_load_kw")

    with tempfile.TemporaryDirectory(prefix="foshan_highs_backtest_") as temporary:
        log_dir = Path(temporary)
        results = {
            "historical_operation": reference_strategy(
                realized,
                "historical_operation",
                "p_actual",
                "soc_actual_est",
            ),
            "original_full_month_dp_perfect_foresight": reference_strategy(
                realized,
                "original_full_month_dp_perfect_foresight",
                "p_opt",
                "soc_opt",
            ),
            "daily_perfect_foresight_milp": run_daily_strategy(
                "daily_perfect_foresight_milp",
                realized,
                perfect_pv,
                perfect_load,
                initial_soc_kwh,
                log_dir,
                mip_relative_gap=mip_relative_gap,
            ),
            "previous_day_pv_previous_day_load_milp": run_daily_strategy(
                "previous_day_pv_previous_day_load_milp",
                realized,
                previous_pv,
                previous_load,
                initial_soc_kwh,
                log_dir,
                mip_relative_gap=mip_relative_gap,
            ),
            "chronos_zero_shot_pv_previous_day_load_milp": run_daily_strategy(
                "chronos_zero_shot_pv_previous_day_load_milp",
                realized,
                chronos_pv,
                previous_load,
                initial_soc_kwh,
                log_dir,
                mip_relative_gap=mip_relative_gap,
            ),
        }

    common_timestamps = validate_common_timestamps(results)
    replay = pd.concat([result.replay for result in results.values()], ignore_index=True)
    summaries = pd.DataFrame(
        [summarize_strategy(name, result, parameters) for name, result in results.items()]
    )
    strategy_audits = {
        name: audit_strategy(name, result, parameters)
        for name, result in results.items()
    }
    leakage_audit = {
        "chronos_pv_actual_columns_loaded": False,
        "chronos_quantile_used": "p50",
        "chronos_postprocessing": POSTPROCESSING,
        "load_forecast_lag_days": 1,
        "tariff_known_future": True,
        "actual_pv_or_load_passed_to_forecast_driven_solver": False,
        "actions_frozen_before_realized_replay": True,
    }
    audits = {
        "load_data_notice": LOAD_DATA_NOTICE,
        "forecast_notice": FORECAST_NOTICE,
        "validation_notice": VALIDATION_NOTICE,
        "common_timestamp_count": len(common_timestamps),
        "common_timestamp_start": common_timestamps[0].isoformat(),
        "common_timestamp_end": common_timestamps[-1].isoformat(),
        "chronos_forecast": chronos_metadata,
        "leakage_controls": leakage_audit,
        "strategies": strategy_audits,
        "daily_runs": {
            name: result.daily_runs for name, result in results.items() if result.daily_runs
        },
    }
    revenue = {
        "load_data_notice": LOAD_DATA_NOTICE,
        "forecast_notice": FORECAST_NOTICE,
        "validation_notice": VALIDATION_NOTICE,
        "window": {
            "start": START.isoformat(),
            "end_exclusive": END_EXCLUSIVE.isoformat(),
            "timestamp_count_per_strategy": len(common_timestamps),
        },
        "new_strategy_initial_soc_kwh": initial_soc_kwh,
        "daily_terminal_policy": (
            "target each day's propagated replay start SOC; report cumulative drift "
            "from the May 2 initial SOC separately"
        ),
        "terminal_energy_value_adjustment_yuan": 0.0,
        "chronos_selection": chronos_metadata,
        "strategy_summaries": json.loads(summaries.to_json(orient="records")),
        "chronos_comparisons": _comparison(summaries),
        "provenance": {
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "dispatch_path": str(dispatch_path.resolve()),
            "dispatch_sha256": _sha256(dispatch_path),
            "predictions_path": str(predictions_path.resolve()),
            "predictions_sha256": _sha256(predictions_path),
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": _sha256(selection_path),
        },
    }

    replay.to_csv(output_dir / "replay_timeseries.csv", index=False, float_format="%.15g")
    summaries.to_csv(output_dir / "strategy_summary.csv", index=False, float_format="%.15g")
    _write_json(output_dir / "revenue_comparison.json", revenue)
    _write_json(output_dir / "constraint_audit.json", audits)
    _write_report(output_dir / "report.md", summaries, audits, chronos_metadata)
    return replay, summaries, revenue, audits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest selected Chronos-2 P50 forecasts through daily HiGHS dispatch."
    )
    parser.add_argument("--dispatch-input", required=True, type=Path)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/foshan_chronos2/predictions_long.csv"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/foshan_chronos2/selected_configuration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimization/foshan_may_forecast_backtest"),
    )
    parser.add_argument("--initial-soc-kwh", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summaries, _, _ = run_backtest(
        dispatch_path=args.dispatch_input,
        predictions_path=args.predictions,
        selection_path=args.selection,
        output_dir=args.output_dir,
        initial_soc_kwh=args.initial_soc_kwh,
        mip_relative_gap=args.mip_relative_gap,
        overwrite=args.overwrite,
    )
    print(
        f"Saved Foshan forecast backtest to {args.output_dir.resolve()} "
        f"for {len(summaries)} strategies."
    )


if __name__ == "__main__":
    main()
