"""Evaluation-only rolling forecast wrapper for the frozen controller_v5 policy."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.data.reconstruct_foshan_residual import TARGET_COLUMN, TIMEZONE
from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    DispatchSolution,
    TerminalBand,
    solve_dispatch,
)
from src.optimization.foshan_feedback_controller_v2 import (
    DAILY_TERMINAL_LOWER_KWH,
    DAILY_TERMINAL_UPPER_KWH,
    FINAL_TERMINAL_LOWER_KWH,
    FINAL_TERMINAL_UPPER_KWH,
    TERMINAL_DEVIATION_PENALTY,
    TERMINAL_SOC_REFERENCE_KWH,
    latest_completed_residual,
    net_equivalent_load_pv,
)
from src.optimization.foshan_forecast_backtest import (
    StrategyResult,
    clipping_energy_kwh,
    replay_day,
)


FROZEN_SOURCE_SHA256 = {
    "src/optimization/foshan_feedback_controller_v5.py": "5acbd5c8ad53ef1c8c6d57a7f5cd45d57c8e75ff077cd70fb23e1e648869ced9",
    "src/optimization/foshan_feedback_controller_v2.py": "1466bc673b5a9c96918f330f724f1fae5e55e113348894ced5247f12e3ce7e34",
    "src/optimization/foshan_forecast_backtest.py": "bd5f15fc977f5bf31c9be1848a565fc11a4e327b3f967027578f4b81bab60974",
    "src/optimization/foshan_battery_milp.py": "d299d1d946340018cde78d0edbc00bdff1dfd706c465647e244f67abedb842f7",
    "src/optimization/foshan_controller_v5_final_benchmark.py": "b13da51e2749fd9f6c386be090c4777c327fdc961db0a8ba508abaaf83b31d29",
}


@dataclass(frozen=True)
class ForecastBook:
    """Issue-aware P50 forecasts for one controller candidate."""

    candidate: str
    kind: Literal["signed_residual", "gross_load"]
    predictions: pd.DataFrame
    target_frequency_minutes: int


Solver = Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_controller_sources(repository_root: Path = Path(".")) -> dict[str, str]:
    observed = {
        relative: file_sha256(repository_root / relative)
        for relative in FROZEN_SOURCE_SHA256
    }
    mismatches = {
        path: {"expected": FROZEN_SOURCE_SHA256[path], "actual": value}
        for path, value in observed.items()
        if value != FROZEN_SOURCE_SHA256[path]
    }
    if mismatches:
        raise RuntimeError(f"Frozen controller/MILP sources changed: {mismatches}")
    return observed


def _local_naive(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if isinstance(timestamps, pd.DatetimeIndex):
        series = pd.Series(timestamps)
    else:
        series = timestamps
    if series.dt.tz is None:
        return series.astype("datetime64[ns]")
    return series.dt.tz_convert(TIMEZONE).dt.tz_localize(None).astype("datetime64[ns]")


def make_forecast_book(
    predictions: pd.DataFrame,
    candidate: str,
    kind: Literal["signed_residual", "gross_load"],
) -> ForecastBook:
    """Normalize a prediction table without collapsing issue-time identity."""
    table = predictions.copy()
    if "candidate" in table:
        table = table.loc[table["candidate"].eq(candidate)].copy()
    required = {"issue_time", "target_time", "p50"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Forecast book is missing columns: {missing}")
    if table.empty:
        raise ValueError(f"Forecast book for {candidate} is empty.")
    table["issue_time"] = _local_naive(table["issue_time"]).to_numpy()
    table["target_time"] = _local_naive(table["target_time"]).to_numpy()
    table["p50"] = pd.to_numeric(table["p50"], errors="raise")
    if not np.isfinite(table["p50"]).all():
        raise ValueError(f"Forecast book {candidate} contains non-finite P50 values.")
    if table.duplicated(["issue_time", "target_time"]).any():
        raise ValueError(f"Forecast book {candidate} contains duplicate issue/target rows.")
    if (table["target_time"] < table["issue_time"]).any():
        raise ValueError(f"Forecast book {candidate} contains targets before issue time.")
    if "context_end" in table:
        table["context_end"] = _local_naive(table["context_end"]).to_numpy()
        if (table["context_end"] >= table["issue_time"]).any():
            raise ValueError(f"Forecast book {candidate} contains non-causal contexts.")
    frequencies: set[int] = set()
    for _, group in table.groupby("issue_time", sort=False):
        differences = group["target_time"].sort_values().diff().dropna()
        if differences.empty:
            continue
        unique = differences.dt.total_seconds().div(60).astype(int).unique()
        frequencies.update(int(value) for value in unique)
    if not frequencies:
        raise ValueError(f"Forecast book {candidate} cannot resolve its target frequency.")
    if frequencies not in ({5}, {15}):
        raise ValueError(
            f"Forecast book {candidate} must use 5- or 15-minute targets: {frequencies}"
        )
    return ForecastBook(
        candidate=candidate,
        kind=kind,
        predictions=table.sort_values(["issue_time", "target_time"]).reset_index(drop=True),
        target_frequency_minutes=next(iter(frequencies)),
    )


def terminal_band_for_evaluation_day(
    day: pd.Timestamp,
    final_day: pd.Timestamp,
) -> TerminalBand:
    is_final = pd.Timestamp(day).normalize() == pd.Timestamp(final_day).normalize()
    return TerminalBand(
        lower_kwh=(FINAL_TERMINAL_LOWER_KWH if is_final else DAILY_TERMINAL_LOWER_KWH),
        upper_kwh=(FINAL_TERMINAL_UPPER_KWH if is_final else DAILY_TERMINAL_UPPER_KWH),
        reference_kwh=TERMINAL_SOC_REFERENCE_KWH,
        deviation_penalty_yuan_per_kwh=TERMINAL_DEVIATION_PENALTY,
    )


def causal_residual_fallback(
    target_15min: pd.DataFrame,
    target_time: pd.Timestamp,
    issue_time: pd.Timestamp,
) -> tuple[float, str]:
    """Use previous day, then a four-week same-slot median, strictly before issue."""
    source = target_15min.copy()
    source["timestamp"] = _local_naive(source["timestamp"]).to_numpy()
    values = source.set_index("timestamp")[TARGET_COLUMN]
    previous_day = pd.Timestamp(target_time) - pd.Timedelta(days=1)
    if previous_day >= issue_time:
        raise AssertionError("Previous-day fallback is not strictly historical.")
    previous_value = values.get(previous_day, np.nan)
    if pd.notna(previous_value):
        return float(previous_value), f"previous_day:{previous_day.isoformat()}"
    weekly_times = [
        pd.Timestamp(target_time) - pd.Timedelta(days=days)
        for days in (7, 14, 21, 28)
    ]
    if any(value >= issue_time for value in weekly_times):
        raise AssertionError("Four-week fallback is not strictly historical.")
    weekly = pd.to_numeric(values.reindex(weekly_times), errors="coerce").dropna()
    if weekly.empty:
        raise ValueError(
            f"No causal residual fallback is available for {target_time} at {issue_time}."
        )
    return float(np.median(weekly.to_numpy(dtype=float))), "four_week_median:" + "|".join(
        value.isoformat() for value in weekly.index
    )


def newest_causal_issue(book: ForecastBook, control_time: pd.Timestamp) -> pd.Timestamp:
    issues = book.predictions.loc[
        book.predictions["issue_time"] <= control_time, "issue_time"
    ]
    if issues.empty:
        raise ValueError(
            f"{book.candidate} has no issued forecast available at {control_time}."
        )
    return pd.Timestamp(issues.max())


def controller_horizon_from_book(
    book: ForecastBook,
    pv_horizon: pd.DataFrame,
    target_15min: pd.DataFrame,
    control_time: pd.Timestamp,
    day_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one remaining-day horizon from the newest forecast available now."""
    expected = pd.date_range(control_time, day_end, freq="5min", inclusive="left")
    pv = pv_horizon.set_index("timestamp").reindex(expected)
    if pv["forecast_pv_kw"].isna().any():
        raise ValueError(f"PV forecast is incomplete at {control_time}.")
    issue = newest_causal_issue(book, control_time)
    issued = book.predictions.loc[book.predictions["issue_time"].eq(issue)].set_index(
        "target_time"
    )
    lookup_times = (
        expected
        if book.target_frequency_minutes == 5
        else expected.floor("15min")
    )
    forecast_values = issued["p50"].reindex(lookup_times).to_numpy(dtype=float)
    fallback_count = 0
    fallback_sources: list[str] = []
    for index in np.flatnonzero(~np.isfinite(forecast_values)):
        value, source = causal_residual_fallback(
            target_15min,
            pd.Timestamp(expected[index].floor("15min")),
            issue,
        )
        if book.kind == "gross_load":
            value = max(0.0, float(pv["forecast_pv_kw"].iloc[index]) + value)
        forecast_values[index] = value
        fallback_count += 1
        fallback_sources.append(source)

    forecast_pv = pv["forecast_pv_kw"].to_numpy(dtype=float)
    if book.kind == "signed_residual":
        signed_residual = forecast_values
        forecast_load = np.maximum(0.0, forecast_pv + signed_residual)
    else:
        forecast_load = np.maximum(0.0, forecast_values)
        signed_residual = forecast_load - forecast_pv
    discharge_limit = np.maximum(0.0, signed_residual)
    horizon = pd.DataFrame(
        {
            "timestamp": expected,
            "forecast_pv_kw": forecast_pv,
            "forecast_load_kw": forecast_load,
            "forecast_signed_residual_kw": signed_residual,
            "safe_residual_kw": discharge_limit,
            "forecast_issue_time": issue,
            "forecast_fallback_used": False,
        }
    )
    if fallback_count:
        missing_positions = ~np.isfinite(
            issued["p50"].reindex(lookup_times).to_numpy(dtype=float)
        )
        horizon.loc[missing_positions, "forecast_fallback_used"] = True
    audit = {
        "forecast_issue_time": issue,
        "forecast_issue_is_causal": bool(issue <= control_time),
        "fallback_intervals": fallback_count,
        "fallback_sources": fallback_sources,
    }
    return horizon, audit


def _final_day_charge_limit(
    current_soc_kwh: float,
    parameters: DispatchParameters,
) -> float:
    return min(
        parameters.power_limit_kw,
        max(
            0.0,
            (FINAL_TERMINAL_UPPER_KWH - current_soc_kwh)
            / (parameters.charge_efficiency * parameters.interval_hours),
        ),
    )


def _daily_run_row(
    day: pd.Timestamp,
    day_start_soc: float,
    current_soc: float,
    terminal_band: TerminalBand,
    replay: pd.DataFrame,
    statuses: list[str],
    runtimes: list[float],
    gaps: list[float],
    planned_terminal_socs: list[float],
    negative_slacks: list[float],
    positive_slacks: list[float],
    parameters: DispatchParameters,
) -> dict[str, Any]:
    return {
        "date": day.date().isoformat(),
        "initial_soc_kwh": day_start_soc,
        "realized_terminal_soc_kwh": current_soc,
        "terminal_soc_reference_kwh": TERMINAL_SOC_REFERENCE_KWH,
        "terminal_band_lower_kwh": terminal_band.lower_kwh,
        "terminal_band_upper_kwh": terminal_band.upper_kwh,
        "planned_terminal_soc_kwh": planned_terminal_socs[-1],
        "planned_terminal_deviation_negative_kwh": negative_slacks[-1],
        "planned_terminal_deviation_positive_kwh": positive_slacks[-1],
        "realized_terminal_deviation_negative_kwh": max(
            terminal_band.lower_kwh - current_soc, 0.0
        ),
        "realized_terminal_deviation_positive_kwh": max(
            current_soc - terminal_band.upper_kwh, 0.0
        ),
        "replan_count": len(replay),
        "solver_failure_count": sum(status != "Optimal" for status in statuses),
        "solver_status": sorted(set(statuses))[0],
        "maximum_solver_relative_gap": max(gaps),
        "solver_runtime_seconds": sum(runtimes),
        "clipped_intervals": int(replay["was_clipped"].sum()),
        "clipped_energy_kwh": clipping_energy_kwh(
            replay["total_clip_kw"], parameters.interval_hours
        ),
        "anti_export_clipped_intervals": int(
            (replay["anti_export_clip_kw"] > 1e-7).sum()
        ),
        "anti_export_clipped_kwh": clipping_energy_kwh(
            replay["anti_export_clip_kw"], parameters.interval_hours
        ),
    }


def run_rolling_v5_evaluation(
    name: str,
    realized_with_history: pd.DataFrame,
    pv_forecast: pd.DataFrame,
    forecast_book: ForecastBook,
    residual_target_15min: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    solver_log_dir: Path,
    *,
    solver: Solver = solve_dispatch,
    parameters: DispatchParameters = DispatchParameters(),
    initial_soc_kwh: float = 900.0,
    mip_relative_gap: float = 1e-7,
    show_progress: bool = False,
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    """Reproduce v5 with issue-aware forecasts; execute one five-minute action."""
    verify_frozen_controller_sources()
    if not np.isclose(initial_soc_kwh, 900.0):
        raise ValueError("Frozen controller_v5 evaluation requires initial SOC 900 kWh.")
    realized = realized_with_history.copy()
    realized["timestamp"] = _local_naive(realized["timestamp"]).to_numpy()
    pv = pv_forecast.copy()
    pv["timestamp"] = _local_naive(pv["timestamp"]).to_numpy()
    expected = pd.date_range(start, end_exclusive, freq="5min", inclusive="left")
    actual_window = realized.loc[realized["timestamp"].isin(expected)]
    if actual_window["timestamp"].tolist() != expected.tolist():
        raise ValueError("Realized evaluation data does not match the complete grid.")
    if pv["timestamp"].tolist() != expected.tolist():
        raise ValueError("Frozen PV forecast does not match the complete grid.")

    current_soc = float(initial_soc_kwh)
    final_day = pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)
    replay_parts: list[pd.DataFrame] = []
    daily_runs: list[dict[str, Any]] = []
    replans: list[dict[str, Any]] = []
    solver_log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for day in pd.date_range(start, end_exclusive, freq="1D", inclusive="left"):
        day_end = day + pd.Timedelta(days=1)
        day_start_soc = current_soc
        terminal_band = terminal_band_for_evaluation_day(day, final_day)
        day_pv = pv.loc[(pv["timestamp"] >= day) & (pv["timestamp"] < day_end)].copy()
        day_actual = realized.loc[
            (realized["timestamp"] >= day) & (realized["timestamp"] < day_end)
        ].copy()
        if len(day_pv) != 288 or len(day_actual) != 288:
            raise ValueError(f"{day.date()} must contain 288 five-minute rows.")
        day_replays: list[pd.DataFrame] = []
        statuses: list[str] = []
        runtimes: list[float] = []
        gaps: list[float] = []
        planned_terminal_socs: list[float] = []
        negative_slacks: list[float] = []
        positive_slacks: list[float] = []

        for control_time in pd.date_range(day, day_end, freq="5min", inclusive="left"):
            remaining, forecast_audit = controller_horizon_from_book(
                forecast_book,
                day_pv.loc[day_pv["timestamp"] >= control_time],
                residual_target_15min,
                control_time,
                day_end,
            )
            source_timestamp, measured_residual = latest_completed_residual(
                realized, control_time
            )
            first_load, first_pv = net_equivalent_load_pv(measured_residual)
            remaining.loc[remaining.index[0], "forecast_load_kw"] = first_load
            remaining.loc[remaining.index[0], "forecast_pv_kw"] = first_pv
            remaining.loc[remaining.index[0], "safe_residual_kw"] = max(
                measured_residual, 0.0
            )
            remaining["charge_limit_kw"] = parameters.power_limit_kw
            guard_applied = day.normalize() == final_day.normalize()
            immediate_charge_limit = parameters.power_limit_kw
            if guard_applied:
                immediate_charge_limit = _final_day_charge_limit(current_soc, parameters)
                remaining.loc[remaining.index[0], "charge_limit_kw"] = immediate_charge_limit

            prices = day_actual.loc[
                day_actual["timestamp"] >= control_time, ["timestamp", "price"]
            ]
            milp_input = remaining[
                [
                    "timestamp",
                    "forecast_pv_kw",
                    "forecast_load_kw",
                    "safe_residual_kw",
                    "charge_limit_kw",
                ]
            ].merge(prices, on="timestamp", how="inner", validate="one_to_one").rename(
                columns={
                    "forecast_pv_kw": "pv",
                    "forecast_load_kw": "load",
                    "safe_residual_kw": "discharge_limit_kw",
                }
            )
            replan_parameters = replace(
                parameters,
                initial_soc_kwh=current_soc,
                terminal_soc_kwh=TERMINAL_SOC_REFERENCE_KWH,
            )
            solved: DispatchSolution = solver(
                milp_input,
                solver_log_dir / f"{name}.log",
                replan_parameters,
                mip_relative_gap=mip_relative_gap,
                log_to_console=False,
                terminal_band=terminal_band,
            )
            status = str(solved.solver_metadata.get("solver_status"))
            if status != "Optimal":
                raise RuntimeError(f"{name} {control_time} did not solve optimally: {status}.")
            schedule = solved.dispatch.iloc[:1].copy()

            # Actual PV/load are sliced only after the schedule has been solved.
            realized_block = day_actual.loc[
                day_actual["timestamp"].eq(control_time),
                ["timestamp", "pv", "load", "price"],
            ]
            replay = replay_day(
                schedule,
                realized_block,
                current_soc,
                replan_parameters,
            ).merge(remaining.iloc[:1], on="timestamp", how="left", validate="one_to_one")
            replay.insert(0, "strategy", name)
            replay["controller"] = "controller_v5_evaluation_wrapper"
            replay["issue_time"] = forecast_audit["forecast_issue_time"]
            replay["replan_time"] = control_time
            replay["cadence_minutes"] = 5
            replay["terminal_soc_reference_kwh"] = TERMINAL_SOC_REFERENCE_KWH
            replay["terminal_band_lower_kwh"] = terminal_band.lower_kwh
            replay["terminal_band_upper_kwh"] = terminal_band.upper_kwh
            replay["q10_discharge_limit_enabled"] = False
            replay["terminal_recovery_charge_ban_enabled"] = False
            replay["intraday_load_bias_correction_enabled"] = False
            replay["final_day_immediate_charge_guard_enabled"] = True
            replay["final_day_immediate_charge_guard_applied"] = guard_applied
            replay["planned_terminal_deviation_negative_kwh"] = float(
                solved.solver_metadata["terminal_deviation_negative_kwh"]
            )
            replay["planned_terminal_deviation_positive_kwh"] = float(
                solved.solver_metadata["terminal_deviation_positive_kwh"]
            )
            replay["planned_terminal_soc_kwh"] = float(
                solved.dispatch["soc_kwh"].iloc[-1]
            )
            replay["safety_filter_applied"] = True
            replay["counterfactual_provisional"] = True
            replay["validation_period_demo"] = True
            day_replays.append(replay)
            replay_parts.append(replay)
            current_soc = float(replay["realized_soc_end_kwh"].iloc[-1])

            runtime = float(solved.solver_metadata.get("wall_clock_runtime_seconds") or 0.0)
            gap = float(solved.solver_metadata.get("optimality_gap") or 0.0)
            negative_slack = float(
                solved.solver_metadata["terminal_deviation_negative_kwh"]
            )
            positive_slack = float(
                solved.solver_metadata["terminal_deviation_positive_kwh"]
            )
            statuses.append(status)
            runtimes.append(runtime)
            gaps.append(gap)
            planned_terminal_socs.append(float(solved.dispatch["soc_kwh"].iloc[-1]))
            negative_slacks.append(negative_slack)
            positive_slacks.append(positive_slack)
            replans.append(
                {
                    "strategy": name,
                    "control_time": control_time.isoformat(),
                    "forecast_issue_time": pd.Timestamp(
                        forecast_audit["forecast_issue_time"]
                    ).isoformat(),
                    "forecast_issue_is_causal": forecast_audit[
                        "forecast_issue_is_causal"
                    ],
                    "forecast_candidate": forecast_book.candidate,
                    "forecast_kind": forecast_book.kind,
                    "fallback_intervals_in_horizon": forecast_audit[
                        "fallback_intervals"
                    ],
                    "first_step_residual_source_timestamp": source_timestamp.isoformat(),
                    "first_step_source_strictly_before_control": bool(
                        source_timestamp < control_time
                    ),
                    "first_step_measured_residual_kw": measured_residual,
                    "future_realized_pv_or_load_passed": False,
                    "known_future_tariff_passed": True,
                    "initial_soc_kwh": float(replan_parameters.initial_soc_kwh),
                    "realized_soc_after_execution_kwh": current_soc,
                    "terminal_band_lower_kwh": terminal_band.lower_kwh,
                    "terminal_band_upper_kwh": terminal_band.upper_kwh,
                    "terminal_deviation_negative_kwh": negative_slack,
                    "terminal_deviation_positive_kwh": positive_slack,
                    "final_day_immediate_charge_guard_applied": guard_applied,
                    "immediate_charge_limit_kw": immediate_charge_limit,
                    "solver_status": status,
                    "solver_relative_gap": gap,
                    "solver_runtime_seconds": runtime,
                }
            )

        day_replay = pd.concat(day_replays, ignore_index=True)
        daily_runs.append(
            _daily_run_row(
                day,
                day_start_soc,
                current_soc,
                terminal_band,
                day_replay,
                statuses,
                runtimes,
                gaps,
                planned_terminal_socs,
                negative_slacks,
                positive_slacks,
                parameters,
            )
        )
        if show_progress:
            print(
                f"Completed {name} through {day.date()} "
                f"(SOC={current_soc:.2f} kWh, elapsed={time.perf_counter() - started:.1f}s)."
            )
    return StrategyResult(pd.concat(replay_parts, ignore_index=True), daily_runs), replans
