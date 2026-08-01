"""Foshan controller_v2 with fixed terminal SOC and causal discharge limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.optimization.foshan_battery_milp import (
    LOAD_DATA_NOTICE,
    DispatchParameters,
    DispatchSolution,
    TerminalBand,
    recalculate_objective,
    solve_dispatch,
)
from src.optimization.foshan_forecast_backtest import (
    END_EXCLUSIVE,
    FORECAST_NOTICE,
    START,
    VALIDATION_NOTICE,
    StrategyResult,
    clipping_energy_kwh,
    load_reference_dispatch,
    load_selected_chronos_p50,
    previous_day_forecast,
    replay_day,
    validate_common_timestamps,
)
from src.utils.runtime import git_commit, git_is_dirty


TERMINAL_SOC_REFERENCE_KWH = 900.0
DAILY_TERMINAL_LOWER_KWH = 850.0
DAILY_TERMINAL_UPPER_KWH = 950.0
FINAL_TERMINAL_LOWER_KWH = 895.0
FINAL_TERMINAL_UPPER_KWH = 905.0
FINAL_DAY = pd.Timestamp("2026-05-31")
TERMINAL_DEVIATION_PENALTY = 1.0
Q10_QUANTILE = 0.10
Q10_QUANTILE_METHOD = "linear"
OUTPUT_FILENAMES = (
    "replay_timeseries.csv",
    "strategy_summary.csv",
    "daily_summary.csv",
    "comparison.json",
    "constraint_audit.json",
    "controller_metadata.json",
    "report.md",
)
OLD_STRATEGIES = (
    "fixed_actual_pv_actual_load_oracle",
    "feedback_previous_day_pv_previous_day_load",
    "feedback_chronos_pv_previous_day_load",
)
V2_STRATEGIES = (
    "controller_v2_previous_day_pv_previous_day_load",
    "controller_v2_chronos_pv_previous_day_load",
)
STRATEGY_LABELS = {
    "fixed_actual_pv_actual_load_oracle": {
        "controller": "fixed_daily_perfect_foresight",
        "pv_source": "actual_future_pv",
        "load_source": "actual_future_provisional_load",
        "deployable": False,
        "oracle_diagnostic": True,
    },
    "feedback_previous_day_pv_previous_day_load": {
        "controller": "feedback_v1_15min",
        "pv_source": "previous_day_pv",
        "load_source": "previous_day_provisional_load",
        "deployable": True,
        "oracle_diagnostic": False,
    },
    "feedback_chronos_pv_previous_day_load": {
        "controller": "feedback_v1_15min",
        "pv_source": "chronos2_zero_shot_postprocessed_p50",
        "load_source": "previous_day_provisional_load",
        "deployable": True,
        "oracle_diagnostic": False,
    },
    "controller_v2_previous_day_pv_previous_day_load": {
        "controller": "controller_v2",
        "pv_source": "previous_day_pv",
        "load_source": "previous_day_provisional_load",
        "deployable": True,
        "oracle_diagnostic": False,
    },
    "controller_v2_chronos_pv_previous_day_load": {
        "controller": "controller_v2",
        "pv_source": "chronos2_zero_shot_postprocessed_p50",
        "load_source": "previous_day_provisional_load",
        "deployable": True,
        "oracle_diagnostic": False,
    },
}


Solver = Callable[..., DispatchSolution]


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


def terminal_band_for_day(day: pd.Timestamp) -> TerminalBand:
    normalized = pd.Timestamp(day).normalize()
    if normalized == FINAL_DAY:
        lower = FINAL_TERMINAL_LOWER_KWH
        upper = FINAL_TERMINAL_UPPER_KWH
    else:
        lower = DAILY_TERMINAL_LOWER_KWH
        upper = DAILY_TERMINAL_UPPER_KWH
    return TerminalBand(
        lower_kwh=lower,
        upper_kwh=upper,
        reference_kwh=TERMINAL_SOC_REFERENCE_KWH,
        deviation_penalty_yuan_per_kwh=TERMINAL_DEVIATION_PENALTY,
    )


def completed_day_q10(
    residual_error_history: list[tuple[pd.Timestamp, float]],
    day: pd.Timestamp,
) -> tuple[float, int, bool]:
    """Calculate q10 from errors whose calendar day is already complete."""
    normalized_day = pd.Timestamp(day).normalize()
    eligible = [
        float(error)
        for timestamp, error in residual_error_history
        if pd.Timestamp(timestamp).normalize() < normalized_day
    ]
    if not eligible:
        return 0.0, 0, True
    return (
        float(
            np.quantile(
                eligible,
                Q10_QUANTILE,
                method=Q10_QUANTILE_METHOD,
            )
        ),
        len(eligible),
        False,
    )


def safe_residual_limit(
    forecast_load_kw: pd.Series | np.ndarray,
    forecast_pv_kw: pd.Series | np.ndarray,
    q10_residual_error_kw: float,
) -> np.ndarray:
    """Return max(0, forecast residual + frozen historical q10)."""
    load = np.asarray(forecast_load_kw, dtype=float)
    pv = np.asarray(forecast_pv_kw, dtype=float)
    if load.shape != pv.shape:
        raise ValueError("Forecast load and PV arrays must have identical shapes.")
    if not np.isfinite(load).all() or not np.isfinite(pv).all():
        raise ValueError("Forecast load and PV must be finite.")
    if not np.isfinite(q10_residual_error_kw):
        raise ValueError("q10 residual error must be finite.")
    forecast_residual = load - pv
    return np.maximum(0.0, forecast_residual + q10_residual_error_kw)


def latest_completed_residual(
    realized: pd.DataFrame,
    control_time: pd.Timestamp,
    interval_minutes: int = 5,
) -> tuple[pd.Timestamp, float]:
    """Return the residual from the interval immediately before control time."""
    expected_timestamp = pd.Timestamp(control_time) - pd.Timedelta(
        minutes=interval_minutes
    )
    completed = realized.loc[
        realized["timestamp"] < control_time,
        ["timestamp", "pv", "load"],
    ].tail(1)
    if completed.empty:
        raise ValueError(
            f"No completed residual measurement is available before {control_time}."
        )
    source_timestamp = pd.Timestamp(completed["timestamp"].iloc[0])
    if source_timestamp != expected_timestamp:
        raise ValueError(
            "Latest residual measurement must be exactly one five-minute "
            f"interval before control time; expected {expected_timestamp}, "
            f"found {source_timestamp}."
        )
    residual_kw = float(completed["load"].iloc[0] - completed["pv"].iloc[0])
    if not np.isfinite(residual_kw):
        raise ValueError("Latest completed residual measurement must be finite.")
    return source_timestamp, residual_kw


def net_equivalent_load_pv(residual_kw: float) -> tuple[float, float]:
    """Represent signed residual load as a nonnegative net-equivalent load/PV pair."""
    if not np.isfinite(residual_kw):
        raise ValueError("Residual load must be finite.")
    return max(residual_kw, 0.0), max(-residual_kw, 0.0)


def intraday_load_bias(
    realized: pd.DataFrame,
    frozen_load_forecast: pd.DataFrame,
    control_time: pd.Timestamp,
    *,
    window_size: int = 12,
    minimum_samples: int = 3,
) -> tuple[float, int, pd.Timestamp | None, pd.Timestamp | None]:
    """Return a causal same-day median load error over completed intervals."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if minimum_samples <= 0 or minimum_samples > window_size:
        raise ValueError("minimum_samples must be in [1, window_size].")
    day_start = pd.Timestamp(control_time).normalize()
    completed = realized.loc[
        (realized["timestamp"] >= day_start)
        & (realized["timestamp"] < control_time),
        ["timestamp", "load"],
    ].merge(
        frozen_load_forecast[["timestamp", "forecast_load_kw"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp").tail(window_size)
    sample_count = len(completed)
    if sample_count == 0:
        return 0.0, 0, None, None
    if not (completed["timestamp"] < control_time).all():
        raise ValueError("Intraday load bias included a non-completed timestamp.")
    errors = (
        completed["load"].to_numpy(dtype=float)
        - completed["forecast_load_kw"].to_numpy(dtype=float)
    )
    if not np.isfinite(errors).all():
        raise ValueError("Intraday load errors must be finite.")
    bias_kw = float(np.median(errors)) if sample_count >= minimum_samples else 0.0
    return (
        bias_kw,
        sample_count,
        pd.Timestamp(completed["timestamp"].iloc[0]),
        pd.Timestamp(completed["timestamp"].iloc[-1]),
    )


def apply_intraday_load_bias(
    frozen_forecast_load_kw: pd.Series | np.ndarray,
    bias_kw: float,
) -> np.ndarray:
    """Apply a scalar load correction and retain nonnegative load forecasts."""
    load = np.asarray(frozen_forecast_load_kw, dtype=float)
    if not np.isfinite(load).all() or not np.isfinite(bias_kw):
        raise ValueError("Frozen load forecasts and bias must be finite.")
    return np.maximum(0.0, load + bias_kw)


def final_day_immediate_charge_limit_kw(
    current_soc_kwh: float,
    parameters: DispatchParameters = DispatchParameters(),
) -> float:
    """Return the May 31 first-action charge cap for the 905-kWh ceiling."""
    if not np.isfinite(current_soc_kwh):
        raise ValueError("Current SOC must be finite.")
    return min(
        parameters.power_limit_kw,
        max(
            0.0,
            (FINAL_TERMINAL_UPPER_KWH - current_soc_kwh)
            / (parameters.charge_efficiency * parameters.interval_hours),
        ),
    )


def _add_soc_clip_components(
    replay: pd.DataFrame,
    parameters: DispatchParameters,
) -> pd.DataFrame:
    result = replay.copy()
    if {"upper_soc_clip_kw", "lower_soc_clip_kw"}.issubset(result.columns):
        return result
    charge_after_power = np.minimum(
        result["scheduled_charge_kw"], parameters.power_limit_kw
    )
    discharge_after_power = np.minimum(
        result["scheduled_discharge_kw"], parameters.power_limit_kw
    )
    result["upper_soc_clip_kw"] = np.maximum(
        charge_after_power - result["applied_charge_kw"], 0.0
    )
    discharge_after_soc = (
        result["applied_discharge_kw"] + result["anti_export_clip_kw"]
    )
    result["lower_soc_clip_kw"] = np.maximum(
        discharge_after_power - discharge_after_soc, 0.0
    )
    reconstructed = result["upper_soc_clip_kw"] + result["lower_soc_clip_kw"]
    if not np.allclose(reconstructed, result["soc_clip_kw"], atol=1e-7):
        raise ValueError("Could not reconstruct upper/lower SOC clipping from old replay.")
    return result


def load_old_results(
    path: Path,
    parameters: DispatchParameters = DispatchParameters(),
) -> dict[str, StrategyResult]:
    table = pd.read_csv(path, low_memory=False)
    table["timestamp"] = pd.to_datetime(table["timestamp"], errors="raise")
    metadata_path = path.with_name("controller_metadata.json")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    replan_keys = {
        "feedback_previous_day_pv_previous_day_load": "previous_day_replans",
        "feedback_chronos_pv_previous_day_load": "chronos_replans",
    }
    results: dict[str, StrategyResult] = {}
    for name in OLD_STRATEGIES:
        replay = table.loc[table["strategy"].eq(name)].copy()
        replay = replay.sort_values("timestamp").reset_index(drop=True)
        if replay.empty:
            raise ValueError(f"Old feedback replay is missing {name}.")
        replay = _add_soc_clip_components(replay, parameters)
        if not np.isclose(replay["realized_soc_start_kwh"].iloc[0], 900.0):
            raise ValueError(f"Old strategy {name} does not start at 900 kWh.")
        replans = metadata.get(replan_keys.get(name, ""), [])
        daily_runs: list[dict[str, Any]] = []
        replan_table: dict[str, list[dict[str, Any]]] = {}
        for row in replans:
            day = pd.Timestamp(row["forecast_frozen_at"]).date().isoformat()
            replan_table.setdefault(day, []).append(row)
        for day, rows in sorted(replan_table.items()):
            last = rows[-1]
            daily_runs.append(
                {
                    "date": day,
                    "planned_terminal_soc_kwh": last.get("terminal_target_kwh"),
                    "solver_runtime_seconds": sum(
                        float(row.get("solver_runtime_seconds") or 0.0)
                        for row in rows
                    ),
                    "replan_count": len(rows),
                    "solver_failure_count": sum(
                        str(row.get("solver_status")) != "Optimal" for row in rows
                    ),
                }
            )
        results[name] = StrategyResult(replay=replay, daily_runs=daily_runs)
    validate_common_timestamps(results)
    return results


def _merge_forecasts(
    pv_forecast: pd.DataFrame,
    load_forecast: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    forecast = pv_forecast.merge(
        load_forecast,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp").reset_index(drop=True)
    expected = pd.date_range(start, end_exclusive, freq="5min", inclusive="left")
    if forecast["timestamp"].tolist() != expected.tolist():
        raise ValueError("Controller_v2 forecasts do not match the complete grid.")
    if "pv_forecast_issue_time" in forecast:
        issue = pd.to_datetime(forecast["pv_forecast_issue_time"], errors="raise")
        if (issue > forecast["timestamp"]).any():
            raise ValueError("PV forecast issue time follows a target timestamp.")
        if (
            pd.DataFrame(
                {"day": forecast["timestamp"].dt.normalize(), "issue": issue}
            )
            .groupby("day")["issue"]
            .nunique()
            .gt(1)
            .any()
        ):
            raise ValueError("PV forecasts must remain frozen for each day.")
    for column in (
        "forecast_pv_kw_source_timestamp",
        "forecast_load_kw_source_timestamp",
    ):
        if column in forecast:
            source = pd.to_datetime(forecast[column], errors="raise")
            if (source >= forecast["timestamp"]).any():
                raise ValueError(f"{column} must be strictly historical.")
    return forecast


def run_controller_v2(
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
    cadence_minutes: int = 15,
    mip_relative_gap: float = 1e-7,
    show_progress: bool = False,
    initial_residual_error_history: list[tuple[pd.Timestamp, float]] | None = None,
    use_q10_discharge_limit: bool = True,
    use_terminal_recovery_charge_ban: bool = True,
    use_latest_completed_residual_for_first_step: bool = False,
    use_intraday_load_bias_correction: bool = False,
    use_final_day_immediate_charge_guard: bool = False,
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    """Run fixed-forecast receding control with causal residual protection."""
    if name not in V2_STRATEGIES:
        raise ValueError(f"Unsupported controller_v2 strategy: {name}")
    if cadence_minutes not in (5, 15):
        raise ValueError("cadence_minutes must be 5 or 15.")
    execution_intervals = cadence_minutes // 5
    forecast = _merge_forecasts(
        pv_forecast, load_forecast, start, end_exclusive
    )
    days = pd.date_range(start, end_exclusive, freq="1D", inclusive="left")
    current_soc = float(initial_soc_kwh)
    residual_error_history = [
        (pd.Timestamp(timestamp), float(error))
        for timestamp, error in (initial_residual_error_history or [])
    ]
    if any(
        timestamp >= start or not np.isfinite(error)
        for timestamp, error in residual_error_history
    ):
        raise ValueError(
            "Initial residual errors must be finite and strictly before start."
        )
    replay_parts: list[pd.DataFrame] = []
    daily_runs: list[dict[str, Any]] = []
    replans: list[dict[str, Any]] = []

    for day in days:
        day_end = day + pd.Timedelta(days=1)
        day_start_soc = current_soc
        terminal_band = terminal_band_for_day(day)
        terminal_recovery_active = bool(
            use_terminal_recovery_charge_ban
            and current_soc > terminal_band.upper_kwh + 1e-7
        )
        q10_error, sample_count, q10_fallback_used = completed_day_q10(
            residual_error_history,
            day,
        )
        applied_q10_error = q10_error if use_q10_discharge_limit else 0.0
        day_forecast = forecast.loc[
            (forecast["timestamp"] >= day) & (forecast["timestamp"] < day_end)
        ].copy()
        day_tariff = realized.loc[
            (realized["timestamp"] >= day) & (realized["timestamp"] < day_end),
            ["timestamp", "price"],
        ].copy()
        if len(day_forecast) != 288 or len(day_tariff) != 288:
            raise ValueError(f"{day.date()} must contain 288 five-minute rows.")

        day_replays: list[pd.DataFrame] = []
        statuses: list[str] = []
        gaps: list[float] = []
        absolute_gaps: list[float] = []
        runtimes: list[float] = []
        planned_negative_slacks: list[float] = []
        planned_positive_slacks: list[float] = []
        planned_terminal_socs: list[float] = []
        day_residual_errors: list[tuple[pd.Timestamp, float]] = []
        control_times = pd.date_range(
            day, day_end, freq=f"{cadence_minutes}min", inclusive="left"
        )
        for control_time in control_times:
            if use_terminal_recovery_charge_ban:
                if current_soc > terminal_band.upper_kwh + 1e-7:
                    terminal_recovery_active = True
                elif (
                    terminal_recovery_active
                    and current_soc <= TERMINAL_SOC_REFERENCE_KWH + 1e-7
                ):
                    terminal_recovery_active = False
            else:
                terminal_recovery_active = False
            remaining = day_forecast.loc[
                day_forecast["timestamp"] >= control_time
            ].copy()
            intraday_bias_kw = 0.0
            intraday_bias_sample_count = 0
            intraday_bias_oldest_timestamp: pd.Timestamp | None = None
            intraday_bias_newest_timestamp: pd.Timestamp | None = None
            if use_intraday_load_bias_correction:
                (
                    intraday_bias_kw,
                    intraday_bias_sample_count,
                    intraday_bias_oldest_timestamp,
                    intraday_bias_newest_timestamp,
                ) = intraday_load_bias(
                    realized,
                    day_forecast,
                    control_time,
                )
            remaining["frozen_forecast_load_kw"] = remaining[
                "forecast_load_kw"
            ]
            remaining["intraday_load_bias_kw"] = intraday_bias_kw
            remaining["intraday_load_bias_sample_count"] = (
                intraday_bias_sample_count
            )
            remaining["intraday_load_bias_oldest_timestamp"] = (
                intraday_bias_oldest_timestamp
            )
            remaining["intraday_load_bias_newest_timestamp"] = (
                intraday_bias_newest_timestamp
            )
            remaining["adjusted_forecast_load_kw"] = apply_intraday_load_bias(
                remaining["frozen_forecast_load_kw"], intraday_bias_kw
            )
            remaining["adjusted_residual_forecast_kw"] = np.maximum(
                remaining["adjusted_forecast_load_kw"]
                - remaining["forecast_pv_kw"],
                0.0,
            )
            remaining["forecast_load_kw"] = remaining[
                "adjusted_forecast_load_kw"
            ]
            first_step_source_timestamp: pd.Timestamp | None = None
            first_step_residual_kw: float | None = None
            remaining["first_step_residual_override_applied"] = False
            remaining["first_step_residual_source_timestamp"] = pd.NaT
            remaining["first_step_measured_residual_kw"] = np.nan
            if use_latest_completed_residual_for_first_step:
                first_step_source_timestamp, first_step_residual_kw = (
                    latest_completed_residual(realized, control_time)
                )
                first_step_load_kw, first_step_pv_kw = net_equivalent_load_pv(
                    first_step_residual_kw
                )
                first_index = remaining.index[0]
                remaining.at[first_index, "forecast_load_kw"] = first_step_load_kw
                remaining.at[first_index, "forecast_pv_kw"] = first_step_pv_kw
                remaining.at[
                    first_index, "first_step_residual_override_applied"
                ] = True
                remaining.at[
                    first_index, "first_step_residual_source_timestamp"
                ] = first_step_source_timestamp
                remaining.at[
                    first_index, "first_step_measured_residual_kw"
                ] = first_step_residual_kw
            safe_limit = safe_residual_limit(
                remaining["forecast_load_kw"],
                remaining["forecast_pv_kw"],
                applied_q10_error,
            )
            remaining["safe_residual_kw"] = safe_limit
            remaining["residual_error_q10_kw"] = q10_error
            remaining["applied_residual_error_q10_kw"] = applied_q10_error
            remaining["q10_discharge_limit_enabled"] = use_q10_discharge_limit
            remaining["historical_error_sample_count"] = sample_count
            remaining["q10_fallback_used"] = q10_fallback_used
            remaining["q10_quantile_method"] = Q10_QUANTILE_METHOD
            remaining["terminal_recovery_active"] = terminal_recovery_active
            remaining["terminal_recovery_charge_ban_enabled"] = (
                use_terminal_recovery_charge_ban
            )
            remaining["charge_limit_kw"] = (
                0.0 if terminal_recovery_active else parameters.power_limit_kw
            )
            guard_applied = bool(
                use_final_day_immediate_charge_guard
                and pd.Timestamp(control_time).normalize() == FINAL_DAY
            )
            immediate_charge_limit_kw = float(
                remaining["charge_limit_kw"].iloc[0]
            )
            if guard_applied:
                immediate_charge_limit_kw = min(
                    immediate_charge_limit_kw,
                    final_day_immediate_charge_limit_kw(
                        current_soc,
                        parameters,
                    ),
                )
                remaining.at[
                    remaining.index[0], "charge_limit_kw"
                ] = immediate_charge_limit_kw
            future_charge_limits_unrestricted = bool(
                len(remaining) <= 1
                or remaining["charge_limit_kw"].iloc[1:].eq(
                    parameters.power_limit_kw
                ).all()
            )
            remaining_tariff = day_tariff.loc[
                day_tariff["timestamp"] >= control_time
            ].copy()
            milp_input = remaining[
                [
                    "timestamp",
                    "forecast_pv_kw",
                    "forecast_load_kw",
                    "safe_residual_kw",
                    "charge_limit_kw",
                ]
            ].merge(
                remaining_tariff,
                on="timestamp",
                how="inner",
                validate="one_to_one",
            ).rename(
                columns={
                    "forecast_pv_kw": "pv",
                    "forecast_load_kw": "load",
                    "safe_residual_kw": "discharge_limit_kw",
                }
            )
            replan_initial_soc = current_soc
            replan_parameters = replace(
                parameters,
                initial_soc_kwh=replan_initial_soc,
                terminal_soc_kwh=TERMINAL_SOC_REFERENCE_KWH,
            )
            solved = solver(
                milp_input,
                solver_log_dir / f"{name}_{cadence_minutes}min.log",
                replan_parameters,
                mip_relative_gap=mip_relative_gap,
                log_to_console=False,
                terminal_band=terminal_band,
            )
            status = str(solved.solver_metadata.get("solver_status"))
            if status != "Optimal":
                raise RuntimeError(f"{name} {control_time} did not solve optimally: {status}.")

            execution_end = control_time + pd.Timedelta(minutes=cadence_minutes)
            schedule = solved.dispatch.loc[
                solved.dispatch["timestamp"] < execution_end
            ].copy()
            if len(schedule) != execution_intervals:
                raise ValueError(
                    f"Controller must execute {execution_intervals} five-minute actions."
                )

            # Future actual PV/load are accessed only after the solve is complete.
            realized_block = realized.loc[
                (realized["timestamp"] >= control_time)
                & (realized["timestamp"] < execution_end),
                ["timestamp", "pv", "load", "price"],
            ].copy()
            replay = replay_day(
                schedule,
                realized_block,
                replan_initial_soc,
                replan_parameters,
            )
            execution_forecast = remaining.loc[
                remaining["timestamp"] < execution_end
            ].copy()
            replay = replay.merge(
                execution_forecast,
                on="timestamp",
                how="left",
                validate="one_to_one",
            )
            replay.insert(0, "strategy", name)
            replay["controller"] = "controller_v2"
            replay["issue_time"] = day
            replay["replan_time"] = control_time
            replay["cadence_minutes"] = cadence_minutes
            replay["terminal_soc_reference_kwh"] = TERMINAL_SOC_REFERENCE_KWH
            replay["terminal_band_lower_kwh"] = terminal_band.lower_kwh
            replay["terminal_band_upper_kwh"] = terminal_band.upper_kwh
            replay["terminal_recovery_active"] = terminal_recovery_active
            replay["q10_discharge_limit_enabled"] = use_q10_discharge_limit
            replay["terminal_recovery_charge_ban_enabled"] = (
                use_terminal_recovery_charge_ban
            )
            replay["final_day_immediate_charge_guard_enabled"] = (
                use_final_day_immediate_charge_guard
            )
            replay["final_day_immediate_charge_guard_applied"] = guard_applied
            replay["charge_limit_kw"] = remaining["charge_limit_kw"].iloc[0]
            replay["planned_terminal_deviation_negative_kwh"] = float(
                solved.solver_metadata["terminal_deviation_negative_kwh"]
            )
            replay["planned_terminal_deviation_positive_kwh"] = float(
                solved.solver_metadata["terminal_deviation_positive_kwh"]
            )
            planned_terminal_soc = float(solved.dispatch["soc_kwh"].iloc[-1])
            replay["planned_terminal_soc_kwh"] = planned_terminal_soc
            replay["safety_filter_applied"] = True
            replay["counterfactual_provisional"] = True
            replay["validation_period_demo"] = True
            replay_parts.append(replay)
            day_replays.append(replay)

            realized_residual = (
                replay["realized_load_kw"] - replay["realized_pv_kw"]
            ).to_numpy(dtype=float)
            forecast_residual = (
                replay["forecast_load_kw"] - replay["forecast_pv_kw"]
            ).to_numpy(dtype=float)
            day_residual_errors.extend(
                zip(
                    replay["timestamp"].tolist(),
                    (realized_residual - forecast_residual).tolist(),
                )
            )
            current_soc = float(replay["realized_soc_end_kwh"].iloc[-1])

            gap = float(solved.solver_metadata.get("optimality_gap") or 0.0)
            dual_bound = solved.solver_metadata.get("mip_dual_bound")
            absolute_gap = (
                abs(float(solved.solver_objective_yuan) - float(dual_bound))
                if dual_bound is not None
                else 0.0
            )
            runtime = float(
                solved.solver_metadata.get("wall_clock_runtime_seconds") or 0.0
            )
            negative_slack = float(
                solved.solver_metadata["terminal_deviation_negative_kwh"]
            )
            positive_slack = float(
                solved.solver_metadata["terminal_deviation_positive_kwh"]
            )
            statuses.append(status)
            gaps.append(gap)
            absolute_gaps.append(absolute_gap)
            runtimes.append(runtime)
            planned_negative_slacks.append(negative_slack)
            planned_positive_slacks.append(positive_slack)
            planned_terminal_socs.append(planned_terminal_soc)
            replans.append(
                {
                    "strategy": name,
                    "control_time": control_time.isoformat(),
                    "forecast_frozen_at": day.isoformat(),
                    "cadence_minutes": cadence_minutes,
                    "horizon_intervals": len(milp_input),
                    "executed_intervals": execution_intervals,
                    "initial_soc_kwh": replan_initial_soc,
                    "terminal_soc_reference_kwh": TERMINAL_SOC_REFERENCE_KWH,
                    "terminal_band_lower_kwh": terminal_band.lower_kwh,
                    "terminal_band_upper_kwh": terminal_band.upper_kwh,
                    "planned_terminal_soc_kwh": planned_terminal_soc,
                    "terminal_deviation_negative_kwh": negative_slack,
                    "terminal_deviation_positive_kwh": positive_slack,
                    "terminal_deviation_penalty_yuan": float(
                        solved.solver_metadata["terminal_deviation_penalty_yuan"]
                    ),
                    "historical_error_sample_count": sample_count,
                    "residual_error_q10_kw": q10_error,
                    "applied_residual_error_q10_kw": applied_q10_error,
                    "q10_discharge_limit_enabled": use_q10_discharge_limit,
                    "q10_quantile": Q10_QUANTILE,
                    "q10_quantile_method": Q10_QUANTILE_METHOD,
                    "q10_fallback_used": q10_fallback_used,
                    "q10_history_scope": "completed_previous_days",
                    "terminal_recovery_active": terminal_recovery_active,
                    "terminal_recovery_charge_ban_enabled": (
                        use_terminal_recovery_charge_ban
                    ),
                    "latest_completed_residual_first_step_enabled": (
                        use_latest_completed_residual_for_first_step
                    ),
                    "intraday_load_bias_correction_enabled": (
                        use_intraday_load_bias_correction
                    ),
                    "final_day_immediate_charge_guard_enabled": (
                        use_final_day_immediate_charge_guard
                    ),
                    "final_day_immediate_charge_guard_applied": guard_applied,
                    "immediate_charge_limit_kw": immediate_charge_limit_kw,
                    "future_charge_limits_unrestricted": (
                        future_charge_limits_unrestricted
                    ),
                    "intraday_load_bias_kw": intraday_bias_kw,
                    "intraday_load_bias_sample_count": (
                        intraday_bias_sample_count
                    ),
                    "intraday_load_bias_oldest_timestamp": (
                        intraday_bias_oldest_timestamp.isoformat()
                        if intraday_bias_oldest_timestamp is not None
                        else None
                    ),
                    "intraday_load_bias_newest_timestamp": (
                        intraday_bias_newest_timestamp.isoformat()
                        if intraday_bias_newest_timestamp is not None
                        else None
                    ),
                    "first_step_frozen_load_forecast_kw": float(
                        remaining["frozen_forecast_load_kw"].iloc[0]
                    ),
                    "first_step_frozen_pv_forecast_kw": float(
                        day_forecast.loc[
                            day_forecast["timestamp"].eq(control_time),
                            "forecast_pv_kw",
                        ].iloc[0]
                    ),
                    "first_step_adjusted_load_forecast_kw": float(
                        remaining["adjusted_forecast_load_kw"].iloc[0]
                    ),
                    "first_step_adjusted_residual_forecast_kw": float(
                        remaining["adjusted_residual_forecast_kw"].iloc[0]
                    ),
                    "first_step_residual_source_timestamp": (
                        first_step_source_timestamp.isoformat()
                        if first_step_source_timestamp is not None
                        else None
                    ),
                    "first_step_measured_residual_kw": first_step_residual_kw,
                    "charge_limit_kw": float(remaining["charge_limit_kw"].iloc[0]),
                    "future_realized_pv_or_load_passed": False,
                    "known_future_tariff_passed": True,
                    "solver_status": status,
                    "solver_relative_gap": gap,
                    "solver_absolute_gap_yuan": absolute_gap,
                    "solver_runtime_seconds": runtime,
                    "realized_soc_after_execution_kwh": current_soc,
                }
            )

        residual_error_history.extend(day_residual_errors)
        day_replay = pd.concat(day_replays, ignore_index=True)
        realized_negative_slack = max(terminal_band.lower_kwh - current_soc, 0.0)
        realized_positive_slack = max(current_soc - terminal_band.upper_kwh, 0.0)
        daily_runs.append(
            {
                "date": day.date().isoformat(),
                "initial_soc_kwh": day_start_soc,
                "realized_terminal_soc_kwh": current_soc,
                "terminal_soc_reference_kwh": TERMINAL_SOC_REFERENCE_KWH,
                "terminal_band_lower_kwh": terminal_band.lower_kwh,
                "terminal_band_upper_kwh": terminal_band.upper_kwh,
                "planned_terminal_soc_kwh": planned_terminal_socs[-1],
                "planned_terminal_deviation_negative_kwh": (
                    planned_negative_slacks[-1]
                ),
                "planned_terminal_deviation_positive_kwh": (
                    planned_positive_slacks[-1]
                ),
                "realized_terminal_deviation_negative_kwh": (
                    realized_negative_slack
                ),
                "realized_terminal_deviation_positive_kwh": (
                    realized_positive_slack
                ),
                "maximum_planned_terminal_deviation_kwh": max(
                    np.asarray(planned_negative_slacks)
                    + np.asarray(planned_positive_slacks)
                ),
                "terminal_recovery_replans": sum(
                    bool(row["terminal_recovery_active"])
                    for row in replans[-len(control_times) :]
                ),
                "charging_disabled_intervals": int(
                    (day_replay["charge_limit_kw"] <= 1e-7).sum()
                ),
                "clipped_intervals": int(day_replay["was_clipped"].sum()),
                "clipped_energy_kwh": clipping_energy_kwh(
                    day_replay["total_clip_kw"], parameters.interval_hours
                ),
                "anti_export_clipped_intervals": int(
                    (day_replay["anti_export_clip_kw"] > 1e-7).sum()
                ),
                "anti_export_clipped_kwh": clipping_energy_kwh(
                    day_replay["anti_export_clip_kw"], parameters.interval_hours
                ),
                "upper_soc_clipped_intervals": int(
                    (day_replay["upper_soc_clip_kw"] > 1e-7).sum()
                ),
                "upper_soc_clipped_kwh": clipping_energy_kwh(
                    day_replay["upper_soc_clip_kw"], parameters.interval_hours
                ),
                "lower_soc_clipped_intervals": int(
                    (day_replay["lower_soc_clip_kw"] > 1e-7).sum()
                ),
                "lower_soc_clipped_kwh": clipping_energy_kwh(
                    day_replay["lower_soc_clip_kw"], parameters.interval_hours
                ),
                "historical_error_samples_at_day_end": len(residual_error_history),
                "historical_error_samples_at_day_start": sample_count,
                "residual_error_q10_kw": q10_error,
                "applied_residual_error_q10_kw": applied_q10_error,
                "q10_discharge_limit_enabled": use_q10_discharge_limit,
                "terminal_recovery_charge_ban_enabled": (
                    use_terminal_recovery_charge_ban
                ),
                "q10_quantile_method": Q10_QUANTILE_METHOD,
                "q10_fallback_used": q10_fallback_used,
                "replan_count": len(control_times),
                "solver_failure_count": sum(status != "Optimal" for status in statuses),
                "solver_status": sorted(set(statuses))[0],
                "maximum_solver_relative_gap": max(gaps),
                "maximum_solver_absolute_gap_yuan": max(absolute_gaps),
                "solver_runtime_seconds": sum(runtimes),
            }
        )
        if show_progress:
            print(
                f"Completed {name} through {day.date()} "
                f"({len(replans)} replans, SOC={current_soc:.2f} kWh)."
            )

    return (
        StrategyResult(
            replay=pd.concat(replay_parts, ignore_index=True),
            daily_runs=daily_runs,
        ),
        replans,
    )


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


def _physical_violations(
    replay: pd.DataFrame,
    parameters: DispatchParameters,
) -> dict[str, float]:
    minimum_soc = float(
        replay[["realized_soc_start_kwh", "realized_soc_end_kwh"]].min().min()
    )
    maximum_soc = float(
        replay[["realized_soc_start_kwh", "realized_soc_end_kwh"]].max().max()
    )
    residual_load = np.maximum(
        replay["realized_load_kw"] - replay["realized_pv_kw"], 0.0
    )
    continuity = np.abs(
        replay["realized_soc_start_kwh"].iloc[1:].to_numpy(dtype=float)
        - replay["realized_soc_end_kwh"].iloc[:-1].to_numpy(dtype=float)
    )
    values = {
        "soc_lower_violation_kwh": max(0.0, -minimum_soc),
        "soc_upper_violation_kwh": max(
            0.0, maximum_soc - parameters.capacity_kwh
        ),
        "charge_power_violation_kw": max(
            0.0, float(replay["applied_charge_kw"].max()) - parameters.power_limit_kw
        ),
        "discharge_power_violation_kw": max(
            0.0,
            float(replay["applied_discharge_kw"].max()) - parameters.power_limit_kw,
        ),
        "simultaneous_charge_discharge_kw": float(
            np.minimum(
                replay["applied_charge_kw"], replay["applied_discharge_kw"]
            ).max()
        ),
        "anti_export_violation_kw": float(
            np.maximum(replay["applied_discharge_kw"] - residual_load, 0.0).max()
        ),
        "grid_import_export_overlap_kw": float(
            np.minimum(replay["grid_import_kw"], replay["grid_export_kw"]).max()
        ),
        "soc_continuity_violation_kwh": float(continuity.max()),
    }
    if "safe_residual_kw" in replay:
        values["safe_discharge_limit_violation_kw"] = float(
            np.maximum(
                replay["scheduled_discharge_kw"] - replay["safe_residual_kw"], 0.0
            ).max()
        )
    values["maximum_constraint_violation"] = max(values.values())
    return values


def _terminal_slack_for_soc(
    soc_kwh: float,
    day: pd.Timestamp,
) -> tuple[float, float]:
    band = terminal_band_for_day(day)
    return max(band.lower_kwh - soc_kwh, 0.0), max(soc_kwh - band.upper_kwh, 0.0)


def build_daily_summary(
    results: dict[str, StrategyResult],
    parameters: DispatchParameters,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        replay = result.replay.assign(date=result.replay["timestamp"].dt.normalize())
        planned_by_day = {row["date"]: row for row in result.daily_runs}
        for day, table in replay.groupby("date", sort=True):
            accounting = recalculate_objective(_accounting_frame(table), parameters)
            final_soc = float(table["realized_soc_end_kwh"].iloc[-1])
            negative, positive = _terminal_slack_for_soc(final_soc, day)
            planned = planned_by_day.get(day.date().isoformat(), {})
            rows.append(
                {
                    "strategy": name,
                    "date": day.date().isoformat(),
                    "initial_soc_kwh": float(
                        table["realized_soc_start_kwh"].iloc[0]
                    ),
                    "final_soc_kwh": final_soc,
                    "planned_terminal_soc_kwh": planned.get(
                        "planned_terminal_soc_kwh"
                    ),
                    "planned_terminal_deviation_negative_kwh": planned.get(
                        "planned_terminal_deviation_negative_kwh"
                    ),
                    "planned_terminal_deviation_positive_kwh": planned.get(
                        "planned_terminal_deviation_positive_kwh"
                    ),
                    "realized_terminal_deviation_negative_kwh": negative,
                    "realized_terminal_deviation_positive_kwh": positive,
                    "planned_charge_kwh": float(
                        table["scheduled_charge_kw"].sum()
                        * parameters.interval_hours
                    ),
                    "planned_discharge_kwh": float(
                        table["scheduled_discharge_kw"].sum()
                        * parameters.interval_hours
                    ),
                    "executed_charge_kwh": float(
                        table["applied_charge_kw"].sum()
                        * parameters.interval_hours
                    ),
                    "executed_discharge_kwh": float(
                        table["applied_discharge_kw"].sum()
                        * parameters.interval_hours
                    ),
                    "clipped_intervals": int(table["was_clipped"].sum()),
                    "clipped_energy_kwh": clipping_energy_kwh(
                        table["total_clip_kw"], parameters.interval_hours
                    ),
                    "anti_export_clipped_intervals": int(
                        (table["anti_export_clip_kw"] > 1e-7).sum()
                    ),
                    "anti_export_clipped_kwh": clipping_energy_kwh(
                        table["anti_export_clip_kw"], parameters.interval_hours
                    ),
                    "upper_soc_clipped_intervals": int(
                        (table["upper_soc_clip_kw"] > 1e-7).sum()
                    ),
                    "upper_soc_clipped_kwh": clipping_energy_kwh(
                        table["upper_soc_clip_kw"], parameters.interval_hours
                    ),
                    "lower_soc_clipped_intervals": int(
                        (table["lower_soc_clip_kw"] > 1e-7).sum()
                    ),
                    "lower_soc_clipped_kwh": clipping_energy_kwh(
                        table["lower_soc_clip_kw"], parameters.interval_hours
                    ),
                    "solver_replans": planned.get("replan_count"),
                    "solver_failures": planned.get("solver_failure_count"),
                    "solver_runtime_seconds": planned.get(
                        "solver_runtime_seconds"
                    ),
                    "residual_error_q10_kw": planned.get(
                        "residual_error_q10_kw"
                    ),
                    "applied_residual_error_q10_kw": planned.get(
                        "applied_residual_error_q10_kw"
                    ),
                    "terminal_recovery_replans": planned.get(
                        "terminal_recovery_replans", 0
                    ),
                    "charging_disabled_intervals": planned.get(
                        "charging_disabled_intervals", 0
                    ),
                    **accounting,
                }
            )
    return pd.DataFrame(rows)


def build_strategy_summary(
    results: dict[str, StrategyResult],
    daily: pd.DataFrame,
    parameters: DispatchParameters,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for name, result in results.items():
        replay = result.replay
        accounting = recalculate_objective(_accounting_frame(replay), parameters)
        strategy_daily = daily.loc[daily["strategy"].eq(name)]
        violations = _physical_violations(replay, parameters)
        audits[name] = violations
        labels = STRATEGY_LABELS[name]
        has_solver_metadata = any(
            "solver_runtime_seconds" in row for row in result.daily_runs
        )
        rows.append(
            {
                "strategy": name,
                **labels,
                "revenue_status": "counterfactual_provisional",
                "validation_period_demo": True,
                "timestamp_count": len(replay),
                "initial_soc_kwh": float(
                    replay["realized_soc_start_kwh"].iloc[0]
                ),
                "final_soc_kwh": float(replay["realized_soc_end_kwh"].iloc[-1]),
                "planned_charge_kwh": float(
                    replay["scheduled_charge_kw"].sum() * parameters.interval_hours
                ),
                "planned_discharge_kwh": float(
                    replay["scheduled_discharge_kw"].sum()
                    * parameters.interval_hours
                ),
                "executed_charge_kwh": float(
                    replay["applied_charge_kw"].sum() * parameters.interval_hours
                ),
                "executed_discharge_kwh": float(
                    replay["applied_discharge_kw"].sum()
                    * parameters.interval_hours
                ),
                "clipped_intervals": int(replay["was_clipped"].sum()),
                "clipped_energy_kwh": clipping_energy_kwh(
                    replay["total_clip_kw"], parameters.interval_hours
                ),
                "anti_export_clipped_intervals": int(
                    (replay["anti_export_clip_kw"] > 1e-7).sum()
                ),
                "anti_export_clipped_fraction": float(
                    (replay["anti_export_clip_kw"] > 1e-7).mean()
                ),
                "anti_export_clipped_kwh": clipping_energy_kwh(
                    replay["anti_export_clip_kw"], parameters.interval_hours
                ),
                "anti_export_clipped_kwh_per_planned_discharge_kwh": (
                    clipping_energy_kwh(
                        replay["anti_export_clip_kw"], parameters.interval_hours
                    )
                    / (
                        float(replay["scheduled_discharge_kw"].sum())
                        * parameters.interval_hours
                    )
                )
                if float(replay["scheduled_discharge_kw"].sum()) > 0.0
                else 0.0,
                "upper_soc_clipped_intervals": int(
                    (replay["upper_soc_clip_kw"] > 1e-7).sum()
                ),
                "upper_soc_clipped_kwh": clipping_energy_kwh(
                    replay["upper_soc_clip_kw"], parameters.interval_hours
                ),
                "lower_soc_clipped_intervals": int(
                    (replay["lower_soc_clip_kw"] > 1e-7).sum()
                ),
                "lower_soc_clipped_kwh": clipping_energy_kwh(
                    replay["lower_soc_clip_kw"], parameters.interval_hours
                ),
                "total_realized_terminal_slack_kwh": float(
                    strategy_daily[
                        [
                            "realized_terminal_deviation_negative_kwh",
                            "realized_terminal_deviation_positive_kwh",
                        ]
                    ].to_numpy(dtype=float).sum()
                ),
                "may31_realized_terminal_slack_kwh": float(
                    strategy_daily.iloc[-1][
                        "realized_terminal_deviation_negative_kwh"
                    ]
                    + strategy_daily.iloc[-1][
                        "realized_terminal_deviation_positive_kwh"
                    ]
                ),
                "may31_realized_slack_below_kwh": float(
                    strategy_daily.iloc[-1][
                        "realized_terminal_deviation_negative_kwh"
                    ]
                ),
                "may31_realized_slack_above_kwh": float(
                    strategy_daily.iloc[-1][
                        "realized_terminal_deviation_positive_kwh"
                    ]
                ),
                "total_planned_terminal_slack_kwh": float(
                    strategy_daily[
                        [
                            "planned_terminal_deviation_negative_kwh",
                            "planned_terminal_deviation_positive_kwh",
                        ]
                    ].fillna(0.0).to_numpy(dtype=float).sum()
                ),
                "may31_planned_terminal_slack_kwh": float(
                    strategy_daily.iloc[-1][
                        "planned_terminal_deviation_negative_kwh"
                    ]
                    + strategy_daily.iloc[-1][
                        "planned_terminal_deviation_positive_kwh"
                    ]
                )
                if strategy_daily.iloc[-1][
                    [
                        "planned_terminal_deviation_negative_kwh",
                        "planned_terminal_deviation_positive_kwh",
                    ]
                ].notna().all()
                else 0.0,
                "may31_planned_slack_below_kwh": (
                    float(
                        strategy_daily.iloc[-1][
                            "planned_terminal_deviation_negative_kwh"
                        ]
                    )
                    if pd.notna(
                        strategy_daily.iloc[-1][
                            "planned_terminal_deviation_negative_kwh"
                        ]
                    )
                    else None
                ),
                "may31_planned_slack_above_kwh": (
                    float(
                        strategy_daily.iloc[-1][
                            "planned_terminal_deviation_positive_kwh"
                        ]
                    )
                    if pd.notna(
                        strategy_daily.iloc[-1][
                            "planned_terminal_deviation_positive_kwh"
                        ]
                    )
                    else None
                ),
                "may31_planned_terminal_soc_kwh": (
                    float(strategy_daily.iloc[-1]["planned_terminal_soc_kwh"])
                    if pd.notna(
                        strategy_daily.iloc[-1]["planned_terminal_soc_kwh"]
                    )
                    else None
                ),
                "terminal_recovery_replans": int(
                    replay.loc[
                        replay.get(
                            "terminal_recovery_active",
                            pd.Series(False, index=replay.index),
                        ).astype(bool),
                        "replan_time",
                    ].nunique()
                )
                if "replan_time" in replay
                else 0,
                "charging_disabled_intervals": int(
                    (replay.get("charge_limit_kw", parameters.power_limit_kw) <= 1e-7)
                    .sum()
                )
                if "charge_limit_kw" in replay
                else 0,
                "solver_replans": (
                    sum(int(row.get("replan_count") or 0) for row in result.daily_runs)
                    if has_solver_metadata
                    else None
                ),
                "solver_failures": (
                    sum(
                        int(row.get("solver_failure_count") or 0)
                        for row in result.daily_runs
                    )
                    if has_solver_metadata
                    else None
                ),
                "solver_runtime_seconds": (
                    sum(
                        float(row.get("solver_runtime_seconds") or 0.0)
                        for row in result.daily_runs
                    )
                    if has_solver_metadata
                    else None
                ),
                "maximum_constraint_violation": violations[
                    "maximum_constraint_violation"
                ],
                **accounting,
            }
        )
    return pd.DataFrame(rows), audits


def compare_v2(summary: pd.DataFrame) -> dict[str, Any]:
    indexed = summary.set_index("strategy")
    pairs = {
        "previous_day": (
            "feedback_previous_day_pv_previous_day_load",
            "controller_v2_previous_day_pv_previous_day_load",
        ),
        "chronos": (
            "feedback_chronos_pv_previous_day_load",
            "controller_v2_chronos_pv_previous_day_load",
        ),
    }
    comparisons: dict[str, Any] = {}
    optional_five_minute = False
    for label, (old_name, new_name) in pairs.items():
        old = indexed.loc[old_name]
        new = indexed.loc[new_name]
        anti_fraction = float(new["anti_export_clipped_fraction"])
        anti_energy_fraction = float(
            new["anti_export_clipped_kwh_per_planned_discharge_kwh"]
        )
        optional_five_minute = optional_five_minute or (
            anti_fraction > 0.10 or anti_energy_fraction > 0.10
        )
        comparisons[label] = {
            "old_strategy": old_name,
            "controller_v2_strategy": new_name,
            "revenue_change_yuan": float(
                new["objective_yuan"] - old["objective_yuan"]
            ),
            "final_soc_change_kwh": float(new["final_soc_kwh"] - old["final_soc_kwh"]),
            "anti_export_clipped_energy_change_kwh": float(
                new["anti_export_clipped_kwh"] - old["anti_export_clipped_kwh"]
            ),
            "anti_export_clipped_fraction": anti_fraction,
            "anti_export_clipped_kwh_per_planned_discharge_kwh": (
                anti_energy_fraction
            ),
            "old_revenue_yuan": float(old["objective_yuan"]),
            "controller_v2_revenue_yuan": float(new["objective_yuan"]),
            "old_final_soc_kwh": float(old["final_soc_kwh"]),
            "controller_v2_final_soc_kwh": float(new["final_soc_kwh"]),
            "old_solver_failures": int(old["solver_failures"]),
            "controller_v2_solver_failures": int(new["solver_failures"]),
            "old_solver_runtime_seconds": float(old["solver_runtime_seconds"]),
            "controller_v2_solver_runtime_seconds": float(
                new["solver_runtime_seconds"]
            ),
            "final_soc_within_900_plus_minus_50": bool(
                abs(float(new["final_soc_kwh"]) - 900.0) <= 50.0
            ),
            "final_soc_within_900_plus_minus_5": bool(
                abs(float(new["final_soc_kwh"]) - 900.0) <= 5.0
            ),
            "zero_anti_export_violation": bool(
                float(new["maximum_constraint_violation"]) <= 1e-7
            ),
        }
    return {
        "comparisons": comparisons,
        "optional_five_minute_experiment_recommended": optional_five_minute,
        "recommendation_threshold": (
            "anti-export clipped intervals / total intervals > 10% OR "
            "anti-export clipped kWh / planned discharge kWh > 10%"
        ),
        "terminal_energy_value_adjustment_yuan": 0.0,
    }


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    comparison: dict[str, Any],
    cadence_minutes: int,
) -> None:
    def display(value: Any, digits: int = 1) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"

    lines = [
        "# Foshan Controller V2",
        "",
        f"**Status:** {FORECAST_NOTICE}",
        "",
        f"**Evaluation warning:** {VALIDATION_NOTICE}",
        "",
        f"**Load caveat:** {LOAD_DATA_NOTICE}",
        "",
        f"Controller cadence: {cadence_minutes} minutes.",
        "",
        "Terminal SOC reference is fixed at 900 kWh. Daily bands are 850-950 kWh; "
        "May 31 uses 895-905 kWh. Terminal deviation outside the band costs 1.0 "
        "yuan/kWh in the optimization objective but is not added to reported revenue.",
        "",
        "Residual-error q10 uses NumPy's linear empirical quantile over completed "
        "previous days and remains frozen within each day. May 2 uses the recorded "
        "zero-sample q10=0 fallback because April 30 load is unavailable.",
        "",
        "## Comparison",
        "",
        (
            "| Strategy | Revenue | Initial / final SOC | Planned C/D | Executed C/D | "
            "Anti-export clips | Upper / lower SOC clips | "
            "May 31 planned / realized SOC |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.objective_yuan:.2f} | "
            f"{row.initial_soc_kwh:.1f} / {row.final_soc_kwh:.1f} | "
            f"{row.planned_charge_kwh:.1f} / {row.planned_discharge_kwh:.1f} | "
            f"{row.executed_charge_kwh:.1f} / {row.executed_discharge_kwh:.1f} | "
            f"{row.anti_export_clipped_intervals} / "
            f"{row.anti_export_clipped_kwh:.1f} kWh | "
            f"{row.upper_soc_clipped_intervals} / {row.lower_soc_clipped_intervals}; "
            f"{row.upper_soc_clipped_kwh:.1f} / {row.lower_soc_clipped_kwh:.1f} kWh | "
            f"{display(row.may31_planned_terminal_soc_kwh)} / "
            f"{row.final_soc_kwh:.1f} kWh |"
        )
    lines.extend(
        [
            "",
            "## Terminal Slack",
            "",
            "| Strategy | Planned below / above | Realized below / above |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {display(row.may31_planned_slack_below_kwh)} / "
            f"{display(row.may31_planned_slack_above_kwh)} kWh | "
            f"{row.may31_realized_slack_below_kwh:.1f} / "
            f"{row.may31_realized_slack_above_kwh:.1f} kWh |"
        )
    lines.extend(
        [
            "",
            "## Solver",
            "",
            "| Strategy | Replans | Failures | Runtime (seconds) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {display(row.solver_replans, 0)} | "
            f"{display(row.solver_failures, 0)} | "
            f"{display(row.solver_runtime_seconds, 2)} |"
        )
    lines.extend(
        [
            "",
            "## Five-Minute Follow-Up",
            "",
            (
                "Optional five-minute experiment recommended: "
                f"{comparison['optional_five_minute_experiment_recommended']}."
            ),
            "This result never changes the recorded controller cadence silently.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Controller_v2 outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace only this directory."
        )


def run_backtest(
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    old_feedback_replay_path: Path,
    output_dir: Path,
    *,
    initial_soc_kwh: float = 900.0,
    cadence_minutes: int = 15,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_started = time.perf_counter()
    if not np.isclose(initial_soc_kwh, TERMINAL_SOC_REFERENCE_KWH):
        raise ValueError("Controller_v2 benchmark requires initial_soc_kwh=900.")
    _ensure_output_available(output_dir, overwrite)
    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)
    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path
    )
    previous_pv = previous_day_forecast(realized, "pv", "forecast_pv_kw")
    previous_load = previous_day_forecast(realized, "load", "forecast_load_kw")
    old_results = load_old_results(old_feedback_replay_path, parameters)

    with tempfile.TemporaryDirectory(prefix="foshan_controller_v2_highs_") as temporary:
        log_dir = Path(temporary)
        previous_result, previous_replans = run_controller_v2(
            "controller_v2_previous_day_pv_previous_day_load",
            realized,
            previous_pv,
            previous_load,
            initial_soc_kwh,
            log_dir,
            cadence_minutes=cadence_minutes,
            mip_relative_gap=mip_relative_gap,
            show_progress=show_progress,
        )
        chronos_result, chronos_replans = run_controller_v2(
            "controller_v2_chronos_pv_previous_day_load",
            realized,
            chronos_pv,
            previous_load,
            initial_soc_kwh,
            log_dir,
            cadence_minutes=cadence_minutes,
            mip_relative_gap=mip_relative_gap,
            show_progress=show_progress,
        )

    results = {
        "fixed_actual_pv_actual_load_oracle": old_results[
            "fixed_actual_pv_actual_load_oracle"
        ],
        "feedback_previous_day_pv_previous_day_load": old_results[
            "feedback_previous_day_pv_previous_day_load"
        ],
        "controller_v2_previous_day_pv_previous_day_load": previous_result,
        "feedback_chronos_pv_previous_day_load": old_results[
            "feedback_chronos_pv_previous_day_load"
        ],
        "controller_v2_chronos_pv_previous_day_load": chronos_result,
    }
    common_timestamps = validate_common_timestamps(results)
    initial_states = {
        name: float(result.replay["realized_soc_start_kwh"].iloc[0])
        for name, result in results.items()
    }
    if any(not np.isclose(value, 900.0) for value in initial_states.values()):
        raise ValueError(f"Strategies do not share initial SOC: {initial_states}")

    replay = pd.concat([result.replay for result in results.values()], ignore_index=True)
    daily = build_daily_summary(results, parameters)
    summary, strategy_audits = build_strategy_summary(results, daily, parameters)
    comparison = compare_v2(summary)
    audit = {
        "common_timestamp_count": len(common_timestamps),
        "common_timestamp_start": common_timestamps[0].isoformat(),
        "common_timestamp_end": common_timestamps[-1].isoformat(),
        "identical_timestamp_sets": True,
        "identical_initial_soc": True,
        "shared_initial_soc_kwh": initial_soc_kwh,
        "terminal_soc_reference_kwh": TERMINAL_SOC_REFERENCE_KWH,
        "daily_terminal_band_kwh": [
            DAILY_TERMINAL_LOWER_KWH,
            DAILY_TERMINAL_UPPER_KWH,
        ],
        "final_terminal_band_kwh": [
            FINAL_TERMINAL_LOWER_KWH,
            FINAL_TERMINAL_UPPER_KWH,
        ],
        "terminal_deviation_penalty_yuan_per_kwh": (
            TERMINAL_DEVIATION_PENALTY
        ),
        "clipping_energy_formula": "sum(clip_kw * (1/12))",
        "leakage_controls": {
            "forecast_frozen_for_day": True,
            "future_realized_pv_or_load_passed": False,
            "residual_error_history_completed_previous_days_only": True,
            "q10_frozen_within_day": True,
            "q10_quantile": Q10_QUANTILE,
            "q10_quantile_method": Q10_QUANTILE_METHOD,
            "known_future_tariff_passed": True,
            "real_time_anti_export_filter_retained": True,
            "terminal_recovery_uses_realized_soc_only": True,
        },
        "strategies": strategy_audits,
    }
    controller_metadata = {
        "cadence_minutes": cadence_minutes,
        "residual_error_policy": {
            "history_scope": "completed_previous_days",
            "q10_quantile": Q10_QUANTILE,
            "q10_quantile_method": Q10_QUANTILE_METHOD,
            "q10_frozen_within_day": True,
            "may2_prior_day_seed_count": 0,
            "may2_fallback_used": True,
            "may2_fallback_reason": (
                "May 1 residual errors require an April 30 previous-day load "
                "forecast, but the reference input begins on May 1."
            ),
        },
        "chronos_forecast": chronos_metadata,
        "previous_day_replans": previous_replans,
        "chronos_replans": chronos_replans,
        "provenance": {
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "dispatch_path": str(dispatch_path.resolve()),
            "dispatch_sha256": _sha256(dispatch_path),
            "predictions_path": str(predictions_path.resolve()),
            "predictions_sha256": _sha256(predictions_path),
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": _sha256(selection_path),
            "old_feedback_replay_path": str(old_feedback_replay_path.resolve()),
            "old_feedback_replay_sha256": _sha256(old_feedback_replay_path),
        },
        "wall_clock_runtime_seconds": time.perf_counter() - run_started,
    }

    replay.to_csv(output_dir / "replay_timeseries.csv", index=False, float_format="%.15g")
    summary.to_csv(output_dir / "strategy_summary.csv", index=False, float_format="%.15g")
    daily.to_csv(output_dir / "daily_summary.csv", index=False, float_format="%.15g")
    _write_json(output_dir / "comparison.json", comparison)
    _write_json(output_dir / "constraint_audit.json", audit)
    _write_json(output_dir / "controller_metadata.json", controller_metadata)
    _write_report(output_dir / "report.md", summary, comparison, cadence_minutes)
    return replay, summary, daily, comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe Foshan controller_v2 with HiGHS."
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
        "--old-feedback-replay",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_state_feedback/replay_timeseries.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_controller_v2_daily_frozen_q10"
        ),
    )
    parser.add_argument("--initial-soc-kwh", type=float, default=900.0)
    parser.add_argument("--cadence-minutes", type=int, choices=(5, 15), default=15)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary, _, comparison = run_backtest(
        dispatch_path=args.dispatch_input,
        predictions_path=args.predictions,
        selection_path=args.selection,
        old_feedback_replay_path=args.old_feedback_replay,
        output_dir=args.output_dir,
        initial_soc_kwh=args.initial_soc_kwh,
        cadence_minutes=args.cadence_minutes,
        mip_relative_gap=args.mip_relative_gap,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    print(
        f"Saved controller_v2 to {args.output_dir.resolve()} for {len(summary)} "
        f"strategies; five_minute_recommended="
        f"{comparison['optional_five_minute_experiment_recommended']}."
    )


if __name__ == "__main__":
    main()
