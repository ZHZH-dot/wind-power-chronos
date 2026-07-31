"""Diagnose fixed dispatch and run causal 15-minute state-feedback control."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
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
from src.optimization.foshan_forecast_backtest import (
    END_EXCLUSIVE,
    FORECAST_NOTICE,
    START,
    VALIDATION_NOTICE,
    StrategyResult,
    audit_strategy,
    clipping_energy_kwh,
    load_reference_dispatch,
    load_selected_chronos_p50,
    perfect_forecast,
    previous_day_forecast,
    replay_day,
    run_daily_strategy,
    validate_common_timestamps,
)
from src.utils.runtime import git_commit, git_is_dirty


OUTPUT_FILENAMES = (
    "replay_timeseries.csv",
    "strategy_summary.csv",
    "daily_revenue.csv",
    "constraint_audit.json",
    "diagnosis.json",
    "controller_metadata.json",
    "report.md",
)
CONSTRAINT_VIOLATION_FIELDS = (
    "soc_lower_violation_kwh",
    "soc_upper_violation_kwh",
    "charge_power_violation_kw",
    "discharge_power_violation_kw",
    "simultaneous_charge_discharge_kw",
    "anti_export_violation_kw",
    "pv_export_limit_violation_kw",
    "soc_continuity_violation_kwh",
)
FIXED_RESULT_NAMES = {
    "daily_perfect_foresight_milp": "fixed_actual_pv_actual_load_oracle",
    "previous_day_pv_previous_day_load_milp": (
        "fixed_previous_day_pv_previous_day_load_reference"
    ),
    "chronos_zero_shot_pv_previous_day_load_milp": (
        "fixed_chronos_pv_previous_day_load"
    ),
}
STRATEGY_LABELS = {
    "fixed_actual_pv_actual_load_oracle": {
        "pv_source": "actual_future_pv",
        "load_source": "actual_future_provisional_load",
        "controller": "fixed_daily",
        "deployable": False,
        "oracle_diagnostic": True,
    },
    "fixed_chronos_pv_actual_load_oracle": {
        "pv_source": "chronos2_zero_shot_postprocessed_p50",
        "load_source": "actual_future_provisional_load",
        "controller": "fixed_daily",
        "deployable": False,
        "oracle_diagnostic": True,
    },
    "fixed_actual_pv_previous_day_load_oracle": {
        "pv_source": "actual_future_pv",
        "load_source": "previous_day_provisional_load",
        "controller": "fixed_daily",
        "deployable": False,
        "oracle_diagnostic": True,
    },
    "fixed_chronos_pv_previous_day_load": {
        "pv_source": "chronos2_zero_shot_postprocessed_p50",
        "load_source": "previous_day_provisional_load",
        "controller": "fixed_daily",
        "deployable": True,
        "oracle_diagnostic": False,
    },
    "fixed_previous_day_pv_previous_day_load_reference": {
        "pv_source": "previous_day_pv",
        "load_source": "previous_day_provisional_load",
        "controller": "fixed_daily",
        "deployable": True,
        "oracle_diagnostic": False,
    },
    "feedback_previous_day_pv_previous_day_load": {
        "pv_source": "previous_day_pv",
        "load_source": "previous_day_provisional_load",
        "controller": "receding_15min",
        "deployable": True,
        "oracle_diagnostic": False,
    },
    "feedback_chronos_pv_previous_day_load": {
        "pv_source": "chronos2_zero_shot_postprocessed_p50",
        "load_source": "previous_day_provisional_load",
        "controller": "receding_15min",
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


def _label_result(name: str, result: StrategyResult) -> StrategyResult:
    labels = STRATEGY_LABELS[name]
    replay = result.replay.copy()
    replay["strategy"] = name
    replay["pv_forecast_source"] = labels["pv_source"]
    replay["load_forecast_source"] = labels["load_source"]
    replay["controller"] = labels["controller"]
    replay["deployable"] = labels["deployable"]
    replay["oracle_diagnostic"] = labels["oracle_diagnostic"]
    return StrategyResult(replay=replay, daily_runs=result.daily_runs)


def load_existing_fixed_results(path: Path) -> dict[str, StrategyResult]:
    """Load immutable fixed schedules from the prior backtest output."""
    table = pd.read_csv(path, low_memory=False)
    table["timestamp"] = pd.to_datetime(table["timestamp"], errors="raise")
    results: dict[str, StrategyResult] = {}
    for source_name, output_name in FIXED_RESULT_NAMES.items():
        replay = table.loc[table["strategy"].eq(source_name)].copy()
        replay = replay.sort_values("timestamp").reset_index(drop=True)
        if replay.empty:
            raise ValueError(f"Existing fixed replay is missing {source_name}.")
        if not np.isclose(replay["realized_soc_start_kwh"].iloc[0], 900.0):
            raise ValueError(f"Existing fixed strategy {source_name} does not start at 900 kWh.")
        results[output_name] = _label_result(
            output_name, StrategyResult(replay=replay, daily_runs=[])
        )
    validate_common_timestamps(results)
    return results


def validate_causal_forecast_sources(
    pv_forecast: pd.DataFrame,
    load_forecast: pd.DataFrame,
) -> None:
    """Reject forecast provenance that would reveal future measured values."""
    if "pv_forecast_issue_time" in pv_forecast:
        issue = pd.to_datetime(pv_forecast["pv_forecast_issue_time"], errors="raise")
        target = pd.to_datetime(pv_forecast["timestamp"], errors="raise")
        if (issue > target).any():
            raise ValueError("PV forecast issue times must not follow their target times.")
        issue_per_day = pd.DataFrame({"day": target.dt.normalize(), "issue": issue}).groupby(
            "day"
        )["issue"].nunique()
        if (issue_per_day > 1).any():
            raise ValueError("The day PV forecast must remain frozen during feedback control.")
    for frame, column in (
        (pv_forecast, "forecast_pv_kw_source_timestamp"),
        (load_forecast, "forecast_load_kw_source_timestamp"),
    ):
        if column not in frame:
            continue
        source = pd.to_datetime(frame[column], errors="raise")
        target = pd.to_datetime(frame["timestamp"], errors="raise")
        if (source >= target).any():
            raise ValueError(f"{column} must be strictly historical.")


def _merged_forecast(
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
        raise ValueError("Feedback forecasts do not match the complete control grid.")
    return forecast


def validate_shared_initial_soc(
    results: dict[str, StrategyResult],
    expected_soc_kwh: float,
) -> dict[str, float]:
    initial_states = {
        name: float(result.replay["realized_soc_start_kwh"].iloc[0])
        for name, result in results.items()
    }
    if any(
        not np.isclose(value, expected_soc_kwh) for value in initial_states.values()
    ):
        raise ValueError(f"Strategies do not share initial SOC: {initial_states}")
    return initial_states


def run_feedback_strategy(
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
    show_progress: bool = False,
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    """Replan every 15 minutes and execute only the next three actions."""
    validate_causal_forecast_sources(pv_forecast, load_forecast)
    forecast = _merged_forecast(pv_forecast, load_forecast, start, end_exclusive)
    days = pd.date_range(start, end_exclusive, freq="1D", inclusive="left")
    replay_parts: list[pd.DataFrame] = []
    daily_runs: list[dict[str, Any]] = []
    replans: list[dict[str, Any]] = []
    current_soc = float(initial_soc_kwh)

    for day in days:
        day_end = day + pd.Timedelta(days=1)
        day_start_soc = current_soc
        day_forecast = forecast.loc[
            (forecast["timestamp"] >= day) & (forecast["timestamp"] < day_end)
        ].copy()
        day_tariff = realized.loc[
            (realized["timestamp"] >= day) & (realized["timestamp"] < day_end),
            ["timestamp", "price"],
        ].copy()
        if len(day_forecast) != 288 or len(day_tariff) != 288:
            raise ValueError(f"{day.date()} must contain 288 five-minute rows.")

        statuses: list[str] = []
        gaps: list[float] = []
        runtimes: list[float] = []
        last_terminal_target = current_soc
        day_replay_parts: list[pd.DataFrame] = []
        control_times = pd.date_range(day, day_end, freq="15min", inclusive="left")
        for control_time in control_times:
            remaining = day_forecast.loc[
                day_forecast["timestamp"] >= control_time
            ].copy()
            remaining_tariff = day_tariff.loc[
                day_tariff["timestamp"] >= control_time
            ].copy()
            milp_input = remaining[
                ["timestamp", "forecast_pv_kw", "forecast_load_kw"]
            ].merge(
                remaining_tariff,
                on="timestamp",
                how="inner",
                validate="one_to_one",
            ).rename(
                columns={
                    "forecast_pv_kw": "pv",
                    "forecast_load_kw": "load",
                }
            )
            replan_initial_soc = current_soc
            replan_parameters = replace(
                parameters,
                initial_soc_kwh=replan_initial_soc,
                terminal_soc_kwh=replan_initial_soc,
            )
            solved = solver(
                milp_input,
                solver_log_dir / f"{name}.log",
                replan_parameters,
                mip_relative_gap=mip_relative_gap,
                log_to_console=False,
            )
            status = str(solved.solver_metadata.get("solver_status"))
            if status != "Optimal":
                raise RuntimeError(f"{name} {control_time} did not solve optimally: {status}.")

            execution_end = control_time + pd.Timedelta(minutes=15)
            schedule = solved.dispatch.loc[
                solved.dispatch["timestamp"] < execution_end
            ].copy()
            if len(schedule) != 3:
                raise ValueError("Each feedback solve must execute exactly three actions.")

            # Realized PV/load are sliced only after the remaining-horizon solve.
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
            execution_forecast = day_forecast.loc[
                (day_forecast["timestamp"] >= control_time)
                & (day_forecast["timestamp"] < execution_end)
            ].copy()
            replay = replay.merge(
                execution_forecast,
                on="timestamp",
                how="left",
                validate="one_to_one",
            )
            replay.insert(0, "strategy", name)
            replay["issue_time"] = day
            replay["replan_time"] = control_time
            replay["safety_filter_applied"] = True
            replay["counterfactual_provisional"] = True
            replay["validation_period_demo"] = True
            replay_parts.append(replay)
            day_replay_parts.append(replay)

            current_soc = float(replay["realized_soc_end_kwh"].iloc[-1])
            last_terminal_target = replan_initial_soc
            gap = float(solved.solver_metadata.get("optimality_gap") or 0.0)
            runtime = float(
                solved.solver_metadata.get("wall_clock_runtime_seconds") or 0.0
            )
            statuses.append(status)
            gaps.append(gap)
            runtimes.append(runtime)
            replans.append(
                {
                    "strategy": name,
                    "control_time": control_time.isoformat(),
                    "forecast_frozen_at": day.isoformat(),
                    "horizon_intervals": len(milp_input),
                    "initial_soc_kwh": replan_initial_soc,
                    "terminal_target_kwh": replan_initial_soc,
                    "realized_soc_after_execution_kwh": current_soc,
                    "executed_intervals": 3,
                    "future_realized_pv_or_load_passed": False,
                    "known_future_tariff_passed": True,
                    "solver_status": status,
                    "solver_gap": gap,
                    "solver_runtime_seconds": runtime,
                }
            )

        day_replay = pd.concat(day_replay_parts, ignore_index=True)
        daily_runs.append(
            {
                "date": day.date().isoformat(),
                "initial_soc_kwh": day_start_soc,
                "scheduled_terminal_soc_kwh": last_terminal_target,
                "realized_terminal_soc_kwh": current_soc,
                "terminal_difference_kwh": current_soc - last_terminal_target,
                "clipped_intervals": int(day_replay["was_clipped"].sum()),
                "clipped_energy_kwh": clipping_energy_kwh(
                    day_replay["total_clip_kw"], parameters.interval_hours
                ),
                "replan_count": len(control_times),
                "solver_status": sorted(set(statuses))[0],
                "solver_runtime_seconds": sum(runtimes),
                "solver_gap": max(gaps),
            }
        )
        if show_progress:
            print(f"Completed {name} through {day.date()} ({len(replans)} replans).")

    result = StrategyResult(
        replay=pd.concat(replay_parts, ignore_index=True),
        daily_runs=daily_runs,
    )
    return _label_result(name, result), replans


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


def summarize_result(
    name: str,
    result: StrategyResult,
    parameters: DispatchParameters = DispatchParameters(),
) -> dict[str, Any]:
    replay = result.replay
    labels = STRATEGY_LABELS[name]
    accounting = recalculate_objective(_accounting_frame(replay), parameters)
    dt = parameters.interval_hours
    initial_soc = float(replay["realized_soc_start_kwh"].iloc[0])
    final_soc = float(replay["realized_soc_end_kwh"].iloc[-1])
    return {
        "strategy": name,
        "pv_source": labels["pv_source"],
        "load_source": labels["load_source"],
        "controller": labels["controller"],
        "deployable": labels["deployable"],
        "oracle_diagnostic": labels["oracle_diagnostic"],
        "revenue_status": "counterfactual_provisional",
        "validation_period_demo": True,
        "timestamp_count": len(replay),
        "initial_soc_kwh": initial_soc,
        "final_soc_kwh": final_soc,
        "final_soc_difference_from_initial_kwh": final_soc - initial_soc,
        "planned_charge_kwh": float(replay["scheduled_charge_kw"].sum() * dt),
        "planned_discharge_kwh": float(replay["scheduled_discharge_kw"].sum() * dt),
        "executed_charge_kwh": float(replay["applied_charge_kw"].sum() * dt),
        "executed_discharge_kwh": float(replay["applied_discharge_kw"].sum() * dt),
        "clipped_intervals": int(replay["was_clipped"].sum()),
        "clipped_energy_kwh": clipping_energy_kwh(replay["total_clip_kw"], dt),
        "power_clip_intervals": int((replay["power_clip_kw"] > 1e-7).sum()),
        "power_clipped_kwh": clipping_energy_kwh(replay["power_clip_kw"], dt),
        "soc_clip_intervals": int((replay["soc_clip_kw"] > 1e-7).sum()),
        "soc_clipped_kwh": clipping_energy_kwh(replay["soc_clip_kw"], dt),
        "anti_export_clip_intervals": int(
            (replay["anti_export_clip_kw"] > 1e-7).sum()
        ),
        "anti_export_clipped_kwh": clipping_energy_kwh(
            replay["anti_export_clip_kw"], dt
        ),
        **accounting,
    }


def daily_revenue_table(
    results: dict[str, StrategyResult],
    parameters: DispatchParameters = DispatchParameters(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        replay = result.replay.assign(date=result.replay["timestamp"].dt.date)
        for day, table in replay.groupby("date", sort=True):
            accounting = recalculate_objective(_accounting_frame(table), parameters)
            rows.append(
                {
                    "strategy": name,
                    "date": day.isoformat(),
                    "deployable": STRATEGY_LABELS[name]["deployable"],
                    "oracle_diagnostic": STRATEGY_LABELS[name]["oracle_diagnostic"],
                    "final_soc_kwh": float(table["realized_soc_end_kwh"].iloc[-1]),
                    "planned_charge_kwh": float(
                        table["scheduled_charge_kw"].sum() * parameters.interval_hours
                    ),
                    "planned_discharge_kwh": float(
                        table["scheduled_discharge_kw"].sum()
                        * parameters.interval_hours
                    ),
                    "executed_charge_kwh": float(
                        table["applied_charge_kw"].sum() * parameters.interval_hours
                    ),
                    "executed_discharge_kwh": float(
                        table["applied_discharge_kw"].sum()
                        * parameters.interval_hours
                    ),
                    "clipped_intervals": int(table["was_clipped"].sum()),
                    "clipped_energy_kwh": clipping_energy_kwh(
                        table["total_clip_kw"], parameters.interval_hours
                    ),
                    **accounting,
                }
            )
    return pd.DataFrame(rows)


def diagnose_bottleneck(summary: pd.DataFrame) -> dict[str, Any]:
    objective = summary.set_index("strategy")["objective_yuan"].to_dict()
    actual_actual = float(objective["fixed_actual_pv_actual_load_oracle"])
    chronos_actual = float(objective["fixed_chronos_pv_actual_load_oracle"])
    actual_previous = float(
        objective["fixed_actual_pv_previous_day_load_oracle"]
    )
    chronos_previous = float(objective["fixed_chronos_pv_previous_day_load"])
    feedback_chronos = float(
        objective["feedback_chronos_pv_previous_day_load"]
    )
    fixed_previous = float(
        objective["fixed_previous_day_pv_previous_day_load_reference"]
    )
    feedback_previous = float(
        objective["feedback_previous_day_pv_previous_day_load"]
    )

    pv_loss = actual_actual - chronos_actual
    load_loss = actual_actual - actual_previous
    combined_loss = actual_actual - chronos_previous
    interaction = combined_loss - pv_loss - load_loss
    chronos_feedback_gain = feedback_chronos - chronos_previous
    previous_feedback_gain = feedback_previous - fixed_previous
    positive = {
        "pv_forecast": max(pv_loss, 0.0),
        "provisional_load_forecast": max(load_loss, 0.0),
        "open_loop_controller": max(chronos_feedback_gain, previous_feedback_gain, 0.0),
        "forecast_interaction": max(interaction, 0.0),
    }
    ordered = sorted(positive.items(), key=lambda item: item[1], reverse=True)
    material_components = [
        name for name, value in ordered if value >= 0.6 * ordered[0][1]
    ]
    if ordered[0][1] <= 0.0:
        bottleneck = "combination_or_unresolved"
        material_components = []
    elif len(material_components) > 1:
        bottleneck = "combination"
    else:
        bottleneck = ordered[0][0]
    return {
        "bottleneck": bottleneck,
        "bottleneck_components": material_components,
        "controlled_revenue_differences_yuan": {
            "pv_forecast_loss_with_actual_load": pv_loss,
            "load_forecast_loss_with_actual_pv": load_loss,
            "combined_forecast_loss": combined_loss,
            "nonadditive_interaction": interaction,
            "chronos_feedback_gain_over_fixed": chronos_feedback_gain,
            "previous_day_feedback_gain_over_fixed": previous_feedback_gain,
        },
        "interpretation_rule": (
            "Largest positive controlled loss or feedback recovery; include every factor "
            "that is at least 60% of the largest and report combination when multiple "
            "factors qualify."
        ),
        "terminal_energy_adjustment_yuan": 0.0,
    }


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    diagnosis: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    lines = [
        "# Foshan 15-Minute State-Feedback Revenue Diagnostic",
        "",
        f"**Status:** {FORECAST_NOTICE}",
        "",
        f"**Evaluation warning:** {VALIDATION_NOTICE}",
        "",
        f"**Load caveat:** {LOAD_DATA_NOTICE}",
        "",
        "Strategies containing actual future PV or load are oracle diagnostics and are "
        "not deployable.",
        "",
        "## Results",
        "",
        (
            "| Strategy | Deployable | Revenue (yuan) | Final SOC (kWh) | "
            "Planned charge/discharge (kWh) | Executed charge/discharge (kWh) | "
            "Clipped intervals |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.deployable} | {row.objective_yuan:.2f} | "
            f"{row.final_soc_kwh:.3f} | {row.planned_charge_kwh:.1f} / "
            f"{row.planned_discharge_kwh:.1f} | {row.executed_charge_kwh:.1f} / "
            f"{row.executed_discharge_kwh:.1f} | {row.clipped_intervals} |"
        )
    lines.extend(
        [
            "",
            "## Clipping Reasons",
            "",
            "| Strategy | Power (kWh) | SOC (kWh) | Anti-export (kWh) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.power_clipped_kwh:.3f} | "
            f"{row.soc_clipped_kwh:.3f} | {row.anti_export_clipped_kwh:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Constraint Audit",
            "",
            "| Strategy | Maximum post-replay violation | Solver status |",
            "| --- | ---: | --- |",
        ]
    )
    for row in summary.itertuples(index=False):
        strategy_audit = audit["strategies"][row.strategy]
        statuses = strategy_audit["daily_solver_statuses"] or ["reused prior result"]
        lines.append(
            f"| {row.strategy} | {row.maximum_constraint_violation:.3e} | "
            f"{', '.join(statuses)} |"
        )
    differences = diagnosis["controlled_revenue_differences_yuan"]
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"- Bottleneck classification: `{diagnosis['bottleneck']}`.",
            (
                "- Material components: "
                f"{', '.join(diagnosis['bottleneck_components'])}."
            ),
            (
                "- PV forecast loss with actual load: "
                f"{differences['pv_forecast_loss_with_actual_load']:.2f} yuan."
            ),
            (
                "- Previous-day load loss with actual PV: "
                f"{differences['load_forecast_loss_with_actual_pv']:.2f} yuan."
            ),
            (
                "- Chronos state-feedback gain over fixed daily control: "
                f"{differences['chronos_feedback_gain_over_fixed']:.2f} yuan."
            ),
            (
                "- Previous-day state-feedback gain over fixed daily control: "
                f"{differences['previous_day_feedback_gain_over_fixed']:.2f} yuan."
            ),
            "",
            "No terminal-energy value adjustment is applied. Final SOC and clipping are "
            "reported directly, so raw revenue differences must be read with those states.",
            "",
            "The feedback controller keeps each day forecast frozen, replans every 15 "
            "minutes from realized SOC, executes three five-minute actions, and never passes "
            "future realized PV or load to HiGHS.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Feedback outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace only this result directory."
        )


def run_backtest(
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    fixed_replay_path: Path,
    output_dir: Path,
    *,
    initial_soc_kwh: float = 900.0,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not np.isclose(initial_soc_kwh, 900.0):
        raise ValueError("This benchmark requires initial_soc_kwh=900.")
    _ensure_output_available(output_dir, overwrite)
    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)
    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path
    )
    previous_pv = previous_day_forecast(realized, "pv", "forecast_pv_kw")
    previous_load = previous_day_forecast(realized, "load", "forecast_load_kw")
    actual_pv = perfect_forecast(realized, "pv", "forecast_pv_kw")
    actual_load = perfect_forecast(realized, "load", "forecast_load_kw")
    existing = load_existing_fixed_results(fixed_replay_path)

    with tempfile.TemporaryDirectory(prefix="foshan_feedback_highs_") as temporary:
        log_dir = Path(temporary)
        chronos_actual = _label_result(
            "fixed_chronos_pv_actual_load_oracle",
            run_daily_strategy(
                "fixed_chronos_pv_actual_load_oracle",
                realized,
                chronos_pv,
                actual_load,
                initial_soc_kwh,
                log_dir,
                mip_relative_gap=mip_relative_gap,
            ),
        )
        actual_previous = _label_result(
            "fixed_actual_pv_previous_day_load_oracle",
            run_daily_strategy(
                "fixed_actual_pv_previous_day_load_oracle",
                realized,
                actual_pv,
                previous_load,
                initial_soc_kwh,
                log_dir,
                mip_relative_gap=mip_relative_gap,
            ),
        )
        feedback_previous, previous_replans = run_feedback_strategy(
            "feedback_previous_day_pv_previous_day_load",
            realized,
            previous_pv,
            previous_load,
            initial_soc_kwh,
            log_dir,
            mip_relative_gap=mip_relative_gap,
            show_progress=show_progress,
        )
        feedback_chronos, chronos_replans = run_feedback_strategy(
            "feedback_chronos_pv_previous_day_load",
            realized,
            chronos_pv,
            previous_load,
            initial_soc_kwh,
            log_dir,
            mip_relative_gap=mip_relative_gap,
            show_progress=show_progress,
        )

    results = {
        "fixed_actual_pv_actual_load_oracle": existing[
            "fixed_actual_pv_actual_load_oracle"
        ],
        "fixed_chronos_pv_actual_load_oracle": chronos_actual,
        "fixed_actual_pv_previous_day_load_oracle": actual_previous,
        "fixed_chronos_pv_previous_day_load": existing[
            "fixed_chronos_pv_previous_day_load"
        ],
        "fixed_previous_day_pv_previous_day_load_reference": existing[
            "fixed_previous_day_pv_previous_day_load_reference"
        ],
        "feedback_previous_day_pv_previous_day_load": feedback_previous,
        "feedback_chronos_pv_previous_day_load": feedback_chronos,
    }
    common_timestamps = validate_common_timestamps(results)
    validate_shared_initial_soc(results, initial_soc_kwh)

    replay = pd.concat([result.replay for result in results.values()], ignore_index=True)
    summary = pd.DataFrame(
        [summarize_result(name, result, parameters) for name, result in results.items()]
    )
    daily = daily_revenue_table(results, parameters)
    strategy_audits = {
        name: {
            **audit_strategy(name, result, parameters),
            "deployable": STRATEGY_LABELS[name]["deployable"],
            "oracle_diagnostic": STRATEGY_LABELS[name]["oracle_diagnostic"],
        }
        for name, result in results.items()
    }
    maximum_violations = {
        name: max(float(audit[field]) for field in CONSTRAINT_VIOLATION_FIELDS)
        for name, audit in strategy_audits.items()
    }
    summary["maximum_constraint_violation"] = summary["strategy"].map(
        maximum_violations
    )
    diagnosis = diagnose_bottleneck(summary)
    audit = {
        "common_timestamp_count": len(common_timestamps),
        "common_timestamp_start": common_timestamps[0].isoformat(),
        "common_timestamp_end": common_timestamps[-1].isoformat(),
        "shared_initial_soc_kwh": initial_soc_kwh,
        "identical_initial_soc": True,
        "identical_timestamp_sets": True,
        "chronos_forecast": chronos_metadata,
        "leakage_controls": {
            "day_forecasts_frozen": True,
            "replan_frequency": "15min",
            "executed_intervals_per_replan": 3,
            "future_realized_pv_or_load_passed_to_feedback_solver": False,
            "known_future_tariff_passed": True,
            "realized_soc_propagated": True,
            "oracle_strategies_clearly_non_deployable": True,
        },
        "strategies": strategy_audits,
    }
    controller_metadata = {
        "terminal_policy": (
            "each shrinking-horizon solve starts and ends at the latest realized SOC"
        ),
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
            "fixed_replay_path": str(fixed_replay_path.resolve()),
            "fixed_replay_sha256": _sha256(fixed_replay_path),
        },
    }

    replay.to_csv(output_dir / "replay_timeseries.csv", index=False, float_format="%.15g")
    summary.to_csv(output_dir / "strategy_summary.csv", index=False, float_format="%.15g")
    daily.to_csv(output_dir / "daily_revenue.csv", index=False, float_format="%.15g")
    _write_json(output_dir / "constraint_audit.json", audit)
    _write_json(output_dir / "diagnosis.json", diagnosis)
    _write_json(output_dir / "controller_metadata.json", controller_metadata)
    _write_report(output_dir / "report.md", summary, diagnosis, audit)
    return replay, summary, daily, diagnosis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Foshan fixed dispatch and run causal 15-minute HiGHS feedback."
        )
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
        "--fixed-replay",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_forecast_backtest/replay_timeseries.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimization/foshan_may_state_feedback"),
    )
    parser.add_argument("--initial-soc-kwh", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary, _, diagnosis = run_backtest(
        dispatch_path=args.dispatch_input,
        predictions_path=args.predictions,
        selection_path=args.selection,
        fixed_replay_path=args.fixed_replay,
        output_dir=args.output_dir,
        initial_soc_kwh=args.initial_soc_kwh,
        mip_relative_gap=args.mip_relative_gap,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    print(
        f"Saved Foshan state-feedback backtest to {args.output_dir.resolve()} "
        f"for {len(summary)} strategies; bottleneck={diagnosis['bottleneck']}."
    )


if __name__ == "__main__":
    main()
