"""Final equal-condition full-May benchmark for frozen controller_v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.reconstruct_foshan_provisional_load import (
    load_provisional_load_signals,
    reconstruct_provisional_load,
    validate_against_reference_load,
    validate_april30_reconstruction,
)
from src.optimization.foshan_battery_milp import (
    LOAD_DATA_NOTICE,
    DispatchParameters,
    recalculate_objective,
    solve_dispatch,
)
from src.optimization.foshan_feedback_controller_v2 import (
    FINAL_TERMINAL_LOWER_KWH,
    FINAL_TERMINAL_UPPER_KWH,
    _accounting_frame,
    _physical_violations,
    run_controller_v2,
)
from src.optimization.foshan_feedback_controller_v3 import _solver_metrics
from src.optimization.foshan_feedback_controller_v5 import (
    V5_NAME,
    _guard_audit,
    _rename_v5_result,
)
from src.optimization.foshan_forecast_backtest import (
    StrategyResult,
    load_reference_dispatch,
    load_selected_chronos_p50,
    previous_day_forecast,
    validate_common_timestamps,
)
from src.utils.runtime import git_commit, git_is_dirty


START = pd.Timestamp("2026-05-01 00:00:00")
END_EXCLUSIVE = pd.Timestamp("2026-06-01 00:00:00")
APRIL30 = pd.Timestamp("2026-04-30 00:00:00")
HISTORICAL_NAME = "historical_actual_operation"
ORACLE_NAME = "perfect_foresight_oracle"
ACTUAL_REFERENCE_YUAN = 112029.39146820732
ORACLE_REFERENCE_YUAN = 122211.25230969457
OUTPUT_FILENAMES = (
    "full_may_comparison.csv",
    "revenue_components.csv",
    "energy_flow_comparison.csv",
    "technical_metrics.csv",
    "interval_results.csv",
    "summary.json",
    "final_benchmark_report.md",
    "monthly_revenue_components.png",
    "may15_dispatch.png",
    "may15_soc.png",
    "may15_net_load.png",
    "april30_provisional_load_reconstruction.csv",
    "may1_reconstruction_validation.csv",
    "perfect_foresight_oracle_solver.log",
)


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


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Final benchmark outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace only this output directory."
        )


def build_historical_result(
    realized: pd.DataFrame,
    parameters: DispatchParameters = DispatchParameters(),
    *,
    initial_soc_kwh: float = 900.0,
) -> tuple[StrategyResult, dict[str, Any]]:
    """Replay observed PCS commands unchanged and reconstruct bounded SOC."""
    window = realized.loc[
        (realized["timestamp"] >= START)
        & (realized["timestamp"] < END_EXCLUSIVE)
    ].copy()
    expected = pd.date_range(START, END_EXCLUSIVE, freq="5min", inclusive="left")
    if window["timestamp"].tolist() != expected.tolist():
        raise ValueError("Historical source does not cover the full May grid.")

    power = window["p_actual"].to_numpy(dtype=float)
    charge = np.maximum(-power, 0.0)
    discharge = np.maximum(power, 0.0)
    soc_starts: list[float] = []
    soc_ends: list[float] = []
    unconstrained_ends: list[float] = []
    saturation_adjustments: list[float] = []
    current_soc = float(initial_soc_kwh)
    for charge_kw, discharge_kw in zip(charge, discharge):
        soc_starts.append(current_soc)
        unconstrained_end = (
            current_soc
            + parameters.charge_efficiency
            * parameters.interval_hours
            * charge_kw
            - parameters.interval_hours
            / parameters.discharge_efficiency
            * discharge_kw
        )
        bounded_end = min(
            parameters.capacity_kwh, max(0.0, unconstrained_end)
        )
        unconstrained_ends.append(unconstrained_end)
        saturation_adjustments.append(bounded_end - unconstrained_end)
        soc_ends.append(bounded_end)
        current_soc = bounded_end

    net_grid = (
        window["load"].to_numpy(dtype=float)
        - window["pv"].to_numpy(dtype=float)
        - discharge
        + charge
    )
    replay = pd.DataFrame(
        {
            "strategy": HISTORICAL_NAME,
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
            "scheduled_soc_start_kwh": soc_starts,
            "scheduled_soc_end_kwh": soc_ends,
            "realized_soc_start_kwh": soc_starts,
            "realized_soc_end_kwh": soc_ends,
            "unconstrained_soc_end_kwh": unconstrained_ends,
            "soc_saturation_adjustment_kwh": saturation_adjustments,
            "grid_import_kw": np.maximum(net_grid, 0.0),
            "grid_export_kw": np.maximum(-net_grid, 0.0),
            "power_clip_kw": 0.0,
            "soc_clip_kw": 0.0,
            "upper_soc_clip_kw": 0.0,
            "lower_soc_clip_kw": 0.0,
            "anti_export_clip_kw": 0.0,
            "total_clip_kw": 0.0,
            "was_clipped": False,
            "issue_time": pd.NaT,
            "safety_filter_applied": False,
            "counterfactual_provisional": False,
            "validation_period_demo": True,
        }
    )
    source_soc = window["soc_actual_est"].to_numpy(dtype=float)
    source_soc_error = np.abs(np.asarray(soc_starts) - source_soc)
    saturation = np.abs(np.asarray(saturation_adjustments))
    audit = {
        "observed_commands_modified": False,
        "initial_soc_kwh": initial_soc_kwh,
        "reconstructed_final_soc_kwh": current_soc,
        "source_soc_start_max_abs_difference_kwh": float(source_soc_error.max()),
        "soc_saturation_intervals": int((saturation > 1e-9).sum()),
        "soc_saturation_energy_kwh": float(saturation.sum()),
        "maximum_soc_saturation_adjustment_kwh": float(saturation.max()),
    }
    return StrategyResult(replay=replay, daily_runs=[]), audit


def build_oracle_result(
    realized: pd.DataFrame,
    solver_log_path: Path,
    parameters: DispatchParameters = DispatchParameters(),
    *,
    mip_relative_gap: float = 1e-7,
) -> tuple[StrategyResult, dict[str, Any]]:
    """Solve one full-month perfect-foresight MILP with terminal SOC 900."""
    milp_input = realized.loc[
        (realized["timestamp"] >= START)
        & (realized["timestamp"] < END_EXCLUSIVE),
        ["timestamp", "pv", "load", "price"],
    ].copy()
    solved = solve_dispatch(
        milp_input,
        solver_log_path,
        parameters,
        mip_relative_gap=mip_relative_gap,
        log_to_console=False,
    )
    if solved.solver_metadata.get("solver_status") != "Optimal":
        raise RuntimeError(
            "Perfect-foresight oracle did not solve optimally: "
            f"{solved.solver_metadata}"
        )
    dispatch = solved.dispatch
    replay = pd.DataFrame(
        {
            "strategy": ORACLE_NAME,
            "timestamp": dispatch["timestamp"],
            "realized_pv_kw": dispatch["pv"],
            "realized_load_kw": dispatch["load"],
            "price_yuan_per_kwh": dispatch["price"],
            "forecast_pv_kw": dispatch["pv"],
            "forecast_load_kw": dispatch["load"],
            "scheduled_charge_kw": dispatch["charge_kw"],
            "scheduled_discharge_kw": dispatch["discharge_kw"],
            "applied_charge_kw": dispatch["charge_kw"],
            "applied_discharge_kw": dispatch["discharge_kw"],
            "scheduled_soc_start_kwh": dispatch["soc_start_kwh"],
            "scheduled_soc_end_kwh": dispatch["soc_kwh"],
            "realized_soc_start_kwh": dispatch["soc_start_kwh"],
            "realized_soc_end_kwh": dispatch["soc_kwh"],
            "grid_import_kw": dispatch["grid_import_kw"],
            "grid_export_kw": dispatch["grid_export_kw"],
            "power_clip_kw": 0.0,
            "soc_clip_kw": 0.0,
            "upper_soc_clip_kw": 0.0,
            "lower_soc_clip_kw": 0.0,
            "anti_export_clip_kw": 0.0,
            "total_clip_kw": 0.0,
            "was_clipped": False,
            "issue_time": START,
            "safety_filter_applied": False,
            "counterfactual_provisional": True,
            "validation_period_demo": True,
        }
    )
    return StrategyResult(replay=replay, daily_runs=[]), solved.solver_metadata


def interval_accounting(
    replay: pd.DataFrame,
    parameters: DispatchParameters = DispatchParameters(),
) -> pd.DataFrame:
    """Independently calculate interval energy and revenue components."""
    dt = parameters.interval_hours
    pv = replay["realized_pv_kw"].to_numpy(dtype=float)
    load = replay["realized_load_kw"].to_numpy(dtype=float)
    price = replay["price_yuan_per_kwh"].to_numpy(dtype=float)
    grid_import = replay["grid_import_kw"].to_numpy(dtype=float)
    grid_export = replay["grid_export_kw"].to_numpy(dtype=float)
    pv_self = pv - grid_export
    baseline_import = np.maximum(load - pv, 0.0)
    baseline_cost = baseline_import * price * dt
    actual_cost = grid_import * price * dt
    savings = baseline_cost - actual_cost
    direct_revenue = (
        pv_self * parameters.pv_self_price_yuan_per_kwh * dt
    )
    export_revenue = (
        grid_export * parameters.pv_export_price_yuan_per_kwh * dt
    )
    institute_share = parameters.storage_revenue_share * savings
    user_share = (1.0 - parameters.storage_revenue_share) * savings
    return pd.DataFrame(
        {
            "pv_generation_kwh": pv * dt,
            "pv_self_consumption_kwh": pv_self * dt,
            "pv_export_kwh": grid_export * dt,
            "grid_import_kwh": grid_import * dt,
            "battery_charge_kwh": replay["applied_charge_kw"].to_numpy(
                dtype=float
            )
            * dt,
            "battery_discharge_kwh": replay[
                "applied_discharge_kw"
            ].to_numpy(dtype=float)
            * dt,
            "baseline_grid_import_cost_yuan": baseline_cost,
            "grid_import_cost_yuan": actual_cost,
            "pv_direct_supply_revenue_yuan": direct_revenue,
            "surplus_pv_export_revenue_yuan": export_revenue,
            "total_pv_revenue_yuan": direct_revenue + export_revenue,
            "total_storage_bill_savings_yuan": savings,
            "design_institute_storage_share_yuan": institute_share,
            "user_storage_share_yuan": user_share,
            "design_institute_comprehensive_revenue_yuan": (
                direct_revenue + export_revenue + institute_share
            ),
        }
    )


def summarize_result(
    name: str,
    result: StrategyResult,
    parameters: DispatchParameters,
    *,
    runtime_seconds: float,
    solver_replans: int,
    solver_failures: int,
    terminal_policy: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float]]:
    replay = result.replay
    accounting = interval_accounting(replay, parameters)
    totals = accounting.sum().to_dict()
    reported = recalculate_objective(_accounting_frame(replay), parameters)
    independent_revenue = totals[
        "design_institute_comprehensive_revenue_yuan"
    ]
    error = abs(float(reported["objective_yuan"]) - independent_revenue)
    if error > 0.01:
        raise ValueError(
            f"{name} independent revenue differs by {error:.6f} yuan."
        )
    comprehensive_identity_error = abs(
        independent_revenue
        - totals["total_pv_revenue_yuan"]
        - totals["design_institute_storage_share_yuan"]
    )
    if comprehensive_identity_error > 1e-9:
        raise ValueError(f"{name} comprehensive revenue double-count check failed.")

    violations = _physical_violations(replay, parameters)
    expected_soc_end = (
        replay["realized_soc_start_kwh"].to_numpy(dtype=float)
        + parameters.charge_efficiency
        * parameters.interval_hours
        * replay["applied_charge_kw"].to_numpy(dtype=float)
        - parameters.interval_hours
        / parameters.discharge_efficiency
        * replay["applied_discharge_kw"].to_numpy(dtype=float)
    )
    violations["soc_dynamics_violation_kwh"] = float(
        np.max(
            np.abs(
                replay["realized_soc_end_kwh"].to_numpy(dtype=float)
                - expected_soc_end
            )
        )
    )
    violations["maximum_constraint_violation"] = max(violations.values())
    final_soc = float(replay["realized_soc_end_kwh"].iloc[-1])
    if terminal_policy == "band_895_905":
        terminal_slack = max(FINAL_TERMINAL_LOWER_KWH - final_soc, 0.0) + max(
            final_soc - FINAL_TERMINAL_UPPER_KWH, 0.0
        )
    else:
        terminal_slack = abs(final_soc - parameters.terminal_soc_kwh)
    planned_discharge = float(
        replay["scheduled_discharge_kw"].sum() * parameters.interval_hours
    )
    anti_export_clipped = float(
        replay["anti_export_clip_kw"].sum() * parameters.interval_hours
    )
    row = {
        "strategy": name,
        "raw_revenue_yuan": independent_revenue,
        "reported_revenue_yuan": float(reported["objective_yuan"]),
        "revenue_recalculation_abs_error_yuan": error,
        "initial_soc_kwh": float(replay["realized_soc_start_kwh"].iloc[0]),
        "final_soc_kwh": final_soc,
        "terminal_slack_kwh": terminal_slack,
        "terminal_policy": terminal_policy,
        "planned_charge_kwh": float(
            replay["scheduled_charge_kw"].sum() * parameters.interval_hours
        ),
        "planned_discharge_kwh": planned_discharge,
        "executed_charge_kwh": totals["battery_charge_kwh"],
        "executed_discharge_kwh": totals["battery_discharge_kwh"],
        "anti_export_clipped_intervals": int(
            (replay["anti_export_clip_kw"] > 1e-7).sum()
        ),
        "anti_export_clipped_kwh": anti_export_clipped,
        "clipped_discharge_fraction": (
            anti_export_clipped / planned_discharge
            if planned_discharge > 0.0
            else 0.0
        ),
        "soc_violation_kwh": max(
            violations["soc_lower_violation_kwh"],
            violations["soc_upper_violation_kwh"],
            violations["soc_dynamics_violation_kwh"],
            violations["soc_continuity_violation_kwh"],
        ),
        "power_violation_kw": max(
            violations["charge_power_violation_kw"],
            violations["discharge_power_violation_kw"],
        ),
        "anti_export_violation_kw": violations[
            "anti_export_violation_kw"
        ],
        "maximum_constraint_violation": violations[
            "maximum_constraint_violation"
        ],
        "solver_replans": solver_replans,
        "solver_failures": solver_failures,
        "runtime_seconds": runtime_seconds,
        **totals,
    }
    return row, accounting, violations


def _comparison_table(
    components: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for label, values in components.items():
        actual = float(values[HISTORICAL_NAME])
        controller = float(values[V5_NAME])
        oracle = float(values[ORACLE_NAME])
        rows.append(
            {
                "Component": label,
                "Historical actual": actual,
                "Controller_v5": controller,
                "Perfect-foresight oracle": oracle,
                "v5 minus actual": controller - actual,
                "v5 minus oracle": controller - oracle,
            }
        )
    return pd.DataFrame(rows)


def _write_charts(
    output_dir: Path,
    summary: pd.DataFrame,
    interval_results: pd.DataFrame,
) -> None:
    plot_cache = output_dir / ".matplotlib-cache"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Historical actual", "Controller_v5", "Perfect foresight"]
    names = [HISTORICAL_NAME, V5_NAME, ORACLE_NAME]
    indexed = summary.set_index("strategy")
    revenue_columns = [
        ("pv_direct_supply_revenue_yuan", "PV direct supply"),
        ("surplus_pv_export_revenue_yuan", "PV export"),
        ("design_institute_storage_share_yuan", "Storage share (80%)"),
    ]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(names))
    for column, label in revenue_columns:
        values = indexed.loc[names, column].to_numpy(dtype=float)
        axis.bar(labels, values, bottom=bottom, label=label)
        bottom += values
    axis.set_ylabel("Revenue (yuan)")
    axis.set_title("Full-May design institute revenue components")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "monthly_revenue_components.png", dpi=160)
    plt.close(figure)

    may15 = interval_results.loc[
        pd.to_datetime(interval_results["timestamp"]).dt.date
        == pd.Timestamp("2026-05-15").date()
    ].copy()
    may15["timestamp"] = pd.to_datetime(may15["timestamp"])
    colors = {
        HISTORICAL_NAME: "#4c566a",
        V5_NAME: "#007c91",
        ORACLE_NAME: "#c44e52",
    }
    display = dict(zip(names, labels))

    figure, axis = plt.subplots(figsize=(12, 5))
    for name in names:
        table = may15.loc[may15["strategy"].eq(name)]
        net_battery = (
            table["applied_discharge_kw"] - table["applied_charge_kw"]
        )
        axis.step(
            table["timestamp"], net_battery, where="post", label=display[name],
            color=colors[name], linewidth=1.0,
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_ylabel("Battery power (kW)\n(+ discharge, - charge)")
    axis.set_title("May 15 battery dispatch")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / "may15_dispatch.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    for name in names:
        table = may15.loc[may15["strategy"].eq(name)]
        axis.step(
            table["timestamp"], table["realized_soc_end_kwh"], where="post",
            label=display[name], color=colors[name], linewidth=1.0,
        )
    axis.set_ylabel("SOC (kWh)")
    axis.set_title("May 15 reconstructed/realized SOC")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / "may15_soc.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    base = may15.loc[may15["strategy"].eq(HISTORICAL_NAME)]
    axis.plot(
        base["timestamp"],
        base["realized_load_kw"] - base["realized_pv_kw"],
        label="Before storage",
        color="black",
        linewidth=1.0,
        alpha=0.7,
    )
    for name in names:
        table = may15.loc[may15["strategy"].eq(name)]
        net_grid = table["grid_import_kw"] - table["grid_export_kw"]
        axis.step(
            table["timestamp"], net_grid, where="post", label=display[name],
            color=colors[name], linewidth=1.0,
        )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_ylabel("Net grid exchange (kW)")
    axis.set_title("May 15 net load and post-storage grid exchange")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / "may15_net_load.png", dpi=160)
    plt.close(figure)


def _write_report(
    path: Path,
    revenue: pd.DataFrame,
    energy: pd.DataFrame,
    technical: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    def markdown_table(table: pd.DataFrame, digits: int = 2) -> str:
        headers = [str(column) for column in table.columns]
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for values in table.itertuples(index=False, name=None):
            formatted = []
            for value in values:
                if pd.isna(value):
                    formatted.append("")
                elif isinstance(value, (float, np.floating)):
                    formatted.append(f"{float(value):.{digits}f}")
                else:
                    formatted.append(str(value))
            rows.append("| " + " | ".join(formatted) + " |")
        return "\n".join(rows)

    lines = [
        "# Foshan Full-May Controller V5 Benchmark",
        "",
        f"**Load warning:** {LOAD_DATA_NOTICE}",
        "",
        "May 2026 was used for Chronos configuration selection. Forecast-driven "
        "revenue remains a counterfactual validation-period result, not observed "
        "actual revenue.",
        "",
        "## Revenue Components",
        "",
        markdown_table(revenue),
        "",
        "## Energy Flows",
        "",
        markdown_table(energy),
        "",
        "## Technical Metrics",
        "",
        markdown_table(technical, digits=6),
        "",
        "## Reconciliation",
        "",
        (
            f"Historical difference from task package: "
            f"{summary['reconciliation']['historical_difference_yuan']:.6f} yuan."
        ),
        "",
        (
            f"Oracle difference from task package: "
            f"{summary['reconciliation']['oracle_difference_yuan']:.6f} yuan."
        ),
        "",
        "The oracle uses continuous SOC in HiGHS. The task-package reference used "
        "a 10-kWh dynamic-programming SOC grid and its stored schedule includes a "
        "reported charge-power excursion above the current 1000-kW MILP limit. "
        "These model differences explain a non-trivial oracle reconciliation gap.",
        "",
        "The historical PCS commands were not altered. Its SOC estimate was "
        "independently propagated with explicit bound saturation diagnostics.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    site_workbook: Path,
    storage_workbook: Path,
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    output_dir: Path,
    *,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_started = time.perf_counter()
    _ensure_output_available(output_dir, overwrite)
    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)

    signals = load_provisional_load_signals(site_workbook, storage_workbook)
    april30, april30_audit = reconstruct_provisional_load(
        signals, APRIL30, START
    )
    validate_april30_reconstruction(april30_audit)
    may1_reconstruction, may1_audit = reconstruct_provisional_load(
        signals, START, START + pd.Timedelta(days=1)
    )
    may1_reference = realized.loc[
        (realized["timestamp"] >= START)
        & (realized["timestamp"] < START + pd.Timedelta(days=1)),
        ["timestamp", "load"],
    ].copy()
    may1_validation = validate_against_reference_load(
        may1_reconstruction, may1_reference
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    historical_started = time.perf_counter()
    historical, historical_audit = build_historical_result(realized, parameters)
    historical_runtime = time.perf_counter() - historical_started

    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path, START, END_EXCLUSIVE
    )
    april_history = april30[
        ["timestamp", "pv_kw", "provisional_load_kw"]
    ].rename(columns={"pv_kw": "pv", "provisional_load_kw": "load"})
    april_history["timestamp"] = april_history["timestamp"].dt.tz_localize(None)
    april_history["price"] = np.nan
    realized_with_history = pd.concat(
        [april_history, realized], ignore_index=True, sort=False
    ).sort_values("timestamp")
    previous_load = previous_day_forecast(
        realized_with_history,
        "load",
        "forecast_load_kw",
        START,
        END_EXCLUSIVE,
    )
    may1_load_sources = previous_load.loc[
        previous_load["timestamp"] < START + pd.Timedelta(days=1),
        "forecast_load_kw_source_timestamp",
    ]
    if not (may1_load_sources < START).all():
        raise ValueError("May 1 load forecast contains a non-historical source.")

    controller_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="foshan_final_v5_highs_") as temp:
        controller_base, controller_replans = run_controller_v2(
            "controller_v2_chronos_pv_previous_day_load",
            realized_with_history,
            chronos_pv,
            previous_load,
            900.0,
            Path(temp),
            parameters=parameters,
            start=START,
            end_exclusive=END_EXCLUSIVE,
            cadence_minutes=5,
            mip_relative_gap=mip_relative_gap,
            show_progress=show_progress,
            use_q10_discharge_limit=False,
            use_terminal_recovery_charge_ban=False,
            use_latest_completed_residual_for_first_step=True,
            use_intraday_load_bias_correction=False,
            use_final_day_immediate_charge_guard=True,
        )
    controller, controller_replans = _rename_v5_result(
        controller_base, controller_replans
    )
    controller_runtime = time.perf_counter() - controller_started
    controller_policy_audit = _guard_audit(controller_replans, parameters)
    controller_solver_metrics = _solver_metrics(controller)

    oracle_started = time.perf_counter()
    oracle, oracle_solver_metadata = build_oracle_result(
        realized,
        output_dir / "perfect_foresight_oracle_solver.log",
        parameters,
        mip_relative_gap=mip_relative_gap,
    )
    oracle_runtime = time.perf_counter() - oracle_started

    results = {
        HISTORICAL_NAME: historical,
        V5_NAME: controller,
        ORACLE_NAME: oracle,
    }
    common = validate_common_timestamps(results, START, END_EXCLUSIVE)
    initial_states = {
        name: float(result.replay["realized_soc_start_kwh"].iloc[0])
        for name, result in results.items()
    }
    if any(not np.isclose(value, 900.0) for value in initial_states.values()):
        raise ValueError(f"Strategies do not share initial SOC: {initial_states}")

    metric_inputs = {
        HISTORICAL_NAME: (historical_runtime, 0, 0, "reference_900_not_enforced"),
        V5_NAME: (
            controller_solver_metrics[2],
            controller_solver_metrics[0],
            controller_solver_metrics[1],
            "band_895_905",
        ),
        ORACLE_NAME: (
            float(oracle_solver_metadata.get("wall_clock_runtime_seconds") or 0.0),
            1,
            0,
            "target_900",
        ),
    }
    rows: list[dict[str, Any]] = []
    interval_parts: list[pd.DataFrame] = []
    physical_audits: dict[str, Any] = {}
    for name, result in results.items():
        runtime, replans, failures, terminal_policy = metric_inputs[name]
        row, accounting, physical = summarize_result(
            name,
            result,
            parameters,
            runtime_seconds=float(runtime),
            solver_replans=int(replans),
            solver_failures=int(failures),
            terminal_policy=str(terminal_policy),
        )
        rows.append(row)
        physical_audits[name] = physical
        interval_parts.append(
            pd.concat(
                [result.replay.reset_index(drop=True), accounting], axis=1
            )
        )
    full_summary = pd.DataFrame(rows)
    indexed = full_summary.set_index("strategy")
    controller_final_soc = float(indexed.loc[V5_NAME, "final_soc_kwh"])
    if not (
        FINAL_TERMINAL_LOWER_KWH - 1e-7
        <= controller_final_soc
        <= FINAL_TERMINAL_UPPER_KWH + 1e-7
    ):
        raise ValueError(
            "Frozen controller_v5 missed the required final SOC band: "
            f"{controller_final_soc:.6f} kWh."
        )

    revenue_components = _comparison_table(
        {
            "PV direct-supply revenue (yuan)": {
                name: indexed.loc[name, "pv_direct_supply_revenue_yuan"]
                for name in results
            },
            "Surplus-PV export revenue (yuan)": {
                name: indexed.loc[name, "surplus_pv_export_revenue_yuan"]
                for name in results
            },
            "Total PV revenue (yuan)": {
                name: indexed.loc[name, "total_pv_revenue_yuan"]
                for name in results
            },
            "Total storage bill savings (yuan)": {
                name: indexed.loc[name, "total_storage_bill_savings_yuan"]
                for name in results
            },
            "Design institute 80% storage share (yuan)": {
                name: indexed.loc[name, "design_institute_storage_share_yuan"]
                for name in results
            },
            "User 20% storage share (yuan)": {
                name: indexed.loc[name, "user_storage_share_yuan"]
                for name in results
            },
            "Design institute comprehensive revenue (yuan)": {
                name: indexed.loc[
                    name, "design_institute_comprehensive_revenue_yuan"
                ]
                for name in results
            },
        }
    )
    energy_flow = _comparison_table(
        {
            "PV generation (kWh)": {
                name: indexed.loc[name, "pv_generation_kwh"] for name in results
            },
            "PV self-consumption (kWh)": {
                name: indexed.loc[name, "pv_self_consumption_kwh"]
                for name in results
            },
            "PV export (kWh)": {
                name: indexed.loc[name, "pv_export_kwh"] for name in results
            },
            "Grid import (kWh)": {
                name: indexed.loc[name, "grid_import_kwh"] for name in results
            },
            "Battery charging energy (kWh)": {
                name: indexed.loc[name, "battery_charge_kwh"] for name in results
            },
            "Battery discharging energy (kWh)": {
                name: indexed.loc[name, "battery_discharge_kwh"]
                for name in results
            },
            "Initial SOC (kWh)": {
                name: indexed.loc[name, "initial_soc_kwh"] for name in results
            },
            "Final SOC (kWh)": {
                name: indexed.loc[name, "final_soc_kwh"] for name in results
            },
        }
    )
    technical_columns = [
        "raw_revenue_yuan",
        "final_soc_kwh",
        "terminal_slack_kwh",
        "planned_charge_kwh",
        "planned_discharge_kwh",
        "executed_charge_kwh",
        "executed_discharge_kwh",
        "anti_export_clipped_intervals",
        "anti_export_clipped_kwh",
        "clipped_discharge_fraction",
        "soc_violation_kwh",
        "power_violation_kw",
        "anti_export_violation_kw",
        "maximum_constraint_violation",
        "solver_failures",
        "runtime_seconds",
    ]
    technical = full_summary[["strategy", *technical_columns]].copy()
    interval_results = pd.concat(interval_parts, ignore_index=True, sort=False)

    actual_revenue = float(indexed.loc[HISTORICAL_NAME, "raw_revenue_yuan"])
    controller_revenue = float(indexed.loc[V5_NAME, "raw_revenue_yuan"])
    oracle_revenue = float(indexed.loc[ORACLE_NAME, "raw_revenue_yuan"])
    historical_difference = actual_revenue - ACTUAL_REFERENCE_YUAN
    oracle_difference = oracle_revenue - ORACLE_REFERENCE_YUAN
    summary = {
        "status": (
            "counterfactual validation-period benchmark; forecast revenue is not "
            "observed actual revenue"
        ),
        "load_data_warning": LOAD_DATA_NOTICE,
        "period": {
            "start": START.isoformat(),
            "end_exclusive": END_EXCLUSIVE.isoformat(),
            "timestamp_count": len(common),
        },
        "april30_reconstruction": april30_audit,
        "may1_reconstruction": may1_audit,
        "may1_reference_validation": may1_validation,
        "historical_soc_reconstruction": historical_audit,
        "controller_policy_audit": controller_policy_audit,
        "physical_audits": physical_audits,
        "revenues_yuan": {
            HISTORICAL_NAME: actual_revenue,
            V5_NAME: controller_revenue,
            ORACLE_NAME: oracle_revenue,
        },
        "comparison": {
            "v5_improvement_over_historical_yuan": (
                controller_revenue - actual_revenue
            ),
            "v5_percentage_improvement_over_historical": (
                100.0 * (controller_revenue - actual_revenue) / actual_revenue
            ),
            "v5_oracle_attainment_percent": (
                100.0 * controller_revenue / oracle_revenue
            ),
            "v5_gap_to_oracle_yuan": oracle_revenue - controller_revenue,
        },
        "reconciliation": {
            "historical_reference_yuan": ACTUAL_REFERENCE_YUAN,
            "historical_difference_yuan": historical_difference,
            "historical_within_one_yuan": abs(historical_difference) <= 1.0,
            "oracle_reference_yuan": ORACLE_REFERENCE_YUAN,
            "oracle_difference_yuan": oracle_difference,
            "oracle_within_one_yuan": abs(oracle_difference) <= 1.0,
            "oracle_difference_diagnosis": (
                "HiGHS uses continuous SOC under the current MILP constraints; "
                "the task-package oracle used a 10-kWh DP SOC grid and its stored "
                "schedule has a reported charge-power excursion above 1000 kW."
                if abs(oracle_difference) > 1.0
                else None
            ),
        },
        "chronos_forecast": chronos_metadata,
        "oracle_solver_metadata": oracle_solver_metadata,
        "parameters": parameters.to_dict(),
        "provenance": {
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "site_workbook": str(site_workbook.resolve()),
            "site_workbook_sha256": _sha256(site_workbook),
            "storage_workbook": str(storage_workbook.resolve()),
            "storage_workbook_sha256": _sha256(storage_workbook),
            "dispatch_path": str(dispatch_path.resolve()),
            "dispatch_sha256": _sha256(dispatch_path),
            "predictions_path": str(predictions_path.resolve()),
            "predictions_sha256": _sha256(predictions_path),
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": _sha256(selection_path),
        },
        "wall_clock_runtime_seconds": time.perf_counter() - run_started,
    }

    full_summary.to_csv(
        output_dir / "full_may_comparison.csv", index=False, float_format="%.15g"
    )
    revenue_components.to_csv(
        output_dir / "revenue_components.csv", index=False, float_format="%.15g"
    )
    energy_flow.to_csv(
        output_dir / "energy_flow_comparison.csv", index=False, float_format="%.15g"
    )
    technical.to_csv(
        output_dir / "technical_metrics.csv", index=False, float_format="%.15g"
    )
    interval_results.to_csv(
        output_dir / "interval_results.csv", index=False, float_format="%.15g"
    )
    april30.to_csv(
        output_dir / "april30_provisional_load_reconstruction.csv",
        index=False,
        float_format="%.15g",
    )
    may1_validation_table = may1_reconstruction.copy()
    may1_validation_table["timestamp"] = may1_validation_table[
        "timestamp"
    ].dt.tz_localize(None)
    may1_validation_table = may1_validation_table.merge(
        may1_reference,
        on="timestamp",
        how="inner",
        validate="one_to_one",
        suffixes=("_reconstructed", "_reference"),
    )
    may1_validation_table["absolute_difference_kw"] = (
        may1_validation_table["provisional_load_kw"]
        - may1_validation_table["load"]
    ).abs()
    may1_validation_table.to_csv(
        output_dir / "may1_reconstruction_validation.csv",
        index=False,
        float_format="%.15g",
    )
    _write_json(output_dir / "summary.json", summary)
    _write_report(
        output_dir / "final_benchmark_report.md",
        revenue_components,
        energy_flow,
        technical,
        summary,
    )
    _write_charts(output_dir, full_summary, interval_results)
    return full_summary, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen controller_v5 full-May Foshan benchmark."
    )
    parser.add_argument("--site-workbook", required=True, type=Path)
    parser.add_argument("--storage-workbook", required=True, type=Path)
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
        default=Path(
            "results/final_benchmark/foshan_may2026_controller_v5"
        ),
    )
    parser.add_argument("--mip-relative-gap", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    table, summary = run_benchmark(
        site_workbook=args.site_workbook,
        storage_workbook=args.storage_workbook,
        dispatch_path=args.dispatch_input,
        predictions_path=args.predictions,
        selection_path=args.selection,
        output_dir=args.output_dir,
        mip_relative_gap=args.mip_relative_gap,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    print(
        f"Saved full-May benchmark for {len(table)} strategies to "
        f"{args.output_dir.resolve()}: {summary['revenues_yuan']}"
    )


if __name__ == "__main__":
    main()
