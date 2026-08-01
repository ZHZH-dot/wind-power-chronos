"""Revenue-first five-minute Foshan battery feedback controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    recalculate_objective,
)
from src.optimization.foshan_feedback_controller_v2 import (
    FINAL_TERMINAL_LOWER_KWH,
    FINAL_TERMINAL_UPPER_KWH,
    TERMINAL_SOC_REFERENCE_KWH,
    _accounting_frame,
    _physical_violations,
    load_old_results,
    run_controller_v2,
)
from src.optimization.foshan_forecast_backtest import (
    StrategyResult,
    clipping_energy_kwh,
    load_reference_dispatch,
    load_selected_chronos_p50,
    previous_day_forecast,
    validate_common_timestamps,
)
from src.utils.runtime import git_commit, git_is_dirty


SOURCE_CONFIG = {
    "previous_day": {
        "base_strategy": "controller_v2_previous_day_pv_previous_day_load",
        "v3_strategy": "controller_v3_previous_day_pv_previous_day_load",
        "feedback_strategy": "feedback_previous_day_pv_previous_day_load",
        "variant_a_strategy": "controller_v2_ablation_a_previous_day_pv",
        "variant_d_strategy": "controller_v2_ablation_d_previous_day_pv",
        "pv_source": "previous_day_pv",
    },
    "chronos": {
        "base_strategy": "controller_v2_chronos_pv_previous_day_load",
        "v3_strategy": "controller_v3_chronos_pv_previous_day_load",
        "feedback_strategy": "feedback_chronos_pv_previous_day_load",
        "variant_a_strategy": "controller_v2_ablation_a_chronos_pv",
        "variant_d_strategy": "controller_v2_ablation_d_chronos_pv",
        "pv_source": "chronos2_zero_shot_postprocessed_p50",
    },
}
OUTPUT_FILENAMES = (
    "replay_timeseries.csv",
    "strategy_summary.csv",
    "comparison.json",
    "constraint_audit.json",
    "controller_metadata.json",
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


def independent_replay_revenue(
    replay: pd.DataFrame,
    parameters: DispatchParameters = DispatchParameters(),
) -> float:
    """Recalculate physical replay revenue without solver objective metadata."""
    dt = parameters.interval_hours
    baseline_import = np.maximum(
        replay["realized_load_kw"].to_numpy(dtype=float)
        - replay["realized_pv_kw"].to_numpy(dtype=float),
        0.0,
    )
    grid_import = replay["grid_import_kw"].to_numpy(dtype=float)
    grid_export = replay["grid_export_kw"].to_numpy(dtype=float)
    pv = replay["realized_pv_kw"].to_numpy(dtype=float)
    price = replay["price_yuan_per_kwh"].to_numpy(dtype=float)
    pv_self_consumption = pv - grid_export
    return float(
        pv_self_consumption.sum()
        * dt
        * parameters.pv_self_price_yuan_per_kwh
        + grid_export.sum() * dt * parameters.pv_export_price_yuan_per_kwh
        + parameters.storage_revenue_share
        * (
            np.sum(baseline_import * price) * dt
            - np.sum(grid_import * price) * dt
        )
    )


def _load_ablation_results(path: Path) -> dict[str, StrategyResult]:
    table = pd.read_csv(path, low_memory=False)
    table["timestamp"] = pd.to_datetime(table["timestamp"], errors="raise")
    expected = {
        str(config[key])
        for config in SOURCE_CONFIG.values()
        for key in ("variant_a_strategy", "variant_d_strategy")
    }
    results: dict[str, StrategyResult] = {}
    for name in sorted(expected):
        replay = table.loc[table["strategy"].eq(name)].copy()
        replay = replay.sort_values("timestamp").reset_index(drop=True)
        if replay.empty:
            raise ValueError(f"Ablation replay is missing {name}.")
        results[name] = StrategyResult(replay=replay, daily_runs=[])
    validate_common_timestamps(results)
    return results


def _rename_v3_result(
    result: StrategyResult,
    replans: list[dict[str, Any]],
    name: str,
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    replay = result.replay.copy()
    replay["strategy"] = name
    replay["controller"] = "controller_v3_5min"
    replay["q10_discharge_limit_enabled"] = False
    replay["terminal_recovery_charge_ban_enabled"] = False
    renamed_replans = []
    for row in replans:
        updated = dict(row)
        updated["strategy"] = name
        renamed_replans.append(updated)
    daily_runs = []
    for row in result.daily_runs:
        updated = dict(row)
        updated["strategy"] = name
        daily_runs.append(updated)
    return StrategyResult(replay=replay, daily_runs=daily_runs), renamed_replans


def _solver_metrics(result: StrategyResult) -> tuple[int, int, float]:
    return (
        sum(int(row.get("replan_count") or 0) for row in result.daily_runs),
        sum(int(row.get("solver_failure_count") or 0) for row in result.daily_runs),
        sum(float(row.get("solver_runtime_seconds") or 0.0) for row in result.daily_runs),
    )


def _summary_row(
    name: str,
    role: str,
    pv_source: str,
    result: StrategyResult,
    parameters: DispatchParameters,
    solver_metrics: tuple[int | None, int | None, float | None],
) -> tuple[dict[str, Any], dict[str, float]]:
    replay = result.replay
    reported = recalculate_objective(_accounting_frame(replay), parameters)[
        "objective_yuan"
    ]
    independent = independent_replay_revenue(replay, parameters)
    violations = _physical_violations(replay, parameters)
    planned_discharge = float(
        replay["scheduled_discharge_kw"].sum() * parameters.interval_hours
    )
    anti_export_clipped = clipping_energy_kwh(
        replay["anti_export_clip_kw"], parameters.interval_hours
    )
    final_soc = float(replay["realized_soc_end_kwh"].iloc[-1])
    terminal_slack = max(FINAL_TERMINAL_LOWER_KWH - final_soc, 0.0) + max(
        final_soc - FINAL_TERMINAL_UPPER_KWH, 0.0
    )
    replans, failures, runtime = solver_metrics
    return (
        {
            "strategy": name,
            "role": role,
            "pv_source": pv_source,
            "load_source": "previous_day_provisional_load",
            "objective_yuan": reported,
            "independent_recalculated_objective_yuan": independent,
            "revenue_recalculation_abs_error_yuan": abs(reported - independent),
            "initial_soc_kwh": float(replay["realized_soc_start_kwh"].iloc[0]),
            "final_soc_kwh": final_soc,
            "terminal_slack_kwh": terminal_slack,
            "terminal_comparable_895_905": bool(
                FINAL_TERMINAL_LOWER_KWH - 1e-7
                <= final_soc
                <= FINAL_TERMINAL_UPPER_KWH + 1e-7
            ),
            "planned_charge_kwh": float(
                replay["scheduled_charge_kw"].sum() * parameters.interval_hours
            ),
            "planned_discharge_kwh": planned_discharge,
            "executed_charge_kwh": float(
                replay["applied_charge_kw"].sum() * parameters.interval_hours
            ),
            "executed_discharge_kwh": float(
                replay["applied_discharge_kw"].sum() * parameters.interval_hours
            ),
            "anti_export_clipped_intervals": int(
                (replay["anti_export_clip_kw"] > 1e-7).sum()
            ),
            "anti_export_clipped_kwh": anti_export_clipped,
            "anti_export_clipped_kwh_per_planned_discharge_kwh": (
                anti_export_clipped / planned_discharge
                if planned_discharge > 0.0
                else 0.0
            ),
            "physical_valid": bool(
                violations["maximum_constraint_violation"] <= 1e-7
            ),
            "maximum_constraint_violation": violations[
                "maximum_constraint_violation"
            ],
            "solver_replans": replans,
            "solver_failures": failures,
            "solver_runtime_seconds": runtime,
        },
        violations,
    )


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Controller_v3 outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace only this controller_v3 directory."
        )


def run_backtest(
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    old_feedback_replay_path: Path,
    ablation_replay_path: Path,
    ablation_summary_path: Path,
    output_dir: Path,
    *,
    initial_soc_kwh: float = 900.0,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    if not np.isclose(initial_soc_kwh, TERMINAL_SOC_REFERENCE_KWH):
        raise ValueError("Controller_v3 benchmark requires initial_soc_kwh=900.")
    _ensure_output_available(output_dir, overwrite)

    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)
    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path
    )
    forecasts = {
        "previous_day": previous_day_forecast(realized, "pv", "forecast_pv_kw"),
        "chronos": chronos_pv,
    }
    previous_load = previous_day_forecast(
        realized, "load", "forecast_load_kw"
    )
    old_results = load_old_results(old_feedback_replay_path, parameters)
    ablation_results = _load_ablation_results(ablation_replay_path)
    ablation_summary = pd.read_csv(ablation_summary_path).set_index("strategy")

    v3_results: dict[str, StrategyResult] = {}
    all_replans: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="foshan_controller_v3_highs_") as temp:
        log_dir = Path(temp)
        for source, config in SOURCE_CONFIG.items():
            result, replans = run_controller_v2(
                str(config["base_strategy"]),
                realized,
                forecasts[source],
                previous_load,
                initial_soc_kwh,
                log_dir,
                cadence_minutes=5,
                mip_relative_gap=mip_relative_gap,
                show_progress=show_progress,
                use_q10_discharge_limit=False,
                use_terminal_recovery_charge_ban=False,
                use_latest_completed_residual_for_first_step=True,
            )
            name = str(config["v3_strategy"])
            result, replans = _rename_v3_result(result, replans, name)
            v3_results[name] = result
            all_replans.extend(replans)

    comparison_results: dict[str, StrategyResult] = {}
    roles: dict[str, tuple[str, str]] = {}
    reference_solver_metrics: dict[str, tuple[int | None, int | None, float | None]] = {}
    for config in SOURCE_CONFIG.values():
        source_label = str(config["pv_source"])
        feedback_name = str(config["feedback_strategy"])
        variant_a_name = str(config["variant_a_strategy"])
        variant_d_name = str(config["variant_d_strategy"])
        v3_name = str(config["v3_strategy"])
        comparison_results[feedback_name] = old_results[feedback_name]
        comparison_results[variant_a_name] = ablation_results[variant_a_name]
        comparison_results[variant_d_name] = ablation_results[variant_d_name]
        comparison_results[v3_name] = v3_results[v3_name]
        roles.update(
            {
                feedback_name: ("feedback_v1", source_label),
                variant_a_name: ("ablation_variant_a", source_label),
                variant_d_name: ("controller_v2_variant_d", source_label),
                v3_name: ("controller_v3", source_label),
            }
        )
        reference_solver_metrics[feedback_name] = _solver_metrics(
            old_results[feedback_name]
        )
        for name in (variant_a_name, variant_d_name):
            row = ablation_summary.loc[name]
            reference_solver_metrics[name] = (
                int(row["solver_replans"]),
                int(row["solver_failures"]),
                float(row["solver_runtime_seconds"]),
            )

    common = validate_common_timestamps(comparison_results)
    initial_states = {
        name: float(result.replay["realized_soc_start_kwh"].iloc[0])
        for name, result in comparison_results.items()
    }
    if any(not np.isclose(value, initial_soc_kwh) for value in initial_states.values()):
        raise ValueError(f"Strategies do not share initial SOC: {initial_states}")

    rows: list[dict[str, Any]] = []
    physical_audits: dict[str, Any] = {}
    for name, result in comparison_results.items():
        role, source_label = roles[name]
        metrics = (
            _solver_metrics(result)
            if role == "controller_v3"
            else reference_solver_metrics[name]
        )
        row, violations = _summary_row(
            name, role, source_label, result, parameters, metrics
        )
        rows.append(row)
        physical_audits[name] = violations
    summary = pd.DataFrame(rows)

    comparison: dict[str, Any] = {}
    for source_label, table in summary.groupby("pv_source", sort=False):
        eligible = table.loc[
            table["physical_valid"] & table["terminal_comparable_895_905"]
        ]
        comparison[source_label] = {
            "highest_raw_revenue_comparable_strategy": (
                str(eligible.loc[eligible["objective_yuan"].idxmax(), "strategy"])
                if not eligible.empty
                else None
            ),
            "comparable_strategy_count": len(eligible),
            "controller_v3_comparable": bool(
                table.loc[table["role"].eq("controller_v3"),
                          "terminal_comparable_895_905"].iloc[0]
            ),
        }

    replan_table = pd.DataFrame(all_replans)
    source_times = pd.to_datetime(
        replan_table["first_step_residual_source_timestamp"], errors="raise"
    )
    control_times = pd.to_datetime(replan_table["control_time"], errors="raise")
    source_lag_seconds = (control_times - source_times).dt.total_seconds()
    leakage_audit = {
        "future_realized_pv_or_load_passed": bool(
            replan_table["future_realized_pv_or_load_passed"].any()
        ),
        "first_step_source_strictly_before_control": bool(
            (source_times < control_times).all()
        ),
        "first_step_source_lag_seconds_unique": sorted(
            source_lag_seconds.unique().tolist()
        ),
        "q10_discharge_limit_enabled": bool(
            replan_table["q10_discharge_limit_enabled"].any()
        ),
        "terminal_recovery_charge_ban_enabled": bool(
            replan_table["terminal_recovery_charge_ban_enabled"].any()
        ),
        "charging_disabled_intervals": int(
            sum(
                (result.replay["charge_limit_kw"] <= 1e-7).sum()
                for result in v3_results.values()
            )
        ),
    }
    if (
        leakage_audit["future_realized_pv_or_load_passed"]
        or not leakage_audit["first_step_source_strictly_before_control"]
        or leakage_audit["first_step_source_lag_seconds_unique"] != [300.0]
        or leakage_audit["q10_discharge_limit_enabled"]
        or leakage_audit["terminal_recovery_charge_ban_enabled"]
        or leakage_audit["charging_disabled_intervals"] != 0
    ):
        raise ValueError(f"Controller_v3 causal-policy audit failed: {leakage_audit}")

    v3_replay = pd.concat(
        [result.replay for result in v3_results.values()], ignore_index=True
    )
    audit = {
        "common_timestamp_count": len(common),
        "common_timestamp_start": common[0].isoformat(),
        "common_timestamp_end": common[-1].isoformat(),
        "identical_timestamp_sets": True,
        "identical_initial_soc": True,
        "shared_initial_soc_kwh": initial_soc_kwh,
        "terminal_comparability_band_kwh": [
            FINAL_TERMINAL_LOWER_KWH,
            FINAL_TERMINAL_UPPER_KWH,
        ],
        "leakage": leakage_audit,
        "physical": physical_audits,
    }
    metadata = {
        "controller": "controller_v3_5min",
        "cadence_minutes": 5,
        "executed_intervals_per_replan": 1,
        "terminal_soc_reference_kwh": TERMINAL_SOC_REFERENCE_KWH,
        "q10_discharge_correction": False,
        "terminal_recovery_charge_ban": False,
        "first_step_estimate": "latest_completed_five_minute_residual",
        "future_steps": "unchanged_frozen_forecasts",
        "chronos_forecast": chronos_metadata,
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
            "ablation_replay_path": str(ablation_replay_path.resolve()),
            "ablation_replay_sha256": _sha256(ablation_replay_path),
        },
        "wall_clock_runtime_seconds": time.perf_counter() - started,
    }

    v3_replay.to_csv(
        output_dir / "replay_timeseries.csv", index=False, float_format="%.15g"
    )
    summary.to_csv(
        output_dir / "strategy_summary.csv", index=False, float_format="%.15g"
    )
    _write_json(output_dir / "comparison.json", comparison)
    _write_json(output_dir / "constraint_audit.json", audit)
    _write_json(output_dir / "controller_metadata.json", metadata)
    return v3_replay, summary, comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run revenue-first five-minute Foshan controller_v3."
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
        "--ablation-replay",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_controller_v2_ablations/"
            "replay_timeseries.csv"
        ),
    )
    parser.add_argument(
        "--ablation-summary",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_controller_v2_ablations/"
            "strategy_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimization/foshan_may_controller_v3"),
    )
    parser.add_argument("--initial-soc-kwh", type=float, default=900.0)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary, comparison = run_backtest(
        dispatch_path=args.dispatch_input,
        predictions_path=args.predictions,
        selection_path=args.selection,
        old_feedback_replay_path=args.old_feedback_replay,
        ablation_replay_path=args.ablation_replay,
        ablation_summary_path=args.ablation_summary,
        output_dir=args.output_dir,
        initial_soc_kwh=args.initial_soc_kwh,
        mip_relative_gap=args.mip_relative_gap,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    print(
        f"Saved controller_v3 comparison for {len(summary)} strategies to "
        f"{args.output_dir.resolve()}; comparison={comparison}."
    )


if __name__ == "__main__":
    main()
