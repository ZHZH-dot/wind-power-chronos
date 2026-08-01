"""Controller_v3 with a May 31 immediate-action SOC ceiling guard."""

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

from src.optimization.foshan_battery_milp import DispatchParameters
from src.optimization.foshan_feedback_controller_v2 import (
    FINAL_DAY,
    FINAL_TERMINAL_LOWER_KWH,
    FINAL_TERMINAL_UPPER_KWH,
    TERMINAL_SOC_REFERENCE_KWH,
    final_day_immediate_charge_limit_kw,
    run_controller_v2,
)
from src.optimization.foshan_feedback_controller_v3 import (
    _solver_metrics,
    _summary_row,
)
from src.optimization.foshan_forecast_backtest import (
    END_EXCLUSIVE,
    START,
    StrategyResult,
    load_reference_dispatch,
    load_selected_chronos_p50,
    previous_day_forecast,
    validate_common_timestamps,
)
from src.utils.runtime import git_commit, git_is_dirty


ORACLE_NAME = "fixed_actual_pv_actual_load_oracle"
V2_NAME = "controller_v2_ablation_d_chronos_pv"
V3_NAME = "controller_v3_chronos_pv_previous_day_load"
V5_NAME = "controller_v5_chronos_pv_previous_day_load"
OUTPUT_FILENAMES = (
    "replay_timeseries.csv",
    "strategy_summary.csv",
    "replan_audit.csv",
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


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Controller_v5 outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace only this controller_v5 directory."
        )


def _load_result(path: Path, strategy: str) -> StrategyResult:
    table = pd.read_csv(path, low_memory=False)
    table["timestamp"] = pd.to_datetime(table["timestamp"], errors="raise")
    replay = table.loc[table["strategy"].eq(strategy)].copy()
    replay = replay.sort_values("timestamp").reset_index(drop=True)
    if replay.empty:
        raise ValueError(f"Replay {path} is missing strategy {strategy}.")
    return StrategyResult(replay=replay, daily_runs=[])


def _saved_solver_metrics(
    summary_path: Path,
    strategy: str,
) -> tuple[int | None, int | None, float | None]:
    summary = pd.read_csv(summary_path)
    rows = summary.loc[summary["strategy"].eq(strategy)]
    if len(rows) != 1:
        raise ValueError(
            f"Summary {summary_path} must contain one row for {strategy}."
        )
    row = rows.iloc[0]
    return (
        int(row["solver_replans"]),
        int(row["solver_failures"]),
        float(row["solver_runtime_seconds"]),
    )


def _rename_v5_result(
    result: StrategyResult,
    replans: list[dict[str, Any]],
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    replay = result.replay.copy()
    replay["strategy"] = V5_NAME
    replay["controller"] = "controller_v5_5min_final_day_guard"
    daily_runs = []
    for row in result.daily_runs:
        updated = dict(row)
        updated["strategy"] = V5_NAME
        daily_runs.append(updated)
    renamed_replans = []
    for row in replans:
        updated = dict(row)
        updated["strategy"] = V5_NAME
        renamed_replans.append(updated)
    return StrategyResult(replay=replay, daily_runs=daily_runs), renamed_replans


def _guard_audit(
    replans: list[dict[str, Any]],
    parameters: DispatchParameters,
) -> dict[str, Any]:
    table = pd.DataFrame(replans)
    control_time = pd.to_datetime(table["control_time"], errors="raise")
    source_time = pd.to_datetime(
        table["first_step_residual_source_timestamp"], errors="raise"
    )
    is_final_day = control_time.dt.normalize().eq(FINAL_DAY)
    expected_limits = np.where(
        is_final_day,
        [
            final_day_immediate_charge_limit_kw(soc, parameters)
            for soc in table["initial_soc_kwh"].to_numpy(dtype=float)
        ],
        parameters.power_limit_kw,
    )
    actual_limits = table["immediate_charge_limit_kw"].to_numpy(dtype=float)
    audit = {
        "future_realized_pv_or_load_passed": bool(
            table["future_realized_pv_or_load_passed"].any()
        ),
        "first_step_source_strictly_before_control": bool(
            (source_time < control_time).all()
        ),
        "first_step_source_lag_seconds_unique": sorted(
            (control_time - source_time).dt.total_seconds().unique().tolist()
        ),
        "q10_discharge_limit_enabled": bool(
            table["q10_discharge_limit_enabled"].any()
        ),
        "terminal_recovery_charge_ban_enabled": bool(
            table["terminal_recovery_charge_ban_enabled"].any()
        ),
        "intraday_load_bias_correction_enabled": bool(
            table["intraday_load_bias_correction_enabled"].any()
        ),
        "final_day_guard_enabled": bool(
            table["final_day_immediate_charge_guard_enabled"].all()
        ),
        "guard_applied_only_on_may31": bool(
            np.array_equal(
                table["final_day_immediate_charge_guard_applied"].to_numpy(
                    dtype=bool
                ),
                is_final_day.to_numpy(dtype=bool),
            )
        ),
        "may31_guarded_replans": int(is_final_day.sum()),
        "pre_may31_charge_limits_unrestricted": bool(
            np.allclose(actual_limits[~is_final_day], parameters.power_limit_kw)
        ),
        "future_planned_charge_limits_unrestricted": bool(
            table["future_charge_limits_unrestricted"].all()
        ),
        "immediate_charge_limits_match_formula": bool(
            np.allclose(actual_limits, expected_limits, atol=1e-9)
        ),
        "guard_binding_replans": int(
            (actual_limits[is_final_day] < parameters.power_limit_kw - 1e-7).sum()
        ),
        "solver_failures": int((table["solver_status"] != "Optimal").sum()),
    }
    expected = {
        "future_realized_pv_or_load_passed": False,
        "first_step_source_strictly_before_control": True,
        "first_step_source_lag_seconds_unique": [300.0],
        "q10_discharge_limit_enabled": False,
        "terminal_recovery_charge_ban_enabled": False,
        "intraday_load_bias_correction_enabled": False,
        "final_day_guard_enabled": True,
        "guard_applied_only_on_may31": True,
        "may31_guarded_replans": 288,
        "pre_may31_charge_limits_unrestricted": True,
        "future_planned_charge_limits_unrestricted": True,
        "immediate_charge_limits_match_formula": True,
        "solver_failures": 0,
    }
    observed = {key: audit[key] for key in expected}
    if observed != expected:
        raise ValueError(f"Controller_v5 policy audit failed: {audit}")
    return audit


def run_backtest(
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    v3_replay_path: Path,
    v3_summary_path: Path,
    v2_replay_path: Path,
    v2_summary_path: Path,
    oracle_replay_path: Path,
    output_dir: Path,
    *,
    initial_soc_kwh: float = 900.0,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    if not np.isclose(initial_soc_kwh, TERMINAL_SOC_REFERENCE_KWH):
        raise ValueError("Controller_v5 benchmark requires initial_soc_kwh=900.")
    _ensure_output_available(output_dir, overwrite)

    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)
    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path
    )
    previous_load = previous_day_forecast(
        realized, "load", "forecast_load_kw"
    )
    references = {
        V3_NAME: _load_result(v3_replay_path, V3_NAME),
        V2_NAME: _load_result(v2_replay_path, V2_NAME),
        ORACLE_NAME: _load_result(oracle_replay_path, ORACLE_NAME),
    }

    with tempfile.TemporaryDirectory(prefix="foshan_controller_v5_highs_") as temp:
        v5_result, replans = run_controller_v2(
            "controller_v2_chronos_pv_previous_day_load",
            realized,
            chronos_pv,
            previous_load,
            initial_soc_kwh,
            Path(temp),
            cadence_minutes=5,
            mip_relative_gap=mip_relative_gap,
            show_progress=show_progress,
            use_q10_discharge_limit=False,
            use_terminal_recovery_charge_ban=False,
            use_latest_completed_residual_for_first_step=True,
            use_intraday_load_bias_correction=False,
            use_final_day_immediate_charge_guard=True,
        )
    v5_result, replans = _rename_v5_result(v5_result, replans)

    results = {**references, V5_NAME: v5_result}
    common = validate_common_timestamps(results)
    initial_states = {
        name: float(result.replay["realized_soc_start_kwh"].iloc[0])
        for name, result in results.items()
    }
    if any(not np.isclose(value, initial_soc_kwh) for value in initial_states.values()):
        raise ValueError(f"Strategies do not share initial SOC: {initial_states}")

    metric_config = {
        V3_NAME: (
            "controller_v3",
            "chronos2_zero_shot_postprocessed_p50",
            _saved_solver_metrics(v3_summary_path, V3_NAME),
        ),
        V2_NAME: (
            "controller_v2_variant_d",
            "chronos2_zero_shot_postprocessed_p50",
            _saved_solver_metrics(v2_summary_path, V2_NAME),
        ),
        ORACLE_NAME: (
            "equal_soc_oracle_non_deployable",
            "actual_future_pv",
            (None, None, None),
        ),
        V5_NAME: (
            "controller_v5",
            "chronos2_zero_shot_postprocessed_p50",
            _solver_metrics(v5_result),
        ),
    }
    rows: list[dict[str, Any]] = []
    physical: dict[str, Any] = {}
    for name, result in results.items():
        role, pv_source, solver_metrics = metric_config[name]
        row, violations = _summary_row(
            name, role, pv_source, result, parameters, solver_metrics
        )
        if name == ORACLE_NAME:
            row["load_source"] = "actual_future_provisional_load"
        rows.append(row)
        physical[name] = violations
    summary = pd.DataFrame(rows)
    indexed = summary.set_index("strategy")
    v5_row = indexed.loc[V5_NAME]
    comparison = {
        "controller_v5_terminal_comparable": bool(
            v5_row["terminal_comparable_895_905"]
        ),
        "controller_v5_physical_valid": bool(v5_row["physical_valid"]),
        "revenue_differences_v5_yuan": {
            name: float(v5_row["objective_yuan"] - indexed.loc[name, "objective_yuan"])
            for name in (V3_NAME, V2_NAME, ORACLE_NAME)
        },
        "equal_soc_oracle_is_non_deployable": True,
    }
    guard_audit = _guard_audit(replans, parameters)
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
        "policy": guard_audit,
        "physical": physical,
    }
    metadata = {
        "controller": "controller_v5_5min_final_day_guard",
        "cadence_minutes": 5,
        "executed_intervals_per_replan": 1,
        "terminal_soc_reference_kwh": TERMINAL_SOC_REFERENCE_KWH,
        "q10_discharge_correction": False,
        "terminal_recovery_charge_ban": False,
        "intraday_load_bias_correction": False,
        "final_day_immediate_charge_ceiling_kwh": FINAL_TERMINAL_UPPER_KWH,
        "future_planned_actions_restricted_by_guard": False,
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
            "v3_replay_path": str(v3_replay_path.resolve()),
            "v2_replay_path": str(v2_replay_path.resolve()),
            "oracle_replay_path": str(oracle_replay_path.resolve()),
        },
        "wall_clock_runtime_seconds": time.perf_counter() - started,
    }

    v5_result.replay.to_csv(
        output_dir / "replay_timeseries.csv", index=False, float_format="%.15g"
    )
    summary.to_csv(
        output_dir / "strategy_summary.csv", index=False, float_format="%.15g"
    )
    pd.DataFrame(replans).to_csv(
        output_dir / "replan_audit.csv", index=False, float_format="%.15g"
    )
    _write_json(output_dir / "comparison.json", comparison)
    _write_json(output_dir / "constraint_audit.json", audit)
    _write_json(output_dir / "controller_metadata.json", metadata)
    return v5_result.replay, summary, comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Foshan controller_v5 with a May 31 immediate charge guard."
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
        "--v3-replay",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_controller_v3/replay_timeseries.csv"
        ),
    )
    parser.add_argument(
        "--v3-summary",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_controller_v3/strategy_summary.csv"
        ),
    )
    parser.add_argument(
        "--v2-replay",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_controller_v2_ablations/"
            "replay_timeseries.csv"
        ),
    )
    parser.add_argument(
        "--v2-summary",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_controller_v2_ablations/"
            "strategy_summary.csv"
        ),
    )
    parser.add_argument(
        "--oracle-replay",
        type=Path,
        default=Path(
            "results/optimization/foshan_may_state_feedback/replay_timeseries.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimization/foshan_may_controller_v5"),
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
        v3_replay_path=args.v3_replay,
        v3_summary_path=args.v3_summary,
        v2_replay_path=args.v2_replay,
        v2_summary_path=args.v2_summary,
        oracle_replay_path=args.oracle_replay,
        output_dir=args.output_dir,
        initial_soc_kwh=args.initial_soc_kwh,
        mip_relative_gap=args.mip_relative_gap,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    print(
        f"Saved controller_v5 comparison for {len(summary)} strategies to "
        f"{args.output_dir.resolve()}: {comparison}."
    )


if __name__ == "__main__":
    main()
