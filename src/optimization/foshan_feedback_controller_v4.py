"""Five-minute controller_v4 with a causal intraday load-bias correction."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.foshan_battery_milp import DispatchParameters
from src.optimization.foshan_feedback_controller_v2 import (
    FINAL_TERMINAL_LOWER_KWH,
    FINAL_TERMINAL_UPPER_KWH,
    _physical_violations,
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


MAY31 = pd.Timestamp("2026-05-31")
JUNE1 = pd.Timestamp("2026-06-01")
V3_NAME = "controller_v3_chronos_pv_previous_day_load"
V4_NAME = "controller_v4_chronos_pv_previous_day_load"
OUTPUT_FILENAMES = (
    "replay_timeseries.csv",
    "strategy_summary.csv",
    "bias_audit.csv",
    "comparison.json",
    "constraint_audit.json",
    "controller_metadata.json",
)


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
            f"Controller_v4 outputs already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace only this controller_v4 directory."
        )


def _load_v3_replay(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    table["timestamp"] = pd.to_datetime(table["timestamp"], errors="raise")
    replay = table.loc[table["strategy"].eq(V3_NAME)].copy()
    replay = replay.sort_values("timestamp").reset_index(drop=True)
    if replay.empty:
        raise ValueError(f"Controller_v3 replay is missing {V3_NAME}.")
    return replay


def _rename_result(
    result: StrategyResult,
    replans: list[dict[str, Any]],
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    replay = result.replay.copy()
    replay["strategy"] = V4_NAME
    replay["controller"] = "controller_v4_5min_intraday_load_bias"
    daily_runs = []
    for row in result.daily_runs:
        updated = dict(row)
        updated["strategy"] = V4_NAME
        daily_runs.append(updated)
    renamed_replans = []
    for row in replans:
        updated = dict(row)
        updated["strategy"] = V4_NAME
        renamed_replans.append(updated)
    return StrategyResult(replay=replay, daily_runs=daily_runs), renamed_replans


def _bias_audit_table(replans: list[dict[str, Any]]) -> pd.DataFrame:
    columns = (
        "strategy",
        "control_time",
        "intraday_load_bias_kw",
        "intraday_load_bias_sample_count",
        "intraday_load_bias_oldest_timestamp",
        "intraday_load_bias_newest_timestamp",
        "first_step_frozen_load_forecast_kw",
        "first_step_frozen_pv_forecast_kw",
        "first_step_adjusted_load_forecast_kw",
        "first_step_adjusted_residual_forecast_kw",
        "first_step_residual_source_timestamp",
        "first_step_measured_residual_kw",
        "initial_soc_kwh",
        "realized_soc_after_execution_kwh",
        "solver_status",
        "solver_runtime_seconds",
    )
    return pd.DataFrame(replans).loc[:, columns]


def _causal_audit(replans: list[dict[str, Any]]) -> dict[str, Any]:
    table = pd.DataFrame(replans)
    control = pd.to_datetime(table["control_time"], errors="raise")
    first_step_source = pd.to_datetime(
        table["first_step_residual_source_timestamp"], errors="raise"
    )
    oldest = pd.to_datetime(
        table["intraday_load_bias_oldest_timestamp"], errors="coerce"
    )
    newest = pd.to_datetime(
        table["intraday_load_bias_newest_timestamp"], errors="coerce"
    )
    counts = table["intraday_load_bias_sample_count"].to_numpy(dtype=int)
    bias = table["intraday_load_bias_kw"].to_numpy(dtype=float)
    frozen_load = table["first_step_frozen_load_forecast_kw"].to_numpy(
        dtype=float
    )
    frozen_pv = table["first_step_frozen_pv_forecast_kw"].to_numpy(dtype=float)
    adjusted_load = table["first_step_adjusted_load_forecast_kw"].to_numpy(
        dtype=float
    )
    adjusted_residual = table[
        "first_step_adjusted_residual_forecast_kw"
    ].to_numpy(dtype=float)
    nonempty = counts > 0
    audit = {
        "future_realized_pv_or_load_passed": bool(
            table["future_realized_pv_or_load_passed"].any()
        ),
        "first_step_source_strictly_before_control": bool(
            (first_step_source < control).all()
        ),
        "first_step_source_lag_seconds_unique": sorted(
            (control - first_step_source).dt.total_seconds().unique().tolist()
        ),
        "bias_sources_strictly_before_control": bool(
            (oldest[nonempty] < control[nonempty]).all()
            and (newest[nonempty] < control[nonempty]).all()
        ),
        "bias_sources_current_day_only": bool(
            (oldest[nonempty].dt.normalize() == control[nonempty].dt.normalize()).all()
            and (newest[nonempty].dt.normalize() == control[nonempty].dt.normalize()).all()
        ),
        "maximum_bias_sample_count": int(counts.max()),
        "fewer_than_three_samples_zero_bias": bool(
            np.allclose(bias[counts < 3], 0.0)
        ),
        "adjusted_load_nonnegative": bool((adjusted_load >= 0.0).all()),
        "adjusted_load_matches_bias": bool(
            np.allclose(adjusted_load, np.maximum(0.0, frozen_load + bias))
        ),
        "adjusted_residual_matches_load_and_pv": bool(
            np.allclose(
                adjusted_residual,
                np.maximum(0.0, adjusted_load - frozen_pv),
            )
        ),
        "q10_discharge_limit_enabled": bool(
            table["q10_discharge_limit_enabled"].any()
        ),
        "terminal_recovery_charge_ban_enabled": bool(
            table["terminal_recovery_charge_ban_enabled"].any()
        ),
        "charging_disabled_replans": int(
            (table["charge_limit_kw"] <= 1e-7).sum()
        ),
        "solver_failures": int((table["solver_status"] != "Optimal").sum()),
    }
    expected = {
        "future_realized_pv_or_load_passed": False,
        "first_step_source_strictly_before_control": True,
        "first_step_source_lag_seconds_unique": [300.0],
        "bias_sources_strictly_before_control": True,
        "bias_sources_current_day_only": True,
        "maximum_bias_sample_count": 12,
        "fewer_than_three_samples_zero_bias": True,
        "adjusted_load_nonnegative": True,
        "adjusted_load_matches_bias": True,
        "adjusted_residual_matches_load_and_pv": True,
        "q10_discharge_limit_enabled": False,
        "terminal_recovery_charge_ban_enabled": False,
        "charging_disabled_replans": 0,
        "solver_failures": 0,
    }
    if audit != expected:
        raise ValueError(f"Controller_v4 causal audit failed: {audit}")
    return audit


def _run_period(
    realized: pd.DataFrame,
    chronos_pv: pd.DataFrame,
    previous_load: pd.DataFrame,
    initial_soc_kwh: float,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    mip_relative_gap: float,
    show_progress: bool,
) -> tuple[StrategyResult, list[dict[str, Any]]]:
    period_pv = chronos_pv.loc[
        (chronos_pv["timestamp"] >= start)
        & (chronos_pv["timestamp"] < end_exclusive)
    ].copy()
    period_load = previous_load.loc[
        (previous_load["timestamp"] >= start)
        & (previous_load["timestamp"] < end_exclusive)
    ].copy()
    with tempfile.TemporaryDirectory(prefix="foshan_controller_v4_highs_") as temp:
        result, replans = run_controller_v2(
            "controller_v2_chronos_pv_previous_day_load",
            realized,
            period_pv,
            period_load,
            initial_soc_kwh,
            Path(temp),
            start=start,
            end_exclusive=end_exclusive,
            cadence_minutes=5,
            mip_relative_gap=mip_relative_gap,
            show_progress=show_progress,
            use_q10_discharge_limit=False,
            use_terminal_recovery_charge_ban=False,
            use_latest_completed_residual_for_first_step=True,
            use_intraday_load_bias_correction=True,
        )
    return _rename_result(result, replans)


def _save_outputs(
    output_dir: Path,
    replay: pd.DataFrame,
    summary: pd.DataFrame,
    bias_audit: pd.DataFrame,
    comparison: dict[str, Any],
    audit: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    replay.to_csv(
        output_dir / "replay_timeseries.csv", index=False, float_format="%.15g"
    )
    summary.to_csv(
        output_dir / "strategy_summary.csv", index=False, float_format="%.15g"
    )
    bias_audit.to_csv(
        output_dir / "bias_audit.csv", index=False, float_format="%.15g"
    )
    _write_json(output_dir / "comparison.json", comparison)
    _write_json(output_dir / "constraint_audit.json", audit)
    _write_json(output_dir / "controller_metadata.json", metadata)


def run_may31_diagnostic(
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    v3_replay_path: Path,
    output_dir: Path,
    *,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    _ensure_output_available(output_dir, overwrite)
    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)
    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path
    )
    previous_load = previous_day_forecast(
        realized, "load", "forecast_load_kw"
    )
    v3_replay = _load_v3_replay(v3_replay_path)
    v3_may31 = v3_replay.loc[v3_replay["timestamp"].ge(MAY31)]
    if len(v3_may31) != 288:
        raise ValueError("Controller_v3 May 31 replay must contain 288 rows.")
    initial_soc = float(v3_may31["realized_soc_start_kwh"].iloc[0])

    result, replans = _run_period(
        realized,
        chronos_pv,
        previous_load,
        initial_soc,
        MAY31,
        JUNE1,
        mip_relative_gap,
        show_progress,
    )
    row, physical = _summary_row(
        V4_NAME,
        "controller_v4_may31_diagnostic",
        "chronos2_zero_shot_postprocessed_p50",
        result,
        parameters,
        _solver_metrics(result),
    )
    summary = pd.DataFrame([row])
    causal = _causal_audit(replans)
    final_soc = float(row["final_soc_kwh"])
    passed = bool(
        FINAL_TERMINAL_LOWER_KWH <= final_soc <= FINAL_TERMINAL_UPPER_KWH
        and row["physical_valid"]
        and row["revenue_recalculation_abs_error_yuan"] <= 1e-7
    )
    comparison = {
        "may31_terminal_gate_passed": passed,
        "full_run_permitted": passed,
        "final_soc_kwh": final_soc,
        "terminal_slack_kwh": float(row["terminal_slack_kwh"]),
    }
    audit = {"causal": causal, "physical": physical}
    metadata = {
        "stage": "may31_diagnostic",
        "starting_soc_from_controller_v3_kwh": initial_soc,
        "chronos_forecast": chronos_metadata,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "wall_clock_runtime_seconds": time.perf_counter() - started,
    }
    _save_outputs(
        output_dir,
        result.replay,
        summary,
        _bias_audit_table(replans),
        comparison,
        audit,
        metadata,
    )
    return result.replay, summary, comparison


def run_full_backtest(
    dispatch_path: Path,
    predictions_path: Path,
    selection_path: Path,
    v3_replay_path: Path,
    v3_summary_path: Path,
    output_dir: Path,
    *,
    mip_relative_gap: float = 1e-7,
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    _ensure_output_available(output_dir, overwrite)
    parameters = DispatchParameters()
    realized = load_reference_dispatch(dispatch_path)
    chronos_pv, chronos_metadata = load_selected_chronos_p50(
        predictions_path, selection_path
    )
    previous_load = previous_day_forecast(
        realized, "load", "forecast_load_kw"
    )
    v3_replay = _load_v3_replay(v3_replay_path)
    v3_result = StrategyResult(replay=v3_replay, daily_runs=[])
    v3_summary = pd.read_csv(v3_summary_path)
    v3_row = v3_summary.loc[v3_summary["strategy"].eq(V3_NAME)].iloc[0]

    v4_result, replans = _run_period(
        realized,
        chronos_pv,
        previous_load,
        900.0,
        START,
        END_EXCLUSIVE,
        mip_relative_gap,
        show_progress,
    )
    validate_common_timestamps({V3_NAME: v3_result, V4_NAME: v4_result})
    v3_metrics, v3_physical = _summary_row(
        V3_NAME,
        "controller_v3",
        "chronos2_zero_shot_postprocessed_p50",
        v3_result,
        parameters,
        (
            int(v3_row["solver_replans"]),
            int(v3_row["solver_failures"]),
            float(v3_row["solver_runtime_seconds"]),
        ),
    )
    v4_metrics, v4_physical = _summary_row(
        V4_NAME,
        "controller_v4",
        "chronos2_zero_shot_postprocessed_p50",
        v4_result,
        parameters,
        _solver_metrics(v4_result),
    )
    summary = pd.DataFrame([v3_metrics, v4_metrics])
    causal = _causal_audit(replans)
    prior_comparable = v3_summary.loc[
        v3_summary["terminal_comparable_895_905"].astype(bool)
        & v3_summary["physical_valid"].astype(bool)
    ]
    prior_comparable_max = (
        float(prior_comparable["objective_yuan"].max())
        if not prior_comparable.empty
        else None
    )
    v4_success = bool(
        v4_metrics["terminal_comparable_895_905"]
        and v4_metrics["physical_valid"]
        and v4_metrics["revenue_recalculation_abs_error_yuan"] <= 1e-7
        and (
            prior_comparable_max is None
            or v4_metrics["objective_yuan"] > prior_comparable_max
        )
    )
    comparison = {
        "controller_v4_success": v4_success,
        "revenue_difference_v4_minus_v3_yuan": float(
            v4_metrics["objective_yuan"] - v3_metrics["objective_yuan"]
        ),
        "prior_terminal_comparable_controller_count": len(prior_comparable),
        "prior_terminal_comparable_max_revenue_yuan": prior_comparable_max,
    }
    audit = {
        "causal": causal,
        "physical": {V3_NAME: v3_physical, V4_NAME: v4_physical},
    }
    metadata = {
        "stage": "full_may",
        "chronos_forecast": chronos_metadata,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "wall_clock_runtime_seconds": time.perf_counter() - started,
    }
    _save_outputs(
        output_dir,
        v4_result.replay,
        summary,
        _bias_audit_table(replans),
        comparison,
        audit,
        metadata,
    )
    return v4_result.replay, summary, comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Foshan controller_v4.")
    parser.add_argument("--stage", choices=("diagnostic", "full"), required=True)
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mip-relative-gap", type=float, default=1e-7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or Path(
        "results/optimization/foshan_may_controller_v4_may31_diagnostic"
        if args.stage == "diagnostic"
        else "results/optimization/foshan_may_controller_v4"
    )
    common = {
        "dispatch_path": args.dispatch_input,
        "predictions_path": args.predictions,
        "selection_path": args.selection,
        "v3_replay_path": args.v3_replay,
        "output_dir": output_dir,
        "mip_relative_gap": args.mip_relative_gap,
        "overwrite": args.overwrite,
        "show_progress": not args.quiet,
    }
    if args.stage == "diagnostic":
        _, _, comparison = run_may31_diagnostic(**common)
    else:
        _, _, comparison = run_full_backtest(
            **common,
            v3_summary_path=args.v3_summary,
        )
    print(f"Saved controller_v4 {args.stage} to {output_dir.resolve()}: {comparison}")


if __name__ == "__main__":
    main()
