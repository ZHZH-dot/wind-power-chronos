from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    DispatchSolution,
)
from src.optimization.foshan_feedback_controller_v2 import (
    FINAL_TERMINAL_UPPER_KWH,
    final_day_immediate_charge_limit_kw,
    run_controller_v2,
)
from src.optimization.foshan_feedback_controller_v5 import _guard_audit


def _day_inputs(
    day: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range(day, periods=288, freq="5min")
    previous = pd.DataFrame(
        {
            "timestamp": [day - pd.Timedelta(minutes=5)],
            "pv": [0.0],
            "load": [100.0],
            "price": [0.5],
        }
    )
    current = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pv": 0.0,
            "load": 100.0,
            "price": 0.5,
        }
    )
    realized = pd.concat([previous, current], ignore_index=True)
    pv_forecast = pd.DataFrame(
        {
            "timestamp": timestamps,
            "forecast_pv_kw": 0.0,
            "pv_forecast_issue_time": day,
        }
    )
    load_forecast = pd.DataFrame(
        {
            "timestamp": timestamps,
            "forecast_load_kw": 100.0,
            "forecast_load_kw_source_timestamp": timestamps
            - pd.Timedelta(days=1),
        }
    )
    return realized, pv_forecast, load_forecast


def _zero_solution(
    table: pd.DataFrame,
    parameters: DispatchParameters,
) -> DispatchSolution:
    dispatch = table.copy()
    dispatch["charge_kw"] = 0.0
    dispatch["discharge_kw"] = 0.0
    dispatch["soc_start_kwh"] = parameters.initial_soc_kwh
    dispatch["soc_kwh"] = parameters.initial_soc_kwh
    dispatch["grid_import_kw"] = dispatch["load"] - dispatch["pv"]
    dispatch["grid_export_kw"] = 0.0
    dispatch["battery_mode"] = 0
    dispatch["grid_import_mode"] = 1
    return DispatchSolution(
        dispatch=dispatch,
        solver_objective_yuan=0.0,
        solver_metadata={
            "solver_status": "Optimal",
            "optimality_gap": 0.0,
            "mip_dual_bound": 0.0,
            "wall_clock_runtime_seconds": 0.0,
            "terminal_deviation_negative_kwh": 0.0,
            "terminal_deviation_positive_kwh": 0.0,
            "terminal_deviation_penalty_yuan": 0.0,
        },
    )


def test_final_day_charge_limit_matches_soc_formula() -> None:
    parameters = DispatchParameters()
    expected = (
        FINAL_TERMINAL_UPPER_KWH - 900.0
    ) / (parameters.charge_efficiency * parameters.interval_hours)

    assert final_day_immediate_charge_limit_kw(900.0, parameters) == pytest.approx(
        expected
    )
    assert final_day_immediate_charge_limit_kw(905.0, parameters) == 0.0
    assert final_day_immediate_charge_limit_kw(950.0, parameters) == 0.0
    assert final_day_immediate_charge_limit_kw(0.0, parameters) == 1000.0


def test_v5_guard_caps_only_immediate_may31_action(
    tmp_path: Path,
) -> None:
    day = pd.Timestamp("2026-05-31")
    realized, pv_forecast, load_forecast = _day_inputs(day)
    calls: list[pd.DataFrame] = []

    def fake_solver(table, log_path, parameters, terminal_band, **kwargs):
        calls.append(table.copy())
        return _zero_solution(table, parameters)

    result, replans = run_controller_v2(
        "controller_v2_chronos_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=fake_solver,
        start=day,
        end_exclusive=day + pd.Timedelta(days=1),
        cadence_minutes=5,
        use_q10_discharge_limit=False,
        use_terminal_recovery_charge_ban=False,
        use_latest_completed_residual_for_first_step=True,
        use_intraday_load_bias_correction=False,
        use_final_day_immediate_charge_guard=True,
    )

    expected = final_day_immediate_charge_limit_kw(900.0)
    assert len(calls) == len(replans) == len(result.replay) == 288
    assert calls[0].iloc[0]["charge_limit_kw"] == pytest.approx(expected)
    assert calls[0].iloc[1:]["charge_limit_kw"].eq(1000.0).all()
    assert all(row["final_day_immediate_charge_guard_applied"] for row in replans)
    assert all(row["future_charge_limits_unrestricted"] for row in replans)
    assert all(not row["q10_discharge_limit_enabled"] for row in replans)
    assert all(
        not row["terminal_recovery_charge_ban_enabled"] for row in replans
    )
    assert all(
        not row["intraday_load_bias_correction_enabled"] for row in replans
    )
    audit = _guard_audit(replans, DispatchParameters())
    assert audit["guard_applied_only_on_may31"]
    assert audit["may31_guarded_replans"] == 288


def test_v5_guard_does_not_restrict_earlier_day(tmp_path: Path) -> None:
    day = pd.Timestamp("2026-05-30")
    realized, pv_forecast, load_forecast = _day_inputs(day)
    calls: list[pd.DataFrame] = []

    def fake_solver(table, log_path, parameters, terminal_band, **kwargs):
        calls.append(table.copy())
        return _zero_solution(table, parameters)

    _, replans = run_controller_v2(
        "controller_v2_chronos_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=fake_solver,
        start=day,
        end_exclusive=day + pd.Timedelta(days=1),
        cadence_minutes=5,
        use_q10_discharge_limit=False,
        use_terminal_recovery_charge_ban=False,
        use_latest_completed_residual_for_first_step=True,
        use_intraday_load_bias_correction=False,
        use_final_day_immediate_charge_guard=True,
    )

    assert all(call["charge_limit_kw"].eq(1000.0).all() for call in calls)
    assert all(
        not row["final_day_immediate_charge_guard_applied"] for row in replans
    )
