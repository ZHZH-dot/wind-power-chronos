from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    DispatchSolution,
)
from src.optimization.foshan_feedback_controller_v2 import (
    apply_intraday_load_bias,
    intraday_load_bias,
    run_controller_v2,
)


def _forecast_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-31", periods=288, freq="5min"),
            "forecast_load_kw": 100.0,
        }
    )


def test_intraday_bias_uses_only_latest_twelve_strictly_completed_samples() -> None:
    forecast = _forecast_table()
    timestamps = pd.date_range("2026-05-31", periods=16, freq="5min")
    errors = np.arange(16, dtype=float)
    errors[13:] = 1000.0
    realized = pd.DataFrame(
        {
            "timestamp": timestamps,
            "load": 100.0 + errors,
            "pv": 0.0,
            "price": 0.5,
        }
    )
    control_time = pd.Timestamp("2026-05-31 01:05")

    bias, count, oldest, newest = intraday_load_bias(
        realized, forecast, control_time
    )

    assert count == 12
    assert oldest == pd.Timestamp("2026-05-31 00:05")
    assert newest == pd.Timestamp("2026-05-31 01:00")
    assert newest < control_time
    assert bias == pytest.approx(6.5)


def test_intraday_bias_is_zero_with_fewer_than_three_samples() -> None:
    forecast = _forecast_table()
    realized = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-31", periods=3, freq="5min"),
            "load": [10.0, 20.0, 10000.0],
            "pv": 0.0,
            "price": 0.5,
        }
    )

    bias, count, oldest, newest = intraday_load_bias(
        realized, forecast, pd.Timestamp("2026-05-31 00:10")
    )

    assert bias == 0.0
    assert count == 2
    assert oldest == pd.Timestamp("2026-05-31 00:00")
    assert newest == pd.Timestamp("2026-05-31 00:05")


def test_adjusted_load_is_nonnegative() -> None:
    adjusted = apply_intraday_load_bias([5.0, 20.0, 100.0], -30.0)
    assert adjusted.tolist() == pytest.approx([0.0, 0.0, 70.0])


def test_controller_v4_adjusts_all_future_load_without_future_leakage(
    tmp_path: Path,
) -> None:
    previous = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-30 23:55")],
            "pv": [0.0],
            "load": [50.0],
            "price": [0.5],
        }
    )
    current = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-31", periods=288, freq="5min"),
            "pv": 0.0,
            "load": 60.0,
            "price": 0.5,
        }
    )
    current.loc[:2, "load"] = [80.0, 70.0, 60.0]
    current.loc[3:, "load"] = 10000.0
    realized = pd.concat([previous, current], ignore_index=True)
    timestamps = current["timestamp"]
    pv_forecast = pd.DataFrame(
        {
            "timestamp": timestamps,
            "forecast_pv_kw": 11.0,
            "pv_forecast_issue_time": timestamps.iloc[0],
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
    calls: list[pd.DataFrame] = []

    def fake_solver(table, log_path, parameters, terminal_band, **kwargs):
        calls.append(table.copy())
        dispatch = table.copy()
        dispatch["charge_kw"] = 0.0
        dispatch["discharge_kw"] = 0.0
        dispatch["soc_start_kwh"] = parameters.initial_soc_kwh
        dispatch["soc_kwh"] = parameters.initial_soc_kwh
        residual = dispatch["load"] - dispatch["pv"]
        dispatch["grid_import_kw"] = np.maximum(residual, 0.0)
        dispatch["grid_export_kw"] = np.maximum(-residual, 0.0)
        dispatch["battery_mode"] = 0
        dispatch["grid_import_mode"] = (dispatch["grid_import_kw"] > 0).astype(int)
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

    _, replans = run_controller_v2(
        "controller_v2_chronos_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=fake_solver,
        start=pd.Timestamp("2026-05-31"),
        end_exclusive=pd.Timestamp("2026-06-01"),
        cadence_minutes=5,
        use_q10_discharge_limit=False,
        use_terminal_recovery_charge_ban=False,
        use_latest_completed_residual_for_first_step=True,
        use_intraday_load_bias_correction=True,
    )

    replan = replans[3]
    table = calls[3]
    assert replan["control_time"] == "2026-05-31T00:15:00"
    assert replan["intraday_load_bias_kw"] == pytest.approx(-30.0)
    assert replan["intraday_load_bias_sample_count"] == 3
    assert replan["intraday_load_bias_oldest_timestamp"] == (
        "2026-05-31T00:00:00"
    )
    assert replan["intraday_load_bias_newest_timestamp"] == (
        "2026-05-31T00:10:00"
    )
    assert replan["first_step_frozen_load_forecast_kw"] == 100.0
    assert replan["first_step_adjusted_load_forecast_kw"] == 70.0
    assert replan["first_step_adjusted_residual_forecast_kw"] == 59.0
    assert table.iloc[0]["load"] == 60.0
    assert table.iloc[0]["pv"] == 0.0
    assert table.iloc[1:]["load"].eq(70.0).all()
    assert table.iloc[1:]["pv"].eq(11.0).all()
    assert table.iloc[1:]["discharge_limit_kw"].eq(59.0).all()
    assert all(
        pd.Timestamp(row["intraday_load_bias_newest_timestamp"])
        < pd.Timestamp(row["control_time"])
        for row in replans
        if row["intraday_load_bias_newest_timestamp"] is not None
    )
