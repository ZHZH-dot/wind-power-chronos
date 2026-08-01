from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    DispatchSolution,
    recalculate_objective,
)
from src.optimization.foshan_feedback_controller_v2 import (
    _accounting_frame,
    latest_completed_residual,
    net_equivalent_load_pv,
    run_controller_v2,
)
from src.optimization.foshan_feedback_controller_v3 import (
    independent_replay_revenue,
)


def _realized_day() -> pd.DataFrame:
    previous = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-01 23:55")],
            "pv": [100.0],
            "load": [700.0],
            "price": [0.5],
        }
    )
    current = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-02", periods=288, freq="5min"),
            "pv": 999.0,
            "load": 1.0,
            "price": 0.5,
        }
    )
    return pd.concat([previous, current], ignore_index=True)


def _forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range("2026-05-02", periods=288, freq="5min")
    return (
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "forecast_pv_kw": 11.0,
                "pv_forecast_issue_time": timestamps[0],
            }
        ),
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "forecast_load_kw": 22.0,
                "forecast_load_kw_source_timestamp": timestamps
                - pd.Timedelta(days=1),
            }
        ),
    )


def _zero_dispatch(
    table: pd.DataFrame,
    parameters: DispatchParameters,
) -> pd.DataFrame:
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
    return dispatch


def test_latest_completed_residual_is_strictly_historical() -> None:
    realized = _realized_day()

    source, residual = latest_completed_residual(
        realized, pd.Timestamp("2026-05-02 00:00")
    )

    assert source == pd.Timestamp("2026-05-01 23:55")
    assert residual == 600.0
    assert net_equivalent_load_pv(residual) == (600.0, 0.0)
    assert net_equivalent_load_pv(-998.0) == (0.0, 998.0)


def test_controller_v3_policy_uses_only_previous_residual_for_first_step(
    tmp_path: Path,
) -> None:
    realized = _realized_day()
    pv_forecast, load_forecast = _forecasts()
    calls: list[tuple[pd.DataFrame, DispatchParameters]] = []

    def fake_solver(table, log_path, parameters, terminal_band, **kwargs):
        calls.append((table.copy(), parameters))
        return DispatchSolution(
            dispatch=_zero_dispatch(table, parameters),
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

    result, replans = run_controller_v2(
        "controller_v2_previous_day_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=fake_solver,
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-03"),
        cadence_minutes=5,
        use_q10_discharge_limit=False,
        use_terminal_recovery_charge_ban=False,
        use_latest_completed_residual_for_first_step=True,
    )

    assert len(calls) == len(replans) == len(result.replay) == 288
    first_table = calls[0][0]
    assert first_table.iloc[0]["load"] == 600.0
    assert first_table.iloc[0]["pv"] == 0.0
    assert first_table.iloc[0]["discharge_limit_kw"] == 600.0
    assert first_table.iloc[1:]["load"].eq(22.0).all()
    assert first_table.iloc[1:]["pv"].eq(11.0).all()

    second_table = calls[1][0]
    assert second_table.iloc[0]["load"] == 0.0
    assert second_table.iloc[0]["pv"] == 998.0
    assert second_table.iloc[1:]["load"].eq(22.0).all()
    assert second_table.iloc[1:]["pv"].eq(11.0).all()

    control_times = pd.to_datetime([row["control_time"] for row in replans])
    source_times = pd.to_datetime(
        [row["first_step_residual_source_timestamp"] for row in replans]
    )
    assert ((control_times - source_times).total_seconds() == 300.0).all()
    assert all(not row["q10_discharge_limit_enabled"] for row in replans)
    assert all(not row["terminal_recovery_charge_ban_enabled"] for row in replans)
    assert all(row["charge_limit_kw"] == 1000.0 for row in replans)
    assert all(not row["future_realized_pv_or_load_passed"] for row in replans)
    assert result.replay["first_step_residual_override_applied"].all()
    assert result.replay["realized_soc_end_kwh"].iloc[-1] == pytest.approx(900.0)


def test_controller_v3_independent_revenue_matches_existing_accounting(
    tmp_path: Path,
) -> None:
    realized = _realized_day()
    pv_forecast, load_forecast = _forecasts()

    def fake_solver(table, log_path, parameters, terminal_band, **kwargs):
        return DispatchSolution(
            dispatch=_zero_dispatch(table, parameters),
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

    result, _ = run_controller_v2(
        "controller_v2_chronos_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=fake_solver,
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-03"),
        cadence_minutes=5,
        use_q10_discharge_limit=False,
        use_terminal_recovery_charge_ban=False,
        use_latest_completed_residual_for_first_step=True,
    )

    expected = recalculate_objective(_accounting_frame(result.replay))["objective_yuan"]
    assert independent_replay_revenue(result.replay) == pytest.approx(expected)
