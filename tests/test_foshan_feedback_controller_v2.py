from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    DispatchSolution,
    TerminalBand,
    recalculate_objective,
    solve_dispatch,
)
from src.optimization.foshan_feedback_controller_v2 import (
    FINAL_TERMINAL_LOWER_KWH,
    FINAL_TERMINAL_UPPER_KWH,
    TERMINAL_SOC_REFERENCE_KWH,
    _accounting_frame,
    completed_day_q10,
    run_controller_v2,
    safe_residual_limit,
    terminal_band_for_day,
)
from src.optimization.foshan_forecast_backtest import (
    clipping_energy_kwh,
    replay_day,
)


def _day(start: str, pv: float = 900.0, load: float = 800.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=288, freq="5min"),
            "pv": pv,
            "load": load,
            "price": 0.5,
        }
    )


def _forecasts(day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = day["timestamp"].reset_index(drop=True)
    pv = pd.DataFrame(
        {
            "timestamp": timestamps,
            "forecast_pv_kw": 11.0,
            "pv_forecast_issue_time": timestamps.iloc[0],
        }
    )
    load = pd.DataFrame(
        {
            "timestamp": timestamps,
            "forecast_load_kw": 22.0,
            "forecast_load_kw_source_timestamp": timestamps
            - pd.Timedelta(days=1),
        }
    )
    return pv, load


def _fake_dispatch(
    table: pd.DataFrame,
    parameters: DispatchParameters,
    charge_first_three: bool,
) -> pd.DataFrame:
    dispatch = table.copy()
    charge = np.zeros(len(table), dtype=float)
    if charge_first_three:
        charge[:3] = 120.0
    discharge = np.zeros(len(table), dtype=float)
    soc_start = []
    soc_end = []
    soc = parameters.initial_soc_kwh
    for charge_kw in charge:
        soc_start.append(soc)
        soc += parameters.charge_efficiency * parameters.interval_hours * charge_kw
        soc_end.append(soc)
    dispatch["charge_kw"] = charge
    dispatch["discharge_kw"] = discharge
    dispatch["soc_start_kwh"] = soc_start
    dispatch["soc_kwh"] = soc_end
    dispatch["grid_import_kw"] = np.maximum(
        dispatch["load"] - dispatch["pv"] + charge, 0.0
    )
    dispatch["grid_export_kw"] = np.maximum(
        dispatch["pv"] - dispatch["load"] - charge, 0.0
    )
    dispatch["battery_mode"] = (charge > 0).astype(int)
    dispatch["grid_import_mode"] = (dispatch["grid_import_kw"] > 0).astype(int)
    return dispatch


def test_clipping_energy_uses_five_minute_conversion_and_splits_soc_reason() -> None:
    assert clipping_energy_kwh([12.0, 24.0]) == pytest.approx(3.0)
    parameters = DispatchParameters()

    upper_realized = _day("2026-05-02").iloc[:1]
    upper_schedule = pd.DataFrame(
        {
            "timestamp": upper_realized["timestamp"],
            "charge_kw": [1000.0],
            "discharge_kw": [0.0],
            "soc_start_kwh": [1999.0],
            "soc_kwh": [2000.0],
        }
    )
    upper = replay_day(upper_schedule, upper_realized, 1999.0, parameters)
    assert upper["upper_soc_clip_kw"].iloc[0] > 0.0
    assert upper["lower_soc_clip_kw"].iloc[0] == 0.0

    lower_realized = _day("2026-05-03", pv=0.0, load=1000.0).iloc[:1]
    lower_schedule = pd.DataFrame(
        {
            "timestamp": lower_realized["timestamp"],
            "charge_kw": [0.0],
            "discharge_kw": [1000.0],
            "soc_start_kwh": [1.0],
            "soc_kwh": [0.0],
        }
    )
    lower = replay_day(lower_schedule, lower_realized, 1.0, parameters)
    assert lower["lower_soc_clip_kw"].iloc[0] > 0.0
    assert lower["upper_soc_clip_kw"].iloc[0] == 0.0

    export_realized = _day("2026-05-04", pv=900.0, load=800.0).iloc[:1]
    export_schedule = pd.DataFrame(
        {
            "timestamp": export_realized["timestamp"],
            "charge_kw": [0.0],
            "discharge_kw": [500.0],
            "soc_start_kwh": [900.0],
            "soc_kwh": [900.0],
        }
    )
    export = replay_day(export_schedule, export_realized, 900.0, parameters)
    actual_residual = np.maximum(
        export["realized_load_kw"] - export["realized_pv_kw"], 0.0
    )
    assert export["anti_export_clip_kw"].iloc[0] > 0.0
    assert (
        np.maximum(export["applied_discharge_kw"] - actual_residual, 0.0).max()
        == 0.0
    )


def test_highs_terminal_band_slack_and_discharge_limit(tmp_path: Path) -> None:
    parameters = DispatchParameters(initial_soc_kwh=2000.0, terminal_soc_kwh=900.0)
    table = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-02")],
            "pv": [0.0],
            "load": [1000.0],
            "price": [0.5],
            "discharge_limit_kw": [0.0],
        }
    )
    band = TerminalBand(850.0, 950.0, 900.0, 1.0)

    solution = solve_dispatch(
        table,
        tmp_path / "solver.log",
        parameters,
        terminal_band=band,
        log_to_console=False,
    )

    assert solution.dispatch["discharge_kw"].iloc[0] == pytest.approx(0.0)
    assert solution.dispatch["soc_kwh"].iloc[-1] == pytest.approx(2000.0)
    assert solution.solver_metadata["terminal_deviation_negative_kwh"] == 0.0
    assert solution.solver_metadata["terminal_deviation_positive_kwh"] == pytest.approx(
        1050.0
    )
    physical_revenue = recalculate_objective(solution.dispatch, parameters)[
        "objective_yuan"
    ]
    assert solution.solver_objective_yuan == pytest.approx(
        physical_revenue - 1050.0
    )


def test_highs_terminal_recovery_charge_limit(tmp_path: Path) -> None:
    parameters = DispatchParameters(initial_soc_kwh=1000.0, terminal_soc_kwh=900.0)
    table = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-02")],
            "pv": [0.0],
            "load": [1000.0],
            "price": [0.5],
            "charge_limit_kw": [0.0],
            "discharge_limit_kw": [1000.0],
        }
    )

    solution = solve_dispatch(
        table,
        tmp_path / "solver.log",
        parameters,
        terminal_band=TerminalBand(850.0, 950.0, 900.0, 1.0),
        log_to_console=False,
    )

    assert solution.dispatch["charge_kw"].iloc[0] == pytest.approx(0.0)
    assert solution.dispatch["discharge_kw"].iloc[0] > 0.0
    assert solution.dispatch["soc_kwh"].iloc[-1] <= 950.0 + 1e-7
    assert solution.solver_metadata["charge_limit_column_used"] is True


def test_safe_residual_limit_matches_required_example_and_floors_at_zero(
    tmp_path: Path,
) -> None:
    safe = safe_residual_limit(
        forecast_load_kw=[700.0],
        forecast_pv_kw=[200.0],
        q10_residual_error_kw=-120.0,
    )
    assert safe.tolist() == pytest.approx([380.0])

    solution = solve_dispatch(
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-05-02")],
                "pv": [200.0],
                "load": [700.0],
                "price": [0.5],
                "charge_limit_kw": [0.0],
                "discharge_limit_kw": safe,
            }
        ),
        tmp_path / "safe_residual.log",
        DispatchParameters(initial_soc_kwh=1000.0, terminal_soc_kwh=900.0),
        terminal_band=TerminalBand(850.0, 950.0, 900.0, 1.0),
        log_to_console=False,
    )
    assert solution.dispatch["discharge_kw"].iloc[0] <= 380.0 + 1e-7

    floored = safe_residual_limit(
        forecast_load_kw=[700.0],
        forecast_pv_kw=[200.0],
        q10_residual_error_kw=-600.0,
    )
    assert floored.tolist() == [0.0]


def test_completed_day_q10_excludes_current_day_and_records_fallback() -> None:
    history = [
        (pd.Timestamp("2026-05-01 00:00"), -120.0),
        (pd.Timestamp("2026-05-01 00:05"), -80.0),
        (pd.Timestamp("2026-05-02 00:00"), 999.0),
    ]

    q10, count, fallback = completed_day_q10(
        history, pd.Timestamp("2026-05-02")
    )
    assert q10 == pytest.approx(-116.0)
    assert count == 2
    assert fallback is False

    empty_q10, empty_count, empty_fallback = completed_day_q10(
        [], pd.Timestamp("2026-05-02")
    )
    assert empty_q10 == 0.0
    assert empty_count == 0
    assert empty_fallback is True


def test_terminal_reference_is_fixed_and_may31_band_is_strict() -> None:
    normal = terminal_band_for_day(pd.Timestamp("2026-05-30"))
    final = terminal_band_for_day(pd.Timestamp("2026-05-31"))

    assert normal.reference_kwh == TERMINAL_SOC_REFERENCE_KWH
    assert (normal.lower_kwh, normal.upper_kwh) == (850.0, 950.0)
    assert final.reference_kwh == TERMINAL_SOC_REFERENCE_KWH
    assert (final.lower_kwh, final.upper_kwh) == (
        FINAL_TERMINAL_LOWER_KWH,
        FINAL_TERMINAL_UPPER_KWH,
    )


def test_controller_v2_is_causal_and_does_not_ratchet_terminal_reference(
    tmp_path: Path,
) -> None:
    realized = _day("2026-05-02")
    pv_forecast, load_forecast = _forecasts(realized)
    calls: list[tuple[pd.DataFrame, DispatchParameters, TerminalBand]] = []

    def fake_solver(table, log_path, parameters, terminal_band, **kwargs):
        calls.append((table.copy(), parameters, terminal_band))
        dispatch = _fake_dispatch(
            table,
            parameters,
            charge_first_three=len(calls) == 1,
        )
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

    result, replans = run_controller_v2(
        "controller_v2_chronos_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=fake_solver,
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-03"),
    )

    assert len(calls) == 96
    assert calls[0][0]["pv"].eq(11.0).all()
    assert calls[0][0]["load"].eq(22.0).all()
    assert not calls[0][0]["pv"].eq(900.0).any()
    assert not calls[0][0]["load"].eq(800.0).any()
    assert calls[0][0]["discharge_limit_kw"].eq(11.0).all()
    assert calls[0][0]["charge_limit_kw"].eq(1000.0).all()
    first_end_soc = float(result.replay["realized_soc_end_kwh"].iloc[2])
    assert calls[1][1].initial_soc_kwh == pytest.approx(first_end_soc)
    assert calls[1][1].terminal_soc_kwh == TERMINAL_SOC_REFERENCE_KWH
    assert all(call[2].reference_kwh == 900.0 for call in calls)
    assert [row["historical_error_sample_count"] for row in replans[:3]] == [0, 0, 0]
    assert all(row["residual_error_q10_kw"] == 0.0 for row in replans)
    assert all(row["q10_fallback_used"] for row in replans)
    for index, row in enumerate(replans):
        expected = calls[index][1].initial_soc_kwh
        if index == 0:
            expected += (
                3
                * 120.0
                * calls[index][1].charge_efficiency
                * calls[index][1].interval_hours
            )
        assert row["planned_terminal_soc_kwh"] == pytest.approx(expected)
    assert all(not row["future_realized_pv_or_load_passed"] for row in replans)
    assert (
        result.replay["realized_soc_start_kwh"].iloc[1:].to_numpy()
        == pytest.approx(result.replay["realized_soc_end_kwh"].iloc[:-1].to_numpy())
    )


def test_controller_v2_disables_charging_until_high_soc_recovers(
    tmp_path: Path,
) -> None:
    realized = _day("2026-05-02", pv=0.0, load=1000.0)
    pv_forecast, load_forecast = _forecasts(realized)
    calls: list[pd.DataFrame] = []

    def fake_solver(table, log_path, parameters, terminal_band, **kwargs):
        calls.append(table.copy())
        dispatch = _fake_dispatch(table, parameters, charge_first_three=False)
        return DispatchSolution(
            dispatch=dispatch,
            solver_objective_yuan=0.0,
            solver_metadata={
                "solver_status": "Optimal",
                "optimality_gap": 0.0,
                "mip_dual_bound": 0.0,
                "wall_clock_runtime_seconds": 0.0,
                "terminal_deviation_negative_kwh": 0.0,
                "terminal_deviation_positive_kwh": 50.0,
                "terminal_deviation_penalty_yuan": 50.0,
            },
        )

    result, replans = run_controller_v2(
        "controller_v2_previous_day_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        1000.0,
        tmp_path,
        solver=fake_solver,
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-03"),
    )

    assert len(calls) == 96
    assert all(call["charge_limit_kw"].eq(0.0).all() for call in calls)
    assert all(row["terminal_recovery_active"] for row in replans)
    assert result.replay["terminal_recovery_active"].all()
    assert result.replay["realized_soc_end_kwh"].iloc[-1] == pytest.approx(1000.0)


def test_q10_is_frozen_within_day_and_soc_propagates_between_days(
    tmp_path: Path,
) -> None:
    realized = pd.concat(
        [_day("2026-05-02"), _day("2026-05-03")],
        ignore_index=True,
    )
    pv_forecast, load_forecast = _forecasts(realized)
    calls: list[DispatchParameters] = []

    def fake_solver(table, log_path, parameters, terminal_band, **kwargs):
        calls.append(parameters)
        dispatch = _fake_dispatch(table, parameters, charge_first_three=False)
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

    result, replans = run_controller_v2(
        "controller_v2_previous_day_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=fake_solver,
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-04"),
        initial_residual_error_history=[
            (pd.Timestamp("2026-05-01 00:00"), -120.0),
            (pd.Timestamp("2026-05-01 00:05"), -80.0),
        ],
    )

    may2 = replans[:96]
    may3 = replans[96:]
    assert {row["historical_error_sample_count"] for row in may2} == {2}
    assert all(row["residual_error_q10_kw"] == pytest.approx(-116.0) for row in may2)
    assert {row["historical_error_sample_count"] for row in may3} == {290}
    assert all(row["residual_error_q10_kw"] == pytest.approx(-111.0) for row in may3)
    assert calls[96].initial_soc_kwh == pytest.approx(
        result.replay.loc[
            result.replay["timestamp"] < pd.Timestamp("2026-05-03"),
            "realized_soc_end_kwh",
        ].iloc[-1]
    )


def test_replayed_revenue_recalculates_independently(tmp_path: Path) -> None:
    realized = _day("2026-05-02")
    pv_forecast, load_forecast = _forecasts(realized)

    def zero_solver(table, log_path, parameters, terminal_band, **kwargs):
        dispatch = _fake_dispatch(table, parameters, charge_first_three=False)
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

    result, _ = run_controller_v2(
        "controller_v2_chronos_pv_previous_day_load",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=zero_solver,
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-03"),
    )
    replay = result.replay
    calculated = recalculate_objective(_accounting_frame(replay))["objective_yuan"]
    dt = 1.0 / 12.0
    baseline = np.maximum(
        replay["realized_load_kw"] - replay["realized_pv_kw"], 0.0
    )
    pv_self = replay["realized_pv_kw"] - replay["grid_export_kw"]
    direct = (
        pv_self.sum() * dt * 0.58
        + replay["grid_export_kw"].sum() * dt * 0.453
        + 0.8
        * (
            (baseline * replay["price_yuan_per_kwh"]).sum() * dt
            - (replay["grid_import_kw"] * replay["price_yuan_per_kwh"]).sum()
            * dt
        )
    )
    assert calculated == pytest.approx(direct)
    actual_residual = np.maximum(
        replay["realized_load_kw"] - replay["realized_pv_kw"], 0.0
    )
    assert np.maximum(replay["applied_discharge_kw"] - actual_residual, 0.0).max() == 0.0
