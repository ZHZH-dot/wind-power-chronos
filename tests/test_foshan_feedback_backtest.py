from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    DispatchSolution,
)
from src.optimization.foshan_feedback_backtest import (
    STRATEGY_LABELS,
    run_feedback_strategy,
    summarize_result,
    validate_causal_forecast_sources,
    validate_shared_initial_soc,
)
from src.optimization.foshan_forecast_backtest import (
    StrategyResult,
    replay_day,
    validate_common_timestamps,
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


def _forecast(day: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def test_feedback_uses_frozen_forecasts_and_propagates_realized_soc(
    tmp_path: Path,
) -> None:
    realized = _day("2026-05-02")
    pv_forecast, load_forecast = _forecast(realized)
    calls: list[tuple[pd.DataFrame, DispatchParameters]] = []

    def fake_solver(table, log_path, parameters, **kwargs):
        calls.append((table.copy(), parameters))
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
                "wall_clock_runtime_seconds": 0.0,
            },
        )

    result, replans = run_feedback_strategy(
        "feedback_chronos_pv_previous_day_load",
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
    assert len(calls[0][0]) == 288
    assert len(calls[1][0]) == 285
    assert len(calls[-1][0]) == 3
    first_execution_end = float(result.replay["realized_soc_end_kwh"].iloc[2])
    assert calls[1][1].initial_soc_kwh == pytest.approx(first_execution_end)
    assert result.replay["realized_soc_start_kwh"].iloc[0] == 900.0
    assert (
        result.replay["realized_soc_start_kwh"].iloc[1:].to_numpy()
        == pytest.approx(result.replay["realized_soc_end_kwh"].iloc[:-1].to_numpy())
    )
    expected = pd.date_range("2026-05-02", periods=288, freq="5min")
    assert result.replay["timestamp"].tolist() == expected.tolist()
    assert all(not row["future_realized_pv_or_load_passed"] for row in replans)
    assert all(row["executed_intervals"] == 3 for row in replans)


def test_causal_validation_rejects_future_or_updated_forecast_sources() -> None:
    realized = _day("2026-05-02")
    pv, load = _forecast(realized)
    future_issue = pv.copy()
    future_issue["pv_forecast_issue_time"] = future_issue["timestamp"] + pd.Timedelta(
        minutes=5
    )
    with pytest.raises(ValueError, match="must not follow"):
        validate_causal_forecast_sources(future_issue, load)

    updated = pv.copy()
    updated.loc[100:, "pv_forecast_issue_time"] = pd.Timestamp("2026-05-02 00:15")
    with pytest.raises(ValueError, match="remain frozen"):
        validate_causal_forecast_sources(updated, load)


def test_shared_timestamps_and_initial_soc_are_enforced() -> None:
    timestamps = pd.date_range("2026-05-02", periods=288, freq="5min")

    def result(initial_soc: float, values: pd.DatetimeIndex) -> StrategyResult:
        return StrategyResult(
            replay=pd.DataFrame(
                {
                    "timestamp": values,
                    "realized_soc_start_kwh": initial_soc,
                }
            ),
            daily_runs=[],
        )

    valid = {"a": result(900.0, timestamps), "b": result(900.0, timestamps)}
    assert validate_shared_initial_soc(valid, 900.0) == {"a": 900.0, "b": 900.0}
    assert len(
        validate_common_timestamps(
            valid,
            start=pd.Timestamp("2026-05-02"),
            end_exclusive=pd.Timestamp("2026-05-03"),
        )
    ) == 288

    invalid_soc = {**valid, "c": result(901.0, timestamps)}
    with pytest.raises(ValueError, match="do not share initial SOC"):
        validate_shared_initial_soc(invalid_soc, 900.0)

    invalid_time = {**valid, "c": result(900.0, timestamps.delete(-1))}
    with pytest.raises(ValueError, match="identical timestamp set"):
        validate_common_timestamps(
            invalid_time,
            start=pd.Timestamp("2026-05-02"),
            end_exclusive=pd.Timestamp("2026-05-03"),
        )


def test_summary_matches_independent_revenue_calculation() -> None:
    parameters = DispatchParameters()
    realized = _day("2026-05-02").iloc[:3].copy()
    schedule = pd.DataFrame(
        {
            "timestamp": realized["timestamp"],
            "charge_kw": [100.0, 0.0, 0.0],
            "discharge_kw": [0.0, 100.0, 0.0],
            "soc_start_kwh": [900.0, 908.0, 899.0],
            "soc_kwh": [908.0, 899.0, 899.0],
        }
    )
    replay = replay_day(schedule, realized, 900.0, parameters)
    replay.insert(0, "strategy", "feedback_chronos_pv_previous_day_load")
    result = StrategyResult(replay=replay, daily_runs=[])

    summary = summarize_result(
        "feedback_chronos_pv_previous_day_load", result, parameters
    )

    dt = 1.0 / 12.0
    baseline_import = np.maximum(
        replay["realized_load_kw"] - replay["realized_pv_kw"], 0.0
    )
    pv_self = replay["realized_pv_kw"] - replay["grid_export_kw"]
    direct = (
        pv_self.sum() * dt * 0.58
        + replay["grid_export_kw"].sum() * dt * 0.453
        + 0.8
        * (
            (baseline_import * replay["price_yuan_per_kwh"]).sum() * dt
            - (replay["grid_import_kw"] * replay["price_yuan_per_kwh"]).sum()
            * dt
        )
    )
    assert summary["objective_yuan"] == pytest.approx(direct)
    assert summary["planned_charge_kwh"] == pytest.approx(100.0 * dt)
    assert summary["executed_discharge_kwh"] == pytest.approx(0.0)


def test_oracle_diagnostics_are_never_marked_deployable() -> None:
    for name, labels in STRATEGY_LABELS.items():
        if labels["oracle_diagnostic"]:
            assert not labels["deployable"], name
