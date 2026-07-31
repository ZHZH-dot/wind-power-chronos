import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    DispatchSolution,
)
from src.optimization.foshan_forecast_backtest import (
    StrategyResult,
    load_selected_chronos_p50,
    previous_day_forecast,
    replay_day,
    run_daily_strategy,
    summarize_strategy,
    validate_common_timestamps,
)


def _day(start: str, pv: float = 10.0, load: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=288, freq="5min"),
            "pv": pv,
            "load": load,
            "price": 0.5,
        }
    )


def test_selected_chronos_p50_expands_without_shifting_local_clock(tmp_path: Path) -> None:
    issue = pd.Timestamp("2026-05-02T00:00:00+08:00")
    predictions = pd.DataFrame(
        {
            "split": "may_2026_selection",
            "issue_time": issue,
            "target_time": pd.date_range(issue, periods=96, freq="15min"),
            "horizon_step": np.arange(1, 97),
            "target": "pv_kw",
            "model_name": "chronos2_joint_calendar",
            "context_length": 672,
            "postprocessing": "physical_clip_0_1700",
            "p50": np.arange(96, dtype=float),
            "y_true": 9999.0,
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    selection_path = tmp_path / "selection.json"
    predictions.to_csv(predictions_path, index=False)
    selection_path.write_text(
        json.dumps(
            {
                "selected_on": "may_2026_selection",
                "targets": {
                    "pv_kw": {
                        "model_name": "chronos2_joint_calendar",
                        "context_length": 672,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    expanded, metadata = load_selected_chronos_p50(
        predictions_path,
        selection_path,
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-03"),
    )

    assert len(expanded) == 288
    assert expanded["timestamp"].iloc[0] == pd.Timestamp("2026-05-02 00:00")
    assert expanded["timestamp"].iloc[-1] == pd.Timestamp("2026-05-02 23:55")
    assert expanded["forecast_pv_kw"].iloc[:6].tolist() == [0, 0, 0, 1, 1, 1]
    assert metadata["quantile"] == "p50"
    assert "y_true" not in expanded


def test_previous_day_forecast_uses_exact_prior_five_minute_slots() -> None:
    realized = pd.concat(
        [
            _day("2026-05-01", load=0.0),
            _day("2026-05-02", load=1000.0),
        ],
        ignore_index=True,
    )
    realized.loc[:287, "load"] = np.arange(288, dtype=float)

    forecast = previous_day_forecast(
        realized,
        "load",
        "forecast_load_kw",
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-03"),
    )

    assert forecast["forecast_load_kw"].tolist() == list(np.arange(288, dtype=float))
    assert (
        forecast["forecast_load_kw_source_timestamp"]
        == forecast["timestamp"] - pd.Timedelta(days=1)
    ).all()


def test_forecast_driven_solver_never_receives_realized_pv_or_load(tmp_path: Path) -> None:
    realized = _day("2026-05-02", pv=900.0, load=800.0)
    pv_forecast = pd.DataFrame(
        {
            "timestamp": realized["timestamp"],
            "forecast_pv_kw": 11.0,
            "forecast_pv_kw_source_timestamp": realized["timestamp"]
            - pd.Timedelta(days=1),
        }
    )
    load_forecast = pd.DataFrame(
        {
            "timestamp": realized["timestamp"],
            "forecast_load_kw": 22.0,
            "forecast_load_kw_source_timestamp": realized["timestamp"]
            - pd.Timedelta(days=1),
        }
    )
    captured: list[pd.DataFrame] = []

    def fake_solver(table, log_path, parameters, **kwargs):
        captured.append(table.copy())
        dispatch = table.copy()
        dispatch["charge_kw"] = 0.0
        dispatch["discharge_kw"] = 0.0
        dispatch["soc_start_kwh"] = parameters.initial_soc_kwh
        dispatch["soc_kwh"] = parameters.initial_soc_kwh
        dispatch["grid_import_kw"] = np.maximum(dispatch["load"] - dispatch["pv"], 0)
        dispatch["grid_export_kw"] = np.maximum(dispatch["pv"] - dispatch["load"], 0)
        dispatch["battery_mode"] = 0
        dispatch["grid_import_mode"] = 1
        return DispatchSolution(
            dispatch,
            0.0,
            {
                "solver_status": "Optimal",
                "optimality_gap": 0.0,
                "wall_clock_runtime_seconds": 0.0,
            },
        )

    result = run_daily_strategy(
        "forecast_test",
        realized,
        pv_forecast,
        load_forecast,
        900.0,
        tmp_path,
        solver=fake_solver,
        start=pd.Timestamp("2026-05-02"),
        end_exclusive=pd.Timestamp("2026-05-03"),
    )

    assert len(captured) == 1
    assert set(captured[0]["pv"]) == {11.0}
    assert set(captured[0]["load"]) == {22.0}
    assert set(captured[0]["price"]) == {0.5}
    assert set(result.replay["realized_pv_kw"]) == {900.0}
    assert set(result.replay["realized_load_kw"]) == {800.0}
    assert result.daily_runs[0]["solver_status"] == "Optimal"


def test_replay_clips_soc_and_discharge_that_would_create_export() -> None:
    parameters = DispatchParameters()
    realized = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-02", periods=2, freq="5min"),
            "pv": [200.0, 0.0],
            "load": [100.0, 1000.0],
            "price": [0.5, 0.5],
        }
    )
    schedule = pd.DataFrame(
        {
            "timestamp": realized["timestamp"],
            "charge_kw": [0.0, 0.0],
            "discharge_kw": [1000.0, 1000.0],
            "soc_start_kwh": [10.0, 0.0],
            "soc_kwh": [0.0, 0.0],
        }
    )

    replay = replay_day(schedule, realized, 10.0, parameters)

    assert replay.loc[0, "applied_discharge_kw"] == 0.0
    assert replay.loc[0, "anti_export_clip_kw"] > 0.0
    assert replay.loc[1, "realized_soc_end_kwh"] == pytest.approx(0.0)
    assert replay.loc[1, "applied_discharge_kw"] < 1000.0
    assert replay.loc[1, "soc_clip_kw"] > 0.0
    assert replay["realized_soc_end_kwh"].between(0.0, 2000.0).all()
    assert (replay["grid_export_kw"] <= replay["realized_pv_kw"] + 1e-9).all()


def test_replay_clips_charge_at_capacity() -> None:
    parameters = DispatchParameters()
    realized = _day("2026-05-02").iloc[:1]
    schedule = pd.DataFrame(
        {
            "timestamp": realized["timestamp"],
            "charge_kw": [1000.0],
            "discharge_kw": [0.0],
            "soc_start_kwh": [1999.0],
            "soc_kwh": [2000.0],
        }
    )

    replay = replay_day(schedule, realized, 1999.0, parameters)

    assert replay.loc[0, "realized_soc_end_kwh"] == pytest.approx(2000.0)
    assert replay.loc[0, "applied_charge_kw"] < 1000.0
    assert replay.loc[0, "soc_clip_kw"] > 0.0


def test_common_timestamp_validation_rejects_strategy_drift() -> None:
    class Result:
        def __init__(self, timestamps):
            self.replay = pd.DataFrame({"timestamp": timestamps})

    expected = pd.date_range("2026-05-02", periods=288, freq="5min")
    valid = Result(expected)
    invalid = Result(expected.delete(-1))

    with pytest.raises(ValueError, match="identical timestamp set"):
        validate_common_timestamps(
            {"valid": valid, "invalid": invalid},
            start=pd.Timestamp("2026-05-02"),
            end_exclusive=pd.Timestamp("2026-05-03"),
        )


def test_summary_reports_cumulative_soc_drift_separately_from_daily_target() -> None:
    parameters = DispatchParameters()
    realized = _day("2026-05-02").iloc[:1]
    schedule = pd.DataFrame(
        {
            "timestamp": realized["timestamp"],
            "charge_kw": [1000.0],
            "discharge_kw": [0.0],
            "soc_start_kwh": [900.0],
            "soc_kwh": [1000.0],
        }
    )
    replay = replay_day(schedule, realized, 900.0, parameters)
    final_soc = float(replay["realized_soc_end_kwh"].iloc[-1])
    result = StrategyResult(
        replay=replay,
        daily_runs=[
            {
                "scheduled_terminal_soc_kwh": final_soc,
                "terminal_difference_kwh": 0.0,
            }
        ],
    )

    summary = summarize_strategy("forecast_test", result, parameters)

    assert summary["final_terminal_difference_kwh"] == pytest.approx(0.0)
    assert summary["final_soc_difference_from_initial_kwh"] == pytest.approx(
        final_soc - 900.0
    )
    assert summary["final_soc_difference_from_initial_kwh"] > 0.0
