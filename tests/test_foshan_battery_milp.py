from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization.foshan_battery_milp import (
    DispatchParameters,
    constraint_violations,
    load_dispatch_input,
    recalculate_objective,
    solve_dispatch,
)


def _dispatch_input() -> pd.DataFrame:
    periods = 24
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-15", periods=periods, freq="5min"),
            "pv": np.r_[np.zeros(8), np.full(8, 100.0), np.zeros(8)],
            "load": np.full(periods, 300.0),
            "price": np.r_[
                np.full(8, 0.15),
                np.full(8, 0.45),
                np.full(8, 0.65),
            ],
        }
    )


@pytest.fixture
def solved_dispatch(tmp_path: Path):
    parameters = DispatchParameters()
    solution = solve_dispatch(
        _dispatch_input(),
        tmp_path / "solver.log",
        parameters,
        log_to_console=False,
    )
    return solution, parameters


def test_dispatch_obeys_all_physical_and_binary_constraints(solved_dispatch) -> None:
    solution, parameters = solved_dispatch
    dispatch = solution.dispatch
    violations = constraint_violations(dispatch, parameters)
    tolerance = 1e-6

    assert dispatch["soc_start_kwh"].between(0.0, 2000.0).all()
    assert dispatch["soc_kwh"].between(0.0, 2000.0).all()
    assert dispatch["charge_kw"].max() <= 1000.0 + tolerance
    assert dispatch["discharge_kw"].max() <= 1000.0 + tolerance
    assert np.minimum(dispatch["charge_kw"], dispatch["discharge_kw"]).max() <= tolerance
    assert (
        np.minimum(dispatch["grid_import_kw"], dispatch["grid_export_kw"]).max()
        <= tolerance
    )
    assert dispatch["soc_start_kwh"].iloc[0] == pytest.approx(900.0, abs=tolerance)
    assert dispatch["soc_kwh"].iloc[-1] == pytest.approx(900.0, abs=tolerance)
    assert (
        dispatch["discharge_kw"]
        <= np.maximum(dispatch["load"] - dispatch["pv"], 0.0) + tolerance
    ).all()
    assert (dispatch["grid_export_kw"] <= dispatch["pv"] + tolerance).all()
    assert violations["maximum_constraint_violation"] <= tolerance


def test_five_minute_soc_energy_conversion_is_exact() -> None:
    parameters = DispatchParameters()
    charge_kw = 120.0
    expected_change = (
        charge_kw * parameters.interval_hours * parameters.charge_efficiency
    )
    dispatch = pd.DataFrame(
        {
            "pv": [0.0],
            "load": [0.0],
            "price": [0.1],
            "charge_kw": [charge_kw],
            "discharge_kw": [0.0],
            "soc_start_kwh": [900.0],
            "soc_kwh": [900.0 + expected_change],
            "grid_import_kw": [charge_kw],
            "grid_export_kw": [0.0],
            "battery_mode": [1],
            "grid_import_mode": [1],
        }
    )

    violations = constraint_violations(dispatch, parameters)

    assert parameters.interval_hours == pytest.approx(1.0 / 12.0)
    assert dispatch["soc_kwh"].iloc[0] - 900.0 == pytest.approx(expected_change)
    assert violations["soc_dynamics_kwh"] == pytest.approx(0.0, abs=1e-12)


def test_independent_objective_matches_highs_reported_objective(solved_dispatch) -> None:
    solution, parameters = solved_dispatch
    recalculated = recalculate_objective(solution.dispatch, parameters)

    assert recalculated["objective_yuan"] == pytest.approx(
        solution.solver_objective_yuan,
        abs=1e-7,
    )


def test_loader_ignores_original_p_opt_and_soc_opt(tmp_path: Path) -> None:
    source = _dispatch_input()
    source.insert(0, "source_index", np.arange(len(source)))
    source["p_opt"] = 999999.0
    source["soc_opt"] = -999999.0
    path = tmp_path / "reference.csv"
    source.to_csv(path, index=False)

    loaded = load_dispatch_input(path)

    assert list(loaded.columns) == ["timestamp", "pv", "load", "price"]
    assert "p_opt" not in loaded
    assert "soc_opt" not in loaded
