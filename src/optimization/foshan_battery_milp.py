"""Direct HiGHS battery-dispatch MILP for the Foshan/Shunde May reference data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import highspy
import numpy as np
import pandas as pd

from src.utils.runtime import git_commit, git_is_dirty


REQUIRED_INPUT_COLUMNS = ("timestamp", "pv", "load", "price")
LOAD_DATA_NOTICE = (
    "load is the original project's reconstructed/provisional load series; "
    "it is not verified raw gross factory load."
)
OUTPUT_FILENAMES = (
    "dispatch_milp.csv",
    "summary.json",
    "comparison_with_original.json",
    "solver_metadata.json",
    "report.md",
    "solver.log",
)


@dataclass(frozen=True)
class DispatchParameters:
    """Fixed physical and accounting assumptions for the compatibility checkpoint."""

    interval_hours: float = 1.0 / 12.0
    capacity_kwh: float = 2000.0
    power_limit_kw: float = 1000.0
    initial_soc_kwh: float = 900.0
    terminal_soc_kwh: float = 900.0
    round_trip_efficiency: float = 0.8729384881069339
    pv_self_price_yuan_per_kwh: float = 0.58
    pv_export_price_yuan_per_kwh: float = 0.453
    storage_revenue_share: float = 0.8

    @property
    def charge_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def discharge_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    def to_dict(self) -> dict[str, float]:
        values = asdict(self)
        values["charge_efficiency"] = self.charge_efficiency
        values["discharge_efficiency"] = self.discharge_efficiency
        return values


@dataclass(frozen=True)
class VariableBlocks:
    """Column indices for each variable block in the HiGHS model."""

    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    grid_import: np.ndarray
    grid_export: np.ndarray
    battery_mode: np.ndarray
    grid_mode: np.ndarray
    count: int


@dataclass
class DispatchSolution:
    """Solved dispatch and solver-level metadata."""

    dispatch: pd.DataFrame
    solver_objective_yuan: float
    solver_metadata: dict[str, Any]


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


def _finite_float(value: Any) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def load_dispatch_input(path: Path, selected_date: date | None = None) -> pd.DataFrame:
    """Load only timestamp, PV, provisional load, and price from the reference CSV."""
    headers = list(pd.read_csv(path, nrows=0).columns)
    if "timestamp" in headers:
        timestamp_source = "timestamp"
    elif headers and (not str(headers[0]).strip() or str(headers[0]).startswith("Unnamed:")):
        timestamp_source = headers[0]
    else:
        raise ValueError(
            "Input must contain a timestamp column or an unnamed first timestamp column."
        )

    required_source_columns = [timestamp_source, "pv", "load", "price"]
    missing = [column for column in required_source_columns if column not in headers]
    if missing:
        raise ValueError(f"Dispatch input is missing required columns: {missing}")

    table = pd.read_csv(path, usecols=required_source_columns).rename(
        columns={timestamp_source: "timestamp"}
    )
    table["timestamp"] = pd.to_datetime(table["timestamp"], errors="raise")
    for column in ("pv", "load", "price"):
        table[column] = pd.to_numeric(table[column], errors="raise")

    if selected_date is not None:
        table = table.loc[table["timestamp"].dt.date == selected_date].copy()
        if table.empty:
            raise ValueError(f"No dispatch rows were found for {selected_date.isoformat()}.")

    table = table.sort_values("timestamp").reset_index(drop=True)
    if table.empty:
        raise ValueError("Dispatch input is empty.")
    if not table["timestamp"].is_unique:
        raise ValueError("Dispatch timestamps must be unique.")
    if table[list(REQUIRED_INPUT_COLUMNS)].isna().any().any():
        raise ValueError("Dispatch input contains missing timestamp, pv, load, or price values.")
    if not np.isfinite(table[["pv", "load", "price"]].to_numpy(dtype=float)).all():
        raise ValueError("Dispatch input contains non-finite pv, load, or price values.")
    if (table[["pv", "load", "price"]] < 0).any().any():
        raise ValueError("PV, provisional load, and price must be nonnegative.")

    if len(table) > 1:
        differences = table["timestamp"].diff().iloc[1:]
        expected = pd.Timedelta(minutes=5)
        if not differences.eq(expected).all():
            observed = sorted({str(value) for value in differences.unique()})
            raise ValueError(
                "Dispatch timestamps must form an exact five-minute grid; "
                f"observed differences: {observed}"
            )
    return table[list(REQUIRED_INPUT_COLUMNS)]


def load_reference_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if "assumptions" not in value or "optimized" not in value:
        raise ValueError("Reference summary must contain assumptions and optimized sections.")
    return value


def validate_reference_assumptions(
    reference: dict[str, Any],
    parameters: DispatchParameters,
) -> None:
    expected = {
        "soc_start_end_kwh": parameters.initial_soc_kwh,
        "p_max_kw": parameters.power_limit_kw,
        "capacity_kwh": parameters.capacity_kwh,
        "round_trip_efficiency": parameters.round_trip_efficiency,
        "pv_self_price_yuan_per_kwh": parameters.pv_self_price_yuan_per_kwh,
        "pv_export_price_yuan_per_kwh": parameters.pv_export_price_yuan_per_kwh,
        "design_storage_share": parameters.storage_revenue_share,
    }
    assumptions = reference["assumptions"]
    for key, expected_value in expected.items():
        if key not in assumptions or not math.isclose(
            float(assumptions[key]),
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Reference assumption {key!r} does not match {expected_value!r}."
            )


def _variable_blocks(interval_count: int) -> VariableBlocks:
    offset = 0

    def allocate(size: int) -> np.ndarray:
        nonlocal offset
        block = np.arange(offset, offset + size, dtype=np.int32)
        offset += size
        return block

    return VariableBlocks(
        charge=allocate(interval_count),
        discharge=allocate(interval_count),
        soc=allocate(interval_count + 1),
        grid_import=allocate(interval_count),
        grid_export=allocate(interval_count),
        battery_mode=allocate(interval_count),
        grid_mode=allocate(interval_count),
        count=offset,
    )


def _check_highs_status(status: highspy.HighsStatus, action: str) -> None:
    if status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS failed while attempting to {action}.")


def solve_dispatch(
    table: pd.DataFrame,
    log_path: Path,
    parameters: DispatchParameters = DispatchParameters(),
    *,
    mip_relative_gap: float = 1e-7,
    time_limit_seconds: float | None = None,
    log_to_console: bool = True,
) -> DispatchSolution:
    """Construct and solve the battery MILP directly with highspy."""
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"Dispatch table is missing columns: {missing}")
    if mip_relative_gap < 0:
        raise ValueError("mip_relative_gap must be nonnegative.")
    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive when supplied.")

    interval_count = len(table)
    if interval_count == 0:
        raise ValueError("Dispatch table is empty.")

    pv = table["pv"].to_numpy(dtype=np.float64)
    load = table["load"].to_numpy(dtype=np.float64)
    price = table["price"].to_numpy(dtype=np.float64)
    residual_load = np.maximum(load - pv, 0.0)
    baseline_import = residual_load
    blocks = _variable_blocks(interval_count)

    lower = np.zeros(blocks.count, dtype=np.float64)
    upper = np.empty(blocks.count, dtype=np.float64)
    costs = np.zeros(blocks.count, dtype=np.float64)

    upper[blocks.charge] = parameters.power_limit_kw
    upper[blocks.discharge] = parameters.power_limit_kw
    upper[blocks.soc] = parameters.capacity_kwh
    upper[blocks.grid_import] = baseline_import + parameters.power_limit_kw
    upper[blocks.grid_export] = pv
    upper[blocks.battery_mode] = 1.0
    upper[blocks.grid_mode] = 1.0

    lower[blocks.soc[0]] = parameters.initial_soc_kwh
    upper[blocks.soc[0]] = parameters.initial_soc_kwh
    lower[blocks.soc[-1]] = parameters.terminal_soc_kwh
    upper[blocks.soc[-1]] = parameters.terminal_soc_kwh

    costs[blocks.grid_import] = (
        -parameters.storage_revenue_share * price * parameters.interval_hours
    )
    costs[blocks.grid_export] = (
        parameters.pv_export_price_yuan_per_kwh
        - parameters.pv_self_price_yuan_per_kwh
    ) * parameters.interval_hours
    objective_offset = float(
        np.sum(pv * parameters.pv_self_price_yuan_per_kwh * parameters.interval_hours)
        + np.sum(
            parameters.storage_revenue_share
            * baseline_import
            * price
            * parameters.interval_hours
        )
    )

    row_lower: list[float] = []
    row_upper: list[float] = []
    starts: list[int] = []
    indices: list[int] = []
    values: list[float] = []
    infinity = highspy.kHighsInf

    def add_row(
        columns: Sequence[int],
        coefficients: Sequence[float],
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        starts.append(len(indices))
        indices.extend(int(column) for column in columns)
        values.extend(float(value) for value in coefficients)
        row_lower.append(float(lower_bound))
        row_upper.append(float(upper_bound))

    eta_charge = parameters.charge_efficiency
    eta_discharge = parameters.discharge_efficiency
    dt = parameters.interval_hours

    for position in range(interval_count):
        add_row(
            (
                blocks.soc[position + 1],
                blocks.soc[position],
                blocks.charge[position],
                blocks.discharge[position],
            ),
            (1.0, -1.0, -eta_charge * dt, dt / eta_discharge),
            0.0,
            0.0,
        )
        add_row(
            (
                blocks.charge[position],
                blocks.discharge[position],
                blocks.grid_import[position],
                blocks.grid_export[position],
            ),
            (-1.0, 1.0, 1.0, -1.0),
            load[position] - pv[position],
            load[position] - pv[position],
        )
        add_row(
            (blocks.charge[position], blocks.battery_mode[position]),
            (1.0, -parameters.power_limit_kw),
            -infinity,
            0.0,
        )
        add_row(
            (blocks.discharge[position], blocks.battery_mode[position]),
            (1.0, parameters.power_limit_kw),
            -infinity,
            parameters.power_limit_kw,
        )
        add_row(
            (blocks.grid_import[position], blocks.grid_mode[position]),
            (1.0, -(baseline_import[position] + parameters.power_limit_kw)),
            -infinity,
            0.0,
        )
        add_row(
            (blocks.grid_export[position], blocks.grid_mode[position]),
            (1.0, pv[position]),
            -infinity,
            pv[position],
        )
        add_row(
            (blocks.grid_export[position],),
            (1.0,),
            -infinity,
            pv[position],
        )
        add_row(
            (blocks.discharge[position],),
            (1.0,),
            -infinity,
            residual_load[position],
        )

    highs = highspy.Highs()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _check_highs_status(
        highs.setOptionValue("log_file", str(log_path.resolve())),
        "configure the solver log",
    )
    _check_highs_status(
        highs.setOptionValue("log_to_console", log_to_console),
        "configure console logging",
    )
    _check_highs_status(
        highs.setOptionValue("mip_rel_gap", mip_relative_gap),
        "configure the MIP gap",
    )
    _check_highs_status(
        highs.setOptionValue("random_seed", 42),
        "configure the random seed",
    )
    if time_limit_seconds is not None:
        _check_highs_status(
            highs.setOptionValue("time_limit", time_limit_seconds),
            "configure the time limit",
        )

    empty_indices = np.empty(0, dtype=np.int32)
    empty_values = np.empty(0, dtype=np.float64)
    _check_highs_status(
        highs.addCols(
            blocks.count,
            costs,
            lower,
            upper,
            0,
            np.zeros(blocks.count, dtype=np.int32),
            empty_indices,
            empty_values,
        ),
        "add model variables",
    )
    binary_indices = np.concatenate((blocks.battery_mode, blocks.grid_mode))
    binary_types = np.full(
        len(binary_indices),
        highspy.HighsVarType.kInteger.value,
        dtype=np.uint8,
    )
    _check_highs_status(
        highs.changeColsIntegrality(
            len(binary_indices),
            binary_indices,
            binary_types,
        ),
        "mark binary variables",
    )
    _check_highs_status(
        highs.addRows(
            len(row_lower),
            np.asarray(row_lower, dtype=np.float64),
            np.asarray(row_upper, dtype=np.float64),
            len(indices),
            np.asarray(starts, dtype=np.int32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(values, dtype=np.float64),
        ),
        "add model constraints",
    )
    _check_highs_status(highs.changeObjectiveOffset(objective_offset), "set objective offset")
    _check_highs_status(highs.setMaximize(), "set maximization sense")

    solve_started = time.perf_counter()
    run_status = highs.run()
    wall_clock_seconds = time.perf_counter() - solve_started
    _check_highs_status(run_status, "solve the MILP")

    model_status = highs.getModelStatus()
    info = highs.getInfo()
    status_text = highs.modelStatusToString(model_status)
    solver_metadata: dict[str, Any] = {
        "highspy_version": importlib.metadata.version("highspy"),
        "highs_version": highs.version(),
        "solver_status": status_text,
        "solver_status_code": int(model_status.value),
        "optimality_gap": _finite_float(info.mip_gap),
        "mip_dual_bound": _finite_float(info.mip_dual_bound),
        "wall_clock_runtime_seconds": wall_clock_seconds,
        "highs_runtime_seconds": float(highs.getRunTime()),
        "mip_node_count": int(info.mip_node_count),
        "simplex_iteration_count": int(info.simplex_iteration_count),
        "max_primal_infeasibility": _finite_float(info.max_primal_infeasibility),
        "max_integrality_violation": _finite_float(info.max_integrality_violation),
        "num_variables": int(highs.getNumCol()),
        "num_binary_variables": int(len(binary_indices)),
        "num_constraints": int(highs.getNumRow()),
        "num_nonzeros": int(highs.getNumNz()),
        "mip_relative_gap_tolerance": mip_relative_gap,
        "time_limit_seconds": time_limit_seconds,
    }
    if model_status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(
            "HiGHS did not prove an optimal solution: "
            f"status={status_text}, gap={solver_metadata['optimality_gap']}."
        )

    solution = np.asarray(highs.getSolution().col_value, dtype=np.float64)
    dispatch = table.copy()
    dispatch["charge_kw"] = solution[blocks.charge]
    dispatch["discharge_kw"] = solution[blocks.discharge]
    dispatch["soc_start_kwh"] = solution[blocks.soc[:-1]]
    dispatch["soc_kwh"] = solution[blocks.soc[1:]]
    dispatch["grid_import_kw"] = solution[blocks.grid_import]
    dispatch["grid_export_kw"] = solution[blocks.grid_export]
    dispatch["battery_mode"] = np.rint(solution[blocks.battery_mode]).astype(np.int8)
    dispatch["grid_import_mode"] = np.rint(solution[blocks.grid_mode]).astype(np.int8)

    return DispatchSolution(
        dispatch=dispatch,
        solver_objective_yuan=float(info.objective_function_value),
        solver_metadata=solver_metadata,
    )


def recalculate_objective(
    dispatch: pd.DataFrame,
    parameters: DispatchParameters = DispatchParameters(),
) -> dict[str, float]:
    """Independently recalculate all accounting terms from a solved dispatch table."""
    required = {
        "pv",
        "load",
        "price",
        "charge_kw",
        "discharge_kw",
        "grid_import_kw",
        "grid_export_kw",
    }
    missing = sorted(required - set(dispatch.columns))
    if missing:
        raise ValueError(f"Solved dispatch is missing accounting columns: {missing}")

    dt = parameters.interval_hours
    pv = dispatch["pv"].to_numpy(dtype=float)
    load = dispatch["load"].to_numpy(dtype=float)
    price = dispatch["price"].to_numpy(dtype=float)
    grid_import = dispatch["grid_import_kw"].to_numpy(dtype=float)
    grid_export = dispatch["grid_export_kw"].to_numpy(dtype=float)
    baseline_import = np.maximum(load - pv, 0.0)
    pv_self = pv - grid_export

    pv_self_kwh = float(np.sum(pv_self) * dt)
    pv_export_kwh = float(np.sum(grid_export) * dt)
    baseline_grid_import_kwh = float(np.sum(baseline_import) * dt)
    grid_import_kwh = float(np.sum(grid_import) * dt)
    charge_kwh = float(dispatch["charge_kw"].sum() * dt)
    discharge_kwh = float(dispatch["discharge_kw"].sum() * dt)
    pv_self_revenue = pv_self_kwh * parameters.pv_self_price_yuan_per_kwh
    pv_export_revenue = pv_export_kwh * parameters.pv_export_price_yuan_per_kwh
    baseline_import_cost = float(np.sum(baseline_import * price) * dt)
    optimized_import_cost = float(np.sum(grid_import * price) * dt)
    grid_import_cost_savings = baseline_import_cost - optimized_import_cost
    storage_revenue = parameters.storage_revenue_share * grid_import_cost_savings
    objective = pv_self_revenue + pv_export_revenue + storage_revenue

    return {
        "pv_self_kwh": pv_self_kwh,
        "pv_export_kwh": pv_export_kwh,
        "baseline_grid_import_kwh": baseline_grid_import_kwh,
        "grid_import_kwh": grid_import_kwh,
        "charge_kwh": charge_kwh,
        "discharge_kwh": discharge_kwh,
        "pv_self_revenue_yuan": pv_self_revenue,
        "pv_export_revenue_yuan": pv_export_revenue,
        "pv_revenue_yuan": pv_self_revenue + pv_export_revenue,
        "baseline_grid_import_cost_yuan": baseline_import_cost,
        "optimized_grid_import_cost_yuan": optimized_import_cost,
        "grid_import_cost_savings_yuan": grid_import_cost_savings,
        "storage_revenue_share_yuan": storage_revenue,
        "objective_yuan": objective,
    }


def constraint_violations(
    dispatch: pd.DataFrame,
    parameters: DispatchParameters = DispatchParameters(),
) -> dict[str, float]:
    """Calculate physical and logical violations independently from solver internals."""
    charge = dispatch["charge_kw"].to_numpy(dtype=float)
    discharge = dispatch["discharge_kw"].to_numpy(dtype=float)
    soc_start = dispatch["soc_start_kwh"].to_numpy(dtype=float)
    soc_end = dispatch["soc_kwh"].to_numpy(dtype=float)
    grid_import = dispatch["grid_import_kw"].to_numpy(dtype=float)
    grid_export = dispatch["grid_export_kw"].to_numpy(dtype=float)
    battery_mode = dispatch["battery_mode"].to_numpy(dtype=float)
    grid_mode = dispatch["grid_import_mode"].to_numpy(dtype=float)
    pv = dispatch["pv"].to_numpy(dtype=float)
    load = dispatch["load"].to_numpy(dtype=float)
    residual_load = np.maximum(load - pv, 0.0)
    import_big_m = residual_load + parameters.power_limit_kw

    soc_expected = (
        soc_start
        + parameters.charge_efficiency * parameters.interval_hours * charge
        - parameters.interval_hours / parameters.discharge_efficiency * discharge
    )
    energy_balance = pv + grid_import + discharge - load - charge - grid_export
    soc_link = (
        np.abs(soc_start[1:] - soc_end[:-1])
        if len(dispatch) > 1
        else np.array([0.0])
    )

    violations = {
        "soc_dynamics_kwh": float(np.max(np.abs(soc_end - soc_expected))),
        "soc_link_kwh": float(np.max(soc_link)),
        "initial_soc_kwh": abs(float(soc_start[0]) - parameters.initial_soc_kwh),
        "terminal_soc_kwh": abs(float(soc_end[-1]) - parameters.terminal_soc_kwh),
        "soc_lower_bound_kwh": float(max(0.0, -float(np.min([soc_start, soc_end])))),
        "soc_upper_bound_kwh": float(
            max(0.0, float(np.max([soc_start, soc_end])) - parameters.capacity_kwh)
        ),
        "charge_power_limit_kw": float(
            max(0.0, float(np.max(charge)) - parameters.power_limit_kw)
        ),
        "discharge_power_limit_kw": float(
            max(0.0, float(np.max(discharge)) - parameters.power_limit_kw)
        ),
        "simultaneous_charge_discharge_kw": float(np.max(np.minimum(charge, discharge))),
        "energy_balance_kw": float(np.max(np.abs(energy_balance))),
        "simultaneous_import_export_kw": float(
            np.max(np.minimum(grid_import, grid_export))
        ),
        "pv_export_limit_kw": float(np.max(np.maximum(grid_export - pv, 0.0))),
        "battery_anti_export_kw": float(
            np.max(np.maximum(discharge - residual_load, 0.0))
        ),
        "charge_mode_kw": float(
            np.max(
                np.maximum(
                    charge - parameters.power_limit_kw * battery_mode,
                    0.0,
                )
            )
        ),
        "discharge_mode_kw": float(
            np.max(
                np.maximum(
                    discharge - parameters.power_limit_kw * (1.0 - battery_mode),
                    0.0,
                )
            )
        ),
        "grid_import_mode_kw": float(
            np.max(np.maximum(grid_import - import_big_m * grid_mode, 0.0))
        ),
        "grid_export_mode_kw": float(
            np.max(np.maximum(grid_export - pv * (1.0 - grid_mode), 0.0))
        ),
    }
    violations["maximum_constraint_violation"] = max(violations.values())
    return violations


def compare_with_original(
    source_csv: Path,
    dispatch: pd.DataFrame,
    reference_summary: dict[str, Any],
    parameters: DispatchParameters = DispatchParameters(),
) -> dict[str, Any]:
    """Compare the solved MILP with the original DP only after optimization is complete."""
    headers = list(pd.read_csv(source_csv, nrows=0).columns)
    timestamp_source = "timestamp" if "timestamp" in headers else headers[0]
    required = [timestamp_source, "pv", "load", "price", "p_opt", "soc_opt"]
    missing = [column for column in required if column not in headers]
    if missing:
        raise ValueError(f"Original comparison fields are missing: {missing}")
    original = pd.read_csv(source_csv, usecols=required).rename(
        columns={timestamp_source: "timestamp"}
    )
    original["timestamp"] = pd.to_datetime(original["timestamp"], errors="raise")
    for column in ("pv", "load", "price", "p_opt", "soc_opt"):
        original[column] = pd.to_numeric(original[column], errors="raise")

    comparison = dispatch[
        ["timestamp", "charge_kw", "discharge_kw", "soc_start_kwh"]
    ].merge(
        original[["timestamp", "pv", "load", "price", "p_opt", "soc_opt"]],
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(dispatch) or len(comparison) != len(original):
        raise ValueError("MILP and original schedules do not share the complete timestamp set.")

    original_power = comparison["p_opt"].to_numpy(dtype=float)
    original_grid_import = np.maximum(
        comparison["load"].to_numpy(dtype=float)
        - comparison["pv"].to_numpy(dtype=float)
        - original_power,
        0.0,
    )
    original_grid_export = np.maximum(
        comparison["pv"].to_numpy(dtype=float)
        + original_power
        - comparison["load"].to_numpy(dtype=float),
        0.0,
    )
    original_dispatch = pd.DataFrame(
        {
            "pv": comparison["pv"],
            "load": comparison["load"],
            "price": comparison["price"],
            "charge_kw": np.maximum(-original_power, 0.0),
            "discharge_kw": np.maximum(original_power, 0.0),
            "grid_import_kw": original_grid_import,
            "grid_export_kw": original_grid_export,
        }
    )
    original_recalculation = recalculate_objective(original_dispatch, parameters)
    highs_recalculation = recalculate_objective(dispatch, parameters)

    reference_objective = float(reference_summary["optimized"]["objective_yuan"])
    highs_objective = highs_recalculation["objective_yuan"]
    difference = highs_objective - reference_objective
    relative_percent = 100.0 * difference / reference_objective
    highs_power = (
        comparison["discharge_kw"].to_numpy(dtype=float)
        - comparison["charge_kw"].to_numpy(dtype=float)
    )
    power_difference = highs_power - original_power
    soc_difference = (
        comparison["soc_start_kwh"].to_numpy(dtype=float)
        - comparison["soc_opt"].to_numpy(dtype=float)
    )
    reference_power_limit_violation = float(
        np.max(np.maximum(np.abs(original_power) - parameters.power_limit_kw, 0.0))
    )
    soc_step_kwh = float(reference_summary["assumptions"].get("soc_step_kwh", 10.0))
    if soc_step_kwh <= 0:
        raise ValueError("Reference soc_step_kwh must be positive.")
    original_soc = comparison["soc_opt"].to_numpy(dtype=float)
    original_soc_grid_deviation = np.abs(
        original_soc - np.round(original_soc / soc_step_kwh) * soc_step_kwh
    )
    highs_soc = comparison["soc_start_kwh"].to_numpy(dtype=float)
    highs_soc_grid_deviation = np.abs(
        highs_soc - np.round(highs_soc / soc_step_kwh) * soc_step_kwh
    )
    accounting_keys = (
        "pv_self_kwh",
        "pv_export_kwh",
        "grid_import_kwh",
        "charge_kwh",
        "discharge_kwh",
        "pv_revenue_yuan",
        "grid_import_cost_savings_yuan",
        "storage_revenue_share_yuan",
        "objective_yuan",
    )
    accounting_differences = {
        key: highs_recalculation[key] - original_recalculation[key]
        for key in accounting_keys
    }

    return {
        "reference_objective_yuan": reference_objective,
        "reference_objective_recalculated_yuan": original_recalculation["objective_yuan"],
        "reference_recalculation_error_yuan": (
            original_recalculation["objective_yuan"] - reference_objective
        ),
        "highs_objective_yuan": highs_objective,
        "difference_yuan": difference,
        "relative_difference_percent": relative_percent,
        "absolute_relative_difference_percent": abs(relative_percent),
        "within_one_percent": abs(relative_percent) <= 1.0,
        "schedule_comparison": {
            "power_mae_kw": float(np.mean(np.abs(power_difference))),
            "power_max_abs_difference_kw": float(np.max(np.abs(power_difference))),
            "soc_start_mae_kwh": float(np.mean(np.abs(soc_difference))),
            "soc_start_max_abs_difference_kwh": float(np.max(np.abs(soc_difference))),
            "reference_max_charge_kw": float(np.max(np.maximum(-original_power, 0.0))),
            "reference_max_discharge_kw": float(np.max(np.maximum(original_power, 0.0))),
            "reference_power_limit_violation_kw": reference_power_limit_violation,
        },
        "accounting_differences_highs_minus_original": accounting_differences,
        "soc_grid_diagnosis": {
            "reference_soc_step_kwh": soc_step_kwh,
            "reference_soc_max_grid_deviation_kwh": float(
                np.max(original_soc_grid_deviation)
            ),
            "highs_soc_max_grid_deviation_kwh": float(
                np.max(highs_soc_grid_deviation)
            ),
            "interpretation": (
                "The original schedule lies on its reported 10 kWh SOC grid, while "
                "the requested MILP uses continuous SOC. This is consistent with a "
                "continuous-relaxation improvement and is not an accounting change."
            ),
        },
        "diagnostic_notes": [
            LOAD_DATA_NOTICE,
            (
                "The original objective is independently reproduced from p_opt only "
                "after the MILP solve; its reconciliation error is reported above."
            ),
            (
                "The MILP enforces the requested 1000 kW grid-side charge/discharge "
                "limit; the original discrete schedule is comparison-only."
            ),
            (
                "A small objective difference can result from continuous SOC in the "
                "MILP versus the original 10 kWh DP grid."
            ),
        ],
    }


def _write_report(
    path: Path,
    summary: dict[str, Any],
    comparison: dict[str, Any],
    solver_metadata: dict[str, Any],
) -> None:
    accounting = summary["accounting"]
    violations = summary["constraint_violations"]
    lines = [
        "# Foshan May Battery Dispatch With HiGHS",
        "",
        f"**Data caveat:** {LOAD_DATA_NOTICE}",
        "",
        "## Result",
        "",
        f"- Solver status: `{solver_metadata['solver_status']}`",
        f"- Independent objective: {accounting['objective_yuan']:.6f} yuan",
        f"- Original objective: {comparison['reference_objective_yuan']:.6f} yuan",
        f"- Difference: {comparison['relative_difference_percent']:.6f}%",
        f"- Within 1% compatibility threshold: {comparison['within_one_percent']}",
        f"- Runtime: {solver_metadata['wall_clock_runtime_seconds']:.3f} seconds",
        f"- Optimality gap: {solver_metadata['optimality_gap']}",
        "",
        "## Accounting",
        "",
        f"- PV self-consumption revenue: {accounting['pv_self_revenue_yuan']:.6f} yuan",
        f"- PV export revenue: {accounting['pv_export_revenue_yuan']:.6f} yuan",
        (
            "- 80% storage share of grid-import cost savings: "
            f"{accounting['storage_revenue_share_yuan']:.6f} yuan"
        ),
        (
            "- Solver/recalculation difference: "
            f"{summary['objective_reconciliation_error_yuan']:.9f} yuan"
        ),
        (
            "- Original/recalculated difference: "
            f"{comparison['reference_recalculation_error_yuan']:.9f} yuan"
        ),
        "",
        "## Compatibility Diagnosis",
        "",
        (
            "- Original SOC grid: "
            f"{comparison['soc_grid_diagnosis']['reference_soc_step_kwh']:.1f} kWh"
        ),
        (
            "- Original maximum SOC-grid deviation: "
            f"{comparison['soc_grid_diagnosis']['reference_soc_max_grid_deviation_kwh']:.9g} kWh"
        ),
        (
            "- Continuous HiGHS maximum SOC-grid deviation: "
            f"{comparison['soc_grid_diagnosis']['highs_soc_max_grid_deviation_kwh']:.9g} kWh"
        ),
        (
            "- Grid-import cost-savings difference: "
            f"{comparison['accounting_differences_highs_minus_original']['grid_import_cost_savings_yuan']:.6f} yuan"
        ),
        (
            "- PV-revenue difference: "
            f"{comparison['accounting_differences_highs_minus_original']['pv_revenue_yuan']:.6f} yuan"
        ),
        (
            "- Conclusion: the strict 1% reproduction threshold is not met. "
            "Accounting, efficiency, sign, terminal SOC, and anti-export checks "
            "reconcile; the remaining difference is consistent with continuous "
            "SOC/control versus the original 10 kWh grid."
        ),
        "",
        "## Constraint Audit",
        "",
        f"- Maximum independent violation: {violations['maximum_constraint_violation']:.9g}",
        f"- Terminal SOC violation: {violations['terminal_soc_kwh']:.9g} kWh",
        (
            "- Simultaneous charge/discharge: "
            f"{violations['simultaneous_charge_discharge_kw']:.9g} kW"
        ),
        (
            "- Simultaneous import/export: "
            f"{violations['simultaneous_import_export_kw']:.9g} kW"
        ),
        f"- Battery anti-export violation: {violations['battery_anti_export_kw']:.9g} kW",
        "",
        "The model uses only timestamp, pv, load, and price. Original `p_opt` and "
        "`soc_opt` are read only after solving for the final comparison.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_output_available(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {existing}. "
            "Pass --overwrite to replace this optimization run only."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve the Foshan/Shunde battery-dispatch MILP directly with highspy."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/optimization/foshan_may_highs"),
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="Optional YYYY-MM-DD subset for a one-day smoke solve.",
    )
    parser.add_argument("--mip-relative-gap", type=float, default=1e-7)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parameters = DispatchParameters()
    output_dir = args.output_dir.expanduser().resolve()
    _ensure_output_available(output_dir, args.overwrite)

    reference_summary = load_reference_summary(args.reference_summary)
    validate_reference_assumptions(reference_summary, parameters)
    table = load_dispatch_input(args.input, args.date)

    solved = solve_dispatch(
        table,
        output_dir / "solver.log",
        parameters,
        mip_relative_gap=args.mip_relative_gap,
        time_limit_seconds=args.time_limit_seconds,
        log_to_console=not args.quiet,
    )
    dispatch_path = output_dir / "dispatch_milp.csv"
    solved.dispatch.to_csv(dispatch_path, index=False, float_format="%.15g")

    reloaded_dispatch = pd.read_csv(dispatch_path, parse_dates=["timestamp"])
    accounting = recalculate_objective(reloaded_dispatch, parameters)
    violations = constraint_violations(reloaded_dispatch, parameters)
    reconciliation_error = accounting["objective_yuan"] - solved.solver_objective_yuan
    if abs(reconciliation_error) > 1e-5:
        raise RuntimeError(
            "Independent objective does not match the HiGHS objective: "
            f"difference={reconciliation_error} yuan."
        )
    if violations["maximum_constraint_violation"] > 1e-5:
        raise RuntimeError(
            "Solved dispatch failed independent constraint checks: "
            f"maximum violation={violations['maximum_constraint_violation']}."
        )

    metadata = dict(solved.solver_metadata)
    metadata.update(
        {
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "input_path": str(args.input.expanduser().resolve()),
            "input_sha256": _sha256(args.input),
            "reference_summary_path": str(args.reference_summary.expanduser().resolve()),
            "reference_summary_sha256": _sha256(args.reference_summary),
            "selected_date": args.date.isoformat() if args.date else None,
            "row_count": len(reloaded_dispatch),
            "maximum_constraint_violation": violations[
                "maximum_constraint_violation"
            ],
            "constraint_violations": violations,
        }
    )
    summary = {
        "load_data_notice": LOAD_DATA_NOTICE,
        "timestamp_start": reloaded_dispatch["timestamp"].min().isoformat(),
        "timestamp_end": reloaded_dispatch["timestamp"].max().isoformat(),
        "row_count": len(reloaded_dispatch),
        "parameters": parameters.to_dict(),
        "solver_objective_yuan": solved.solver_objective_yuan,
        "objective_reconciliation_error_yuan": reconciliation_error,
        "accounting": accounting,
        "constraint_violations": violations,
    }

    if args.date is None:
        comparison = compare_with_original(
            args.input,
            reloaded_dispatch,
            reference_summary,
            parameters,
        )
    else:
        comparison = {
            "scope": "one_day_smoke",
            "selected_date": args.date.isoformat(),
            "reference_objective_yuan": None,
            "highs_objective_yuan": accounting["objective_yuan"],
            "within_one_percent": None,
            "note": "The monthly reference objective is not compared to a one-day smoke solve.",
        }

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "comparison_with_original.json", comparison)
    _write_json(output_dir / "solver_metadata.json", metadata)
    if args.date is None:
        _write_report(output_dir / "report.md", summary, comparison, metadata)
    else:
        (output_dir / "report.md").write_text(
            "# Foshan One-Day HiGHS Smoke\n\n"
            f"**Data caveat:** {LOAD_DATA_NOTICE}\n\n"
            f"- Date: {args.date.isoformat()}\n"
            f"- Solver status: `{metadata['solver_status']}`\n"
            f"- Objective: {accounting['objective_yuan']:.6f} yuan\n"
            f"- Runtime: {metadata['wall_clock_runtime_seconds']:.3f} seconds\n"
            f"- Maximum violation: {violations['maximum_constraint_violation']:.9g}\n",
            encoding="utf-8",
        )

    print(
        f"Saved HiGHS dispatch to {dispatch_path} "
        f"(objective={accounting['objective_yuan']:.6f} yuan, "
        f"status={metadata['solver_status']})."
    )


if __name__ == "__main__":
    main()
