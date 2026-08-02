from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepare_foshan import ParsedSignal
from src.data.reconstruct_foshan_provisional_load import (
    ProvisionalLoadSignals,
    reconstruct_provisional_load,
    validate_against_reference_load,
    validate_april30_reconstruction,
)
from src.optimization.foshan_battery_milp import DispatchParameters
from src.optimization.foshan_controller_v5_final_benchmark import (
    END_EXCLUSIVE,
    START,
    build_historical_result,
    interval_accounting,
)


def _parsed(
    timestamps: pd.DatetimeIndex,
    column: str,
    values: list[float],
) -> ParsedSignal:
    return ParsedSignal(
        frame=pd.DataFrame({"timestamp": timestamps, column: values}),
        audit={},
        negative_readings=pd.DataFrame(),
    )


def _signals() -> ProvisionalLoadSignals:
    quarter_hours = pd.date_range(
        "2026-04-30", periods=3, freq="15min", tz="Asia/Shanghai"
    )
    five_minutes = pd.date_range(
        "2026-04-30", periods=9, freq="5min", tz="Asia/Shanghai"
    )
    return ProvisionalLoadSignals(
        pv=_parsed(quarter_hours, "pv_kw_raw", [-0.1, 10.0, 20.0]),
        net_grid=_parsed(
            quarter_hours, "net_grid_kw_raw", [5.0, np.nan, 7.0]
        ),
        pcs=_parsed(
            five_minutes,
            "pcs_kw_raw",
            [-10.0, -2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ),
        site_workbook=Path("site.xlsx"),
        storage_workbook=Path("storage.xlsx"),
    )


def test_reconstruction_forward_fills_without_interpolation_and_records_clips() -> None:
    table, audit = reconstruct_provisional_load(
        _signals(),
        "2026-04-30 00:00",
        "2026-04-30 00:45",
    )

    assert len(table) == 9
    assert table.loc[:2, "pv_kw"].eq(0.0).all()
    assert table.loc[3:5, "pv_kw"].eq(10.0).all()
    assert table.loc[3:5, "net_grid_kw"].eq(5.0).all()
    assert table.loc[3:5, "net_grid_was_causally_filled"].all()
    assert table.loc[3:5, "net_grid_source_timestamp"].eq(
        pd.Timestamp("2026-04-30 00:00", tz="Asia/Shanghai")
    ).all()
    assert table.loc[0, "provisional_load_proxy_raw_kw"] == pytest.approx(-5.0)
    assert table.loc[0, "provisional_load_kw"] == 0.0
    assert bool(table.loc[0, "provisional_load_was_clipped"])
    assert audit["raw_proxy_negative_rows"] == 1
    assert audit["alignment_method"].endswith("no interpolation")
    assert audit["pcs_sign_convention"] == "positive=discharge, negative=charge"


def test_reference_validation_requires_all_timestamps_and_tolerance() -> None:
    table, _ = reconstruct_provisional_load(
        _signals(),
        "2026-04-30 00:00",
        "2026-04-30 00:45",
    )
    reference = table[["timestamp", "provisional_load_kw"]].rename(
        columns={"provisional_load_kw": "load"}
    )
    audit = validate_against_reference_load(table, reference)
    assert audit["matched_timestamps"] == 9
    assert audit["maximum_absolute_difference_kw"] == 0.0

    bad = reference.copy()
    bad.loc[0, "load"] += 1e-6
    with pytest.raises(ValueError, match="Reference load validation failed"):
        validate_against_reference_load(table, bad)


def test_april30_gate_rejects_changed_energy_or_counts() -> None:
    audit = {
        "pv_source_rows": 96,
        "net_grid_source_rows": 96,
        "pcs_source_rows": 288,
        "final_reconstructed_rows": 288,
        "raw_proxy_negative_rows": 3,
        "raw_proxy_energy_kwh": 9786.9875,
        "clipped_provisional_load_energy_kwh": 9869.3175,
        "missing_rows": {"provisional_load_kw": 0},
    }
    validate_april30_reconstruction(audit)
    audit["raw_proxy_negative_rows"] = 4
    with pytest.raises(ValueError, match="April 30"):
        validate_april30_reconstruction(audit)


def test_historical_commands_are_unchanged_and_soc_is_reconstructed() -> None:
    timestamps = pd.date_range(START, END_EXCLUSIVE, freq="5min", inclusive="left")
    realized = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pv": 0.0,
            "load": 100.0,
            "price": 0.5,
            "p_actual": 0.0,
            "p_opt": 0.0,
            "soc_actual_est": 900.0,
            "soc_opt": 900.0,
        }
    )
    realized.loc[0, "p_actual"] = -1200.0
    result, audit = build_historical_result(realized)

    replay = result.replay
    assert replay.loc[0, "scheduled_charge_kw"] == 1200.0
    assert replay.loc[0, "applied_charge_kw"] == 1200.0
    assert replay.loc[0, "scheduled_discharge_kw"] == 0.0
    assert audit["observed_commands_modified"] is False
    expected_soc = 900.0 + (
        DispatchParameters().charge_efficiency * (1.0 / 12.0) * 1200.0
    )
    assert replay.loc[0, "realized_soc_end_kwh"] == pytest.approx(expected_soc)


def test_interval_accounting_has_no_comprehensive_revenue_double_count() -> None:
    replay = pd.DataFrame(
        {
            "realized_pv_kw": [100.0, 0.0],
            "realized_load_kw": [80.0, 100.0],
            "price_yuan_per_kwh": [0.5, 0.5],
            "grid_import_kw": [0.0, 90.0],
            "grid_export_kw": [20.0, 0.0],
            "applied_charge_kw": [0.0, 0.0],
            "applied_discharge_kw": [0.0, 10.0],
        }
    )
    accounting = interval_accounting(replay)
    totals = accounting.sum()
    assert totals["design_institute_comprehensive_revenue_yuan"] == pytest.approx(
        totals["total_pv_revenue_yuan"]
        + totals["design_institute_storage_share_yuan"]
    )
    assert totals["design_institute_storage_share_yuan"] == pytest.approx(
        0.8 * totals["total_storage_bill_savings_yuan"]
    )
    assert totals["user_storage_share_yuan"] == pytest.approx(
        0.2 * totals["total_storage_bill_savings_yuan"]
    )
