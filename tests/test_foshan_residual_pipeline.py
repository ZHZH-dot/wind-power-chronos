from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepare_foshan import ParsedSignal
from src.data.reconstruct_foshan_provisional_load import ProvisionalLoadSignals
from src.data.reconstruct_foshan_residual import (
    CALENDAR_COLUMNS,
    TARGET_COLUMN,
    TARGET_LABEL,
    add_residual_calendar_covariates,
    aggregate_signed_residual_15min,
    reconstruct_signed_residual,
)
from src.models.foshan_residual_zero_shot import (
    MODEL_REVISION,
    PREDICTION_COLUMNS,
    build_inference_frames,
    resolve_pinned_snapshot,
    run_residual_candidate,
)
from src.optimization.foshan_battery_milp import DispatchSolution
from src.optimization.foshan_controller_v5_final_benchmark import summarize_result
from src.optimization.foshan_residual_controller_eval import (
    FROZEN_SOURCE_SHA256,
    controller_horizon_from_book,
    make_forecast_book,
    run_rolling_v5_evaluation,
    verify_frozen_controller_sources,
)


TZ = "Asia/Shanghai"


def _parsed_signal(
    timestamps: pd.DatetimeIndex,
    values: list[float],
    name: str,
) -> ParsedSignal:
    return ParsedSignal(
        frame=pd.DataFrame(
            {
                "timestamp": timestamps,
                f"{name}_raw": values,
                f"is_missing_{name}": False,
            }
        ),
        audit={},
        negative_readings=pd.DataFrame(),
    )


def test_signed_residual_algebra_alignment_aggregation_and_negative_preservation(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site.xlsx"
    storage = tmp_path / "storage.xlsx"
    site.write_bytes(b"site")
    storage.write_bytes(b"storage")
    quarter_hours = pd.date_range("2026-04-01", periods=2, freq="15min", tz=TZ)
    five_minutes = pd.date_range("2026-04-01", periods=6, freq="5min", tz=TZ)
    signals = ProvisionalLoadSignals(
        pv=_parsed_signal(quarter_hours, [0.0, 0.0], "pv_kw"),
        net_grid=_parsed_signal(quarter_hours, [10.0, -5.0], "net_grid_kw"),
        pcs=_parsed_signal(five_minutes, [1.0, 2.0, 3.0, -1.0, -2.0, -3.0], "pcs_kw"),
        site_workbook=site,
        storage_workbook=storage,
    )

    five, audit = reconstruct_signed_residual(
        signals, quarter_hours[0], quarter_hours[-1] + pd.Timedelta(minutes=15)
    )
    assert five["net_grid_kw"].tolist() == [10.0, 10.0, 10.0, -5.0, -5.0, -5.0]
    assert five[TARGET_COLUMN].tolist() == [11.0, 12.0, 13.0, -6.0, -7.0, -8.0]
    assert five["net_grid_interval_timestamp"].tolist() == [
        quarter_hours[0],
        quarter_hours[0],
        quarter_hours[0],
        quarter_hours[1],
        quarter_hours[1],
        quarter_hours[1],
    ]
    assert audit["formula"] == "net_grid_kw + pcs_kw"
    assert audit["pcs_sign_convention"] == "positive=discharge, negative=charge"
    assert audit["maximum_algebra_error_kw"] == 0.0

    fifteen = aggregate_signed_residual_15min(five)
    assert fifteen[TARGET_COLUMN].tolist() == [12.0, -7.0]
    assert fifteen["observed_five_minute_target_count"].tolist() == [3, 3]
    assert fifteen["is_negative_target"].tolist() == [False, True]
    assert fifteen["target_label"].eq(TARGET_LABEL).all()


def _residual_table(periods: int = 3200) -> pd.DataFrame:
    timestamps = pd.date_range("2026-03-01", periods=periods, freq="15min", tz=TZ)
    table = pd.DataFrame(
        {
            "timestamp": timestamps,
            TARGET_COLUMN: np.sin(np.arange(periods) / 20.0) * 100.0,
        }
    )
    tariff = {
        minute: ("valley" if minute < 480 else "peak" if minute >= 1080 else "shoulder")
        for minute in range(0, 1440, 15)
    }
    return add_residual_calendar_covariates(table, tariff)


class FakeChronos:
    def __init__(self) -> None:
        self.calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def predict_df(self, context: pd.DataFrame, *, future_df: pd.DataFrame, **_: object) -> pd.DataFrame:
        self.calls.append((context.copy(), future_df.copy()))
        return pd.DataFrame(
            {
                "id": future_df["id"].to_numpy(),
                "timestamp": future_df["timestamp"].to_numpy(),
                "0.1": np.full(len(future_df), -10.0),
                "0.5": np.arange(len(future_df), dtype=float) - 20.0,
                "0.9": np.arange(len(future_df), dtype=float),
            }
        )


def test_context_is_strictly_causal_calendar_only_and_issue_calls_are_independent(
    tmp_path: Path,
) -> None:
    table = _residual_table()
    issues = [
        pd.Timestamp("2026-04-01 00:00", tz=TZ),
        pd.Timestamp("2026-04-01 01:00", tz=TZ),
    ]
    context, future, metadata = build_inference_frames(table, issues[0], 672)
    assert context["timestamp"].max() == pd.Timestamp("2026-03-31 23:45")
    assert metadata["context_end"] < issues[0]
    assert str(context["timestamp"].dtype) == "datetime64[ns]"
    assert str(future["timestamp"].dtype) == "datetime64[ns]"
    assert set(future.columns) == {"id", "timestamp", *CALENDAR_COLUMNS}
    assert not {
        "target",
        TARGET_COLUMN,
        "pv_kw",
        "net_grid_kw",
        "pcs_kw",
        "provisional_load_kw",
    }.intersection(future.columns)

    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    pipeline = FakeChronos()
    predictions, audit = run_residual_candidate(
        pipeline,
        table,
        issues,
        candidate="chronos2_residual_hourly_ctx672",
        split="april_2026_selection",
        context_length=672,
        refresh_cadence_minutes=60,
        snapshot=snapshot,
        inference_batch_size=64,
    )
    assert len(pipeline.calls) == 2
    assert len(predictions) == 2 * 96
    assert set(PREDICTION_COLUMNS).issubset(predictions.columns)
    assert predictions.groupby("inference_call_id")["issue_time"].nunique().eq(1).all()
    assert predictions.groupby("issue_time").size().eq(96).all()
    assert not predictions["used_future_realized_data"].any()
    assert audit["different_issue_times_batched"].eq(False).all()
    assert (pd.to_datetime(audit["context_end"]) < pd.to_datetime(audit["issue_time"])).all()
    assert (predictions["p10"] < 0.0).any(), "Signed forecasts must not be clipped."


def test_pinned_snapshot_requires_exact_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["Chronos2Model"]}), encoding="utf-8"
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    assert resolve_pinned_snapshot(model_path=snapshot) == snapshot.resolve()

    mutable = tmp_path / "main"
    mutable.mkdir()
    (mutable / "config.json").write_text(
        json.dumps({"architectures": ["Chronos2Model"]}), encoding="utf-8"
    )
    (mutable / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="revision-specific"):
        resolve_pinned_snapshot(model_path=mutable)


def test_hourly_replacement_uses_newest_causal_issue_and_residual_conversion() -> None:
    day = pd.Timestamp("2026-04-01")
    predictions = pd.DataFrame(
        {
            "candidate": "hourly",
            "issue_time": [day, day, day + pd.Timedelta(hours=1), day + pd.Timedelta(hours=1)],
            "target_time": [
                day + pd.Timedelta(hours=1),
                day + pd.Timedelta(hours=1, minutes=15),
                day + pd.Timedelta(hours=1),
                day + pd.Timedelta(hours=1, minutes=15),
            ],
            "p50": [50.0, 60.0, -20.0, -10.0],
            "context_end": [
                day - pd.Timedelta(minutes=15),
                day - pd.Timedelta(minutes=15),
                day + pd.Timedelta(minutes=45),
                day + pd.Timedelta(minutes=45),
            ],
        }
    )
    book = make_forecast_book(predictions, "hourly", "signed_residual")
    five_times = pd.date_range(day + pd.Timedelta(hours=1), periods=6, freq="5min")
    pv = pd.DataFrame({"timestamp": five_times, "forecast_pv_kw": 15.0})
    target = pd.DataFrame(
        {
            "timestamp": pd.date_range(day - pd.Timedelta(days=28), periods=3000, freq="15min"),
            TARGET_COLUMN: 0.0,
        }
    )
    horizon, audit = controller_horizon_from_book(
        book,
        pv,
        target,
        day + pd.Timedelta(hours=1),
        day + pd.Timedelta(hours=1, minutes=30),
    )
    assert audit["forecast_issue_time"] == day + pd.Timedelta(hours=1)
    assert horizon["forecast_signed_residual_kw"].tolist() == [-20.0] * 3 + [-10.0] * 3
    assert horizon["safe_residual_kw"].eq(0.0).all()
    assert horizon["forecast_load_kw"].tolist() == [0.0] * 3 + [5.0] * 3


def test_controller_sources_are_frozen() -> None:
    assert verify_frozen_controller_sources() == FROZEN_SOURCE_SHA256


def _zero_solver(
    table: pd.DataFrame,
    _log_path: Path,
    parameters: object,
    **_: object,
) -> DispatchSolution:
    count = len(table)
    soc = float(parameters.initial_soc_kwh)
    dispatch = pd.DataFrame(
        {
            "timestamp": table["timestamp"].to_numpy(),
            "charge_kw": np.zeros(count),
            "discharge_kw": np.zeros(count),
            "soc_start_kwh": np.full(count, soc),
            "soc_kwh": np.full(count, soc),
        }
    )
    return DispatchSolution(
        dispatch=dispatch,
        solver_objective_yuan=0.0,
        solver_metadata={
            "solver_status": "Optimal",
            "optimality_gap": 0.0,
            "wall_clock_runtime_seconds": 0.0,
            "terminal_deviation_negative_kwh": 0.0,
            "terminal_deviation_positive_kwh": 0.0,
        },
    )


def test_evaluation_wrapper_preserves_policy_and_revenue_reconciles(tmp_path: Path) -> None:
    day = pd.Timestamp("2026-04-30")
    actual_times = pd.date_range(day - pd.Timedelta(minutes=5), periods=289, freq="5min")
    realized = pd.DataFrame(
        {"timestamp": actual_times, "pv": 0.0, "load": 100.0, "price": 0.5}
    )
    target_times = pd.date_range(day, periods=288, freq="5min")
    pv = pd.DataFrame({"timestamp": target_times, "forecast_pv_kw": 0.0})
    predictions = pd.DataFrame(
        {
            "candidate": "previous_day",
            "issue_time": day,
            "target_time": target_times,
            "p50": 100.0,
            "context_end": target_times - pd.Timedelta(days=1),
        }
    )
    book = make_forecast_book(predictions, "previous_day", "gross_load")
    residual = pd.DataFrame(
        {
            "timestamp": pd.date_range(day - pd.Timedelta(days=1), periods=192, freq="15min"),
            TARGET_COLUMN: 100.0,
        }
    )
    result, replans = run_rolling_v5_evaluation(
        "test_v5",
        realized,
        pv,
        book,
        residual,
        day,
        day + pd.Timedelta(days=1),
        tmp_path,
        solver=_zero_solver,
    )
    row, _, _ = summarize_result(
        "test_v5",
        result,
        __import__("src.optimization.foshan_battery_milp", fromlist=["DispatchParameters"]).DispatchParameters(),
        runtime_seconds=0.0,
        solver_replans=288,
        solver_failures=0,
        terminal_policy="band_895_905",
    )
    assert len(replans) == 288
    assert all(item["first_step_source_strictly_before_control"] for item in replans)
    assert all(not item["future_realized_pv_or_load_passed"] for item in replans)
    assert row["final_soc_kwh"] == 900.0
    assert row["revenue_recalculation_abs_error_yuan"] <= 0.01
    assert row["maximum_constraint_violation"] == 0.0
