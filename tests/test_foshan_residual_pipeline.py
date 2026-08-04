from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepare_foshan import ParsedSignal, add_calendar_covariates
from src.data.reconstruct_foshan_provisional_load import ProvisionalLoadSignals
from src.data.reconstruct_foshan_residual import (
    CALENDAR_COLUMNS,
    TARGET_COLUMN,
    TARGET_LABEL,
    add_residual_calendar_covariates,
    aggregate_signed_residual_15min,
    load_tariff_clock_profile,
    reconstruct_signed_residual,
    tariff_clock_profile_from_calendar,
)
from src.models.foshan_residual_zero_shot import (
    DEFAULT_PV_CONFIG,
    DEFAULT_PV_SELECTION,
    FROZEN_PV_POSTPROCESSING,
    MODEL_REVISION,
    PREDICTION_COLUMNS,
    build_inference_frames,
    evaluate_residual_predictions,
    generate_frozen_april_pv,
    resolve_pinned_snapshot,
    run_residual_candidate,
    validate_frozen_april_pv_rows,
)
import src.models.foshan_residual_zero_shot as residual_zero_shot
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


class FakePvChronos:
    def predict_df(
        self,
        context_df: pd.DataFrame,
        *,
        future_df: pd.DataFrame | None,
        **kwargs: object,
    ) -> pd.DataFrame:
        targets = kwargs["target"]
        target_names = [targets] if isinstance(targets, str) else list(targets)
        prediction_length = int(kwargs["prediction_length"])
        timestamps = (
            pd.DatetimeIndex(future_df["timestamp"])
            if future_df is not None
            else pd.date_range(
                context_df["timestamp"].max() + pd.Timedelta(minutes=15),
                periods=prediction_length,
                freq="15min",
            )
        )
        rows: list[dict[str, object]] = []
        for target in target_names:
            for timestamp in timestamps:
                pv = target == "pv_kw"
                rows.append(
                    {
                        "id": "foshan_site",
                        "timestamp": timestamp,
                        "target_name": target,
                        "0.1": -10.0 if pv else -50.0,
                        "0.5": 500.0 if pv else 25.0,
                        "0.9": 1800.0 if pv else 100.0,
                    }
                )
        return pd.DataFrame(rows)


def _processed_foshan_table() -> pd.DataFrame:
    periods = 96 * 34
    timestamps = pd.date_range(
        "2026-03-01", periods=periods, freq="15min", tz=TZ
    )
    slot = np.arange(periods) % 96
    pv = np.maximum(0.0, np.sin((slot - 24) * np.pi / 48.0)) * 1000.0
    table = pd.DataFrame(
        {
            "id": "foshan_site",
            "timestamp": timestamps,
            "pv_kw_raw": pv,
            "pv_kw": pv,
            "net_grid_kw_raw": 100.0,
            "net_grid_kw": 100.0,
            "is_missing_pv_kw": False,
            "is_missing_net_grid_kw": False,
            "is_corrected_pv_kw": False,
        }
    )
    return add_calendar_covariates(table)


def _write_pv_selection(
    path: Path,
    model_name: str,
    *,
    postprocessing: str | None,
) -> None:
    selected: dict[str, object] = {
        "model_name": model_name,
        "context_length": 672,
    }
    if postprocessing is not None:
        selected["postprocessing"] = postprocessing
    path.write_text(
        json.dumps({"targets": {"pv_kw": selected}}), encoding="utf-8"
    )


def _run_fake_frozen_pv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_name: str,
    postprocessing: str | None,
    max_origins: int,
) -> tuple[pd.DataFrame, list[int]]:
    table_path = tmp_path / f"processed_{model_name}_{max_origins}.parquet"
    selection_path = tmp_path / f"selection_{model_name}_{max_origins}.json"
    _processed_foshan_table().to_parquet(table_path, index=False)
    _write_pv_selection(
        selection_path, model_name, postprocessing=postprocessing
    )
    internal_counts: list[int] = []
    original = residual_zero_shot.run_chronos_configuration

    def recording_run(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        internal_counts.append(len(result[0]))
        return result

    monkeypatch.setattr(
        residual_zero_shot, "run_chronos_configuration", recording_run
    )
    rows = generate_frozen_april_pv(
        FakePvChronos(),
        table_path,
        DEFAULT_PV_CONFIG,
        selection_path,
        "fake-chronos-2",
        max_origins=max_origins,
    )
    return rows, internal_counts


def test_frozen_pv_legacy_selection_keeps_one_physical_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, internal_counts = _run_fake_frozen_pv(
        tmp_path,
        monkeypatch,
        model_name="chronos2_pv_univariate",
        postprocessing=None,
        max_origins=1,
    )

    assert internal_counts == [192]
    assert len(rows) == 96
    assert rows["postprocessing"].eq(FROZEN_PV_POSTPROCESSING).all()
    assert rows["target"].eq("pv_kw").all()
    assert rows["horizon_step"].tolist() == list(range(1, 97))


def test_frozen_pv_explicit_selection_returns_two_complete_origins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, internal_counts = _run_fake_frozen_pv(
        tmp_path,
        monkeypatch,
        model_name="chronos2_pv_univariate",
        postprocessing=FROZEN_PV_POSTPROCESSING,
        max_origins=2,
    )

    assert internal_counts == [384]
    assert len(rows) == 192
    assert rows.groupby("issue_time").size().eq(96).all()
    assert not rows.duplicated(["issue_time", "target_time", "horizon_step"]).any()


def test_frozen_pv_joint_selection_excludes_provisional_grid_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, internal_counts = _run_fake_frozen_pv(
        tmp_path,
        monkeypatch,
        model_name="chronos2_joint_calendar",
        postprocessing=FROZEN_PV_POSTPROCESSING,
        max_origins=1,
    )

    assert internal_counts == [288]
    assert len(rows) == 96
    assert rows["target"].eq("pv_kw").all()
    assert rows["model_name"].eq("chronos2_joint_calendar").all()


def _valid_frozen_pv_rows() -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    origin = pd.Timestamp("2026-04-01", tz=TZ)
    target_times = pd.date_range(origin, periods=96, freq="15min")
    return (
        pd.DataFrame(
            {
                "issue_time": origin,
                "target_time": target_times,
                "horizon_step": np.arange(1, 97),
                "target": "pv_kw",
                "model_name": "chronos2_pv_univariate",
                "context_length": 672,
                "postprocessing": FROZEN_PV_POSTPROCESSING,
            }
        ),
        [origin],
    )


def test_frozen_pv_validation_rejects_duplicate_and_missing_horizons() -> None:
    rows, origins = _valid_frozen_pv_rows()
    kwargs = {
        "expected_origins": origins,
        "model_name": "chronos2_pv_univariate",
        "context_length": 672,
        "postprocessing": FROZEN_PV_POSTPROCESSING,
    }
    with pytest.raises(RuntimeError, match="duplicate forecast keys"):
        validate_frozen_april_pv_rows(
            pd.concat([rows, rows.iloc[[0]]], ignore_index=True), **kwargs
        )
    with pytest.raises(RuntimeError, match="horizons are incomplete"):
        validate_frozen_april_pv_rows(rows.iloc[:-1], **kwargs)


def test_frozen_pv_rejects_unknown_postprocessing(tmp_path: Path) -> None:
    table_path = tmp_path / "processed.parquet"
    selection_path = tmp_path / "selection.json"
    _processed_foshan_table().to_parquet(table_path, index=False)
    _write_pv_selection(
        selection_path,
        "chronos2_pv_univariate",
        postprocessing="unknown_policy",
    )

    with pytest.raises(ValueError, match="unknown postprocessing"):
        generate_frozen_april_pv(
            FakePvChronos(),
            table_path,
            DEFAULT_PV_CONFIG,
            selection_path,
            "fake-chronos-2",
            max_origins=1,
        )


def test_frozen_pv_default_selection_path_is_corrected() -> None:
    expected = Path("results/zero_shot/foshan_chronos2/selected_configuration.json")
    assert DEFAULT_PV_SELECTION == expected
    launcher = Path("scripts/run_foshan_residual_zero_shot.sh").read_text(
        encoding="utf-8"
    )
    assert f"PV_SELECTION:-{expected.as_posix()}" in launcher


def _write_tariff_csv(
    path: Path,
    *,
    frequency: str,
    days: int = 2,
) -> pd.DataFrame:
    steps_per_day = 288 if frequency == "5min" else 96
    timestamps = pd.date_range(
        "2026-05-01", periods=days * steps_per_day, freq=frequency
    )
    minute_of_day = timestamps.hour * 60 + timestamps.minute
    prices = np.where(
        minute_of_day < 480,
        0.3,
        np.where(minute_of_day < 1080, 0.6, 1.0),
    )
    table = pd.DataFrame(
        {
            "Unnamed: 0": timestamps,
            "pv": 0.0,
            "load": 100.0,
            "price": prices,
            "period": "not-used-for-classification",
        }
    )
    table.to_csv(path, index=False)
    return table


def test_tariff_loader_canonicalizes_five_minutes_and_preserves_fifteen_minutes(
    tmp_path: Path,
) -> None:
    five_path = tmp_path / "dispatch_5min.csv"
    fifteen_path = tmp_path / "dispatch_15min.csv"
    _write_tariff_csv(five_path, frequency="5min", days=31)
    _write_tariff_csv(fifteen_path, frequency="15min", days=31)

    five_profile = load_tariff_clock_profile(five_path)
    fifteen_profile = load_tariff_clock_profile(fifteen_path)

    assert len(pd.read_csv(five_path)) == 8_928
    assert len(five_profile) == 96
    assert set(five_profile) == set(range(0, 24 * 60, 15))
    assert five_profile == fifteen_profile
    assert five_profile[0] == "valley"
    assert five_profile[480] == "shoulder"
    assert five_profile[1080] == "peak"


def test_tariff_loader_rejects_within_quarter_change(tmp_path: Path) -> None:
    path = tmp_path / "within_quarter_change.csv"
    table = _write_tariff_csv(path, frequency="5min", days=1)
    table.loc[table["Unnamed: 0"].eq(pd.Timestamp("2026-05-01 00:05")), "price"] = 0.6
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match="inside a 15-minute block"):
        load_tariff_clock_profile(path)


def test_tariff_loader_rejects_missing_clock_position(tmp_path: Path) -> None:
    path = tmp_path / "missing_clock.csv"
    table = _write_tariff_csv(path, frequency="5min", days=2)
    missing_clock = (
        pd.to_datetime(table["Unnamed: 0"]).dt.hour.eq(12)
        & pd.to_datetime(table["Unnamed: 0"]).dt.minute.eq(5)
    )
    table.loc[~missing_clock].to_csv(path, index=False)

    with pytest.raises(ValueError, match="complete deterministic"):
        load_tariff_clock_profile(path)


def test_tariff_loader_rejects_cross_day_clock_price_conflict(tmp_path: Path) -> None:
    path = tmp_path / "cross_day_conflict.csv"
    table = _write_tariff_csv(path, frequency="5min", days=2)
    conflict = table["Unnamed: 0"].eq(pd.Timestamp("2026-05-02 00:00"))
    table.loc[conflict, "price"] = 0.6
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match="changes across days"):
        load_tariff_clock_profile(path)


def test_tariff_loader_rejects_mixed_or_nonaligned_timestamps(tmp_path: Path) -> None:
    mixed_path = tmp_path / "mixed.csv"
    five = _write_tariff_csv(mixed_path, frequency="5min", days=1)
    fifteen_path = tmp_path / "fifteen.csv"
    fifteen = _write_tariff_csv(fifteen_path, frequency="15min", days=1)
    fifteen["Unnamed: 0"] = pd.to_datetime(fifteen["Unnamed: 0"]) + pd.Timedelta(days=1)
    pd.concat([five, fifteen], ignore_index=True).to_csv(mixed_path, index=False)
    with pytest.raises(ValueError, match="incomplete or mixed"):
        load_tariff_clock_profile(mixed_path)

    nonaligned_path = tmp_path / "nonaligned.csv"
    nonaligned = _write_tariff_csv(nonaligned_path, frequency="5min", days=1)
    nonaligned.loc[0, "Unnamed: 0"] = pd.Timestamp("2026-05-01 00:01")
    nonaligned.to_csv(nonaligned_path, index=False)
    with pytest.raises(ValueError, match="aligned to five-minute"):
        load_tariff_clock_profile(nonaligned_path)


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


def test_late_may_future_calendar_extends_beyond_realized_target_boundary(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range(
        "2026-03-01 00:00", "2026-06-01 00:00", freq="15min", tz=TZ, inclusive="left"
    )
    table = _residual_table(periods=len(timestamps))
    tariff_profile = tariff_clock_profile_from_calendar(table)
    issues = [
        pd.Timestamp("2026-05-31 01:00", tz=TZ),
        pd.Timestamp("2026-05-31 23:00", tz=TZ),
    ]

    for issue in issues:
        context, future, metadata = build_inference_frames(table, issue, 672)
        assert len(future) == 96
        assert future[CALENDAR_COLUMNS].notna().all().all()
        assert metadata["context_end"] < issue
        assert context["timestamp"].max() < issue.tz_localize(None)
        assert set(future.columns) == {"id", "timestamp", *CALENDAR_COLUMNS}

        future_times = pd.DatetimeIndex(future["timestamp"]).tz_localize(TZ)
        minute_of_day = future_times.hour * 60 + future_times.minute
        expected_periods = pd.Series(minute_of_day).map(tariff_profile)
        actual_periods = (
            future[["tariff_is_peak", "tariff_is_shoulder", "tariff_is_valley"]]
            .idxmax(axis=1)
            .str.removeprefix("tariff_is_")
        )
        assert actual_periods.tolist() == expected_periods.tolist()
        assert future[["tariff_is_peak", "tariff_is_shoulder", "tariff_is_valley"]].sum(
            axis=1
        ).eq(1).all()

        in_range = future_times < pd.Timestamp("2026-06-01 00:00", tz=TZ)
        previous_calendar = (
            table.set_index("timestamp").reindex(future_times[in_range])[CALENDAR_COLUMNS]
        )
        pd.testing.assert_frame_equal(
            future.loc[in_range, CALENDAR_COLUMNS].reset_index(drop=True),
            previous_calendar.reset_index(drop=True),
            check_dtype=True,
        )

    _, late_future, late_metadata = build_inference_frames(table, issues[1], 672)
    assert late_metadata["target_end"] == pd.Timestamp("2026-06-01 22:45", tz=TZ)
    assert late_future["timestamp"].iloc[-1] == pd.Timestamp("2026-06-01 22:45")
    assert late_future["month"].iloc[:4].eq(5).all()
    assert late_future["month"].iloc[4:].eq(6).all()

    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    predictions, _ = run_residual_candidate(
        FakeChronos(),
        table,
        issues,
        candidate="chronos2_residual_hourly_ctx672",
        split="may_2026_test",
        context_length=672,
        refresh_cadence_minutes=60,
        snapshot=snapshot,
        inference_batch_size=64,
    )
    june_boundary = pd.Timestamp("2026-06-01 00:00", tz=TZ)
    outside_truth = predictions["target_time"] >= june_boundary
    assert predictions.loc[outside_truth, "y_true_kw"].isna().all()
    assert predictions.loc[outside_truth, "is_missing_target"].all()
    assert predictions.loc[~outside_truth, "y_true_kw"].notna().all()

    metrics = evaluate_residual_predictions(
        predictions,
        pd.Timestamp("2026-05-01 00:00", tz=TZ),
        june_boundary,
    )
    overall = metrics.loc[metrics["metric_scope"].eq("overall")].iloc[0]
    assert int(overall["n_scored"]) == int((~outside_truth).sum()) == 96


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
