import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepare_foshan import (
    CALENDAR_COLUMNS,
    add_calendar_covariates,
    build_site_table,
    read_signal_sheet,
)
from src.evaluation.foshan_benchmark import (
    ForecastWindow,
    build_forecast_window,
    chronos_rows_for_window,
    normalize_chronos_quantiles,
    period_origins,
    postprocess_pv_quantiles,
    run_causal_baselines,
    select_may_configurations,
)
from src.evaluation.metrics import mase, wape
from src.models.foshan_chronos_zero_shot import run_chronos_configuration


def _write_source_workbook(path: Path) -> None:
    pv_rows = [
        ["序号", "用户类别", "数据时间", "有功功率(kW)"],
        ["序号", "用户类别", "数据时间", "总"],
        [1, "光伏发电客户", "2026-03-01 00:45:00", 300.0],
        [2, "光伏发电客户", "2026-03-01 00:15:00", 100.0],
        [3, "光伏发电客户", "2026-03-01 00:15:00", 200.0],
        [4, "光伏发电客户", "2026-03-01 00:00:00", -2.0],
    ]
    grid_rows = [
        ["序号", "用户名称", "数据时间", "有功功率(kW)"],
        ["序号", "用户名称", "数据时间", "总"],
        [1, "志达", "2026-03-01 00:45:00", 20.0],
        [2, "志达", "2026-03-01 00:30:00", -10.0],
        [3, "志达", "2026-03-01 00:15:00", 0.0],
        [4, "志达", "2026-03-01 00:00:00", 5.0],
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(pv_rows).to_excel(writer, sheet_name="光伏", header=False, index=False)
        pd.DataFrame(grid_rows).to_excel(writer, sheet_name="负荷", header=False, index=False)


def _site_table(periods: int = 3000) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-03-01 00:00:00", periods=periods, freq="15min", tz="Asia/Shanghai"
    )
    slot = np.arange(periods) % 96
    pv = np.maximum(0.0, np.sin((slot - 24) * np.pi / 48.0)) * 1000.0
    grid = 100.0 + 0.1 * np.arange(periods)
    table = pd.DataFrame(
        {
            "id": "foshan_site",
            "timestamp": timestamps,
            "pv_kw_raw": pv,
            "pv_kw": pv,
            "net_grid_kw_raw": grid,
            "net_grid_kw": grid,
            "is_missing_pv_kw": False,
            "is_missing_net_grid_kw": False,
            "is_corrected_pv_kw": False,
        }
    )
    return add_calendar_covariates(table)


def test_excel_header_parsing_sorting_timezone_duplicates_and_grid(tmp_path: Path) -> None:
    workbook = tmp_path / "foshan.xlsx"
    _write_source_workbook(workbook)

    parsed = read_signal_sheet(workbook, "光伏", "pv_kw", "15min")

    assert parsed.audit["resolved_header_row_1_based"] == 1
    assert parsed.audit["duplicate_count"] == 1
    assert parsed.audit["missing_intervals"] == 1
    assert parsed.frame["timestamp"].is_monotonic_increasing
    assert str(parsed.frame["timestamp"].dt.tz) == "Asia/Shanghai"
    assert parsed.frame["timestamp"].tolist() == list(
        pd.date_range("2026-03-01", periods=4, freq="15min", tz="Asia/Shanghai")
    )
    assert parsed.frame.loc[2, "is_missing_pv_kw"]


def test_site_table_preserves_signed_grid_and_clips_pv_raw(tmp_path: Path) -> None:
    workbook = tmp_path / "foshan.xlsx"
    _write_source_workbook(workbook)
    pv = read_signal_sheet(workbook, "光伏", "pv_kw", "15min")
    grid = read_signal_sheet(workbook, "负荷", "net_grid_kw", "15min")

    table = build_site_table(pv, grid)

    assert table.loc[0, "pv_kw_raw"] == -2.0
    assert table.loc[0, "pv_kw"] == 0.0
    assert table["net_grid_kw"].min() == -10.0
    assert table["net_grid_kw"].equals(table["net_grid_kw_raw"])
    assert set(CALENDAR_COLUMNS).issubset(table.columns)


def test_causal_context_never_backward_fills() -> None:
    table = _site_table(periods=120)
    table.loc[0, "pv_kw"] = np.nan
    table.loc[0, "is_missing_pv_kw"] = True
    issue = table.loc[4, "timestamp"]

    window, reason = build_forecast_window(
        table,
        issue,
        targets=["pv_kw"],
        context_length=4,
        prediction_length=8,
        known_future_covariates=[],
        causal_fill_limit=2,
    )

    assert window is None
    assert reason == "unresolved_context_gap"


def test_origin_skipped_after_more_than_two_missing_context_slots() -> None:
    table = _site_table(periods=120)
    table.loc[1:3, "pv_kw"] = np.nan
    table.loc[1:3, "is_missing_pv_kw"] = True

    window, reason = build_forecast_window(
        table,
        table.loc[4, "timestamp"],
        targets=["pv_kw"],
        context_length=4,
        prediction_length=8,
        known_future_covariates=[],
        causal_fill_limit=2,
    )

    assert window is None
    assert reason == "unresolved_context_gap"


def test_window_has_strict_context_full_horizon_and_target_free_future() -> None:
    table = _site_table(periods=200)
    issue = table.loc[100, "timestamp"]

    window, reason = build_forecast_window(
        table,
        issue,
        targets=["pv_kw", "net_grid_kw"],
        context_length=96,
        prediction_length=96,
        known_future_covariates=CALENDAR_COLUMNS,
    )

    assert reason is None
    assert window is not None
    assert window.context_df["timestamp"].max() < issue
    assert window.truth_df["timestamp"].min() == issue
    assert len(window.truth_df) == 96
    assert window.future_df is not None
    assert len(window.future_df) == 96
    assert not {"pv_kw", "net_grid_kw", "pv_kw_raw", "net_grid_kw_raw"}.intersection(
        window.future_df.columns
    )


def test_period_origins_require_complete_96_step_days() -> None:
    table = _site_table(periods=96 * 3)
    period = {
        "start": "2026-03-01T00:00:00+08:00",
        "end_exclusive": "2026-03-04T00:00:00+08:00",
    }

    origins = period_origins(table, period, prediction_length=96, stride_steps=96)

    assert origins == list(
        pd.date_range("2026-03-01", periods=3, freq="1D", tz="Asia/Shanghai")
    )


def test_multi_target_target_name_mapping_is_not_positional() -> None:
    timestamps = pd.date_range(
        "2026-05-01", periods=2, freq="15min", tz="Asia/Shanghai"
    )
    forecast = pd.DataFrame(
        {
            "id": ["foshan_site"] * 4,
            "timestamp": [*timestamps, *timestamps],
            "target_name": ["net_grid_kw", "net_grid_kw", "pv_kw", "pv_kw"],
            "0.1": [10.0, 11.0, 1.0, 2.0],
            "0.5": [20.0, 21.0, 3.0, 4.0],
            "0.9": [30.0, 31.0, 5.0, 6.0],
            "predictions": [20.0, 21.0, 3.0, 4.0],
        }
    )

    normalized = normalize_chronos_quantiles(
        forecast,
        targets=["pv_kw", "net_grid_kw"],
        expected_times=timestamps,
        site_id="foshan_site",
    )

    assert normalized.loc[normalized["target"] == "pv_kw", "p50"].tolist() == [3.0, 4.0]
    assert normalized.loc[normalized["target"] == "net_grid_kw", "p50"].tolist() == [20.0, 21.0]


def test_fake_pipeline_receives_target_free_future_dataframe() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def predict_df(self, context_df: pd.DataFrame, future_df: pd.DataFrame, **kwargs):
            self.calls.append({"context": context_df.copy(), "future": future_df.copy(), **kwargs})
            targets = kwargs["target"]
            rows = []
            for target_index, target in enumerate(targets):
                for timestamp in future_df["timestamp"]:
                    rows.append(
                        {
                            "id": "foshan_site",
                            "timestamp": timestamp,
                            "target_name": target,
                            "0.1": 1.0 + 10 * target_index,
                            "0.5": 2.0 + 10 * target_index,
                            "0.9": 3.0 + 10 * target_index,
                        }
                    )
            return pd.DataFrame(rows)

    table = _site_table(periods=120)
    issue = table.loc[16, "timestamp"]
    config = {
        "prediction_length": 4,
        "quantile_levels": [0.1, 0.5, 0.9],
        "inference_batch_size": 8,
        "causal_fill_limit": 2,
        "frequency": "15min",
        "pv_capacity_kw": 1700.0,
        "site_id": "foshan_site",
        "calendar_covariates": CALENDAR_COLUMNS,
    }
    configuration = {
        "name": "chronos2_joint_calendar",
        "targets": ["pv_kw", "net_grid_kw"],
        "known_future_covariates": ["calendar"],
    }
    pipeline = FakePipeline()

    predictions, skipped, _ = run_chronos_configuration(
        pipeline,
        table,
        config,
        configuration,
        context_lengths=[8],
        origins=[issue],
        split_name="may_2026_selection",
        run_id="test",
        model_source="amazon/chronos-2",
    )

    assert not skipped
    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call["target"] == ["pv_kw", "net_grid_kw"]
    assert pd.Timestamp(call["context"]["timestamp"].max()) < issue
    assert not {"pv_kw", "net_grid_kw"}.intersection(call["future"].columns)
    grid = predictions[predictions["target"] == "net_grid_kw"]
    assert grid["p50"].eq(12.0).all()


def test_pv_postprocessing_clips_and_repairs_quantile_order() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-01", tz="Asia/Shanghai")],
            "target": ["pv_kw"],
            "p10": [1800.0],
            "p50": [-10.0],
            "p90": [900.0],
        }
    )

    processed = postprocess_pv_quantiles(raw)

    assert processed.loc[0, ["p10", "p50", "p90"]].tolist() == [0.0, 900.0, 1700.0]


def test_chronos_rows_keep_raw_and_postprocessed_pv_schema() -> None:
    table = _site_table(periods=12)
    issue = table.loc[4, "timestamp"]
    window, _ = build_forecast_window(
        table,
        issue,
        targets=["pv_kw"],
        context_length=4,
        prediction_length=2,
        known_future_covariates=[],
    )
    assert window is not None
    forecast = pd.DataFrame(
        {
            "id": ["foshan_site", "foshan_site"],
            "timestamp": window.truth_df["timestamp"],
            "target_name": ["pv_kw", "pv_kw"],
            "0.1": [-10.0, 10.0],
            "0.5": [-5.0, 20.0],
            "0.9": [1.0, 30.0],
        }
    )

    rows = chronos_rows_for_window(
        forecast,
        window,
        targets=["pv_kw"],
        split_name="may_2026_selection",
        run_id="test",
        model_name="chronos2_pv_univariate",
        model_id="amazon/chronos-2",
        context_length=4,
        known_future_covariates=[],
    )

    assert set(rows["postprocessing"]) == {"raw", "physical_clip_0_1700"}
    assert rows[rows["postprocessing"] == "raw"].iloc[0]["p50"] == -5.0
    assert rows[rows["postprocessing"] == "physical_clip_0_1700"].iloc[0]["p50"] >= 0.0
    assert (rows["y_pred"] == rows["p50"]).all()


def test_causal_baselines_use_only_prior_slots() -> None:
    table = _site_table(periods=96 * 36)
    issue = table.loc[96 * 35, "timestamp"]

    predictions = run_causal_baselines(
        table,
        [issue],
        split_name="may_2026_selection",
        run_id="test",
    )

    first = predictions[predictions["horizon_step"] == 1]
    pv_zero = first[(first["target"] == "pv_kw") & (first["model_name"] == "zero")]
    assert pv_zero.iloc[0]["p50"] == 0.0
    day = first[(first["target"] == "net_grid_kw") & (first["model_name"] == "previous_day")]
    expected = table.set_index("timestamp").loc[issue - pd.Timedelta(days=1), "net_grid_kw"]
    assert day.iloc[0]["p50"] == expected
    assert predictions["p10"].isna().all()
    assert predictions["p90"].isna().all()


def test_wape_and_mase_handle_zero_and_negative_actuals() -> None:
    assert wape([-2.0, 0.0, 2.0], [-1.0, 1.0, 1.0]) == pytest.approx(0.75)
    assert mase([-1.0, 1.0], [0.0, 0.0], [-2.0, 0.0, 2.0], seasonal_period=1) == pytest.approx(0.5)
    assert math.isnan(wape([0.0, 0.0], [1.0, 1.0]))


def test_selection_rejects_june_metrics_and_uses_deterministic_may_tie_break() -> None:
    may = pd.DataFrame(
        {
            "split": ["may_2026_selection", "may_2026_selection"],
            "target": ["pv_kw", "pv_kw"],
            "model_name": ["chronos2_pv_calendar", "chronos2_pv_univariate"],
            "model_id": ["amazon/chronos-2", "amazon/chronos-2"],
            "context_length": [672, 672],
            "postprocessing": ["physical_clip_0_1700", "physical_clip_0_1700"],
            "wape": [0.2, 0.2],
            "pv_active_mae": [10.0, 11.0],
            "mae": [9.0, 9.0],
        }
    )

    selected = select_may_configurations(may)

    assert selected["targets"]["pv_kw"]["model_name"] == "chronos2_pv_calendar"
    june = may.copy()
    june["split"] = "june_2026_test"
    with pytest.raises(ValueError, match="May metrics only"):
        select_may_configurations(pd.concat([may, june], ignore_index=True))
