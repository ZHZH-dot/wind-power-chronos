from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.foshan_load_forecast_ablation import (
    APRIL_SPLIT,
    CHRONOS_672,
    CURRENT_BASELINE,
    FOUR_WEEK_MEDIAN,
    MAY_SPLIT,
    PREVIOUS_WEEK,
    aggregate_provisional_load_15min,
    build_chronos_context_frame,
    chronos_predictions,
    expand_load_predictions_to_five_minutes,
    frozen_v5_policy_kwargs,
    seasonal_predictions,
    select_april_candidates,
)


def _target_table(days: int = 45) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-03-01 00:00:00",
        periods=days * 96,
        freq="15min",
        tz="Asia/Shanghai",
    )
    values = np.arange(len(timestamps), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "provisional_load_proxy_raw_kw": values,
            "provisional_load_kw": values,
            "provisional_load_was_clipped": False,
            "observed_five_minute_target_count": 3,
        }
    )


class FakeChronosPipeline:
    def __init__(self) -> None:
        self.context: pd.DataFrame | None = None
        self.kwargs: dict[str, object] = {}

    def predict_df(self, context: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        self.context = context.copy()
        self.kwargs = kwargs
        rows = []
        for item_id, group in context.groupby("id"):
            start = group["timestamp"].max() + pd.Timedelta(minutes=15)
            value = float(group["target"].dropna().iloc[-1])
            for timestamp in pd.date_range(start, periods=96, freq="15min"):
                rows.append(
                    {
                        "id": item_id,
                        "timestamp": timestamp,
                        "0.1": value - 1.0,
                        "0.5": value,
                        "0.9": value + 1.0,
                    }
                )
        return pd.DataFrame(rows)


def test_reconstruction_aggregation_preserves_raw_clipped_and_flag() -> None:
    timestamps = pd.date_range(
        "2026-04-30 00:00:00", periods=6, freq="5min", tz="Asia/Shanghai"
    )
    raw = np.array([-3.0, 3.0, 6.0, 9.0, 12.0, 15.0])
    five_minute = pd.DataFrame(
        {
            "timestamp": timestamps,
            "provisional_load_proxy_raw_kw": raw,
            "provisional_load_kw": np.maximum(raw, 0.0),
            "provisional_load_was_clipped": raw < 0.0,
        }
    )

    result = aggregate_provisional_load_15min(five_minute)

    assert len(result) == 2
    assert result["provisional_load_proxy_raw_kw"].tolist() == [2.0, 12.0]
    assert result["provisional_load_kw"].tolist() == [3.0, 12.0]
    assert result["provisional_load_was_clipped"].tolist() == [True, False]
    assert result["five_minute_row_count"].tolist() == [3, 3]


def test_chronos_context_is_strictly_causal_and_forecast_has_96_points() -> None:
    target = _target_table()
    issue_time = pd.Timestamp("2026-04-01 00:00:00", tz="Asia/Shanghai")
    context, metadata = build_chronos_context_frame(target, [issue_time], 672)
    original_timezone = target["timestamp"].dt.tz

    assert len(context) == 672
    assert context["timestamp"].dt.tz is None
    assert context["timestamp"].max() == pd.Timestamp("2026-03-31 23:45:00")
    assert metadata.loc[0, "context_end_timestamp"] < issue_time
    assert str(original_timezone) == "Asia/Shanghai"

    pipeline = FakeChronosPipeline()
    predictions = chronos_predictions(
        pipeline,
        target,
        [issue_time],
        CHRONOS_672,
        APRIL_SPLIT,
        "amazon/chronos-2",
    )

    assert len(predictions) == 96
    assert predictions["horizon_step"].tolist() == list(range(1, 97))
    assert predictions["forecast_source_timestamp"].lt(issue_time).all()
    assert not predictions["used_future_realized_data"].any()
    assert pipeline.kwargs["future_df"] is None
    assert pipeline.kwargs["target"] == "target"


def test_five_minute_expansion_repeats_each_quarter_hour() -> None:
    issue = pd.Timestamp("2026-05-01 00:00:00", tz="Asia/Shanghai")
    target_times = pd.date_range(issue, periods=96, freq="15min")
    predictions = pd.DataFrame(
        {
            "issue_time": issue,
            "target_time": target_times,
            "horizon_step": np.arange(1, 97),
            "p50": np.arange(96, dtype=float),
            "forecast_source_timestamp": issue - pd.Timedelta(minutes=15),
        }
    )

    expanded = expand_load_predictions_to_five_minutes(
        predictions,
        issue.tz_localize(None),
        (issue + pd.Timedelta(days=1)).tz_localize(None),
    )

    assert len(expanded) == 288
    assert expanded["forecast_load_kw"].iloc[:6].tolist() == [0, 0, 0, 1, 1, 1]
    assert expanded["timestamp"].iloc[:4].tolist() == list(
        pd.date_range("2026-05-01 00:00:00", periods=4, freq="5min")
    )


def test_previous_week_uses_exact_same_interval() -> None:
    target = _target_table()
    issue = pd.Timestamp("2026-04-01 00:00:00", tz="Asia/Shanghai")

    predictions = seasonal_predictions(
        target, [issue], PREVIOUS_WEEK, APRIL_SPLIT
    )
    lookup = target.set_index("timestamp")["provisional_load_kw"]
    expected_source = predictions["target_time"] - pd.Timedelta(days=7)

    assert np.array_equal(
        predictions["p50"].to_numpy(),
        lookup.reindex(expected_source).to_numpy(),
    )
    assert predictions["forecast_source_timestamp"].tolist() == expected_source.tolist()
    assert predictions["forecast_source_timestamp"].lt(issue).all()


def test_four_week_median_uses_exact_four_week_slots() -> None:
    target = _target_table()
    issue = pd.Timestamp("2026-04-01 00:00:00", tz="Asia/Shanghai")

    predictions = seasonal_predictions(
        target, [issue], FOUR_WEEK_MEDIAN, APRIL_SPLIT
    )
    lookup = target.set_index("timestamp")["provisional_load_kw"]
    first_target = predictions.loc[0, "target_time"]
    source_times = [first_target - pd.Timedelta(days=value) for value in (7, 14, 21, 28)]

    assert predictions.loc[0, "p50"] == float(np.median(lookup.reindex(source_times)))
    assert predictions.loc[0, "forecast_source_timestamps"] == "|".join(
        value.isoformat() for value in source_times
    )


def test_selection_rejects_any_may_metrics() -> None:
    candidates = [PREVIOUS_WEEK, FOUR_WEEK_MEDIAN, CHRONOS_672, "chronos2_univariate_ctx1344"]
    metrics = pd.DataFrame(
        {
            "split": APRIL_SPLIT,
            "candidate": candidates,
            "candidate_kind": "non_baseline",
            "metric_scope": "overall",
            "wape": [0.4, 0.3, 0.2, 0.1],
            "absolute_bias": [1.0, 1.0, 1.0, 1.0],
        }
    )
    contaminated = pd.concat(
        [metrics, metrics.iloc[[0]].assign(split=MAY_SPLIT)], ignore_index=True
    )

    with pytest.raises(ValueError, match="April metrics only"):
        select_april_candidates(contaminated)


def test_frozen_controller_v5_policy_is_unchanged() -> None:
    assert frozen_v5_policy_kwargs() == {
        "cadence_minutes": 5,
        "use_q10_discharge_limit": False,
        "use_terminal_recovery_charge_ban": False,
        "use_latest_completed_residual_for_first_step": True,
        "use_intraday_load_bias_correction": False,
        "use_final_day_immediate_charge_guard": True,
    }


def test_current_baseline_remains_identified_as_baseline() -> None:
    target = _target_table()
    issue = pd.Timestamp("2026-04-01 00:00:00", tz="Asia/Shanghai")
    predictions = seasonal_predictions(
        target, [issue], CURRENT_BASELINE, APRIL_SPLIT
    )

    assert predictions["candidate_kind"].eq("baseline").all()
    assert predictions["forecast_source_timestamp"].lt(issue).all()
