import pandas as pd
import pytest

from src.models.chronos_zero_shot import (
    extract_quantile_predictions,
    run_rolling_forecasts,
)


class FakeChronosPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict_df(self, context_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        prediction_length = int(kwargs["prediction_length"])
        start = pd.Timestamp(context_df["timestamp"].max()) + pd.Timedelta(hours=1)
        timestamps = pd.date_range(start, periods=prediction_length, freq="1h")
        self.calls.append(
            {
                "context_end": pd.Timestamp(context_df["timestamp"].max()),
                "prediction_length": prediction_length,
                "quantile_levels": kwargs["quantile_levels"],
            }
        )
        return pd.DataFrame(
            {
                "id": [str(context_df["id"].iloc[0])] * prediction_length,
                "timestamp": timestamps,
                "target_p10": [10.0 + index for index in range(prediction_length)],
                "target_quantile_0.5": [20.0 + index for index in range(prediction_length)],
                "P90": [30.0 + index for index in range(prediction_length)],
            }
        )


def test_extracts_p10_p50_p90_across_chronos_column_names() -> None:
    timestamp = pd.Timestamp("2020-01-01 01:00:00")
    forecast = pd.DataFrame(
        {
            "id": ["1"],
            "timestamp": [timestamp],
            "target_p10": [10.0],
            "target_quantile_0.5": [20.0],
            "P90": [30.0],
        }
    )

    quantiles = extract_quantile_predictions(forecast, "1", timestamp, horizon=1)

    assert quantiles == {"p10": 10.0, "p50": 20.0, "p90": 30.0}


def test_rolling_forecasts_are_restricted_to_complete_test_horizons() -> None:
    timestamps = pd.date_range("2020-01-01", periods=20, freq="1h")
    data = pd.DataFrame(
        {
            "id": ["1"] * len(timestamps),
            "timestamp": timestamps,
            "target": range(len(timestamps)),
            "is_imputed_target": [False] * 16 + [True] + [False] * 3,
        }
    )
    pipeline = FakeChronosPipeline()

    predictions = run_rolling_forecasts(
        pipeline=pipeline,
        df=data,
        mode="univariate",
        covariates=[],
        horizons=[1, 2],
        context_length=4,
        stride=1,
        quantile_levels=[0.5],
        model_id="amazon/chronos-2",
        test_start=timestamps[16],
        test_end=timestamps[19],
    )

    assert predictions["timestamp"].min() >= timestamps[16]
    assert predictions["timestamp"].max() <= timestamps[19]
    assert all(
        call["context_end"] + pd.Timedelta(hours=int(call["prediction_length"]))
        <= timestamps[19]
        for call in pipeline.calls
    )
    assert predictions.loc[predictions["timestamp"] == timestamps[16], "is_imputed_target"].all()
    assert (predictions["y_pred"] == predictions["p50"]).all()
    assert pipeline.calls[0]["context_end"] == timestamps[15]
    assert pipeline.calls[0]["quantile_levels"] == [0.1, 0.5, 0.9]
