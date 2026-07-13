import pandas as pd
import pytest

from src.data.prepare_sdwpf import prepare_sdwpf_dataframe


def _tiny_raw_sdwpf() -> pd.DataFrame:
    timestamps = [f"{hour:02d}:{minute:02d}" for hour in range(2) for minute in range(0, 60, 10)]
    return pd.DataFrame(
        {
            "TurbID": [1] * len(timestamps),
            "Day": [1] * len(timestamps),
            "Tmstamp": timestamps,
            "Patv": list(range(len(timestamps))),
            "Wspd": [10.0 + index for index in range(len(timestamps))],
            "Wdir": [100.0 + index for index in range(len(timestamps))],
        }
    )


def test_prepare_sdwpf_resamples_hourly() -> None:
    processed = prepare_sdwpf_dataframe(
        _tiny_raw_sdwpf(),
        covariates=["Wspd", "Wdir"],
        freq="1h",
    )

    assert list(processed.columns) == ["id", "timestamp", "target", "Wspd", "Wdir"]
    assert len(processed) == 2
    assert processed.loc[0, "id"] == "1"
    assert processed.loc[0, "timestamp"] == pd.Timestamp("2020-01-01 00:00:00")
    assert processed.loc[1, "timestamp"] == pd.Timestamp("2020-01-01 01:00:00")
    assert processed.loc[0, "target"] == pytest.approx(2.5)


def test_prepare_sdwpf_preserves_covariate_means() -> None:
    processed = prepare_sdwpf_dataframe(
        _tiny_raw_sdwpf(),
        covariates=["Wspd", "Wdir"],
        freq="1h",
    )

    assert processed.loc[0, "Wspd"] == pytest.approx(12.5)
    assert processed.loc[0, "Wdir"] == pytest.approx(102.5)


def test_prepare_sdwpf_regularizes_missing_hour() -> None:
    raw = pd.DataFrame(
        {
            "TurbID": [1, 1],
            "timestamp": ["2020-01-01 00:00:00", "2020-01-01 02:00:00"],
            "Patv": [1.0, 3.0],
            "Wspd": [10.0, 30.0],
        }
    )

    processed = prepare_sdwpf_dataframe(
        raw,
        timestamp_column="timestamp",
        covariates=["Wspd"],
        freq="1h",
        regularize_hourly=True,
    )

    inserted = processed[processed["timestamp"] == pd.Timestamp("2020-01-01 01:00:00")]

    assert len(processed) == 3
    assert len(inserted) == 1
    assert inserted.iloc[0]["target"] == pytest.approx(2.0)
    assert inserted.iloc[0]["Wspd"] == pytest.approx(20.0)
    assert bool(inserted.iloc[0]["is_imputed_target"]) is True


def test_regularization_uses_each_turbines_observed_range() -> None:
    raw = pd.DataFrame(
        {
            "TurbID": [1, 1, 2, 2],
            "timestamp": [
                "2020-01-01 00:00:00",
                "2020-01-01 02:00:00",
                "2020-01-01 01:00:00",
                "2020-01-01 03:00:00",
            ],
            "Patv": [1.0, 3.0, 10.0, 30.0],
            "Wspd": [10.0, 30.0, 100.0, 300.0],
        }
    )

    processed = prepare_sdwpf_dataframe(
        raw,
        timestamp_column="timestamp",
        covariates=["Wspd"],
        regularize_hourly=True,
    )

    turbine_1 = processed[processed["id"] == "1"]
    turbine_2 = processed[processed["id"] == "2"]
    assert turbine_1["timestamp"].tolist() == list(
        pd.date_range("2020-01-01 00:00:00", periods=3, freq="1h")
    )
    assert turbine_2["timestamp"].tolist() == list(
        pd.date_range("2020-01-01 01:00:00", periods=3, freq="1h")
    )
    assert turbine_1["is_imputed_target"].tolist() == [False, True, False]
    assert turbine_2["is_imputed_target"].tolist() == [False, True, False]
