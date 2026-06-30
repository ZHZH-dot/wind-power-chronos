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
