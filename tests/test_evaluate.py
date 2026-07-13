import math

import pandas as pd
import pytest

from src.evaluation.evaluate import evaluate_predictions
from src.evaluation.metrics import (
    bias,
    interval_coverage,
    mean_interval_width,
    nmae_capacity,
    nrmse_capacity,
    pinball_loss,
)


def _predictions_with_imputed_target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mode": ["univariate", "univariate"],
            "horizon": [1, 1],
            "y_true": [100.0, 100.0],
            "p10": [90.0, 90.0],
            "p50": [110.0, 500.0],
            "p90": [120.0, 510.0],
            "y_pred": [999.0, 999.0],
            "is_imputed_target": [False, True],
        }
    )


def test_imputed_targets_are_excluded_by_default() -> None:
    result = evaluate_predictions(_predictions_with_imputed_target()).iloc[0]

    assert result["n_scored"] == 1
    assert result["n_excluded_imputed"] == 1
    assert result["mae"] == pytest.approx(10.0)
    assert result["bias"] == pytest.approx(10.0)
    assert result["nmae_capacity"] == pytest.approx(10.0 / 1500.0)


def test_imputed_targets_can_be_included_for_diagnostics() -> None:
    result = evaluate_predictions(
        _predictions_with_imputed_target(),
        include_imputed_targets=True,
    ).iloc[0]

    assert result["n_scored"] == 2
    assert result["n_excluded_imputed"] == 0
    assert result["mae"] == pytest.approx(205.0)


def test_capacity_normalization_and_bias() -> None:
    y_true = [0.0, 1500.0]
    y_pred = [150.0, 1200.0]

    assert nmae_capacity(y_true, y_pred, 1500.0) == pytest.approx(0.15)
    assert nrmse_capacity(y_true, y_pred, 1500.0) == pytest.approx(
        math.sqrt((150.0**2 + 300.0**2) / 2.0) / 1500.0
    )
    assert bias(y_true, y_pred) == pytest.approx(-75.0)


def test_evaluation_uses_rated_capacity_column_when_present() -> None:
    predictions = pd.DataFrame(
        {
            "mode": ["univariate", "univariate"],
            "horizon": [1, 1],
            "y_true": [0.0, 0.0],
            "p50": [100.0, 200.0],
            "p10": [0.0, 0.0],
            "p90": [200.0, 400.0],
            "rated_capacity_kw": [1000.0, 2000.0],
            "is_imputed_target": [False, False],
        }
    )

    result = evaluate_predictions(predictions, rated_capacity_kw=1500.0).iloc[0]

    assert result["nmae_capacity"] == pytest.approx(0.1)
    assert result["nrmse_capacity"] == pytest.approx(0.1)


def test_pinball_loss_and_p10_p90_interval_metrics() -> None:
    assert pinball_loss([0.0, 10.0], [-1.0, 8.0], 0.1) == pytest.approx(0.15)

    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    p10 = [0.0, 1.0, 2.0, 3.0, 6.0]
    p90 = [2.0, 3.0, 4.0, 5.0, 8.0]
    assert interval_coverage(y_true, p10, p90) == pytest.approx(0.8)
    assert mean_interval_width(p10, p90) == pytest.approx(2.0)
