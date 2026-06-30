import math

import pytest

from src.evaluation.metrics import mae, nmae, nrmse, rmse


def test_metrics_compute_expected_values() -> None:
    y_true = [1.0, 2.0, 4.0]
    y_pred = [2.0, 2.0, 1.0]

    assert mae(y_true, y_pred) == pytest.approx(4.0 / 3.0)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(10.0 / 3.0))
    assert nmae(y_true, y_pred) == pytest.approx(4.0 / 7.0)
    assert nrmse(y_true, y_pred) == pytest.approx(math.sqrt(10.0 / 3.0) / (7.0 / 3.0))


def test_metrics_ignore_nan_pairs() -> None:
    assert mae([1.0, math.nan, 3.0], [2.0, 10.0, 1.0]) == pytest.approx(1.5)
