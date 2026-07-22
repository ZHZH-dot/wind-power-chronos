"""Forecast error metrics for zero-shot wind power experiments."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def _clean_arrays(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    mask = np.isfinite(actual) & np.isfinite(predicted)
    return actual[mask], predicted[mask]


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual, predicted = _clean_arrays(y_true, y_pred)
    if actual.size == 0:
        return math.nan
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual, predicted = _clean_arrays(y_true, y_pred)
    if actual.size == 0:
        return math.nan
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def bias(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean signed error, positive when forecasts overpredict."""
    actual, predicted = _clean_arrays(y_true, y_pred)
    if actual.size == 0:
        return math.nan
    return float(np.mean(predicted - actual))


def wape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Weighted absolute percentage error with an absolute-actual denominator."""
    actual, predicted = _clean_arrays(y_true, y_pred)
    if actual.size == 0:
        return math.nan
    denominator = float(np.sum(np.abs(actual)))
    if denominator == 0:
        return math.nan
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def mase(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    insample: Iterable[float],
    seasonal_period: int = 1,
) -> float:
    """Mean absolute scaled error using a strictly historical seasonal scale."""
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive.")
    history = np.asarray(list(insample), dtype=float)
    if history.size <= seasonal_period:
        return math.nan
    current = history[seasonal_period:]
    lagged = history[:-seasonal_period]
    mask = np.isfinite(current) & np.isfinite(lagged)
    if not np.any(mask):
        return math.nan
    scale = float(np.mean(np.abs(current[mask] - lagged[mask])))
    if not math.isfinite(scale) or scale == 0:
        return math.nan
    return mae(y_true, y_pred) / scale


def mase_from_scales(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    scales: Iterable[float],
) -> float:
    """Aggregate absolute scaled errors when each origin has a causal scale."""
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    scale_values = np.asarray(list(scales), dtype=float)
    if actual.shape != predicted.shape or actual.shape != scale_values.shape:
        raise ValueError("y_true, y_pred, and scales must have the same shape.")
    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & np.isfinite(scale_values)
        & (scale_values > 0)
    )
    if not np.any(mask):
        return math.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / scale_values[mask]))


def _default_normalizer(y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true)))


def nmae(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    normalizer: float | None = None,
) -> float:
    actual, predicted = _clean_arrays(y_true, y_pred)
    if actual.size == 0:
        return math.nan
    denominator = float(normalizer) if normalizer is not None else _default_normalizer(actual)
    if denominator == 0:
        return math.nan
    return mae(actual, predicted) / denominator


def nrmse(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    normalizer: float | None = None,
) -> float:
    actual, predicted = _clean_arrays(y_true, y_pred)
    if actual.size == 0:
        return math.nan
    denominator = float(normalizer) if normalizer is not None else _default_normalizer(actual)
    if denominator == 0:
        return math.nan
    return rmse(actual, predicted) / denominator


def _clean_capacity_arrays(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    rated_capacity_kw: float | Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if np.isscalar(rated_capacity_kw):
        capacity = np.full(actual.shape, float(rated_capacity_kw), dtype=float)
    else:
        capacity = np.asarray(list(rated_capacity_kw), dtype=float)
        if capacity.shape != actual.shape:
            raise ValueError("rated_capacity_kw must be scalar or match y_true shape.")

    mask = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(capacity) & (capacity > 0)
    return actual[mask], predicted[mask], capacity[mask]


def nmae_capacity(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    rated_capacity_kw: float | Iterable[float],
) -> float:
    actual, predicted, capacity = _clean_capacity_arrays(
        y_true,
        y_pred,
        rated_capacity_kw,
    )
    if actual.size == 0:
        return math.nan
    return float(np.mean(np.abs(actual - predicted) / capacity))


def nrmse_capacity(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    rated_capacity_kw: float | Iterable[float],
) -> float:
    actual, predicted, capacity = _clean_capacity_arrays(
        y_true,
        y_pred,
        rated_capacity_kw,
    )
    if actual.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(((actual - predicted) / capacity) ** 2)))


def pinball_loss(
    y_true: Iterable[float],
    y_quantile: Iterable[float],
    quantile: float,
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")
    actual, predicted = _clean_arrays(y_true, y_quantile)
    if actual.size == 0:
        return math.nan
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def mean_pinball_loss(
    y_true: Iterable[float],
    quantile_predictions: Mapping[float, Iterable[float]],
) -> float:
    if not quantile_predictions:
        return math.nan
    actual = list(y_true)
    losses = [
        pinball_loss(actual, prediction, quantile)
        for quantile, prediction in quantile_predictions.items()
    ]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        return math.nan
    return float(np.mean(losses))


def interval_coverage(
    y_true: Iterable[float],
    lower: Iterable[float],
    upper: Iterable[float],
) -> float:
    actual = np.asarray(list(y_true), dtype=float)
    lower_values = np.asarray(list(lower), dtype=float)
    upper_values = np.asarray(list(upper), dtype=float)
    if actual.shape != lower_values.shape or actual.shape != upper_values.shape:
        raise ValueError("y_true, lower, and upper must have the same shape.")
    mask = np.isfinite(actual) & np.isfinite(lower_values) & np.isfinite(upper_values)
    if not np.any(mask):
        return math.nan
    return float(np.mean((actual[mask] >= lower_values[mask]) & (actual[mask] <= upper_values[mask])))


def mean_interval_width(lower: Iterable[float], upper: Iterable[float]) -> float:
    lower_values, upper_values = _clean_arrays(lower, upper)
    if lower_values.size == 0:
        return math.nan
    return float(np.mean(upper_values - lower_values))
