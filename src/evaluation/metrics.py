"""Forecast error metrics for zero-shot wind power experiments."""

from __future__ import annotations

import math
from typing import Iterable

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
