"""Leakage-safe rolling evaluation utilities for the Foshan benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.data.prepare_foshan import CALENDAR_COLUMNS, TIMEZONE
from src.evaluation.metrics import (
    bias,
    interval_coverage,
    mae,
    mase_from_scales,
    mean_interval_width,
    mean_pinball_loss,
    pinball_loss,
    rmse,
    wape,
)
from src.models.chronos_zero_shot import choose_quantile_column


PREDICTION_COLUMNS = [
    "run_id",
    "split",
    "issue_time",
    "target_time",
    "horizon_step",
    "target",
    "model_name",
    "model_id",
    "context_length",
    "y_true_raw",
    "y_true",
    "is_missing_target",
    "p10",
    "p50",
    "p90",
    "y_pred",
    "mase_scale",
    "postprocessing",
    "used_future_covariates",
    "future_covariate_columns",
    "provisional_target",
]


@dataclass
class ForecastWindow:
    issue_time: pd.Timestamp
    context_df: pd.DataFrame
    future_df: pd.DataFrame | None
    truth_df: pd.DataFrame
    mase_scales: dict[str, float]


def load_foshan_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "model_id",
        "frequency",
        "timezone",
        "prediction_length",
        "quantile_levels",
        "context_lengths",
        "selection_period",
        "test_period",
        "configurations",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Foshan config is missing keys: {missing}")
    if int(config["prediction_length"]) != 96:
        raise ValueError("The Foshan benchmark prediction_length must remain 96.")
    if sorted(float(value) for value in config["quantile_levels"]) != [0.1, 0.5, 0.9]:
        raise ValueError("The Foshan benchmark quantiles must remain 0.1, 0.5, and 0.9.")
    return config


def site_timestamp(value: str | pd.Timestamp, timezone: str = TIMEZONE) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def validate_processed_table(table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required = {
        "id",
        "timestamp",
        "pv_kw_raw",
        "pv_kw",
        "net_grid_kw_raw",
        "net_grid_kw",
        "is_missing_pv_kw",
        "is_missing_net_grid_kw",
        *config["calendar_covariates"],
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Processed Foshan table is missing columns: {missing}")
    data = table.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if data["timestamp"].dt.tz is None:
        raise ValueError("Processed Foshan timestamps must be timezone-aware.")
    data["timestamp"] = data["timestamp"].dt.tz_convert(config["timezone"])
    data = data.sort_values(["id", "timestamp"]).reset_index(drop=True)
    if data.duplicated(["id", "timestamp"]).any():
        raise ValueError("Processed Foshan table contains duplicate timestamps.")
    for _, group in data.groupby("id", sort=False):
        differences = group["timestamp"].diff().dropna()
        if not differences.eq(pd.Timedelta(config["frequency"])).all():
            raise ValueError("Processed Foshan table is not on an exact 15-minute grid.")
    return data


def period_origins(
    table: pd.DataFrame,
    period: dict[str, str],
    prediction_length: int = 96,
    frequency: str = "15min",
    stride_steps: int = 96,
    timezone: str = TIMEZONE,
) -> list[pd.Timestamp]:
    if stride_steps <= 0:
        raise ValueError("stride_steps must be positive.")
    start = site_timestamp(period["start"], timezone)
    end_exclusive = site_timestamp(period["end_exclusive"], timezone)
    step = pd.Timedelta(frequency)
    final_origin = end_exclusive - prediction_length * step
    if final_origin < start:
        return []
    candidates = list(pd.date_range(start, final_origin, freq=stride_steps * step))
    available = set(pd.DatetimeIndex(table["timestamp"]))
    origins: list[pd.Timestamp] = []
    for issue_time in candidates:
        expected = pd.date_range(issue_time, periods=prediction_length, freq=frequency)
        if expected[-1] >= end_exclusive:
            continue
        if all(timestamp in available for timestamp in expected):
            origins.append(issue_time)
    return origins


def causal_seasonal_scale(
    table: pd.DataFrame,
    target: str,
    issue_time: pd.Timestamp,
    seasonal_period: int = 96,
) -> float:
    """Compute a seasonal-naive scale from observations strictly before an origin."""
    history = pd.to_numeric(
        table.loc[table["timestamp"] < issue_time, target], errors="coerce"
    ).to_numpy(dtype=float)
    if len(history) <= seasonal_period:
        return math.nan
    current = history[seasonal_period:]
    lagged = history[:-seasonal_period]
    valid = np.isfinite(current) & np.isfinite(lagged)
    if not np.any(valid):
        return math.nan
    scale = float(np.mean(np.abs(current[valid] - lagged[valid])))
    return scale if scale > 0 else math.nan


def build_forecast_window(
    table: pd.DataFrame,
    issue_time: pd.Timestamp,
    targets: list[str],
    context_length: int,
    prediction_length: int,
    known_future_covariates: list[str],
    causal_fill_limit: int = 2,
    frequency: str = "15min",
) -> tuple[ForecastWindow | None, str | None]:
    """Build one regular Chronos window using only causally available target values."""
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if causal_fill_limit < 0:
        raise ValueError("causal_fill_limit must be nonnegative.")
    data = table.sort_values("timestamp").set_index("timestamp", drop=False)
    step = pd.Timedelta(frequency)
    context_index = pd.date_range(
        end=issue_time - step,
        periods=context_length,
        freq=frequency,
    )
    future_index = pd.date_range(issue_time, periods=prediction_length, freq=frequency)
    if not context_index.isin(data.index).all():
        return None, "insufficient_context"
    if not future_index.isin(data.index).all():
        return None, "incomplete_horizon"

    context = data.reindex(context_index).copy()
    for target in targets:
        context[target] = pd.to_numeric(context[target], errors="coerce").ffill(
            limit=causal_fill_limit
        )
    if context[targets].isna().any().any():
        return None, "unresolved_context_gap"

    context_columns = ["id", "timestamp", *targets, *known_future_covariates]
    context_df = context[context_columns].reset_index(drop=True)
    truth_columns = [
        "timestamp",
        *targets,
        *[f"{target}_raw" for target in targets],
        *[f"is_missing_{target}" for target in targets],
    ]
    truth_df = data.reindex(future_index)[truth_columns].reset_index(drop=True)
    future_df: pd.DataFrame | None = None
    if known_future_covariates:
        future_df = data.reindex(future_index)[
            ["id", "timestamp", *known_future_covariates]
        ].reset_index(drop=True)

    assert context_df["timestamp"].max() < issue_time
    assert truth_df["timestamp"].min() == issue_time
    if future_df is not None:
        forbidden = set(targets) | {"pv_kw", "net_grid_kw", "pv_kw_raw", "net_grid_kw_raw"}
        assert not forbidden.intersection(future_df.columns)
    scales = {
        target: causal_seasonal_scale(table, target, issue_time, seasonal_period=96)
        for target in targets
    }
    return (
        ForecastWindow(
            issue_time=issue_time,
            context_df=context_df,
            future_df=future_df,
            truth_df=truth_df,
            mase_scales=scales,
        ),
        None,
    )


def _past_exact(series: pd.Series, timestamp: pd.Timestamp) -> float:
    try:
        value = series.loc[timestamp]
    except KeyError:
        return math.nan
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else math.nan


def _last_causal_value(
    series: pd.Series,
    issue_time: pd.Timestamp,
    fill_limit: int,
) -> float:
    history = pd.to_numeric(series.loc[series.index < issue_time], errors="coerce")
    if history.empty:
        return math.nan
    tail = history.tail(fill_limit + 1).ffill(limit=fill_limit)
    value = tail.iloc[-1]
    return float(value) if pd.notna(value) else math.nan


def _prediction_record(
    *,
    run_id: str,
    split: str,
    issue_time: pd.Timestamp,
    target_time: pd.Timestamp,
    horizon_step: int,
    target: str,
    model_name: str,
    model_id: str,
    context_length: int,
    y_true_raw: float,
    y_true: float,
    is_missing_target: bool,
    p10: float,
    p50: float,
    p90: float,
    mase_scale: float,
    postprocessing: str,
    used_future_covariates: bool,
    future_covariate_columns: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "split": split,
        "issue_time": issue_time,
        "target_time": target_time,
        "horizon_step": int(horizon_step),
        "target": target,
        "model_name": model_name,
        "model_id": model_id,
        "context_length": int(context_length),
        "y_true_raw": y_true_raw,
        "y_true": y_true,
        "is_missing_target": bool(is_missing_target),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "y_pred": p50,
        "mase_scale": mase_scale,
        "postprocessing": postprocessing,
        "used_future_covariates": bool(used_future_covariates),
        "future_covariate_columns": ",".join(future_covariate_columns),
        "provisional_target": target == "net_grid_kw",
    }


def run_causal_baselines(
    table: pd.DataFrame,
    origins: list[pd.Timestamp],
    split_name: str,
    run_id: str,
    prediction_length: int = 96,
    causal_fill_limit: int = 2,
    frequency: str = "15min",
) -> pd.DataFrame:
    """Run deterministic reference methods without future filling."""
    indexed = table.sort_values("timestamp").set_index("timestamp")
    records: list[dict[str, Any]] = []
    baseline_specs = {
        "zero": 0,
        "persistence": 1,
        "previous_day": 96,
        "previous_week": 672,
        "four_week_slot_mean": 2688,
    }
    for issue_time in origins:
        future_times = pd.date_range(issue_time, periods=prediction_length, freq=frequency)
        for target in ("pv_kw", "net_grid_kw"):
            source = pd.to_numeric(indexed[target], errors="coerce")
            scale = causal_seasonal_scale(table, target, issue_time, seasonal_period=96)
            persistence = _last_causal_value(source, issue_time, causal_fill_limit)
            for horizon_step, target_time in enumerate(future_times, start=1):
                actual_row = indexed.loc[target_time]
                y_true = float(actual_row[target]) if pd.notna(actual_row[target]) else math.nan
                raw_column = f"{target}_raw"
                y_true_raw = (
                    float(actual_row[raw_column])
                    if pd.notna(actual_row[raw_column])
                    else math.nan
                )
                missing = bool(actual_row[f"is_missing_{target}"])
                predictions: dict[str, float] = {
                    "persistence": persistence,
                    "previous_day": _past_exact(source, target_time - pd.Timedelta(days=1)),
                    "previous_week": _past_exact(source, target_time - pd.Timedelta(days=7)),
                }
                four_week_values = [
                    _past_exact(source, target_time - pd.Timedelta(days=7 * week))
                    for week in range(1, 5)
                ]
                predictions["four_week_slot_mean"] = (
                    float(np.mean(four_week_values))
                    if all(math.isfinite(value) for value in four_week_values)
                    else math.nan
                )
                if target == "pv_kw":
                    predictions["zero"] = 0.0
                for model_name, point in predictions.items():
                    records.append(
                        _prediction_record(
                            run_id=run_id,
                            split=split_name,
                            issue_time=issue_time,
                            target_time=target_time,
                            horizon_step=horizon_step,
                            target=target,
                            model_name=model_name,
                            model_id="causal_baseline",
                            context_length=baseline_specs[model_name],
                            y_true_raw=y_true_raw,
                            y_true=y_true,
                            is_missing_target=missing,
                            p10=math.nan,
                            p50=point,
                            p90=math.nan,
                            mase_scale=scale,
                            postprocessing="physical_target" if target == "pv_kw" else "none",
                            used_future_covariates=False,
                        )
                    )
    return pd.DataFrame(records, columns=PREDICTION_COLUMNS)


def normalize_chronos_quantiles(
    forecast_df: pd.DataFrame,
    targets: list[str],
    expected_times: pd.DatetimeIndex,
    site_id: str,
) -> pd.DataFrame:
    """Map Chronos-2 output to target-aware P10/P50/P90 rows."""
    forecast = forecast_df.reset_index()
    if "timestamp" not in forecast.columns and "index" in forecast.columns:
        forecast = forecast.rename(columns={"index": "timestamp"})
    if "timestamp" not in forecast.columns:
        raise ValueError("Chronos output does not contain a timestamp column.")
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    if forecast["timestamp"].dt.tz is None and expected_times.tz is not None:
        forecast["timestamp"] = forecast["timestamp"].dt.tz_localize(expected_times.tz)
    elif forecast["timestamp"].dt.tz is not None and expected_times.tz is not None:
        forecast["timestamp"] = forecast["timestamp"].dt.tz_convert(expected_times.tz)
    if "id" in forecast.columns:
        forecast = forecast[forecast["id"].astype(str) == str(site_id)]
    if "target_name" not in forecast.columns:
        if len(targets) != 1:
            raise ValueError("Multi-target Chronos output must contain target_name.")
        forecast["target_name"] = targets[0]

    normalized: list[pd.DataFrame] = []
    for target in targets:
        target_frame = forecast[forecast["target_name"].astype(str) == target].copy()
        if target_frame.empty:
            raise ValueError(f"Chronos output is missing target_name={target!r}.")
        quantile_columns = {
            name: choose_quantile_column(target_frame, quantile, target_column=target)
            for name, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))
        }
        target_frame = target_frame.set_index("timestamp").reindex(expected_times)
        if target_frame[list(quantile_columns.values())].isna().any().any():
            raise ValueError(f"Chronos output is incomplete for target {target!r}.")
        normalized.append(
            pd.DataFrame(
                {
                    "timestamp": expected_times,
                    "target": target,
                    "p10": target_frame[quantile_columns["p10"]].to_numpy(dtype=float),
                    "p50": target_frame[quantile_columns["p50"]].to_numpy(dtype=float),
                    "p90": target_frame[quantile_columns["p90"]].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(normalized, ignore_index=True)


def postprocess_pv_quantiles(
    quantiles: pd.DataFrame,
    lower: float = 0.0,
    upper: float = 1700.0,
) -> pd.DataFrame:
    """Clip PV forecasts and repair quantile crossing row by row."""
    result = quantiles.copy()
    ordered = np.sort(
        result[["p10", "p50", "p90"]].clip(lower=lower, upper=upper).to_numpy(dtype=float),
        axis=1,
    )
    result[["p10", "p50", "p90"]] = ordered
    return result


def chronos_rows_for_window(
    forecast_df: pd.DataFrame,
    window: ForecastWindow,
    targets: list[str],
    split_name: str,
    run_id: str,
    model_name: str,
    model_id: str,
    context_length: int,
    known_future_covariates: list[str],
    pv_capacity_kw: float = 1700.0,
    site_id: str = "foshan_site",
) -> pd.DataFrame:
    expected_times = pd.DatetimeIndex(window.truth_df["timestamp"])
    normalized = normalize_chronos_quantiles(
        forecast_df, targets=targets, expected_times=expected_times, site_id=site_id
    )
    records: list[dict[str, Any]] = []
    for target in targets:
        target_quantiles = normalized[normalized["target"] == target].reset_index(drop=True)
        variants = [("raw", target_quantiles)]
        if target == "pv_kw":
            variants.append(
                (
                    "physical_clip_0_1700",
                    postprocess_pv_quantiles(target_quantiles, upper=pv_capacity_kw),
                )
            )
        truth = window.truth_df.reset_index(drop=True)
        for postprocessing, values in variants:
            for position, row in values.iterrows():
                truth_row = truth.iloc[position]
                y_true = truth_row[target]
                raw_target = truth_row[f"{target}_raw"]
                records.append(
                    _prediction_record(
                        run_id=run_id,
                        split=split_name,
                        issue_time=window.issue_time,
                        target_time=pd.Timestamp(row["timestamp"]),
                        horizon_step=position + 1,
                        target=target,
                        model_name=model_name,
                        model_id=model_id,
                        context_length=context_length,
                        y_true_raw=float(raw_target) if pd.notna(raw_target) else math.nan,
                        y_true=float(y_true) if pd.notna(y_true) else math.nan,
                        is_missing_target=bool(truth_row[f"is_missing_{target}"]),
                        p10=float(row["p10"]),
                        p50=float(row["p50"]),
                        p90=float(row["p90"]),
                        mase_scale=window.mase_scales[target],
                        postprocessing=postprocessing,
                        used_future_covariates=bool(known_future_covariates),
                        future_covariate_columns=known_future_covariates,
                    )
                )
    return pd.DataFrame(records, columns=PREDICTION_COLUMNS)


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().isin({"true", "1", "yes"})


def _metric_row(group: pd.DataFrame, pv_active_threshold_kw: float) -> dict[str, Any]:
    missing = _bool_series(group["is_missing_target"])
    point_values = group.loc[~missing].copy()
    finite = np.isfinite(pd.to_numeric(point_values["y_true"], errors="coerce")) & np.isfinite(
        pd.to_numeric(point_values["p50"], errors="coerce")
    )
    scored = point_values[finite]
    actual = scored["y_true"]
    prediction = scored["p50"]
    target = str(group["target"].iloc[0])
    active = (
        scored[scored["y_true"] > pv_active_threshold_kw]
        if target == "pv_kw"
        else scored.iloc[0:0]
    )
    probabilistic = point_values.dropna(subset=["y_true", "p10", "p50", "p90"])
    has_probabilistic = not probabilistic.empty
    return {
        "n_scored": int(len(scored)),
        "n_excluded_missing": int(missing.sum()),
        "mae": mae(actual, prediction),
        "rmse": rmse(actual, prediction),
        "wape": wape(actual, prediction),
        "mase": mase_from_scales(actual, prediction, scored["mase_scale"]),
        "bias": bias(actual, prediction),
        "pv_active_n_scored": int(len(active)),
        "pv_active_mae": mae(active["y_true"], active["p50"]),
        "pv_active_rmse": rmse(active["y_true"], active["p50"]),
        "pv_active_wape": wape(active["y_true"], active["p50"]),
        "pinball_p10": pinball_loss(probabilistic["y_true"], probabilistic["p10"], 0.1)
        if has_probabilistic
        else math.nan,
        "pinball_p50": pinball_loss(probabilistic["y_true"], probabilistic["p50"], 0.5)
        if has_probabilistic
        else math.nan,
        "pinball_p90": pinball_loss(probabilistic["y_true"], probabilistic["p90"], 0.9)
        if has_probabilistic
        else math.nan,
        "mean_pinball_loss": mean_pinball_loss(
            probabilistic["y_true"],
            {
                0.1: probabilistic["p10"],
                0.5: probabilistic["p50"],
                0.9: probabilistic["p90"],
            },
        )
        if has_probabilistic
        else math.nan,
        "p10_p90_coverage": interval_coverage(
            probabilistic["y_true"], probabilistic["p10"], probabilistic["p90"]
        )
        if has_probabilistic
        else math.nan,
        "mean_interval_width": mean_interval_width(
            probabilistic["p10"], probabilistic["p90"]
        )
        if has_probabilistic
        else math.nan,
    }


def evaluate_foshan_predictions(
    predictions: pd.DataFrame,
    by_horizon: bool = False,
    pv_active_threshold_kw: float = 1.0,
) -> pd.DataFrame:
    """Score missing-target-safe point and probabilistic forecast records."""
    if pv_active_threshold_kw < 0:
        raise ValueError("pv_active_threshold_kw must be nonnegative.")
    group_columns = [
        "split",
        "target",
        "model_name",
        "model_id",
        "context_length",
        "postprocessing",
        "provisional_target",
    ]
    if by_horizon:
        group_columns.append("horizon_step")
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, sort=True, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        row.update(_metric_row(group, pv_active_threshold_kw))
        rows.append(row)
    return pd.DataFrame(rows)


def select_may_configurations(metrics: pd.DataFrame) -> dict[str, Any]:
    """Select Chronos configuration/context pairs without consulting June."""
    if metrics.empty:
        raise ValueError("May selection metrics are empty.")
    if not metrics["split"].eq("may_2026_selection").all():
        raise ValueError("Configuration selection may use May metrics only.")
    chronos = metrics[metrics["model_name"].astype(str).str.startswith("chronos2_")].copy()
    selected: dict[str, Any] = {"selected_on": "may_2026_selection", "targets": {}}
    for target in ("pv_kw", "net_grid_kw"):
        candidates = chronos[chronos["target"] == target].copy()
        if target == "pv_kw":
            candidates = candidates[candidates["postprocessing"] == "physical_clip_0_1700"]
            order = ["wape", "pv_active_mae", "model_name", "context_length"]
        else:
            candidates = candidates[candidates["postprocessing"] == "raw"]
            order = ["wape", "mae", "model_name", "context_length"]
        candidates = candidates.dropna(subset=["wape"])
        if candidates.empty:
            continue
        winner = candidates.sort_values(order, kind="mergesort").iloc[0]
        selected["targets"][target] = {
            "model_name": str(winner["model_name"]),
            "context_length": int(winner["context_length"]),
            "selection_wape": float(winner["wape"]),
            "selection_tie_break": float(
                winner["pv_active_mae"] if target == "pv_kw" else winner["mae"]
            ),
            "provisional_target": target == "net_grid_kw",
        }
    if "pv_kw" not in selected["targets"]:
        raise RuntimeError("May selection did not produce a valid PV Chronos configuration.")
    return selected


def configurations_for_frozen_test(
    configurations: list[dict[str, Any]],
    selection: dict[str, Any],
) -> list[tuple[dict[str, Any], int]]:
    """Return the unique May-selected configurations for June evaluation."""
    by_name = {configuration["name"]: configuration for configuration in configurations}
    chosen: list[tuple[dict[str, Any], int]] = []
    seen: set[tuple[str, int]] = set()
    for target_selection in selection["targets"].values():
        key = (target_selection["model_name"], int(target_selection["context_length"]))
        if key in seen:
            continue
        if key[0] not in by_name:
            raise ValueError(f"Selected configuration is not in the benchmark config: {key[0]}")
        seen.add(key)
        chosen.append((by_name[key[0]], key[1]))
    return chosen
