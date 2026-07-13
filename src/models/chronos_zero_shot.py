"""Rolling-window Chronos-2 zero-shot inference for SDWPF."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.splits import (
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_SPLIT_MANIFEST,
    ensure_split_manifest,
    test_period,
)


DEFAULT_MODEL_ID = "amazon/chronos-2"
DEFAULT_DEVICE_MAP = "cuda"
DEFAULT_HORIZONS = [1, 6, 24, 72]
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


def parse_csv_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        parts = value
    else:
        if value.strip() == "":
            return []
        parts = value.split(",")
    return [str(item).strip() for item in parts if str(item).strip()]


def parse_int_list(value: str | list[str] | list[int] | None) -> list[int]:
    if value is None:
        return DEFAULT_HORIZONS.copy()
    if isinstance(value, str):
        if value.strip() == "":
            return DEFAULT_HORIZONS.copy()
        parts: list[str | int] = value.split(",")
    else:
        parts = value

    horizons: list[int] = []
    for part in parts:
        if isinstance(part, int):
            horizons.append(part)
            continue
        horizons.extend(int(item.strip()) for item in part.split(",") if item.strip())

    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Horizons must be positive integers.")
    return sorted(set(horizons))


def parse_float_list(value: str | None) -> list[float]:
    if value is None or value.strip() == "":
        return DEFAULT_QUANTILES.copy()
    quantiles = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not quantiles or any(quantile <= 0 or quantile >= 1 for quantile in quantiles):
        raise ValueError("Quantiles must be between 0 and 1.")
    return quantiles


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def default_prediction_path(mode: str) -> Path:
    return Path("results") / "zero_shot" / f"predictions_{mode}.csv"


def load_chronos2_pipeline(model_id: str = DEFAULT_MODEL_ID, device_map: str = DEFAULT_DEVICE_MAP) -> Any:
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(model_id, device_map=device_map)


def validate_input_columns(df: pd.DataFrame, mode: str, covariates: list[str]) -> list[str]:
    required = ["id", "timestamp", "target"]
    if mode == "multivariate":
        required.extend(covariates)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    if mode == "multivariate" and not covariates:
        raise ValueError("Multivariate mode requires --covariates.")
    return required


def choose_prediction_column(
    forecast_df: pd.DataFrame,
    prediction_column: str | None,
    target_column: str = "target",
) -> Any:
    candidates = [
        prediction_column,
        "predictions",
        "prediction",
        "mean",
        "median",
        "0.5",
        0.5,
        f"{target_column}_0.5",
        f"{target_column}_p50",
        f"{target_column}_median",
    ]
    for candidate in candidates:
        if candidate is not None and candidate in forecast_df.columns:
            return candidate

    metadata_columns = {"id", "timestamp", "index"}
    numeric_columns = [
        column
        for column in forecast_df.columns
        if column not in metadata_columns and pd.api.types.is_numeric_dtype(forecast_df[column])
    ]
    if numeric_columns:
        return numeric_columns[0]
    raise ValueError(
        "Could not identify a prediction column. Pass --prediction-column explicitly."
    )


def _quantile_from_column_name(column: Any, target_column: str = "target") -> float | None:
    if not isinstance(column, str):
        try:
            value = float(column)
        except (TypeError, ValueError):
            return None
        return value if 0 < value < 1 else None

    label = column.strip().lower()
    target = target_column.strip().lower()
    for prefix in (f"{target}_", f"{target}-"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break

    probability_match = re.fullmatch(r"p[_-]?(\d+(?:\.\d+)?)", label)
    if probability_match:
        value = float(probability_match.group(1)) / 100.0
        return value if 0 < value < 1 else None

    quantile_match = re.fullmatch(
        r"(?:q|quantile)[_-]?(\d+(?:\.\d+)?)",
        label,
    )
    if quantile_match:
        value = float(quantile_match.group(1))
        if value > 1:
            value /= 100.0
        return value if 0 < value < 1 else None

    try:
        value = float(label)
    except ValueError:
        return None
    return value if 0 < value < 1 else None


def choose_quantile_column(
    forecast_df: pd.DataFrame,
    quantile: float,
    prediction_column: str | None = None,
    target_column: str = "target",
) -> Any:
    percentile = int(round(quantile * 100))
    candidates: list[Any] = [
        prediction_column if quantile == 0.5 else None,
        quantile,
        str(quantile),
        f"{quantile:g}",
        f"p{percentile}",
        f"P{percentile}",
        f"q{quantile:g}",
        f"quantile_{quantile:g}",
        f"{target_column}_{quantile:g}",
        f"{target_column}_p{percentile}",
        f"{target_column}_q{quantile:g}",
        f"{target_column}_quantile_{quantile:g}",
    ]
    for candidate in candidates:
        if candidate is not None and candidate in forecast_df.columns:
            return candidate

    for column in forecast_df.columns:
        parsed = _quantile_from_column_name(column, target_column=target_column)
        if parsed is not None and abs(parsed - quantile) < 1e-9:
            return column

    if quantile == 0.5:
        return choose_prediction_column(
            forecast_df,
            prediction_column=prediction_column,
            target_column=target_column,
        )
    raise ValueError(f"Could not identify the {quantile:g} quantile column in Chronos output.")


def normalize_forecast_df(forecast_df: pd.DataFrame) -> pd.DataFrame:
    normalized = forecast_df.reset_index()
    if "timestamp" not in normalized.columns and "index" in normalized.columns:
        normalized = normalized.rename(columns={"index": "timestamp"})
    if "timestamp" in normalized.columns:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
    return normalized


def select_forecast_row(
    forecast_df: pd.DataFrame,
    turbine_id: str,
    forecast_timestamp: pd.Timestamp,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    forecast = normalize_forecast_df(forecast_df)
    if "id" in forecast.columns:
        forecast = forecast[forecast["id"].astype(str) == str(turbine_id)]

    if "timestamp" in forecast.columns:
        matching_rows = forecast[forecast["timestamp"] == forecast_timestamp]
        if not matching_rows.empty:
            return forecast, matching_rows.iloc[0]
        forecast = forecast.sort_values("timestamp")

    if forecast.empty or len(forecast) < horizon:
        raise ValueError(f"Forecast output does not contain horizon {horizon}.")
    return forecast, forecast.iloc[horizon - 1]


def extract_prediction(
    forecast_df: pd.DataFrame,
    turbine_id: str,
    forecast_timestamp: pd.Timestamp,
    horizon: int,
    prediction_column: str | None,
) -> float:
    forecast, row = select_forecast_row(
        forecast_df,
        turbine_id=turbine_id,
        forecast_timestamp=forecast_timestamp,
        horizon=horizon,
    )
    column = choose_prediction_column(forecast, prediction_column)
    return float(row[column])


def extract_quantile_predictions(
    forecast_df: pd.DataFrame,
    turbine_id: str,
    forecast_timestamp: pd.Timestamp,
    horizon: int,
    prediction_column: str | None = None,
) -> dict[str, float]:
    """Extract the required Chronos quantiles across supported column naming styles."""
    forecast, row = select_forecast_row(
        forecast_df,
        turbine_id=turbine_id,
        forecast_timestamp=forecast_timestamp,
        horizon=horizon,
    )
    values: dict[str, float] = {}
    for name, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        column = choose_quantile_column(
            forecast,
            quantile=quantile,
            prediction_column=prediction_column,
        )
        values[name] = float(row[column])
    return values


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


def build_context(
    group: pd.DataFrame,
    cutoff_pos: int,
    context_length: int,
    columns: list[str],
) -> pd.DataFrame:
    start_pos = max(0, cutoff_pos - context_length + 1)
    context = group.iloc[start_pos : cutoff_pos + 1][columns].copy()
    return context.dropna(subset=columns)


def run_rolling_forecasts(
    pipeline: Any,
    df: pd.DataFrame,
    mode: str,
    covariates: list[str],
    horizons: list[int],
    context_length: int,
    stride: int,
    quantile_levels: list[float],
    model_id: str,
    prediction_column: str | None = None,
    limit_turbines: int | None = None,
    max_windows_per_turbine: int | None = None,
    allow_future_covariates: bool = False,
    test_start: pd.Timestamp | str | None = None,
    test_end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if context_length <= 0:
        raise ValueError("--context-length must be positive.")
    if stride <= 0:
        raise ValueError("--stride must be positive.")

    context_columns = validate_input_columns(df, mode, covariates)
    max_horizon = max(horizons)
    data = df.copy()
    data["id"] = data["id"].astype(str)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["id", "timestamp"]).reset_index(drop=True)

    if (test_start is None) != (test_end is None):
        raise ValueError("test_start and test_end must be provided together.")
    if test_start is None:
        manifest = ensure_split_manifest(data)
        test_start, test_end = test_period(manifest)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    required_quantiles = sorted(set([*quantile_levels, *DEFAULT_QUANTILES]))

    if limit_turbines is not None:
        turbine_ids = data["id"].drop_duplicates().head(limit_turbines)
        data = data[data["id"].isin(turbine_ids)]

    records: list[dict[str, object]] = []
    for turbine_id, group in data.groupby("id", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < context_length + max_horizon:
            continue

        eligible_cutoffs: list[int] = []
        for cutoff_pos in range(context_length - 1, len(group) - max_horizon):
            first_forecast_timestamp = pd.Timestamp(group.iloc[cutoff_pos + 1]["timestamp"])
            last_forecast_timestamp = pd.Timestamp(group.iloc[cutoff_pos + max_horizon]["timestamp"])
            if first_forecast_timestamp >= test_start and last_forecast_timestamp <= test_end:
                eligible_cutoffs.append(cutoff_pos)

        windows_done = 0
        for cutoff_pos in eligible_cutoffs[::stride]:
            context_df = build_context(group, cutoff_pos, context_length, context_columns)
            if len(context_df) < context_length:
                continue

            future_df = None
            if mode == "multivariate" and allow_future_covariates:
                future_rows = group.iloc[cutoff_pos + 1 : cutoff_pos + 1 + max_horizon]
                future_df = future_rows[["id", "timestamp", *covariates]].copy()

            forecast_df = pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=max_horizon,
                quantile_levels=required_quantiles,
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

            cutoff_timestamp = group.iloc[cutoff_pos]["timestamp"]
            for horizon in horizons:
                actual_row = group.iloc[cutoff_pos + horizon]
                forecast_timestamp = pd.Timestamp(actual_row["timestamp"])
                quantiles = extract_quantile_predictions(
                    forecast_df,
                    turbine_id=str(turbine_id),
                    forecast_timestamp=forecast_timestamp,
                    horizon=horizon,
                    prediction_column=prediction_column,
                )
                record: dict[str, object] = {
                    "id": str(turbine_id),
                    "mode": mode,
                    "horizon": int(horizon),
                    "cutoff_timestamp": cutoff_timestamp,
                    "timestamp": forecast_timestamp,
                    "y_true": float(actual_row["target"]),
                    "is_imputed_target": _as_bool(actual_row.get("is_imputed_target", False)),
                    "p10": quantiles["p10"],
                    "p50": quantiles["p50"],
                    "p90": quantiles["p90"],
                    "y_pred": quantiles["p50"],
                    "test_start": test_start,
                    "test_end": test_end,
                    "model_id": model_id,
                    "used_future_covariates": bool(allow_future_covariates),
                }
                if "rated_capacity_kw" in group.columns:
                    record["rated_capacity_kw"] = actual_row["rated_capacity_kw"]
                records.append(record)

            windows_done += 1
            if max_windows_per_turbine is not None and windows_done >= max_windows_per_turbine:
                break

    if not records:
        raise RuntimeError(
            "No prediction windows were generated. Check context length, horizons, stride, and data size."
        )
    return pd.DataFrame(records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=Path("data/processed/sdwpf_hourly_regularized.parquet"),
        type=Path,
        help="Processed SDWPF parquet/CSV path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Prediction output CSV/parquet path. Defaults to results/zero_shot/predictions_<mode>.csv.",
    )
    parser.add_argument("--model-id", "--model_id", dest="model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP)
    parser.add_argument("--mode", choices=["univariate", "multivariate"], required=True)
    parser.add_argument(
        "--covariates",
        default="",
        help="Comma-separated covariates used only in multivariate mode.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Forecast horizons. Supports '--horizons 1 6 24 72' or '--horizons 1,6,24,72'.",
    )
    parser.add_argument("--context-length", default=168, type=int)
    parser.add_argument("--stride", default=24, type=int)
    parser.add_argument("--quantiles", default="0.1,0.5,0.9")
    parser.add_argument("--prediction-column", default=None)
    parser.add_argument(
        "--benchmark-config",
        default=DEFAULT_BENCHMARK_CONFIG,
        type=Path,
        help="Reusable benchmark configuration defining the chronological split.",
    )
    parser.add_argument(
        "--split-manifest",
        default=DEFAULT_SPLIT_MANIFEST,
        type=Path,
        help="Persisted exact split boundaries shared across benchmark runs.",
    )
    parser.add_argument(
        "--max-turbines",
        "--max_turbines",
        "--limit-turbines",
        dest="max_turbines",
        default=None,
        type=int,
        help="Limit turbine count for smoke tests.",
    )
    parser.add_argument("--max-windows-per-turbine", default=None, type=int)
    parser.add_argument(
        "--allow-future-covariates",
        action="store_true",
        help="Use measured future covariates. Off by default to avoid SDWPF leakage.",
    )
    return parser


def write_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        predictions.to_parquet(output_path, index=False)
    else:
        predictions.to_csv(output_path, index=False)


def main() -> None:
    args = build_arg_parser().parse_args()
    data = read_table(args.input)
    split_manifest = ensure_split_manifest(
        data,
        config_path=args.benchmark_config,
        manifest_path=args.split_manifest,
    )
    test_start, test_end = test_period(split_manifest)
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model_id,
        device_map=args.device_map,
    )
    output_path = args.output or default_prediction_path(args.mode)
    predictions = run_rolling_forecasts(
        pipeline=pipeline,
        df=data,
        mode=args.mode,
        covariates=parse_csv_list(args.covariates),
        horizons=parse_int_list(args.horizons),
        context_length=args.context_length,
        stride=args.stride,
        quantile_levels=parse_float_list(args.quantiles),
        model_id=args.model_id,
        prediction_column=args.prediction_column,
        limit_turbines=args.max_turbines,
        max_windows_per_turbine=args.max_windows_per_turbine,
        allow_future_covariates=args.allow_future_covariates,
        test_start=test_start,
        test_end=test_end,
    )
    write_predictions(predictions, output_path)
    print(f"Wrote {len(predictions):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
