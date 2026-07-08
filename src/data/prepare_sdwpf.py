"""Prepare raw SDWPF CSV data for Chronos-2 zero-shot forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_COVARIATES = [
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
]


def parse_csv_list(value: str | None) -> list[str]:
    """Parse a comma-separated CLI value into a clean list."""
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column and column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _parse_time_offsets(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_timedelta(values, unit="m")

    text = values.astype(str).str.strip()
    has_hours_minutes = text.str.match(r"^\d{1,2}:\d{2}$", na=False)
    text = text.mask(has_hours_minutes, text + ":00")
    return pd.to_timedelta(text)


def build_timestamp(
    df: pd.DataFrame,
    timestamp_column: str | None,
    day_column: str | None,
    time_column: str | None,
    timestamp_origin: str,
) -> pd.Series:
    """Build a timestamp series from either one datetime column or Day/Tmstamp columns."""
    if timestamp_column:
        _validate_columns(df, [timestamp_column])
        return pd.to_datetime(df[timestamp_column], errors="raise")

    if not day_column or not time_column:
        raise ValueError(
            "Provide --timestamp-column, or provide both --day-column and --time-column."
        )

    _validate_columns(df, [day_column, time_column])
    day_values = pd.to_numeric(df[day_column], errors="raise")
    day_zero = 1 if day_values.min() >= 1 else 0
    day_offsets = pd.to_timedelta(day_values - day_zero, unit="D")
    time_offsets = _parse_time_offsets(df[time_column])
    return pd.to_datetime(timestamp_origin) + day_offsets + time_offsets


def resample_forecasting_table(
    table: pd.DataFrame,
    covariates: list[str],
    freq: str,
    regularize_hourly: bool = False,
) -> pd.DataFrame:
    """Resample each turbine independently and average numeric values."""
    value_columns = ["target", *covariates]
    resample_input = table[["id", "timestamp", *value_columns]].copy()
    resample_input["timestamp"] = pd.to_datetime(resample_input["timestamp"])

    for column in value_columns:
        resample_input[column] = pd.to_numeric(resample_input[column], errors="coerce")

    resample_input = resample_input.dropna(subset=["id", "timestamp"])
    resampled = (
        resample_input.set_index("timestamp")
        .groupby("id")[value_columns]
        .resample(freq)
        .mean()
        .reset_index()
    )

    if regularize_hourly:
        return regularize_hourly_table(resampled, covariates)

    resampled = resampled.dropna(subset=["target"])
    resampled = resampled.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return resampled[["id", "timestamp", *value_columns]]


def regularize_hourly_table(resampled: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Create a complete hourly grid per turbine and fill numeric gaps."""
    value_columns = ["target", *covariates]
    output_columns = ["id", "timestamp", *value_columns, "is_imputed_target"]
    if resampled.empty:
        empty = resampled.copy()
        empty["is_imputed_target"] = pd.Series(dtype=bool)
        return empty.reindex(columns=output_columns)

    regularize_input = resampled[["id", "timestamp", *value_columns]].copy()
    regularize_input["id"] = regularize_input["id"].astype(str)
    regularize_input["timestamp"] = pd.to_datetime(regularize_input["timestamp"])

    full_grid = pd.date_range(
        regularize_input["timestamp"].min(),
        regularize_input["timestamp"].max(),
        freq="1h",
    )

    frames: list[pd.DataFrame] = []
    for turbine_id, group in regularize_input.groupby("id", sort=True):
        turbine_frame = (
            group.drop_duplicates(subset=["timestamp"], keep="last")
            .set_index("timestamp")
            .sort_index()
            .reindex(full_grid)
        )
        turbine_frame.index.name = "timestamp"
        turbine_frame["id"] = str(turbine_id)

        for column in value_columns:
            turbine_frame[column] = pd.to_numeric(turbine_frame[column], errors="coerce")

        turbine_frame["is_imputed_target"] = turbine_frame["target"].isna()
        turbine_frame[value_columns] = (
            turbine_frame[value_columns].interpolate(method="linear").ffill().bfill()
        )
        frames.append(turbine_frame.reset_index())

    regularized = pd.concat(frames, ignore_index=True)
    regularized = regularized.sort_values(["id", "timestamp"]).reset_index(drop=True)
    return regularized[output_columns]


def prepare_sdwpf_dataframe(
    raw_df: pd.DataFrame,
    id_column: str = "TurbID",
    target_column: str = "Patv",
    timestamp_column: str | None = None,
    day_column: str | None = "Day",
    time_column: str | None = "Tmstamp",
    timestamp_origin: str = "2020-01-01",
    covariates: list[str] | None = None,
    freq: str = "1h",
    regularize_hourly: bool = False,
) -> pd.DataFrame:
    """Convert raw SDWPF columns into id/timestamp/target plus selected covariates."""
    selected_covariates = list(dict.fromkeys(covariates or []))
    reserved_names = {"id", "timestamp", "target"}
    conflicts = [column for column in selected_covariates if column in reserved_names]
    if conflicts:
        raise ValueError(f"Covariate names conflict with output columns: {conflicts}")

    required_columns = [id_column, target_column, *selected_covariates]
    if timestamp_column:
        required_columns.append(timestamp_column)
    else:
        required_columns.extend([day_column or "", time_column or ""])
    _validate_columns(raw_df, required_columns)

    prepared = pd.DataFrame(
        {
            "id": raw_df[id_column].astype(str),
            "timestamp": build_timestamp(
                raw_df,
                timestamp_column=timestamp_column,
                day_column=day_column,
                time_column=time_column,
                timestamp_origin=timestamp_origin,
            ),
            "target": raw_df[target_column],
        }
    )
    for covariate in selected_covariates:
        prepared[covariate] = raw_df[covariate]

    return resample_forecasting_table(
        prepared,
        selected_covariates,
        freq=freq,
        regularize_hourly=regularize_hourly,
    )


def write_table(df: pd.DataFrame, output_path: Path) -> None:
    raw_dir = Path("data/raw").resolve()
    resolved_output = output_path.resolve()
    if resolved_output == raw_dir or raw_dir in resolved_output.parents:
        raise ValueError("Refusing to write processed output under data/raw/.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return
    df.to_parquet(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw SDWPF CSV path.")
    parser.add_argument(
        "--output",
        default=Path("data/processed/sdwpf_hourly.parquet"),
        type=Path,
        help="Processed parquet or CSV output path.",
    )
    parser.add_argument("--id-column", default="TurbID")
    parser.add_argument("--target-column", default="Patv")
    parser.add_argument(
        "--timestamp-column",
        default=None,
        help="Single datetime column. If omitted, --day-column and --time-column are used.",
    )
    parser.add_argument("--day-column", default="Day")
    parser.add_argument("--time-column", default="Tmstamp")
    parser.add_argument(
        "--timestamp-origin",
        default="2020-01-01",
        help="Origin date used when building timestamps from day/time columns.",
    )
    parser.add_argument(
        "--covariates",
        default=",".join(DEFAULT_COVARIATES),
        help="Comma-separated covariate columns to preserve and average.",
    )
    parser.add_argument("--freq", default="1h", help="Pandas resampling frequency.")
    parser.add_argument(
        "--regularize-hourly",
        action="store_true",
        help="Fill every turbine onto a complete global hourly timestamp grid for Chronos-2.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_df = pd.read_csv(args.input)
    processed = prepare_sdwpf_dataframe(
        raw_df,
        id_column=args.id_column,
        target_column=args.target_column,
        timestamp_column=args.timestamp_column,
        day_column=args.day_column,
        time_column=args.time_column,
        timestamp_origin=args.timestamp_origin,
        covariates=parse_csv_list(args.covariates),
        freq=args.freq,
        regularize_hourly=args.regularize_hourly,
    )
    write_table(processed, args.output)
    print(f"Wrote {len(processed):,} rows to {args.output}")


if __name__ == "__main__":
    main()
