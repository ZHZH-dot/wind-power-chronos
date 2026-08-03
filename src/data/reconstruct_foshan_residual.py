"""Build the provisional signed residual-load target from the Foshan workbooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.reconstruct_foshan_provisional_load import (
    ALIGNMENT_METHOD,
    CAUSAL_FILL_LIMIT,
    PCS_SIGN_CONVENTION,
    TIMEZONE,
    ProvisionalLoadSignals,
    _causally_filled_signal,
    load_provisional_load_signals,
)


TARGET_COLUMN = "signed_residual_load_kw"
TARGET_LABEL = "provisional reconstructed signed residual-load proxy"
RESIDUAL_FORMULA = "net_grid_kw + pcs_kw"
FREQUENCY = "15min"
FIVE_MINUTE_FREQUENCY = "5min"
CALENDAR_COLUMNS = [
    "quarter_hour_sin",
    "quarter_hour_cos",
    "hour_sin",
    "hour_cos",
    "weekday",
    "is_weekend",
    "month",
    "tariff_is_peak",
    "tariff_is_shoulder",
    "tariff_is_valley",
]
TARIFF_PERIODS = ("peak", "shoulder", "valley")


def _site_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TIMEZONE)
    return timestamp.tz_convert(TIMEZONE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tariff_clock_profile(dispatch_path: Path) -> dict[int, str]:
    """Return a canonical quarter-hour tariff profile from 5- or 15-minute data."""
    table = pd.read_csv(dispatch_path, low_memory=False)
    timestamp_column = "timestamp" if "timestamp" in table else str(table.columns[0])
    if "price" not in table:
        raise ValueError(f"Tariff input {dispatch_path} has no price column.")
    timestamps = pd.to_datetime(table[timestamp_column], errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Tariff timestamps must be unique.")
    if (
        timestamps.dt.minute.mod(5).ne(0).any()
        or timestamps.dt.second.ne(0).any()
        or timestamps.dt.microsecond.ne(0).any()
    ):
        raise ValueError("Tariff timestamps must be aligned to five-minute positions.")
    minute = timestamps.dt.hour * 60 + timestamps.dt.minute
    clock = pd.DataFrame(
        {
            "date": timestamps.dt.date,
            "minute_of_day": minute,
            "price": pd.to_numeric(table["price"], errors="raise"),
        }
    )
    five_minute_positions = set(range(0, 24 * 60, 5))
    quarter_hour_positions = set(range(0, 24 * 60, 15))
    actual_positions = set(int(value) for value in clock["minute_of_day"].unique())
    if actual_positions == five_minute_positions:
        expected_positions = five_minute_positions
        cadence_minutes = 5
    elif actual_positions == quarter_hour_positions:
        expected_positions = quarter_hour_positions
        cadence_minutes = 15
    else:
        raise ValueError(
            "Tariff profile must be a complete deterministic 5- or 15-minute schedule; "
            f"found {len(actual_positions)} clock positions."
        )
    for date, daily in clock.groupby("date", sort=True):
        daily_positions = set(int(value) for value in daily["minute_of_day"])
        if daily_positions != expected_positions:
            raise ValueError(
                "Tariff input contains incomplete or mixed timestamp frequencies on "
                f"{date}: expected {len(expected_positions)} positions, "
                f"found {len(daily_positions)}."
            )
    grouped = clock.groupby("minute_of_day", sort=True)["price"]
    if grouped.nunique().gt(1).any():
        raise ValueError("Tariff price changes across days for the same clock interval.")
    prices = grouped.first()
    if cadence_minutes == 5:
        five_minute_prices = prices.rename_axis("minute_of_day").reset_index()
        five_minute_prices["quarter_hour_start"] = (
            five_minute_prices["minute_of_day"] // 15 * 15
        )
        quarter_hours = five_minute_prices.groupby("quarter_hour_start", sort=True)[
            "price"
        ]
        if quarter_hours.nunique().gt(1).any():
            raise ValueError("Tariff price changes inside a 15-minute block.")
        prices = quarter_hours.first()
    unique = sorted(float(value) for value in prices.unique())
    if len(unique) < 2:
        raise ValueError("At least two tariff levels are required.")
    labels: dict[float, str] = {unique[0]: "valley", unique[-1]: "peak"}
    for value in unique[1:-1]:
        labels[value] = "shoulder"
    return validate_tariff_clock_profile(
        {int(key): labels[float(value)] for key, value in prices.items()}
    )


def validate_tariff_clock_profile(
    tariff_profile: Mapping[int, str],
) -> dict[int, str]:
    """Validate one deterministic tariff label for every quarter-hour clock slot."""
    normalized = {int(minute): str(period) for minute, period in tariff_profile.items()}
    expected_minutes = set(range(0, 24 * 60, 15))
    actual_minutes = set(normalized)
    if actual_minutes != expected_minutes:
        missing = sorted(expected_minutes - actual_minutes)
        unexpected = sorted(actual_minutes - expected_minutes)
        raise ValueError(
            "Tariff clock profile must cover all 96 quarter-hour positions exactly once; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}."
        )
    invalid = sorted(set(normalized.values()) - set(TARIFF_PERIODS))
    if invalid:
        raise ValueError(f"Tariff clock profile contains invalid periods: {invalid}")
    return normalized


def tariff_clock_profile_from_calendar(table: pd.DataFrame) -> dict[int, str]:
    """Recover the audited minute-of-day tariff schedule from calendar indicators."""
    indicator_columns = [f"tariff_is_{period}" for period in TARIFF_PERIODS]
    missing = sorted({"timestamp", *indicator_columns} - set(table.columns))
    if missing:
        raise ValueError(f"Calendar table is missing tariff fields: {missing}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(table["timestamp"], errors="raise"))
    indicators = table[indicator_columns].apply(pd.to_numeric, errors="raise")
    if indicators.isna().any().any() or not indicators.isin([0, 1]).all().all():
        raise ValueError("Tariff indicators must be non-null binary values.")
    if not indicators.sum(axis=1).eq(1).all():
        raise ValueError("Every timestamp must map to exactly one tariff period.")
    clock = pd.DataFrame(
        {
            "minute_of_day": timestamps.hour * 60 + timestamps.minute,
            "period": indicators.idxmax(axis=1).str.removeprefix("tariff_is_"),
        }
    )
    grouped = clock.groupby("minute_of_day", sort=True)["period"]
    if grouped.nunique().gt(1).any():
        raise ValueError("Tariff period changes across days for the same clock interval.")
    return validate_tariff_clock_profile(grouped.first().to_dict())


def calendar_covariates_for_timestamps(
    timestamps: pd.Series | pd.DatetimeIndex,
    tariff_profile: Mapping[int, str],
) -> pd.DataFrame:
    """Generate deterministic known-future covariates from quarter-hour timestamps."""
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    if (
        (index.minute % 15 != 0).any()
        or (index.second != 0).any()
        or (index.microsecond != 0).any()
    ):
        raise ValueError("Calendar timestamps must be aligned to quarter hours.")
    profile = validate_tariff_clock_profile(tariff_profile)
    quarter_hour = index.hour * 4 + index.minute // 15
    hour = index.hour + index.minute / 60.0
    minute_of_day = index.hour * 60 + index.minute
    periods = pd.Series(minute_of_day).map(profile)
    if periods.isna().any():
        missing = sorted(set(minute_of_day[periods.isna()]))
        raise ValueError(f"Tariff profile is missing clock minutes: {missing[:5]}")
    result = pd.DataFrame({"timestamp": index})
    result["quarter_hour_sin"] = np.sin(2.0 * np.pi * quarter_hour / 96.0)
    result["quarter_hour_cos"] = np.cos(2.0 * np.pi * quarter_hour / 96.0)
    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["weekday"] = index.dayofweek.astype(np.int8)
    result["is_weekend"] = (index.dayofweek >= 5).astype(np.int8)
    result["month"] = index.month.astype(np.int8)
    for period in TARIFF_PERIODS:
        result[f"tariff_is_{period}"] = periods.eq(period).astype(np.int8)
    return result


def add_residual_calendar_covariates(
    table: pd.DataFrame,
    tariff_profile: Mapping[int, str],
    external_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add known-future calendar features without using realized site variables."""
    result = table.copy()
    calendar = calendar_covariates_for_timestamps(result["timestamp"], tariff_profile)
    for column in CALENDAR_COLUMNS:
        result[column] = calendar[column].to_numpy()

    if external_calendar is not None:
        allowed = {"timestamp", "is_factory_running", "is_holiday"}
        unexpected = sorted(set(external_calendar.columns) - allowed)
        if unexpected:
            raise ValueError(
                "External calendar may contain only known-future calendar fields: "
                f"{unexpected}"
            )
        calendar = external_calendar.copy()
        calendar["timestamp"] = pd.to_datetime(calendar["timestamp"], errors="raise")
        result = result.merge(calendar, on="timestamp", how="left", validate="one_to_one")
        supplied = [column for column in allowed - {"timestamp"} if column in result]
        if supplied and result[supplied].isna().any().any():
            raise ValueError("External calendar does not cover every target timestamp.")
    return result


def reconstruct_signed_residual(
    signals: ProvisionalLoadSignals,
    start: pd.Timestamp | str,
    end_exclusive: pd.Timestamp | str,
    *,
    causal_fill_limit: int = CAUSAL_FILL_LIMIT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct signed net demand at five-minute resolution without clipping."""
    start_ts = _site_timestamp(start)
    end_ts = _site_timestamp(end_exclusive)
    if end_ts <= start_ts:
        raise ValueError("end_exclusive must be after start.")

    timestamps = pd.date_range(
        start_ts, end_ts, freq=FIVE_MINUTE_FREQUENCY, inclusive="left"
    )
    quarter_hours = pd.date_range(
        start_ts, end_ts, freq=FREQUENCY, inclusive="left"
    )
    net_grid = _causally_filled_signal(
        signals.net_grid.frame, "net_grid_kw_raw", causal_fill_limit
    ).set_index("interval_timestamp")
    net_period = net_grid.reindex(quarter_hours)
    interval_source = timestamps.floor(FREQUENCY)
    net_expanded = net_period.reindex(interval_source)
    pcs = signals.pcs.frame.set_index("timestamp")["pcs_kw_raw"].reindex(timestamps)

    table = pd.DataFrame(
        {
            "timestamp": timestamps,
            "net_grid_interval_timestamp": interval_source,
            "net_grid_source_timestamp": net_expanded[
                "effective_source_timestamp"
            ].to_numpy(),
            "net_grid_kw_raw": net_expanded["value_raw"].to_numpy(dtype=float),
            "net_grid_kw": net_expanded["value"].to_numpy(dtype=float),
            "net_grid_was_causally_filled": net_expanded[
                "was_causally_filled"
            ].to_numpy(dtype=bool),
            "pcs_source_timestamp": pd.Series(timestamps).where(
                pcs.notna().to_numpy()
            ),
            "pcs_kw": pcs.to_numpy(dtype=float),
        }
    )
    table[TARGET_COLUMN] = table["net_grid_kw"] + table["pcs_kw"]
    table["signed_residual_is_negative"] = table[TARGET_COLUMN] < 0.0
    table["site_source_filename"] = signals.site_workbook.name
    table["storage_source_filename"] = signals.storage_workbook.name
    table["alignment_method"] = ALIGNMENT_METHOD
    table["residual_formula"] = RESIDUAL_FORMULA
    table["pcs_sign_convention"] = PCS_SIGN_CONVENTION
    table["target_label"] = TARGET_LABEL

    algebra_error = (
        table[TARGET_COLUMN] - (table["net_grid_kw"] + table["pcs_kw"])
    ).abs()
    finite_error = algebra_error.dropna()
    maximum_algebra_error = float(finite_error.max()) if not finite_error.empty else None
    if maximum_algebra_error is not None and maximum_algebra_error > 1e-12:
        raise AssertionError("Signed residual algebra validation failed.")
    audit: dict[str, Any] = {
        "target_label": TARGET_LABEL,
        "verified_gross_factory_load": False,
        "start": start_ts.isoformat(),
        "end_exclusive": end_ts.isoformat(),
        "timezone": TIMEZONE,
        "frequency": FIVE_MINUTE_FREQUENCY,
        "net_grid_source_rows": int(net_period["value"].notna().sum()),
        "pcs_source_rows": int(pcs.notna().sum()),
        "final_rows": len(table),
        "missing_net_grid_rows": int(table["net_grid_kw"].isna().sum()),
        "missing_pcs_rows": int(table["pcs_kw"].isna().sum()),
        "missing_target_rows": int(table[TARGET_COLUMN].isna().sum()),
        "negative_target_rows": int(table["signed_residual_is_negative"].sum()),
        "signed_residual_energy_kwh": float(table[TARGET_COLUMN].sum() / 12.0),
        "maximum_algebra_error_kw": maximum_algebra_error,
        "causal_fill_limit_quarter_hours": causal_fill_limit,
        "alignment_method": ALIGNMENT_METHOD,
        "formula": RESIDUAL_FORMULA,
        "pcs_sign_convention": PCS_SIGN_CONVENTION,
        "site_source_workbook": str(signals.site_workbook.resolve()),
        "site_source_sha256": sha256_file(signals.site_workbook),
        "storage_source_workbook": str(signals.storage_workbook.resolve()),
        "storage_source_sha256": sha256_file(signals.storage_workbook),
    }
    return table, audit


def aggregate_signed_residual_15min(five_minute: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every three aligned five-minute residuals by arithmetic mean."""
    required = {"timestamp", TARGET_COLUMN, "net_grid_kw", "pcs_kw"}
    missing = sorted(required - set(five_minute.columns))
    if missing:
        raise ValueError(f"Five-minute residual table is missing columns: {missing}")
    table = five_minute.sort_values("timestamp").copy()
    timestamps = pd.to_datetime(table["timestamp"], errors="raise")
    if timestamps.duplicated().any() or not timestamps.dt.minute.mod(5).eq(0).all():
        raise ValueError("Residual timestamps must be unique and five-minute aligned.")
    table["timestamp"] = timestamps
    table["quarter_hour"] = timestamps.dt.floor(FREQUENCY)
    grouped = table.groupby("quarter_hour", sort=True)
    result = grouped.agg(
        signed_residual_load_kw=(TARGET_COLUMN, "mean"),
        net_grid_kw=("net_grid_kw", "mean"),
        pcs_kw=("pcs_kw", "mean"),
        five_minute_row_count=("timestamp", "size"),
        observed_five_minute_target_count=(TARGET_COLUMN, "count"),
        first_source_timestamp=("timestamp", "min"),
        last_source_timestamp=("timestamp", "max"),
    ).reset_index(names="timestamp")
    bad = result.loc[~result["five_minute_row_count"].eq(3)]
    if not bad.empty:
        raise ValueError(
            "Every quarter hour must contain exactly three five-minute rows: "
            f"{bad.head().to_dict(orient='records')}"
        )
    result["is_missing_target"] = result[TARGET_COLUMN].isna()
    result["is_negative_target"] = result[TARGET_COLUMN] < 0.0
    result["target_label"] = TARGET_LABEL
    result["residual_formula"] = RESIDUAL_FORMULA
    result["pcs_sign_convention"] = PCS_SIGN_CONVENTION
    return result


def prepare_residual_dataset(
    site_workbook: Path,
    storage_workbook: Path,
    dispatch_path: Path,
    start: pd.Timestamp | str,
    end_exclusive: pd.Timestamp | str,
    *,
    causal_fill_limit: int = CAUSAL_FILL_LIMIT,
    external_calendar_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    signals = load_provisional_load_signals(site_workbook, storage_workbook)
    five_minute, audit = reconstruct_signed_residual(
        signals,
        start,
        end_exclusive,
        causal_fill_limit=causal_fill_limit,
    )
    fifteen_minute = aggregate_signed_residual_15min(five_minute)
    external = pd.read_csv(external_calendar_path) if external_calendar_path else None
    tariff_profile = load_tariff_clock_profile(dispatch_path)
    fifteen_minute = add_residual_calendar_covariates(
        fifteen_minute, tariff_profile, external
    )
    audit["aggregation"] = {
        "method": "left-labelled arithmetic mean of each three five-minute rows",
        "interpolation": False,
        "fifteen_minute_rows": len(fifteen_minute),
        "missing_targets": int(fifteen_minute[TARGET_COLUMN].isna().sum()),
        "negative_targets": int((fifteen_minute[TARGET_COLUMN] < 0.0).sum()),
        "negative_values_preserved": True,
    }
    audit["known_future_calendar_columns"] = [
        *CALENDAR_COLUMNS,
        *(
            [column for column in ("is_factory_running", "is_holiday") if column in fifteen_minute]
        ),
    ]
    audit["external_calendar_enabled"] = external_calendar_path is not None
    audit["dispatch_tariff_source"] = str(dispatch_path.resolve())
    audit["dispatch_tariff_source_sha256"] = sha256_file(dispatch_path)
    return five_minute, fifteen_minute, audit


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-workbook", required=True, type=Path)
    parser.add_argument("--storage-workbook", required=True, type=Path)
    parser.add_argument("--dispatch-input", required=True, type=Path)
    parser.add_argument("--start", default="2026-03-01T00:00:00+08:00")
    parser.add_argument("--end-exclusive", default="2026-06-01T00:00:00+08:00")
    parser.add_argument("--causal-fill-limit", default=CAUSAL_FILL_LIMIT, type=int)
    parser.add_argument("--external-calendar", default=None, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("results/residual_forecast/foshan_chronos2/data"),
        type=Path,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    five, fifteen, audit = prepare_residual_dataset(
        args.site_workbook,
        args.storage_workbook,
        args.dispatch_input,
        args.start,
        args.end_exclusive,
        causal_fill_limit=args.causal_fill_limit,
        external_calendar_path=args.external_calendar,
    )
    five.to_parquet(args.output_dir / "signed_residual_5min.parquet", index=False)
    fifteen.to_parquet(args.output_dir / "signed_residual_15min.parquet", index=False)
    _write_json(args.output_dir / "residual_data_audit.json", audit)
    print(
        f"Wrote {len(five):,} five-minute and {len(fifteen):,} quarter-hour "
        f"signed residual rows to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
