"""Reconstruct the provisional Foshan load proxy from the raw workbooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.prepare_foshan import ParsedSignal, read_signal_sheet


TIMEZONE = "Asia/Shanghai"
PV_CAPACITY_KW = 1700.0
CAUSAL_FILL_LIMIT = 3
ALIGNMENT_METHOD = (
    "15-minute left-label forward-fill to t, t+5min, and t+10min; "
    "no interpolation"
)
PROXY_FORMULA = "net_grid_kw + pv_kw + pcs_kw"
PCS_SIGN_CONVENTION = "positive=discharge, negative=charge"


@dataclass(frozen=True)
class ProvisionalLoadSignals:
    """Parsed native-frequency signals and their source paths."""

    pv: ParsedSignal
    net_grid: ParsedSignal
    pcs: ParsedSignal
    site_workbook: Path
    storage_workbook: Path


def _site_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TIMEZONE)
    return timestamp.tz_convert(TIMEZONE)


def load_provisional_load_signals(
    site_workbook: Path,
    storage_workbook: Path,
) -> ProvisionalLoadSignals:
    """Read PV, signed net-grid, and PCS signals from the source workbooks."""
    return ProvisionalLoadSignals(
        pv=read_signal_sheet(site_workbook, 0, "pv_kw", "15min"),
        net_grid=read_signal_sheet(
            site_workbook, 1, "net_grid_kw", "15min"
        ),
        pcs=read_signal_sheet(storage_workbook, 0, "pcs_kw", "5min"),
        site_workbook=site_workbook,
        storage_workbook=storage_workbook,
    )


def _causally_filled_signal(
    frame: pd.DataFrame,
    value_column: str,
    limit: int,
) -> pd.DataFrame:
    if limit < 0:
        raise ValueError("Causal fill limit must be nonnegative.")
    table = frame[["timestamp", value_column]].copy().set_index("timestamp")
    raw = table[value_column]
    if limit:
        filled = raw.ffill(limit=limit)
        source_timestamp = pd.Series(
            np.where(raw.notna(), raw.index, pd.NaT),
            index=raw.index,
            dtype="datetime64[ns, Asia/Shanghai]",
        ).ffill(limit=limit)
    else:
        filled = raw.copy()
        source_timestamp = pd.Series(raw.index, index=raw.index).where(raw.notna())
    return pd.DataFrame(
        {
            "interval_timestamp": raw.index,
            "value_raw": raw.to_numpy(dtype=float),
            "value": filled.to_numpy(dtype=float),
            "effective_source_timestamp": source_timestamp.to_numpy(),
            "was_causally_filled": (raw.isna() & filled.notna()).to_numpy(),
        }
    )


def reconstruct_provisional_load(
    signals: ProvisionalLoadSignals,
    start: pd.Timestamp | str,
    end_exclusive: pd.Timestamp | str,
    *,
    causal_fill_limit: int = CAUSAL_FILL_LIMIT,
    pv_capacity_kw: float = PV_CAPACITY_KW,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct a five-minute provisional-load proxy without interpolation."""
    start_ts = _site_timestamp(start)
    end_ts = _site_timestamp(end_exclusive)
    if end_ts <= start_ts:
        raise ValueError("end_exclusive must be after start.")
    if pv_capacity_kw <= 0:
        raise ValueError("pv_capacity_kw must be positive.")

    timestamps = pd.date_range(
        start_ts, end_ts, freq="5min", inclusive="left"
    )
    quarter_hour_timestamps = pd.date_range(
        start_ts, end_ts, freq="15min", inclusive="left"
    )
    pv = _causally_filled_signal(
        signals.pv.frame, "pv_kw_raw", causal_fill_limit
    ).set_index("interval_timestamp")
    net_grid = _causally_filled_signal(
        signals.net_grid.frame, "net_grid_kw_raw", causal_fill_limit
    ).set_index("interval_timestamp")
    pcs = signals.pcs.frame.set_index("timestamp")["pcs_kw_raw"]

    pv_period = pv.reindex(quarter_hour_timestamps)
    net_grid_period = net_grid.reindex(quarter_hour_timestamps)
    quarter_hour_source = timestamps.floor("15min")
    pv_expanded = pv_period.reindex(quarter_hour_source)
    net_grid_expanded = net_grid_period.reindex(quarter_hour_source)
    pcs_period = pcs.reindex(timestamps)

    table = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pv_interval_timestamp": quarter_hour_source,
            "pv_source_timestamp": pv_expanded[
                "effective_source_timestamp"
            ].to_numpy(),
            "pv_kw_raw": pv_expanded["value_raw"].to_numpy(dtype=float),
            "pv_kw_filled": pv_expanded["value"].to_numpy(dtype=float),
            "pv_was_causally_filled": pv_expanded[
                "was_causally_filled"
            ].to_numpy(dtype=bool),
            "net_grid_interval_timestamp": quarter_hour_source,
            "net_grid_source_timestamp": net_grid_expanded[
                "effective_source_timestamp"
            ].to_numpy(),
            "net_grid_kw_raw": net_grid_expanded["value_raw"].to_numpy(
                dtype=float
            ),
            "net_grid_kw": net_grid_expanded["value"].to_numpy(dtype=float),
            "net_grid_was_causally_filled": net_grid_expanded[
                "was_causally_filled"
            ].to_numpy(dtype=bool),
            "pcs_source_timestamp": timestamps,
            "pcs_kw": pcs_period.to_numpy(dtype=float),
        }
    )
    table["pv_kw"] = table["pv_kw_filled"].clip(
        lower=0.0, upper=pv_capacity_kw
    )
    table["pv_was_clipped"] = (
        table["pv_kw_filled"].notna()
        & ~np.isclose(table["pv_kw_filled"], table["pv_kw"])
    )
    table["provisional_load_proxy_raw_kw"] = (
        table["net_grid_kw"] + table["pv_kw"] + table["pcs_kw"]
    )
    table["provisional_load_kw"] = table[
        "provisional_load_proxy_raw_kw"
    ].clip(lower=0.0)
    table["provisional_load_was_clipped"] = (
        table["provisional_load_proxy_raw_kw"] < 0.0
    )
    table["site_source_filename"] = signals.site_workbook.name
    table["storage_source_filename"] = signals.storage_workbook.name
    table["alignment_method"] = ALIGNMENT_METHOD
    table["provisional_load_formula"] = PROXY_FORMULA
    table["pcs_sign_convention"] = PCS_SIGN_CONVENTION

    required = [
        "pv_kw_filled",
        "net_grid_kw",
        "pcs_kw",
        "provisional_load_proxy_raw_kw",
        "provisional_load_kw",
    ]
    audit: dict[str, Any] = {
        "start": start_ts.isoformat(),
        "end_exclusive": end_ts.isoformat(),
        "timezone": TIMEZONE,
        "pv_source_rows": int(pv_period["value"].notna().sum()),
        "net_grid_source_rows": int(net_grid_period["value"].notna().sum()),
        "pcs_source_rows": int(pcs_period.notna().sum()),
        "final_reconstructed_rows": len(table),
        "missing_rows": {
            column: int(table[column].isna().sum()) for column in required
        },
        "pv_negative_rows_corrected": int(
            (table["pv_kw_filled"] < 0.0).sum()
        ),
        "pv_causally_filled_five_minute_rows": int(
            table["pv_was_causally_filled"].sum()
        ),
        "net_grid_causally_filled_five_minute_rows": int(
            table["net_grid_was_causally_filled"].sum()
        ),
        "raw_proxy_negative_rows": int(
            table["provisional_load_was_clipped"].sum()
        ),
        "raw_proxy_energy_kwh": float(
            table["provisional_load_proxy_raw_kw"].sum() / 12.0
        ),
        "clipped_provisional_load_energy_kwh": float(
            table["provisional_load_kw"].sum() / 12.0
        ),
        "causal_fill_limit_quarter_hours": causal_fill_limit,
        "alignment_method": ALIGNMENT_METHOD,
        "formula": PROXY_FORMULA,
        "pcs_sign_convention": PCS_SIGN_CONVENTION,
        "site_source_workbook": str(signals.site_workbook.resolve()),
        "storage_source_workbook": str(signals.storage_workbook.resolve()),
    }
    return table, audit


def validate_april30_reconstruction(audit: dict[str, Any]) -> None:
    """Enforce the raw-workbook checks required by the full-May benchmark."""
    expected_counts = {
        "pv_source_rows": 96,
        "net_grid_source_rows": 96,
        "pcs_source_rows": 288,
        "final_reconstructed_rows": 288,
        "raw_proxy_negative_rows": 3,
    }
    mismatches = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in expected_counts.items()
        if audit.get(key) != expected
    }
    if any(audit["missing_rows"].values()):
        mismatches["missing_rows"] = audit["missing_rows"]
    energy_checks = {
        "raw_proxy_energy_kwh": 9786.9875,
        "clipped_provisional_load_energy_kwh": 9869.3175,
    }
    for key, expected in energy_checks.items():
        if not np.isclose(float(audit[key]), expected, atol=1e-9, rtol=0.0):
            mismatches[key] = {"expected": expected, "actual": audit[key]}
    if mismatches:
        raise ValueError(
            "April 30 provisional-load reconstruction checks failed: "
            f"{mismatches}"
        )


def validate_against_reference_load(
    reconstruction: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    tolerance_kw: float = 1e-9,
) -> dict[str, Any]:
    """Require one-to-one timestamp agreement with an existing load series."""
    expected = reconstruction[["timestamp", "provisional_load_kw"]].copy()
    actual = reference[["timestamp", "load"]].copy()
    for frame in (expected, actual):
        timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(TIMEZONE)
        else:
            timestamps = timestamps.dt.tz_convert(TIMEZONE)
        frame["timestamp"] = timestamps
    matched = expected.merge(
        actual, on="timestamp", how="inner", validate="one_to_one"
    )
    differences = (
        matched["provisional_load_kw"] - matched["load"]
    ).abs()
    audit = {
        "expected_rows": len(expected),
        "reference_rows": len(actual),
        "matched_timestamps": len(matched),
        "missing_reconstructed_rows": int(
            expected["provisional_load_kw"].isna().sum()
        ),
        "missing_reference_rows": int(actual["load"].isna().sum()),
        "maximum_absolute_difference_kw": (
            float(differences.max()) if not differences.empty else None
        ),
        "tolerance_kw": tolerance_kw,
    }
    failed = (
        len(matched) != len(expected)
        or len(actual) != len(expected)
        or audit["missing_reconstructed_rows"] != 0
        or audit["missing_reference_rows"] != 0
        or audit["maximum_absolute_difference_kw"] is None
        or audit["maximum_absolute_difference_kw"] > tolerance_kw
    )
    if failed:
        raise ValueError(f"Reference load validation failed: {audit}")
    return audit
