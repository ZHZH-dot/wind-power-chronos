"""Prepare and audit the Foshan PV/grid-exchange workbooks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIMEZONE = "Asia/Shanghai"
SITE_ID = "foshan_site"
PV_SHEET = "光伏"
GRID_SHEET = "负荷"
TIMESTAMP_LABELS = ("数据时间", "时间", "timestamp", "datetime")
POWER_LABELS = ("有功功率(kw)", "有功功率", "总有功功率", "activepower")
CALENDAR_COLUMNS = [
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
]


@dataclass
class ParsedSignal:
    """One source signal on its complete native-frequency grid."""

    frame: pd.DataFrame
    audit: dict[str, Any]
    negative_readings: pd.DataFrame


def _normalized_label(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("\n", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def _matches_any(label: str, candidates: tuple[str, ...]) -> bool:
    normalized_candidates = tuple(_normalized_label(item) for item in candidates)
    return any(candidate == label or candidate in label for candidate in normalized_candidates)


def resolve_header_columns(raw: pd.DataFrame) -> tuple[int, int, int]:
    """Locate timestamp and active-power columns in a sheet with optional subheaders."""
    for row_position in range(min(20, len(raw))):
        labels = [_normalized_label(value) for value in raw.iloc[row_position].tolist()]
        timestamp_positions = [
            index for index, label in enumerate(labels) if _matches_any(label, TIMESTAMP_LABELS)
        ]
        power_positions = [
            index for index, label in enumerate(labels) if _matches_any(label, POWER_LABELS)
        ]
        if timestamp_positions and power_positions:
            return row_position, timestamp_positions[0], power_positions[0]
    raise ValueError(
        "Could not resolve a header row containing timestamp and active-power columns."
    )


def localize_timestamps(values: pd.Series, timezone: str = TIMEZONE) -> pd.Series:
    """Parse timestamps and normalize them to the site timezone."""
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return parsed.dt.tz_convert(timezone)


def _largest_gap(timestamps: pd.Series, frequency: str) -> dict[str, Any]:
    differences = timestamps.sort_values().drop_duplicates().diff().dropna()
    expected = pd.Timedelta(frequency)
    if differences.empty or differences.max() <= expected:
        return {"duration": str(expected), "missing_intervals": 0}
    largest = differences.max()
    return {
        "duration": str(largest),
        "missing_intervals": int(largest / expected) - 1,
    }


def _value_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    percentiles = finite.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "missing_value_count_source_rows": int(numeric.isna().sum()),
        "negative_count": int((finite < 0).sum()),
        "zero_count": int((finite == 0).sum()),
        "positive_count": int((finite > 0).sum()),
        "min": float(finite.min()) if not finite.empty else None,
        "max": float(finite.max()) if not finite.empty else None,
        "mean": float(finite.mean()) if not finite.empty else None,
        "percentiles": {
            f"p{int(level * 100):02d}": float(value)
            for level, value in percentiles.items()
        },
    }


def read_signal_sheet(
    workbook: Path,
    sheet_name: str | int,
    signal_name: str,
    frequency: str,
    timezone: str = TIMEZONE,
) -> ParsedSignal:
    """Read a signal sheet, resolve its headers, and place it on an exact grid."""
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_row, timestamp_position, power_position = resolve_header_columns(raw)
    resolved_timestamp_column = str(raw.iloc[header_row, timestamp_position]).strip()
    resolved_power_column = str(raw.iloc[header_row, power_position]).strip()

    values = raw.iloc[header_row + 1 :, [timestamp_position, power_position]].copy()
    values.columns = ["timestamp_source", "value_source"]
    nonempty_rows = ~values.isna().all(axis=1)
    values = values[nonempty_rows]
    values["timestamp"] = localize_timestamps(values["timestamp_source"], timezone)
    unparsed_timestamp_rows = int(values["timestamp"].isna().sum())
    values = values.dropna(subset=["timestamp"]).copy()
    values["value_raw"] = pd.to_numeric(values["value_source"], errors="coerce")
    values = values.sort_values("timestamp").reset_index(drop=True)

    duplicate_count = int(values["timestamp"].duplicated().sum())
    observed = (
        values.groupby("timestamp", as_index=False, sort=True)["value_raw"]
        .mean()
        .sort_values("timestamp")
    )
    if observed.empty:
        raise ValueError(f"No timestamped rows were found in sheet {sheet_name!r}.")

    full_grid = pd.date_range(
        observed["timestamp"].min(),
        observed["timestamp"].max(),
        freq=frequency,
    )
    grid = observed.set_index("timestamp").reindex(full_grid)
    grid.index.name = "timestamp"
    grid = grid.rename(columns={"value_raw": f"{signal_name}_raw"}).reset_index()
    grid[f"is_missing_{signal_name}"] = grid[f"{signal_name}_raw"].isna()

    negative_readings = values.loc[
        values["value_raw"] < 0,
        ["timestamp", "value_raw"],
    ].rename(columns={"value_raw": f"{signal_name}_raw"})

    value_summary = _value_summary(values["value_raw"])
    audit: dict[str, Any] = {
        "source_workbook": str(workbook),
        "source_sheet": str(sheet_name),
        "resolved_header_row_1_based": header_row + 1,
        "resolved_timestamp_column": resolved_timestamp_column,
        "resolved_power_column": resolved_power_column,
        "frequency": frequency,
        "timezone": timezone,
        "date_start": observed["timestamp"].min().isoformat(),
        "date_end": observed["timestamp"].max().isoformat(),
        "row_count": int(len(values)),
        "unique_timestamp_count": int(values["timestamp"].nunique()),
        "duplicate_count": duplicate_count,
        "unparsed_timestamp_rows": unparsed_timestamp_rows,
        "expected_grid_rows": int(len(full_grid)),
        "missing_intervals": int(len(full_grid) - observed["timestamp"].nunique()),
        "largest_gap": _largest_gap(observed["timestamp"], frequency),
        "missing_value_count_regularized": int(grid[f"{signal_name}_raw"].isna().sum()),
        **value_summary,
    }
    return ParsedSignal(frame=grid, audit=audit, negative_readings=negative_readings)


def add_calendar_covariates(table: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic numeric calendar features to a timestamped table."""
    result = table.copy()
    timestamp = pd.DatetimeIndex(result["timestamp"])
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    day_of_week = timestamp.dayofweek
    month_zero_based = timestamp.month - 1
    result["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    result["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    result["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    result["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    result["is_weekend"] = (day_of_week >= 5).astype(np.int8)
    result["month_sin"] = np.sin(2 * np.pi * month_zero_based / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * month_zero_based / 12.0)
    return result


def build_site_table(
    pv: ParsedSignal,
    grid: ParsedSignal,
    pv_capacity_kw: float = 1700.0,
    site_id: str = SITE_ID,
) -> pd.DataFrame:
    """Combine source signals without applying noncausal missing-value filling."""
    if pv_capacity_kw <= 0:
        raise ValueError("pv_capacity_kw must be positive.")
    start = min(pv.frame["timestamp"].min(), grid.frame["timestamp"].min())
    end = max(pv.frame["timestamp"].max(), grid.frame["timestamp"].max())
    timestamps = pd.date_range(start, end, freq="15min")
    table = pd.DataFrame({"timestamp": timestamps})
    table = table.merge(pv.frame, on="timestamp", how="left")
    table = table.merge(grid.frame, on="timestamp", how="left")
    table["id"] = site_id
    table["pv_kw"] = table["pv_kw_raw"].clip(lower=0.0, upper=pv_capacity_kw)
    table["net_grid_kw"] = table["net_grid_kw_raw"]
    table["is_missing_pv_kw"] = table["pv_kw_raw"].isna()
    table["is_missing_net_grid_kw"] = table["net_grid_kw_raw"].isna()
    table["is_corrected_pv_kw"] = (
        table["pv_kw_raw"].notna() & (table["pv_kw_raw"] != table["pv_kw"])
    )
    table = add_calendar_covariates(table)
    columns = [
        "id",
        "timestamp",
        "pv_kw_raw",
        "pv_kw",
        "net_grid_kw_raw",
        "net_grid_kw",
        "is_missing_pv_kw",
        "is_missing_net_grid_kw",
        "is_corrected_pv_kw",
        *CALENDAR_COLUMNS,
    ]
    return table[columns].sort_values("timestamp").reset_index(drop=True)


def audit_storage_workbook(
    workbook: Path,
    site_table: pd.DataFrame,
    timezone: str = TIMEZONE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit the five-minute PCS signal and a non-target gross-load proxy."""
    parsed = read_signal_sheet(
        workbook=workbook,
        sheet_name=0,
        signal_name="pcs_kw",
        frequency="5min",
        timezone=timezone,
    )
    storage = parsed.frame.set_index("timestamp")
    pcs_15min = storage["pcs_kw_raw"].resample("15min", label="left", closed="left").mean()
    pcs_count = storage["pcs_kw_raw"].resample("15min", label="left", closed="left").count()
    aligned = site_table[["timestamp", "pv_kw", "net_grid_kw"]].copy()
    aligned = aligned.merge(
        pd.DataFrame(
            {
                "timestamp": pcs_15min.index,
                "pcs_kw": pcs_15min.to_numpy(),
                "pcs_samples": pcs_count.to_numpy(),
            }
        ),
        on="timestamp",
        how="left",
    )
    aligned["gross_load_proxy_kw"] = (
        aligned["net_grid_kw"] + aligned["pv_kw"] + aligned["pcs_kw"]
    )
    complete = aligned["gross_load_proxy_kw"].dropna()
    overlap = aligned["pcs_kw"].notna()
    proxy_percentiles = complete.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    audit = {
        **parsed.audit,
        "role": "audit_only",
        "mapped_field": "pcs_kw",
        "sign_convention": "positive=discharge, negative=charge",
        "aggregation_to_15min": "arithmetic mean of available 5-minute power samples in each left-closed interval",
        "aligned_15min_rows": int(overlap.sum()),
        "alignment_total_rows": int(len(aligned)),
        "alignment_coverage": float(overlap.mean()),
        "aligned_missing_intervals": int((~overlap).sum()),
        "intervals_with_fewer_than_3_samples": int((aligned["pcs_samples"].fillna(0) < 3).sum()),
        "gross_load_proxy_formula": "net_grid_kw + pv_kw + pcs_kw",
        "gross_load_proxy_complete_rows": int(len(complete)),
        "gross_load_proxy_negative_count": int((complete < 0).sum()),
        "gross_load_proxy_min": float(complete.min()) if not complete.empty else None,
        "gross_load_proxy_max": float(complete.max()) if not complete.empty else None,
        "gross_load_proxy_percentiles": {
            f"p{int(level * 100):02d}": float(value)
            for level, value in proxy_percentiles.items()
        },
        "mapping_warning": (
            "The storage workbook name and total active-power field support a PCS mapping, "
            "but the data provider has not independently confirmed the meter semantics."
        ),
    }
    return aligned, audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_workbook_path(requested: Path) -> Path:
    """Resolve an explicit workbook path, including a unique '(1)' copy variant."""
    if requested.is_file():
        return requested.resolve()
    parent = requested.parent if requested.parent.is_dir() else Path("data/raw")
    base_stem = requested.stem.removesuffix(" (1)")
    candidates = sorted(
        path.resolve()
        for path in parent.glob("*.xlsx")
        if path.stem.removesuffix(" (1)") == base_stem
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"Workbook does not exist: {requested}")
    joined = ", ".join(str(path) for path in candidates)
    raise ValueError(f"Workbook path is ambiguous; pass one exact path: {joined}")


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Foshan Data Audit",
        "",
        "> `net_grid_kw` is provisional bidirectional grid exchange. It is not confirmed gross load, and negative values are preserved.",
        "",
    ]
    for target in ("pv_kw", "net_grid_kw"):
        item = audit["targets"][target]
        lines.extend(
            [
                f"## {target}",
                "",
                f"- Source: `{item['source_sheet']}` / `{item['resolved_power_column']}`",
                f"- Range: {item['date_start']} through {item['date_end']}",
                f"- Rows / unique timestamps: {item['row_count']} / {item['unique_timestamp_count']}",
                f"- Duplicates: {item['duplicate_count']}",
                f"- Missing 15-minute intervals: {item['missing_intervals']}",
                f"- Largest gap: {item['largest_gap']['duration']} ({item['largest_gap']['missing_intervals']} missing slots)",
                f"- Missing values on regularized grid: {item['missing_value_count_regularized']}",
                f"- Negative / zero / positive: {item['negative_count']} / {item['zero_count']} / {item['positive_count']}",
                f"- Min / mean / max: {item['min']} / {item['mean']} / {item['max']}",
                f"- Percentiles: {item['percentiles']}",
                f"- Physical correction: {item['physical_correction']}",
                f"- Negative-reading ledger: {item.get('negative_readings_file', 'not applicable')}",
                f"- Skipped forecast origins: {item.get('skipped_forecast_origins', 0)} (updated by the benchmark runner)",
                "",
            ]
        )
    storage = audit.get("storage_audit")
    if storage:
        lines.extend(
            [
                "## Storage Audit",
                "",
                f"- 15-minute alignment coverage: {storage['alignment_coverage']:.3%}",
                f"- Missing aligned intervals: {storage['aligned_missing_intervals']}",
                f"- Gross-load proxy negative count: {storage['gross_load_proxy_negative_count']}",
                f"- Proxy min / max: {storage['gross_load_proxy_min']} / {storage['gross_load_proxy_max']}",
                f"- Warning: {storage['mapping_warning']}",
                "",
            ]
        )
    return "\n".join(lines)


def _classification_markdown() -> str:
    return """# Foshan Target Classification

| Source | Forecast field | Classification | Treatment |
|---|---|---|---|
| 光伏 sheet | `pv_kw` | PV active power | Preserve `pv_kw_raw`; clip model target and forecasts to [0, 1700] kW. |
| 负荷 sheet | `net_grid_kw` | Provisional bidirectional grid exchange | Preserve negative, zero, and positive values; never clip. |
| Storage workbook | `pcs_kw` | Audit-only PCS power | Positive is discharge and negative is charge; aggregate 5-minute readings by 15-minute mean. |

`net_grid_kw` is not confirmed gross factory load. It must not be renamed to
`gross_load_kw`, clamped to zero, or presented as gross load until the data provider
confirms the meter definition. The optional `gross_load_proxy_kw` audit is not a
forecasting target.
"""


def write_audit_documents(audit: dict[str, Any], audit_dir: Path) -> None:
    """Persist structured and human-readable audit documents."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    (audit_dir / "data_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    (audit_dir / "target_classification.md").write_text(
        _classification_markdown(), encoding="utf-8"
    )


def prepare_foshan_workbooks(
    source_workbook: Path,
    storage_workbook: Path | None = None,
    pv_capacity_kw: float = 1700.0,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame | None, pd.DataFrame]:
    """Prepare the site table and all structured audit information."""
    source = resolve_workbook_path(source_workbook)
    storage = resolve_workbook_path(storage_workbook) if storage_workbook else None
    pv = read_signal_sheet(source, PV_SHEET, "pv_kw", "15min")
    grid = read_signal_sheet(source, GRID_SHEET, "net_grid_kw", "15min")
    table = build_site_table(pv, grid, pv_capacity_kw=pv_capacity_kw)

    pv.audit["physical_correction"] = f"clip raw values to [0, {pv_capacity_kw:g}] kW"
    pv.audit["corrected_low_count"] = int((table["pv_kw_raw"] < 0).sum())
    pv.audit["corrected_high_count"] = int((table["pv_kw_raw"] > pv_capacity_kw).sum())
    pv.audit["skipped_forecast_origins"] = 0
    grid.audit["physical_correction"] = "none; signed grid exchange is preserved"
    grid.audit["skipped_forecast_origins"] = 0
    audit: dict[str, Any] = {
        "site_id": SITE_ID,
        "timezone": TIMEZONE,
        "frequency": "15min",
        "source_workbook": str(source),
        "source_workbook_sha256": sha256_file(source),
        "classification_warning": (
            "net_grid_kw is provisional bidirectional grid exchange, not confirmed gross load"
        ),
        "targets": {"pv_kw": pv.audit, "net_grid_kw": grid.audit},
    }

    storage_aligned: pd.DataFrame | None = None
    if storage is not None:
        storage_aligned, storage_audit = audit_storage_workbook(storage, table)
        audit["storage_workbook"] = str(storage)
        audit["storage_workbook_sha256"] = sha256_file(storage)
        audit["storage_audit"] = storage_audit
    return table, audit, storage_aligned, pv.negative_readings


def write_prepared_outputs(
    table: pd.DataFrame,
    audit: dict[str, Any],
    negative_readings: pd.DataFrame,
    output_path: Path,
    audit_dir: Path,
    storage_aligned: pd.DataFrame | None = None,
) -> None:
    raw_dir = Path("data/raw").resolve()
    for path in (output_path, audit_dir):
        resolved = path.resolve()
        if resolved == raw_dir or raw_dir in resolved.parents:
            raise ValueError("Refusing to write generated output under data/raw/.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    negative_path = audit_dir / "pv_negative_readings.csv"
    negative_readings.to_csv(negative_path, index=False)
    audit["targets"]["pv_kw"]["negative_readings_file"] = str(negative_path)
    if storage_aligned is not None:
        storage_aligned.to_parquet(audit_dir / "storage_audit_15min.parquet", index=False)
    write_audit_documents(audit, audit_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workbook", required=True, type=Path)
    parser.add_argument("--storage-workbook", default=None, type=Path)
    parser.add_argument(
        "--output",
        default=Path("results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet"),
        type=Path,
    )
    parser.add_argument(
        "--audit-dir",
        default=Path("results/zero_shot/foshan_chronos2"),
        type=Path,
    )
    parser.add_argument("--pv-capacity-kw", default=1700.0, type=float)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    table, audit, storage_aligned, negative_readings = prepare_foshan_workbooks(
        source_workbook=args.source_workbook,
        storage_workbook=args.storage_workbook,
        pv_capacity_kw=args.pv_capacity_kw,
    )
    write_prepared_outputs(
        table=table,
        audit=audit,
        negative_readings=negative_readings,
        output_path=args.output,
        audit_dir=args.audit_dir,
        storage_aligned=storage_aligned,
    )
    print(f"Wrote {len(table):,} rows to {args.output}")
    print(f"Wrote Foshan data audit to {args.audit_dir}")


if __name__ == "__main__":
    main()
