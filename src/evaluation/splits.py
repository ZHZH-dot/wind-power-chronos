"""Create and reuse chronological benchmark split manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_BENCHMARK_CONFIG = Path("configs/sdwpf_benchmark.json")
DEFAULT_SPLIT_MANIFEST = Path("data/processed/sdwpf_split_manifest.json")


def _normalize_split_ratios(config: dict[str, Any]) -> dict[str, Any]:
    normalized = config.copy()
    for split_name in ("train", "validation", "test"):
        fraction_key = f"{split_name}_fraction"
        ratio_key = f"{split_name}_ratio"
        fraction = normalized.get(fraction_key)
        ratio = normalized.get(ratio_key)
        if fraction is None and ratio is None:
            raise ValueError(
                f"Benchmark config must define {fraction_key} or {ratio_key}."
            )
        if fraction is not None and ratio is not None and not math.isclose(
            float(fraction),
            float(ratio),
        ):
            raise ValueError(f"{fraction_key} and {ratio_key} must match.")
        normalized[fraction_key] = float(fraction if fraction is not None else ratio)
    return normalized


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_benchmark_config(path: Path = DEFAULT_BENCHMARK_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {"name", "frequency", "split_strategy"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Benchmark config is missing fields: {missing}")

    config = _normalize_split_ratios(config)

    fractions = [
        float(config["train_fraction"]),
        float(config["validation_fraction"]),
        float(config["test_fraction"]),
    ]
    if any(fraction <= 0 for fraction in fractions) or not math.isclose(sum(fractions), 1.0):
        raise ValueError("Benchmark split fractions must be positive and sum to 1.0.")
    if config["split_strategy"] != "global_chronological_timestamp":
        raise ValueError("Only global_chronological_timestamp is supported.")
    return config


def _sorted_unique_timestamps(values: Iterable[object]) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(pd.Series(values), errors="raise").dropna().drop_duplicates()
    return pd.DatetimeIndex(timestamps.sort_values().tolist())


def build_split_manifest(
    timestamps: Iterable[object],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build exact global timestamp boundaries from benchmark fractions."""
    unique_timestamps = _sorted_unique_timestamps(timestamps)
    total = len(unique_timestamps)
    train_count = int(total * float(config["train_fraction"]))
    validation_count = int(total * float(config["validation_fraction"]))
    test_count = total - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Dataset is too short to create non-empty train, validation, and test splits.")

    train = unique_timestamps[:train_count]
    validation = unique_timestamps[train_count : train_count + validation_count]
    test = unique_timestamps[train_count + validation_count :]

    def split_entry(values: pd.DatetimeIndex) -> dict[str, Any]:
        return {
            "start": values[0].isoformat(),
            "end": values[-1].isoformat(),
            "n_timestamps": len(values),
        }

    return {
        "version": 1,
        "benchmark": str(config["name"]),
        "strategy": str(config["split_strategy"]),
        "frequency": str(config["frequency"]),
        "fractions": {
            "train": float(config["train_fraction"]),
            "validation": float(config["validation_fraction"]),
            "test": float(config["test_fraction"]),
        },
        "global": {
            "start": unique_timestamps[0].isoformat(),
            "end": unique_timestamps[-1].isoformat(),
            "n_timestamps": total,
        },
        "splits": {
            "train": split_entry(train),
            "validation": split_entry(validation),
            "test": split_entry(test),
        },
    }


def write_split_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


def load_split_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_split_manifest(
    manifest: dict[str, Any],
    timestamps: Iterable[object],
    config: dict[str, Any],
) -> None:
    """Verify that a resolved manifest uses the same data boundaries and split ratios."""
    expected = build_split_manifest(timestamps, _normalize_split_ratios(config))
    comparable_keys = ("strategy", "frequency", "fractions", "global", "splits")
    mismatched = [key for key in comparable_keys if manifest.get(key) != expected.get(key)]
    if mismatched:
        raise ValueError(
            "Resolved split manifest does not match the input data/config for fields: "
            f"{mismatched}"
        )


def ensure_split_manifest(
    data: pd.DataFrame,
    config_path: Path = DEFAULT_BENCHMARK_CONFIG,
    manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a manifest or verify that an existing manifest matches the data and config."""
    if "timestamp" not in data.columns:
        raise ValueError("Input data must contain a timestamp column.")

    config = load_benchmark_config(config_path)
    expected = build_split_manifest(data["timestamp"], config)
    if manifest_path.exists() and not overwrite:
        existing = load_split_manifest(manifest_path)
        validate_split_manifest(existing, data["timestamp"], config)
        return existing

    write_split_manifest(expected, manifest_path)
    return expected


def test_period(manifest: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    test_split = manifest["splits"]["test"]
    return pd.Timestamp(test_split["start"]), pd.Timestamp(test_split["end"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", default=DEFAULT_BENCHMARK_CONFIG, type=Path)
    parser.add_argument("--output", default=DEFAULT_SPLIT_MANIFEST, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest after recomputing it from the input data.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data = read_table(args.input)
    manifest = ensure_split_manifest(
        data,
        config_path=args.config,
        manifest_path=args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))
    print(f"Wrote split manifest to {args.output}")


if __name__ == "__main__":
    main()
