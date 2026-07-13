"""Evaluate Chronos-2 zero-shot prediction files."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (
    bias,
    interval_coverage,
    mae,
    mean_interval_width,
    mean_pinball_loss,
    nmae_capacity,
    nrmse_capacity,
    pinball_loss,
    rmse,
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = [read_table(path) for path in paths]
    if not frames:
        raise ValueError("At least one prediction file is required.")
    predictions = pd.concat(frames, ignore_index=True)
    predictions["id"] = predictions["id"].astype(str)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    return predictions


def attach_ground_truth(predictions: pd.DataFrame, ground_truth_path: Path | None) -> pd.DataFrame:
    if ground_truth_path is None:
        if "y_true" in predictions.columns and predictions["y_true"].notna().all():
            return predictions
        raise ValueError("Ground truth is required when prediction files do not include y_true.")

    truth = read_table(ground_truth_path)
    truth["id"] = truth["id"].astype(str)
    truth["timestamp"] = pd.to_datetime(truth["timestamp"])
    optional_columns = [
        column
        for column in ("is_imputed_target", "rated_capacity_kw")
        if column in truth.columns
    ]
    truth = truth[["id", "timestamp", "target", *optional_columns]].rename(
        columns={
            "target": "y_true_from_truth",
            "is_imputed_target": "is_imputed_target_from_truth",
            "rated_capacity_kw": "rated_capacity_kw_from_truth",
        }
    )

    merged = predictions.merge(truth, on=["id", "timestamp"], how="left")
    for column in ("y_true", "is_imputed_target", "rated_capacity_kw"):
        truth_column = f"{column}_from_truth"
        if truth_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[truth_column].combine_first(merged[column])
        else:
            merged[column] = merged[truth_column]
        merged = merged.drop(columns=[truth_column])
    return merged


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    valid_values = {"true", "false", "1", "0", "1.0", "0.0", "yes", "no", ""}
    invalid = normalized.notna() & ~normalized.isin(valid_values)
    if invalid.any():
        invalid_values = sorted(normalized[invalid].dropna().unique().tolist())
        raise ValueError(f"Invalid is_imputed_target values: {invalid_values}")
    return normalized.isin({"true", "1", "1.0", "yes"})


def evaluate_predictions(
    predictions: pd.DataFrame,
    include_imputed_targets: bool = False,
    rated_capacity_kw: float = 1500.0,
) -> pd.DataFrame:
    if rated_capacity_kw <= 0:
        raise ValueError("rated_capacity_kw must be positive.")

    required = {"mode", "horizon", "y_true"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")

    evaluated = predictions.copy()
    if "p50" not in evaluated.columns:
        if "y_pred" not in evaluated.columns:
            raise ValueError("Prediction file must contain p50 or y_pred.")
        evaluated["p50"] = evaluated["y_pred"]
    if "y_pred" not in evaluated.columns:
        evaluated["y_pred"] = evaluated["p50"]
    for column in ("p10", "p90"):
        if column not in evaluated.columns:
            evaluated[column] = math.nan

    if "is_imputed_target" not in evaluated.columns:
        evaluated["is_imputed_target"] = False
    evaluated["is_imputed_target"] = _boolean_series(evaluated["is_imputed_target"])

    if "rated_capacity_kw" not in evaluated.columns:
        evaluated["rated_capacity_kw"] = rated_capacity_kw
    else:
        evaluated["rated_capacity_kw"] = pd.to_numeric(
            evaluated["rated_capacity_kw"],
            errors="coerce",
        ).fillna(rated_capacity_kw)

    rows: list[dict[str, float | int | str]] = []
    for (mode, horizon), group in evaluated.groupby(["mode", "horizon"], sort=True):
        imputed_count = int(group["is_imputed_target"].sum())
        scoring_group = group if include_imputed_targets else group[~group["is_imputed_target"]]
        point_mask = (
            np.isfinite(pd.to_numeric(scoring_group["y_true"], errors="coerce"))
            & np.isfinite(pd.to_numeric(scoring_group["p50"], errors="coerce"))
        )
        scored = scoring_group[point_mask].copy()
        capacities = pd.to_numeric(scored["rated_capacity_kw"], errors="coerce")
        if (capacities <= 0).any():
            raise ValueError("rated_capacity_kw values must be positive.")

        pinball_p10 = pinball_loss(scoring_group["y_true"], scoring_group["p10"], 0.1)
        pinball_p50 = pinball_loss(scoring_group["y_true"], scoring_group["p50"], 0.5)
        pinball_p90 = pinball_loss(scoring_group["y_true"], scoring_group["p90"], 0.9)
        rows.append(
            {
                "mode": str(mode),
                "horizon": int(horizon),
                "n_scored": int(len(scored)),
                "n_excluded_imputed": 0 if include_imputed_targets else imputed_count,
                "mae": mae(scored["y_true"], scored["p50"]),
                "rmse": rmse(scored["y_true"], scored["p50"]),
                "bias": bias(scored["y_true"], scored["p50"]),
                "nmae_capacity": nmae_capacity(
                    scored["y_true"],
                    scored["p50"],
                    capacities,
                ),
                "nrmse_capacity": nrmse_capacity(
                    scored["y_true"],
                    scored["p50"],
                    capacities,
                ),
                "pinball_p10": pinball_p10,
                "pinball_p50": pinball_p50,
                "pinball_p90": pinball_p90,
                "mean_pinball_loss": mean_pinball_loss(
                    scoring_group["y_true"],
                    {
                        0.1: scoring_group["p10"],
                        0.5: scoring_group["p50"],
                        0.9: scoring_group["p90"],
                    },
                ),
                "p10_p90_coverage": interval_coverage(
                    scoring_group["y_true"],
                    scoring_group["p10"],
                    scoring_group["p90"],
                ),
                "mean_interval_width": mean_interval_width(
                    scoring_group["p10"],
                    scoring_group["p90"],
                ),
            }
        )

    columns = [
        "mode",
        "horizon",
        "n_scored",
        "n_excluded_imputed",
        "mae",
        "rmse",
        "bias",
        "nmae_capacity",
        "nrmse_capacity",
        "pinball_p10",
        "pinball_p50",
        "pinball_p90",
        "mean_pinball_loss",
        "p10_p90_coverage",
        "mean_interval_width",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["horizon", "mode"]).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        type=Path,
        help="One or more prediction CSV/parquet files.",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        type=Path,
        help="Processed SDWPF parquet/CSV with id, timestamp, target.",
    )
    parser.add_argument(
        "--output",
        default=Path("results/zero_shot_metrics.csv"),
        type=Path,
        help="Output CSV result table.",
    )
    parser.add_argument(
        "--rated-capacity-kw",
        default=1500.0,
        type=float,
        help="Fallback turbine rated capacity used for capacity-normalized metrics.",
    )
    parser.add_argument(
        "--include-imputed-targets",
        action="store_true",
        help="Score interpolated targets for diagnostics. Off by default.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    predictions = load_predictions(args.predictions)
    predictions = attach_ground_truth(predictions, args.ground_truth)
    result_table = evaluate_predictions(
        predictions,
        include_imputed_targets=args.include_imputed_targets,
        rated_capacity_kw=args.rated_capacity_kw,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(args.output, index=False)
    print(result_table.to_string(index=False))
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
