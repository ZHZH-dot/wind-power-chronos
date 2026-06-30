"""Evaluate Chronos-2 zero-shot prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import mae, nmae, nrmse, rmse


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
    truth = truth[["id", "timestamp", "target"]].rename(columns={"target": "y_true_from_truth"})

    merged = predictions.merge(truth, on=["id", "timestamp"], how="left")
    if "y_true" in merged.columns:
        merged["y_true"] = merged["y_true_from_truth"].fillna(merged["y_true"])
    else:
        merged["y_true"] = merged["y_true_from_truth"]
    return merged.drop(columns=["y_true_from_truth"])


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"mode", "horizon", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")

    rows: list[dict[str, float | int | str]] = []
    valid = predictions.dropna(subset=["y_true", "y_pred"])
    for (mode, horizon), group in valid.groupby(["mode", "horizon"], sort=True):
        rows.append(
            {
                "mode": str(mode),
                "horizon": int(horizon),
                "n": int(len(group)),
                "mae": mae(group["y_true"], group["y_pred"]),
                "rmse": rmse(group["y_true"], group["y_pred"]),
                "nmae": nmae(group["y_true"], group["y_pred"]),
                "nrmse": nrmse(group["y_true"], group["y_pred"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["horizon", "mode"]).reset_index(drop=True)


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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    predictions = load_predictions(args.predictions)
    predictions = attach_ground_truth(predictions, args.ground_truth)
    result_table = evaluate_predictions(predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(args.output, index=False)
    print(result_table.to_string(index=False))
    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
