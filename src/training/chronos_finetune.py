"""LoRA fine-tuning for Chronos-2 on the leakage-safe SDWPF benchmark split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.splits import (
    build_split_manifest,
    load_benchmark_config,
    load_split_manifest,
    validate_split_manifest,
    write_split_manifest,
)


DEFAULT_MODEL_ID = "amazon/chronos-2"
DEFAULT_SPLIT_CONFIG = Path("configs/splits/sdwpf_70_10_20.json")
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
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


@dataclass(frozen=True)
class FineTuneFrames:
    train: pd.DataFrame
    validation_context: pd.DataFrame
    turbine_ids: list[str]
    n_masked_imputed_train: int
    n_masked_imputed_validation: int


def parse_csv_list(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    return list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    valid = {"true", "false", "1", "0", "1.0", "0.0", "yes", "no", ""}
    invalid = normalized.notna() & ~normalized.isin(valid)
    if invalid.any():
        invalid_values = sorted(normalized[invalid].dropna().unique().tolist())
        raise ValueError(f"Invalid is_imputed_target values: {invalid_values}")
    return normalized.isin({"true", "1", "1.0", "yes"})


def validate_model_id(model_id: str) -> str:
    if model_id == DEFAULT_MODEL_ID:
        return model_id
    local_path = Path(model_id).expanduser()
    if not local_path.is_dir():
        raise ValueError(
            "--model-id must be amazon/chronos-2 or an existing local Chronos-2 directory."
        )
    return str(local_path)


def validate_and_normalize_data(
    data: pd.DataFrame,
    mode: str,
    covariates: list[str],
    frequency: str,
) -> pd.DataFrame:
    reserved_covariates = {"id", "timestamp", "target", "is_imputed_target"}
    conflicts = sorted(reserved_covariates.intersection(covariates))
    if conflicts:
        raise ValueError(f"Covariates conflict with reserved columns: {conflicts}")

    required = ["id", "timestamp", "target", "is_imputed_target"]
    if mode == "multivariate":
        if not covariates:
            raise ValueError("Multivariate mode requires --covariates.")
        required.extend(covariates)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    columns = ["id", "timestamp", "target", "is_imputed_target"]
    if mode == "multivariate":
        columns.extend(covariates)
    normalized = data[columns].copy()
    normalized["id"] = normalized["id"].astype(str)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="raise")
    normalized["is_imputed_target"] = _boolean_series(normalized["is_imputed_target"])

    numeric_columns = ["target", *(covariates if mode == "multivariate" else [])]
    for column in numeric_columns:
        source_missing = normalized[column].isna()
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        introduced_missing = normalized[column].isna() & ~source_missing
        if introduced_missing.any():
            raise ValueError(f"Column {column} contains non-numeric values.")

    if normalized.duplicated(["id", "timestamp"]).any():
        raise ValueError("Input data contains duplicate id/timestamp rows.")
    normalized = normalized.sort_values(["id", "timestamp"]).reset_index(drop=True)

    expected_delta = pd.to_timedelta(frequency)
    for turbine_id, group in normalized.groupby("id", sort=False):
        deltas = group["timestamp"].diff().dropna()
        if not deltas.eq(expected_delta).all():
            raise ValueError(
                f"Turbine {turbine_id} is not regular at frequency {frequency}. "
                "Prepare data with --regularize-hourly."
            )
    return normalized


def resolve_split_manifest(
    data: pd.DataFrame,
    split_config: dict[str, Any],
    output_dir: Path,
    split_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if split_manifest_path is not None:
        if not split_manifest_path.exists():
            raise FileNotFoundError(f"Split manifest does not exist: {split_manifest_path}")
        manifest = load_split_manifest(split_manifest_path)
        validate_split_manifest(manifest, data["timestamp"], split_config)
    else:
        manifest = build_split_manifest(data["timestamp"], split_config)

    write_split_manifest(manifest, output_dir / "resolved_split_manifest.json")
    return manifest


def prepare_finetune_frames(
    data: pd.DataFrame,
    manifest: dict[str, Any],
    mode: str,
    covariates: list[str],
    prediction_length: int,
    context_length: int,
    max_turbines: int | None = None,
) -> FineTuneFrames:
    if max_turbines is not None and max_turbines <= 0:
        raise ValueError("--max-turbines must be positive.")

    train_end = pd.Timestamp(manifest["splits"]["train"]["end"])
    validation_end = pd.Timestamp(manifest["splits"]["validation"]["end"])
    test_start = pd.Timestamp(manifest["splits"]["test"]["start"])

    turbine_ids: list[str] = []
    for turbine_id, group in data.groupby("id", sort=True):
        train_count = int((group["timestamp"] <= train_end).sum())
        validation_count = int(
            ((group["timestamp"] > train_end) & (group["timestamp"] <= validation_end)).sum()
        )
        if (
            train_count >= context_length + prediction_length
            and validation_count >= prediction_length
        ):
            turbine_ids.append(str(turbine_id))
    if max_turbines is not None:
        turbine_ids = turbine_ids[:max_turbines]
    if not turbine_ids:
        raise ValueError(
            "No turbine has enough train history and a complete validation forecast window."
        )
    selected = data[data["id"].isin(turbine_ids)].copy()

    model_columns = ["id", "timestamp", "target"]
    if mode == "multivariate":
        model_columns.extend(covariates)

    train = selected[selected["timestamp"] <= train_end][model_columns].copy()
    validation_context = selected[selected["timestamp"] <= validation_end][model_columns].copy()
    train_imputed = selected.loc[train.index, "is_imputed_target"].astype(bool)
    validation_imputed = selected.loc[validation_context.index, "is_imputed_target"].astype(bool)
    train.loc[train_imputed, "target"] = float("nan")
    validation_context.loc[validation_imputed, "target"] = float("nan")

    if train.empty or validation_context.empty:
        raise ValueError("Resolved training or validation data is empty.")
    if train["timestamp"].max() > train_end:
        raise AssertionError("Training data crosses the train boundary.")
    if validation_context["timestamp"].max() > validation_end:
        raise AssertionError("Validation context crosses the validation boundary.")
    if validation_context["timestamp"].max() >= test_start:
        raise AssertionError("Test targets would enter training or model selection.")
    if not (validation_context["timestamp"] > train_end).any():
        raise ValueError("Validation context contains no validation-period rows.")

    return FineTuneFrames(
        train=train.reset_index(drop=True),
        validation_context=validation_context.reset_index(drop=True),
        turbine_ids=turbine_ids,
        n_masked_imputed_train=int(train_imputed.sum()),
        n_masked_imputed_validation=int(validation_imputed.sum()),
    )


def build_chronos2_hyperparameters(
    model_id: str,
    mode: str,
    prediction_length: int,
    context_length: int,
    steps: int,
    learning_rate: float,
    batch_size: int,
    inference_batch_size: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if min(prediction_length, context_length, steps, batch_size, inference_batch_size) <= 0:
        raise ValueError("Lengths, steps, and batch sizes must be positive.")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    return {
        "Chronos2": {
            "model_path": model_id,
            "device": "cuda",
            "batch_size": inference_batch_size,
            "context_length": context_length,
            "fine_tune": True,
            "fine_tune_mode": "lora",
            "fine_tune_lr": learning_rate,
            "fine_tune_steps": steps,
            "fine_tune_batch_size": batch_size,
            "fine_tune_context_length": context_length,
            "eval_during_fine_tune": True,
            "disable_known_covariates": True,
            "disable_past_covariates": mode == "univariate",
            "fine_tune_trainer_kwargs": {"seed": seed, "data_seed": seed},
            "ag_args": {"name_suffix": "LoRA"},
        }
    }


def _load_autogluon() -> tuple[Any, Any]:
    from autogluon.timeseries import (
        TimeSeriesDataFrame,
        TimeSeriesPredictor,
    )

    return TimeSeriesDataFrame, TimeSeriesPredictor


def fit_with_autogluon(
    frames: FineTuneFrames,
    output_dir: Path,
    frequency: str,
    prediction_length: int,
    hyperparameters: dict[str, dict[str, Any]],
    seed: int,
    autogluon_classes: tuple[Any, Any] | None = None,
) -> Any:
    predictor_path = output_dir / "predictor"
    if predictor_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing AutoGluon predictor: {predictor_path}"
        )

    dataframe_class, predictor_class = autogluon_classes or _load_autogluon()
    train_data = dataframe_class.from_data_frame(
        frames.train,
        id_column="id",
        timestamp_column="timestamp",
    )
    tuning_data = dataframe_class.from_data_frame(
        frames.validation_context,
        id_column="id",
        timestamp_column="timestamp",
    )

    predictor = predictor_class(
        path=str(predictor_path),
        prediction_length=prediction_length,
        target="target",
        known_covariates_names=[],
        quantile_levels=QUANTILE_LEVELS,
        eval_metric="WQL",
        freq=frequency,
    )
    predictor.fit(
        train_data=train_data,
        tuning_data=tuning_data,
        hyperparameters=hyperparameters,
        enable_ensemble=False,
        random_seed=seed,
        refit_full=False,
        skip_model_selection=False,
    )
    return predictor


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    split_config: dict[str, Any],
    manifest: dict[str, Any],
    frames: FineTuneFrames,
    hyperparameters: dict[str, dict[str, Any]],
) -> None:
    run_config = {
        "input": str(args.input),
        "split_config": str(args.split_config),
        "split_manifest_source": str(args.split_manifest) if args.split_manifest else None,
        "model_id": args.model_id,
        "mode": args.mode,
        "covariates": parse_csv_list(args.covariates) if args.mode == "multivariate" else [],
        "prediction_length": args.prediction_length,
        "context_length": args.context_length,
        "quantile_levels": QUANTILE_LEVELS,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "split": manifest["splits"],
        "split_strategy": split_config["split_strategy"],
        "n_turbines": len(frames.turbine_ids),
        "n_train_rows": len(frames.train),
        "n_validation_context_rows": len(frames.validation_context),
        "n_masked_imputed_train": frames.n_masked_imputed_train,
        "n_masked_imputed_validation": frames.n_masked_imputed_validation,
        "test_data_passed_to_fit": False,
        "hyperparameters": hyperparameters,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)
        file.write("\n")


def run(args: argparse.Namespace, autogluon_classes: tuple[Any, Any] | None = None) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.model_id = validate_model_id(args.model_id)
    split_config = load_benchmark_config(Path(args.split_config))
    data = read_table(Path(args.input))
    covariates = parse_csv_list(args.covariates)
    data = validate_and_normalize_data(
        data,
        mode=args.mode,
        covariates=covariates,
        frequency=str(split_config["frequency"]),
    )
    manifest = resolve_split_manifest(
        data,
        split_config=split_config,
        output_dir=output_dir,
        split_manifest_path=Path(args.split_manifest) if args.split_manifest else None,
    )
    frames = prepare_finetune_frames(
        data,
        manifest=manifest,
        mode=args.mode,
        covariates=covariates,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        max_turbines=args.max_turbines,
    )
    hyperparameters = build_chronos2_hyperparameters(
        model_id=args.model_id,
        mode=args.mode,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        steps=args.steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        seed=args.seed,
    )
    write_run_config(output_dir, args, split_config, manifest, frames, hyperparameters)

    summary = {
        "output_dir": str(output_dir),
        "n_turbines": len(frames.turbine_ids),
        "n_train_rows": len(frames.train),
        "n_validation_context_rows": len(frames.validation_context),
        "train_end": manifest["splits"]["train"]["end"],
        "validation_end": manifest["splits"]["validation"]["end"],
        "test_start": manifest["splits"]["test"]["start"],
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        fit_with_autogluon(
            frames,
            output_dir=output_dir,
            frequency=str(split_config["frequency"]),
            prediction_length=args.prediction_length,
            hyperparameters=hyperparameters,
            seed=args.seed,
            autogluon_classes=autogluon_classes,
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--split-config", default=DEFAULT_SPLIT_CONFIG, type=Path)
    parser.add_argument(
        "--split-manifest",
        default=None,
        type=Path,
        help="Optional resolved manifest produced by src.evaluation.splits.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--mode", choices=["univariate", "multivariate"], default="multivariate")
    parser.add_argument("--covariates", default=",".join(DEFAULT_COVARIATES))
    parser.add_argument("--prediction-length", default=72, type=int)
    parser.add_argument("--context-length", default=168, type=int)
    parser.add_argument("--steps", default=1000, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--inference-batch-size", default=64, type=int)
    parser.add_argument("--max-turbines", "--max_turbines", dest="max_turbines", type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("Dry run complete. AutoGluon and Chronos-2 were not loaded.")
    else:
        print(f"Fine-tuned predictor saved under {Path(args.output_dir) / 'predictor'}")


if __name__ == "__main__":
    main()
