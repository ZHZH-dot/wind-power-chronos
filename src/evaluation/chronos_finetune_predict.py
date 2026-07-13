"""Leak-free test inference for a saved AutoGluon Chronos-2 LoRA predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluate import evaluate_predictions
from src.evaluation.splits import (
    load_benchmark_config,
    load_split_manifest,
    test_period,
    validate_split_manifest,
)
from src.models.chronos_zero_shot import (
    DEFAULT_HORIZONS,
    DEFAULT_QUANTILES,
    parse_int_list,
    run_rolling_forecasts,
    write_predictions,
)
from src.training.chronos_finetune import (
    DEFAULT_COVARIATES,
    DEFAULT_SPLIT_CONFIG,
    parse_csv_list,
    prepare_finetune_frames,
    read_table,
    validate_and_normalize_data,
)


DEFAULT_INPUT = Path("src/data/processed/sdwpf_hourly_regularized.parquet")
EXPECTED_MODEL_PREFIX = "Chronos2LoRA"


def _load_autogluon() -> tuple[Any, Any]:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    return TimeSeriesDataFrame, TimeSeriesPredictor


def _model_names(predictor: Any) -> list[str]:
    names = predictor.model_names()
    return [str(name) for name in names]


def validate_checkpoint_files(predictor_path: Path) -> list[Path]:
    if not predictor_path.is_dir():
        raise FileNotFoundError(f"AutoGluon predictor directory does not exist: {predictor_path}")

    adapters = sorted(
        path
        for path in predictor_path.rglob("adapter_model.safetensors")
        if path.parent.name == "fine-tuned-ckpt"
    )
    if not adapters:
        raise RuntimeError(
            "Incomplete Chronos-2 LoRA checkpoint: no "
            "fine-tuned-ckpt/adapter_model.safetensors was found under "
            f"{predictor_path}."
        )
    return adapters


def load_verified_predictor(
    predictor_path: Path,
    predictor_class: Any | None = None,
) -> tuple[Any, str, list[Path]]:
    adapters = validate_checkpoint_files(predictor_path)
    if predictor_class is None:
        _, predictor_class = _load_autogluon()

    try:
        predictor = predictor_class.load(str(predictor_path))
    except Exception as error:
        raise RuntimeError(f"Could not reload AutoGluon predictor at {predictor_path}: {error}") from error

    names = _model_names(predictor)
    matching = [name for name in names if name.startswith(EXPECTED_MODEL_PREFIX)]
    if not matching:
        raise RuntimeError(
            f"Loaded predictor does not contain {EXPECTED_MODEL_PREFIX}. Models found: {names}"
        )
    if len(matching) > 1:
        raise RuntimeError(f"Expected one {EXPECTED_MODEL_PREFIX} model, found: {matching}")
    return predictor, matching[0], adapters


def validate_saved_run_config(
    run_config_path: Path,
    prediction_length: int,
    context_length: int,
    inference_batch_size: int,
    mode: str,
    covariates: list[str],
    n_turbines: int,
) -> dict[str, Any]:
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Saved fine-tuning run config does not exist: {run_config_path}")
    with run_config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    chronos = config.get("hyperparameters", {}).get("Chronos2", {})
    expected = {
        "mode": mode,
        "prediction_length": prediction_length,
        "context_length": context_length,
        "n_turbines": n_turbines,
    }
    mismatches = {
        key: {"expected": value, "saved": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if chronos.get("batch_size") != inference_batch_size:
        mismatches["inference_batch_size"] = {
            "expected": inference_batch_size,
            "saved": chronos.get("batch_size"),
        }
    if chronos.get("fine_tune_mode") != "lora":
        mismatches["fine_tune_mode"] = {
            "expected": "lora",
            "saved": chronos.get("fine_tune_mode"),
        }
    if chronos.get("disable_known_covariates") is not True:
        mismatches["disable_known_covariates"] = {
            "expected": True,
            "saved": chronos.get("disable_known_covariates"),
        }
    if list(config.get("covariates", [])) != covariates:
        mismatches["covariates"] = {
            "expected": covariates,
            "saved": config.get("covariates"),
        }
    if list(config.get("quantile_levels", [])) != DEFAULT_QUANTILES:
        mismatches["quantile_levels"] = {
            "expected": DEFAULT_QUANTILES,
            "saved": config.get("quantile_levels"),
        }
    if mismatches:
        raise ValueError(f"Saved predictor configuration does not match evaluation: {mismatches}")
    return config


class AutoGluonChronos2Adapter:
    """Expose a saved TimeSeriesPredictor through the zero-shot predict_df interface."""

    def __init__(self, predictor: Any, dataframe_class: Any, model_name: str) -> None:
        self.predictor = predictor
        self.dataframe_class = dataframe_class
        self.model_name = model_name

    def predict_df(
        self,
        context_df: pd.DataFrame,
        future_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if future_df is not None:
            raise ValueError("Measured SDWPF covariates must not be passed as known-future covariates.")

        prediction_length = int(kwargs["prediction_length"])
        saved_prediction_length = int(self.predictor.prediction_length)
        if prediction_length != saved_prediction_length:
            raise ValueError(
                f"Requested prediction length {prediction_length} does not match the saved "
                f"predictor length {saved_prediction_length}."
            )

        data = self.dataframe_class.from_data_frame(
            context_df,
            id_column=str(kwargs.get("id_column", "id")),
            timestamp_column=str(kwargs.get("timestamp_column", "timestamp")),
        )
        forecast = self.predictor.predict(data, model=self.model_name)
        normalized = pd.DataFrame(forecast).reset_index()
        if "item_id" in normalized.columns and "id" not in normalized.columns:
            normalized = normalized.rename(columns={"item_id": "id"})
        return normalized


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    run_dir = Path(args.predictor_path).parent
    predictions = args.output or run_dir / "test_predictions.csv"
    metrics = args.metrics_output or run_dir / "test_metrics.csv"
    metadata = args.metadata_output or run_dir / "test_prediction_metadata.json"
    return Path(predictions), Path(metrics), Path(metadata)


def run(
    args: argparse.Namespace,
    autogluon_classes: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    predictor_path = Path(args.predictor_path)
    run_dir = predictor_path.parent
    split_manifest_path = Path(args.split_manifest or run_dir / "resolved_split_manifest.json")
    run_config_path = Path(args.run_config or run_dir / "run_config.json")
    predictions_path, metrics_path, metadata_path = _resolve_output_paths(args)

    covariates = parse_csv_list(args.covariates) if args.mode == "multivariate" else []
    horizons = parse_int_list(args.horizons)
    if max(horizons) != args.prediction_length:
        raise ValueError("The maximum evaluated horizon must equal --prediction-length.")
    if args.max_turbines != 5:
        raise ValueError("This smoke predictor must be evaluated on the same five turbines used to train it.")

    split_config = load_benchmark_config(Path(args.split_config))
    data = validate_and_normalize_data(
        read_table(Path(args.input)),
        mode=args.mode,
        covariates=covariates,
        frequency=str(split_config["frequency"]),
    )
    manifest = load_split_manifest(split_manifest_path)
    validate_split_manifest(manifest, data["timestamp"], split_config)
    frames = prepare_finetune_frames(
        data,
        manifest=manifest,
        mode=args.mode,
        covariates=covariates,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        max_turbines=args.max_turbines,
    )
    selected_ids = frames.turbine_ids
    if len(selected_ids) != 5:
        raise ValueError(
            f"Expected the five smoke-training turbines, but selection produced {selected_ids}."
        )
    validate_saved_run_config(
        run_config_path,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        inference_batch_size=args.inference_batch_size,
        mode=args.mode,
        covariates=covariates,
        n_turbines=len(selected_ids),
    )

    if autogluon_classes is None:
        dataframe_class, predictor_class = _load_autogluon()
    else:
        dataframe_class, predictor_class = autogluon_classes
    predictor, model_name, adapter_paths = load_verified_predictor(
        predictor_path,
        predictor_class=predictor_class,
    )
    if int(predictor.prediction_length) != args.prediction_length:
        raise ValueError(
            f"Saved predictor prediction_length={predictor.prediction_length}, "
            f"expected {args.prediction_length}."
        )

    test_start, test_end = test_period(manifest)
    selected_data = data[data["id"].isin(selected_ids)].copy()
    adapter = AutoGluonChronos2Adapter(predictor, dataframe_class, model_name)
    predictions = run_rolling_forecasts(
        pipeline=adapter,
        df=selected_data,
        mode=args.mode,
        covariates=covariates,
        horizons=horizons,
        context_length=args.context_length,
        stride=args.stride,
        quantile_levels=DEFAULT_QUANTILES,
        model_id=model_name,
        max_windows_per_turbine=args.max_windows_per_turbine,
        allow_future_covariates=False,
        test_start=test_start,
        test_end=test_end,
    )
    predictions["mode"] = f"{args.mode}_lora"
    metrics = evaluate_predictions(
        predictions,
        include_imputed_targets=args.include_imputed_targets,
        rated_capacity_kw=args.rated_capacity_kw,
    )

    write_predictions(predictions, predictions_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    metadata = {
        "predictor_path": str(predictor_path.resolve()),
        "model_name": model_name,
        "adapter_checkpoints": [str(path.resolve()) for path in adapter_paths],
        "input": str(Path(args.input).resolve()),
        "split_config": str(Path(args.split_config).resolve()),
        "split_manifest": str(split_manifest_path.resolve()),
        "selected_turbine_ids": selected_ids,
        "n_turbines": len(selected_ids),
        "mode": args.mode,
        "result_mode": f"{args.mode}_lora",
        "past_covariates": covariates,
        "known_future_covariates": [],
        "prediction_length": args.prediction_length,
        "context_length": args.context_length,
        "horizons": horizons,
        "quantiles": DEFAULT_QUANTILES,
        "inference_batch_size": args.inference_batch_size,
        "stride": args.stride,
        "max_windows_per_turbine": args.max_windows_per_turbine,
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "include_imputed_targets": bool(args.include_imputed_targets),
        "n_prediction_rows": len(predictions),
        "predictions": str(predictions_path.resolve()),
        "metrics": str(metrics_path.resolve()),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-path", required=True, type=Path)
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--split-config", default=DEFAULT_SPLIT_CONFIG, type=Path)
    parser.add_argument("--split-manifest", default=None, type=Path)
    parser.add_argument("--run-config", default=None, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--metrics-output", default=None, type=Path)
    parser.add_argument("--metadata-output", default=None, type=Path)
    parser.add_argument("--mode", choices=["univariate", "multivariate"], default="multivariate")
    parser.add_argument("--covariates", default=",".join(DEFAULT_COVARIATES))
    parser.add_argument("--prediction-length", default=72, type=int)
    parser.add_argument("--context-length", default=168, type=int)
    parser.add_argument("--horizons", nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--inference-batch-size", default=64, type=int)
    parser.add_argument("--stride", default=24, type=int)
    parser.add_argument("--max-turbines", default=5, type=int)
    parser.add_argument("--max-windows-per-turbine", default=None, type=int)
    parser.add_argument("--rated-capacity-kw", default=1500.0, type=float)
    parser.add_argument("--include-imputed-targets", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metadata = run(args)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
