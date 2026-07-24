"""Controlled Chronos-2 full-fine-tuning challenger for Foshan PV."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.evaluation.foshan_benchmark import (
    PREDICTION_COLUMNS,
    build_forecast_window,
    chronos_rows_for_window,
    evaluate_common_scored_timestamps,
    evaluate_foshan_predictions,
    period_origins,
    select_may_configurations,
    validate_processed_table,
)
from src.training.chronos_finetune import build_chronos2_hyperparameters
from src.training.foshan_chronos_finetune import (
    FoshanFineTuneFrames,
    _frozen_zero_shot_comparison_rows,
    _load_autogluon,
    _normalize_autogluon_forecast,
    _to_timeseries_frame,
    _window_context_frame,
    _window_future_frame,
    prepare_foshan_finetune_frames,
    resolve_model_source,
)
from src.utils.runtime import git_commit, git_is_dirty


DEFAULT_INPUT = Path(
    "results/zero_shot/foshan_chronos2/processed_foshan_15min.parquet"
)
DEFAULT_CONFIG = Path("configs/foshan_chronos2_full_finetune.json")
DEFAULT_ZERO_SHOT_DIR = Path("results/zero_shot/foshan_chronos2")
DEFAULT_LORA_RUN_DIR = Path(
    "results/fine_tune/foshan_chronos2_lora_20260723T225913Z"
)
DEFAULT_OUTPUT_DIR = Path("results/full_fine_tune/foshan_chronos2_full_manual")
FULL_RESULTS_ROOT = Path("results/full_fine_tune")
PV_VARIANT = {
    "name": "pv_calendar",
    "targets": ["pv_kw"],
    "selection_target": "pv_kw",
}
METRIC_COLUMNS = [
    "mae",
    "rmse",
    "wape",
    "mase",
    "bias",
    "pv_active_n_scored",
    "pv_active_mae",
    "pv_active_rmse",
    "pv_active_wape",
    "pinball_p10",
    "pinball_p50",
    "pinball_p90",
    "mean_pinball_loss",
    "p10_p90_coverage",
    "mean_interval_width",
    "n_scored",
    "n_excluded_missing",
]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_full_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "model_id",
        "site_id",
        "target",
        "frequency",
        "timezone",
        "prediction_length",
        "context_length",
        "quantile_levels",
        "inference_batch_size",
        "causal_fill_limit",
        "pv_capacity_kw",
        "pv_active_threshold_kw",
        "seed",
        "calendar_covariates",
        "train_period",
        "selection_period",
        "engineering_test_period",
        "smoke_candidate",
        "search_candidates",
        "oom_batch_sizes",
        "bootstrap_samples",
        "retention",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Foshan full-tuning config is missing keys: {missing}")
    if config["model_id"] != "amazon/chronos-2":
        raise ValueError("Foshan full tuning must use amazon/chronos-2.")
    if config["target"] != "pv_kw":
        raise ValueError("Foshan full tuning must target pv_kw only.")
    if int(config["prediction_length"]) != 96:
        raise ValueError("prediction_length must remain 96.")
    if int(config["context_length"]) != 672:
        raise ValueError("context_length must remain 672.")
    if str(config["frequency"]) != "15min":
        raise ValueError("frequency must remain 15min.")
    if [float(value) for value in config["quantile_levels"]] != [0.1, 0.5, 0.9]:
        raise ValueError("Quantiles must remain P10/P50/P90.")
    if int(config["causal_fill_limit"]) != 3:
        raise ValueError("causal_fill_limit must remain 3.")
    if [int(value) for value in config["oom_batch_sizes"]] != [4, 2, 1]:
        raise ValueError("OOM batch-size fallback must remain 4, 2, 1.")
    expected_candidates = {
        (1e-6, 50),
        (3e-6, 50),
        (1e-6, 100),
        (3e-6, 100),
    }
    actual_candidates = {
        (float(candidate["learning_rate"]), int(candidate["steps"]))
        for candidate in config["search_candidates"]
    }
    if actual_candidates != expected_candidates or len(config["search_candidates"]) != 4:
        raise ValueError("Full-tuning search candidates differ from the fixed protocol.")
    train_end = pd.Timestamp(config["train_period"]["end_exclusive"])
    selection_start = pd.Timestamp(config["selection_period"]["start"])
    selection_end = pd.Timestamp(config["selection_period"]["end_exclusive"])
    june_start = pd.Timestamp(config["engineering_test_period"]["start"])
    if train_end != selection_start or selection_end != june_start:
        raise ValueError("March-April, May, and June periods must be contiguous.")
    return config


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["test_period"] = dict(config["engineering_test_period"])
    return result


def prepare_full_finetune_frames(
    table: pd.DataFrame,
    config: dict[str, Any],
) -> FoshanFineTuneFrames:
    frames = prepare_foshan_finetune_frames(
        table,
        _training_config(config),
        PV_VARIANT,
    )
    if frames.targets != ["pv_kw"] or set(frames.item_target_map.values()) != {"pv_kw"}:
        raise AssertionError("Full fine-tuning received a non-PV target.")
    return frames


def validate_output_isolation(
    output_dir: Path,
    zero_shot_dir: Path,
    lora_run_dir: Path,
    results_root: Path = FULL_RESULTS_ROOT,
) -> Path:
    output = output_dir.expanduser().resolve()
    allowed_root = results_root.expanduser().resolve()
    if output == allowed_root or allowed_root not in output.parents:
        raise ValueError(
            f"Full-tuning output must be a run directory under {allowed_root}."
        )
    for label, protected in (
        ("zero-shot", zero_shot_dir),
        ("LoRA", lora_run_dir),
    ):
        protected_path = protected.expanduser().resolve()
        if (
            output == protected_path
            or output in protected_path.parents
            or protected_path in output.parents
        ):
            raise ValueError(f"Full-tuning output overlaps protected {label} outputs.")
    return output


def build_full_hyperparameters(
    model_source: str,
    config: dict[str, Any],
    candidate: dict[str, Any],
    batch_size: int,
    dataloader_num_workers: int = 0,
) -> dict[str, dict[str, Any]]:
    return build_chronos2_hyperparameters(
        model_id=model_source,
        mode="univariate",
        prediction_length=int(config["prediction_length"]),
        context_length=int(config["context_length"]),
        steps=int(candidate["steps"]),
        learning_rate=float(candidate["learning_rate"]),
        batch_size=batch_size,
        inference_batch_size=int(config["inference_batch_size"]),
        seed=int(candidate["seed"]),
        dataloader_num_workers=dataloader_num_workers,
        bf16=True,
        fp16=False,
        disable_known_covariates=False,
        disable_past_covariates=True,
        fine_tune_mode="full",
        model_name_suffix="Full",
    )


def parameter_counts(model: Any, require_all_trainable: bool = True) -> dict[str, int]:
    parameters = list(model.parameters())
    if not parameters:
        raise RuntimeError("Loaded Chronos-2 model exposes no parameters.")
    total = sum(int(parameter.numel()) for parameter in parameters)
    trainable = sum(
        int(parameter.numel()) for parameter in parameters if parameter.requires_grad
    )
    frozen = total - trainable
    if require_all_trainable and frozen:
        raise RuntimeError(
            "Full fine-tuning checkpoint has frozen base-model parameters: "
            f"trainable={trainable}, total={total}, frozen={frozen}."
        )
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "frozen_parameters": frozen,
    }


def inspect_loaded_full_model(predictor: Any, model_name: str) -> dict[str, Any]:
    if not model_name.startswith("Chronos2Full"):
        raise RuntimeError(f"Expected a Chronos2Full model, found {model_name!r}.")
    trainer = getattr(predictor, "_trainer", None)
    if trainer is None or not hasattr(trainer, "load_model"):
        raise RuntimeError("Loaded predictor does not expose its trained model.")
    ag_model = trainer.load_model(model_name)
    hyperparameters = ag_model.get_hyperparameters()
    if hyperparameters.get("fine_tune") is not True:
        raise RuntimeError("Loaded Chronos2Full model is not marked as fine-tuned.")
    if hyperparameters.get("fine_tune_mode") != "full":
        raise RuntimeError(
            "Loaded model is not a full-fine-tuning checkpoint: "
            f"{hyperparameters.get('fine_tune_mode')!r}."
        )
    ag_model.load_model_pipeline()
    pipeline = getattr(ag_model, "_model_pipeline", None)
    core_model = getattr(pipeline, "model", None)
    if core_model is None:
        core_model = getattr(pipeline, "_model", None)
    if core_model is None:
        raise RuntimeError("Could not locate the Chronos-2 base model after reload.")
    counts = parameter_counts(core_model, require_all_trainable=True)
    ag_model._model_pipeline = None
    del core_model, pipeline, ag_model
    gc.collect()
    return {
        **counts,
        "fine_tune": True,
        "fine_tune_mode": "full",
        "model_name": model_name,
    }


def load_full_predictor_for_evaluation(
    predictor_class: Any,
    predictor_path: Path,
    model_name: str,
) -> tuple[Any, dict[str, Any]]:
    predictor = predictor_class.load(str(predictor_path))
    model_names = [str(name) for name in predictor.model_names()]
    if model_name not in model_names:
        raise RuntimeError(
            f"Selected full model {model_name!r} is absent after reload: {model_names}."
        )
    inspection = inspect_loaded_full_model(predictor, model_name)
    return predictor, inspection


def find_full_checkpoint(predictor_path: Path, model_name: str) -> dict[str, Any]:
    model_root = predictor_path / "models" / model_name
    checkpoints = sorted(model_root.rglob("fine-tuned-ckpt"))
    checkpoints = [path for path in checkpoints if path.is_dir()]
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected one full fine-tuned checkpoint under {model_root}, "
            f"found {checkpoints}."
        )
    checkpoint = checkpoints[0]
    adapter_files = list(checkpoint.rglob("adapter_model.safetensors"))
    if adapter_files:
        raise RuntimeError(f"Full checkpoint unexpectedly contains LoRA adapters: {adapter_files}")
    weight_files = [
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in checkpoint.rglob(pattern)
        if path.is_file()
    ]
    if not weight_files:
        raise RuntimeError(f"Full checkpoint contains no model weights: {checkpoint}")
    files = [path for path in checkpoint.rglob("*") if path.is_file()]
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": sum(path.stat().st_size for path in files),
        "checkpoint_weight_files": [str(path) for path in sorted(weight_files)],
    }


def full_gpu_preflight() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is not installed; full fine-tuning requires the configured "
            "CUDA environment on the RTX 4090 server."
        ) from error

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 before full fine-tuning.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to emulate full fine-tuning.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected one visible GPU, found {torch.cuda.device_count()}."
        )
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 4090" not in gpu_name:
        raise RuntimeError(
            f"Foshan full fine-tuning requires one RTX 4090; detected {gpu_name}."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible RTX 4090/PyTorch stack does not support BF16.")
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": gpu_name,
        "gpu_total_vram_bytes": int(properties.total_memory),
        "bf16_supported": True,
        "precision": "bf16",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def validate_model_source_for_training(model_source: str) -> None:
    path = Path(model_source).expanduser()
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"Local Chronos-2 directory lacks config.json: {path}"
            )
        weights = list(path.glob("*.safetensors")) + list(
            path.glob("pytorch_model*.bin")
        )
        if not weights:
            raise FileNotFoundError(
                f"Local Chronos-2 directory contains no model weights: {path}"
            )
        return
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) == "1"
    if offline:
        raise FileNotFoundError(
            "Offline mode requires --model-path or CHRONOS_MODEL_PATH pointing "
            "to a complete local Chronos-2 directory."
        )


def _is_cuda_oom(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        name = type(current).__name__.lower()
        message = str(current).lower()
        if "outofmemory" in name or "out of memory" in message or "cuda oom" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def fit_full_candidate(
    frames: FoshanFineTuneFrames,
    config: dict[str, Any],
    candidate: dict[str, Any],
    batch_size: int,
    model_source: str,
    attempt_dir: Path,
    dataloader_num_workers: int = 0,
    autogluon_classes: tuple[Any, Any] | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    predictor_path = attempt_dir / "predictor"
    if predictor_path.exists():
        raise FileExistsError(f"Refusing to overwrite predictor: {predictor_path}")
    dataframe_class, predictor_class = autogluon_classes or _load_autogluon()
    train_data = _to_timeseries_frame(dataframe_class, frames.train)
    tuning_data = _to_timeseries_frame(dataframe_class, frames.tuning)
    hyperparameters = build_full_hyperparameters(
        model_source,
        config,
        candidate,
        batch_size,
        dataloader_num_workers=dataloader_num_workers,
    )
    predictor = predictor_class(
        path=str(predictor_path),
        prediction_length=int(config["prediction_length"]),
        target="target",
        known_covariates_names=list(config["calendar_covariates"]),
        quantile_levels=[float(value) for value in config["quantile_levels"]],
        eval_metric="WQL",
        freq=str(config["frequency"]),
    )

    torch_module: Any | None = None
    if autogluon_classes is None:
        import torch

        torch_module = torch
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    predictor.fit(
        train_data=train_data,
        tuning_data=tuning_data,
        hyperparameters=hyperparameters,
        enable_ensemble=False,
        random_seed=int(candidate["seed"]),
        refit_full=False,
        skip_model_selection=False,
    )
    training_runtime = time.monotonic() - started
    model_names = [str(name) for name in predictor.model_names()]
    trained = [name for name in model_names if name.startswith("Chronos2Full")]
    if len(trained) != 1:
        raise RuntimeError(
            f"Expected exactly one trained Chronos2Full model, found {model_names}."
        )

    loaded, inspection = load_full_predictor_for_evaluation(
        predictor_class,
        predictor_path,
        trained[0],
    )
    loaded_names = [str(name) for name in loaded.model_names()]
    if loaded_names != model_names:
        raise RuntimeError(
            f"Reloaded model names differ: trained={model_names}, loaded={loaded_names}."
        )
    checkpoint = find_full_checkpoint(predictor_path, trained[0])
    stats = {
        "training_runtime_seconds": training_runtime,
        "peak_allocated_gpu_bytes": (
            int(torch_module.cuda.max_memory_allocated(0))
            if torch_module is not None
            else None
        ),
        "peak_reserved_gpu_bytes": (
            int(torch_module.cuda.max_memory_reserved(0))
            if torch_module is not None
            else None
        ),
        "trained_model_name": trained[0],
        "predictor_path": str(predictor_path),
        "hyperparameters": hyperparameters,
        **inspection,
        **checkpoint,
    }
    return loaded, trained[0], stats


def run_full_origins(
    predictor: Any,
    dataframe_class: Any,
    model_name: str,
    table: pd.DataFrame,
    config: dict[str, Any],
    candidate_name: str,
    origins: list[pd.Timestamp],
    split_name: str,
    run_id: str,
    model_source: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], float]:
    calendar = [str(column) for column in config["calendar_covariates"]]
    site_id = str(config["site_id"])
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    started = time.monotonic()
    for issue_time in origins:
        window, reason = build_forecast_window(
            table=table,
            issue_time=issue_time,
            targets=["pv_kw"],
            context_length=int(config["context_length"]),
            prediction_length=int(config["prediction_length"]),
            known_future_covariates=calendar,
            causal_fill_limit=int(config["causal_fill_limit"]),
            frequency=str(config["frequency"]),
        )
        if window is None:
            skipped.append(
                {
                    "split": split_name,
                    "candidate": candidate_name,
                    "issue_time": issue_time.isoformat(),
                    "reason": reason,
                }
            )
            continue
        context = _window_context_frame(window, ["pv_kw"], calendar, site_id)
        future = _window_future_frame(window, ["pv_kw"], calendar, site_id)
        forbidden = {"pv_kw", "net_grid_kw", "pv_kw_raw", "net_grid_kw_raw", "target"}
        if forbidden.intersection(future.columns):
            raise AssertionError("Measured targets entered full-tuning future covariates.")
        context_data = _to_timeseries_frame(dataframe_class, context)
        future_data = _to_timeseries_frame(dataframe_class, future)
        forecast = predictor.predict(
            context_data,
            known_covariates=future_data,
            model=model_name,
        )
        normalized = _normalize_autogluon_forecast(
            forecast,
            item_target_map={site_id: "pv_kw"},
            site_id=site_id,
        )
        rows = chronos_rows_for_window(
            forecast_df=normalized,
            window=window,
            targets=["pv_kw"],
            split_name=split_name,
            run_id=run_id,
            model_name=f"chronos2_full_pv_calendar_{candidate_name}",
            model_id=f"{model_source}+full:{candidate_name}",
            context_length=int(config["context_length"]),
            known_future_covariates=calendar,
            pv_capacity_kw=float(config["pv_capacity_kw"]),
            site_id=site_id,
        )
        frames.append(rows)
    runtime = time.monotonic() - started
    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS), skipped, runtime
    return pd.concat(frames, ignore_index=True), skipped, runtime


def forecast_key_set(predictions: pd.DataFrame) -> set[tuple[str, str, int]]:
    physical = predictions[
        predictions["postprocessing"] == "physical_clip_0_1700"
    ].copy()
    if physical.empty:
        raise ValueError("Candidate has no postprocessed PV predictions.")
    issue = pd.to_datetime(physical["issue_time"], errors="raise", utc=True)
    target = pd.to_datetime(physical["target_time"], errors="raise", utc=True)
    return set(
        zip(
            issue.astype(str),
            target.astype(str),
            physical["horizon_step"].astype(int),
        )
    )


def validate_identical_candidate_keys(
    predictions_by_candidate: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if not predictions_by_candidate:
        raise ValueError("No successful candidate predictions were provided.")
    key_sets = {
        name: forecast_key_set(predictions)
        for name, predictions in predictions_by_candidate.items()
    }
    first_name, first_keys = next(iter(key_sets.items()))
    mismatches = {
        name: {
            "missing_vs_reference": len(first_keys - keys),
            "extra_vs_reference": len(keys - first_keys),
        }
        for name, keys in key_sets.items()
        if keys != first_keys
    }
    if mismatches:
        raise ValueError(
            "Full-tuning candidates have different May forecast keys; "
            f"reference={first_name}, mismatches={mismatches}."
        )
    serialized = "\n".join("|".join(map(str, key)) for key in sorted(first_keys))
    return {
        "reference_candidate": first_name,
        "candidate_count": len(key_sets),
        "common_key_count": len(first_keys),
        "forecast_key_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def select_full_candidate(
    may_metrics: pd.DataFrame,
    candidate_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if may_metrics.empty or not may_metrics["split"].eq("may_2026_selection").all():
        raise ValueError("Full-tuning candidate selection may use May metrics only.")
    selection = select_may_configurations(may_metrics)
    model_name = str(selection["targets"]["pv_kw"]["model_name"])
    matches = [
        record for record in candidate_records if record["model_name"] == model_name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Could not map selected full model to one candidate: {model_name}")
    selected = dict(matches[0])
    selected.update(selection["targets"]["pv_kw"])
    selected["selected_on"] = "may_2026_selection"
    selected["selection_metric"] = "postprocessed_pv_wape"
    selected["tie_break_metric"] = "pv_active_mae"
    return selected


def _lora_search_dir(lora_run_dir: Path) -> Path:
    search = lora_run_dir / "search"
    return search if search.is_dir() else lora_run_dir


def load_frozen_reference_rows(
    zero_shot_dir: Path,
    lora_run_dir: Path,
) -> pd.DataFrame:
    zero_and_baselines = _frozen_zero_shot_comparison_rows(zero_shot_dir)
    search_dir = _lora_search_dir(lora_run_dir)
    selection_path = search_dir / "selected_configuration.json"
    predictions_path = search_dir / "june_predictions.csv"
    if not selection_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(
            f"Frozen LoRA selection or June predictions are missing under {search_dir}."
        )
    selection = _read_json(selection_path)
    predictions = pd.read_csv(predictions_path)
    lora = predictions[
        (predictions["target"] == "pv_kw")
        & (predictions["split"] == "june_2026_test")
        & (predictions["model_name"] == selection["model_name"])
        & (predictions["postprocessing"] == "physical_clip_0_1700")
    ].copy()
    if lora.empty:
        raise ValueError("Frozen selected LoRA June rows are missing.")
    return pd.concat(
        [zero_and_baselines, lora[PREDICTION_COLUMNS]],
        ignore_index=True,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_artifact_manifest(
    zero_shot_dir: Path,
    lora_run_dir: Path,
) -> dict[str, str]:
    lora_search = _lora_search_dir(lora_run_dir)
    paths = [
        zero_shot_dir / "predictions_long.csv",
        zero_shot_dir / "selected_configuration.json",
        lora_search / "june_predictions.csv",
        lora_search / "selected_configuration.json",
        lora_search / "summary.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected comparison artifacts are missing: {missing}")
    return {str(path.resolve()): _file_sha256(path) for path in paths}


def assert_protected_artifacts_unchanged(manifest: dict[str, str]) -> None:
    changed = [
        path
        for path, expected_hash in manifest.items()
        if not Path(path).is_file() or _file_sha256(Path(path)) != expected_hash
    ]
    if changed:
        raise RuntimeError(f"Protected zero-shot or LoRA artifacts changed: {changed}")


def comparison_improvements(
    common_metrics: pd.DataFrame,
    full_model_name: str,
) -> dict[str, Any]:
    full = common_metrics[common_metrics["model_name"] == full_model_name]
    zero = common_metrics[
        common_metrics["model_name"].astype(str).str.startswith("chronos2_")
        & ~common_metrics["model_name"].astype(str).str.startswith("chronos2_lora_")
        & ~common_metrics["model_name"].astype(str).str.startswith("chronos2_full_")
    ]
    lora = common_metrics[
        common_metrics["model_name"].astype(str).str.startswith("chronos2_lora_")
    ]
    baselines = common_metrics[common_metrics["model_id"] == "causal_baseline"]
    if len(full) != 1 or len(zero) != 1 or len(lora) != 1 or baselines.empty:
        raise ValueError(
            "Common metrics require one full model, one zero-shot model, one LoRA "
            "model, and at least one causal baseline."
        )
    full_row = full.iloc[0]
    references = {
        "zero_shot": zero.iloc[0],
        "lora": lora.iloc[0],
        "best_baseline": baselines.sort_values(
            ["wape", "pv_active_mae", "model_name"],
            kind="mergesort",
        ).iloc[0],
    }
    metrics = [
        "mae",
        "rmse",
        "wape",
        "mase",
        "pv_active_mae",
        "pv_active_rmse",
        "pv_active_wape",
        "mean_pinball_loss",
    ]
    result: dict[str, Any] = {"full_model": full_model_name, "references": {}}
    for label, reference in references.items():
        improvements: dict[str, float | None] = {}
        for metric in metrics:
            reference_value = float(reference[metric])
            improvements[metric] = (
                None
                if not math.isfinite(reference_value) or reference_value == 0
                else 100.0
                * (reference_value - float(full_row[metric]))
                / reference_value
            )
        result["references"][label] = {
            "model_name": str(reference["model_name"]),
            "percent_improvement": improvements,
        }
    return result


def paired_daily_errors_vs_zero_shot(
    common_predictions: pd.DataFrame,
    full_model_name: str,
    timezone: str,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    physical = common_predictions[
        common_predictions["postprocessing"] == "physical_clip_0_1700"
    ].copy()
    zero_names = sorted(
        name
        for name in physical["model_name"].astype(str).unique()
        if name.startswith("chronos2_")
        and not name.startswith("chronos2_lora_")
        and not name.startswith("chronos2_full_")
    )
    if len(zero_names) != 1:
        raise ValueError(f"Expected one zero-shot model for daily pairing: {zero_names}")
    zero_name = zero_names[0]
    paired = physical[
        physical["model_name"].isin([full_model_name, zero_name])
    ].copy()
    paired["target_time"] = pd.to_datetime(
        paired["target_time"], errors="raise", utc=True
    )
    paired["date"] = paired["target_time"].dt.tz_convert(timezone).dt.date
    paired["absolute_error"] = (
        pd.to_numeric(paired["y_true"], errors="raise")
        - pd.to_numeric(paired["p50"], errors="raise")
    ).abs()
    key_columns = ["issue_time", "target_time", "horizon_step"]
    if paired.duplicated([*key_columns, "model_name"]).any():
        raise ValueError("Daily paired comparison contains duplicate forecast keys.")
    rows: list[dict[str, Any]] = []
    for day, group in paired.groupby("date", sort=True):
        by_model = {
            name: model_group
            for name, model_group in group.groupby("model_name", sort=False)
        }
        if set(by_model) != {full_model_name, zero_name}:
            raise ValueError(f"Daily pairing is incomplete for {day}.")
        full = by_model[full_model_name]
        zero = by_model[zero_name]
        full_keys = set(full[key_columns].itertuples(index=False, name=None))
        zero_keys = set(zero[key_columns].itertuples(index=False, name=None))
        if full_keys != zero_keys:
            raise ValueError(f"Daily forecast keys differ for {day}.")
        actual_sum = float(pd.to_numeric(full["y_true"]).abs().sum())
        full_mae = float(full["absolute_error"].mean())
        zero_mae = float(zero["absolute_error"].mean())
        full_wape = (
            float(full["absolute_error"].sum()) / actual_sum
            if actual_sum > 0
            else math.nan
        )
        zero_wape = (
            float(zero["absolute_error"].sum()) / actual_sum
            if actual_sum > 0
            else math.nan
        )
        rows.append(
            {
                "date": day.isoformat(),
                "n_scored": len(full),
                "full_mae": full_mae,
                "zero_shot_mae": zero_mae,
                "mae_difference_full_minus_zero": full_mae - zero_mae,
                "full_wape": full_wape,
                "zero_shot_wape": zero_wape,
                "wape_difference_full_minus_zero": full_wape - zero_wape,
            }
        )
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("No daily paired errors were produced.")
    rng = np.random.default_rng(seed)

    def bootstrap(column: str) -> dict[str, float]:
        values = pd.to_numeric(daily[column], errors="coerce").dropna().to_numpy()
        if len(values) < 2:
            raise ValueError(f"Not enough daily values to bootstrap {column}.")
        indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        means = values[indices].mean(axis=1)
        return {
            "mean_difference": float(values.mean()),
            "ci_lower_95": float(np.quantile(means, 0.025)),
            "ci_upper_95": float(np.quantile(means, 0.975)),
            "n_days": int(len(values)),
            "bootstrap_samples": int(bootstrap_samples),
        }

    summary = {
        "full_model": full_model_name,
        "zero_shot_model": zero_name,
        "difference_definition": "full_minus_zero_shot",
        "daily_mae": bootstrap("mae_difference_full_minus_zero"),
        "daily_wape": bootstrap("wape_difference_full_minus_zero"),
        "seed": seed,
    }
    return daily, summary


def retention_decision(
    common_metrics: pd.DataFrame,
    full_model_name: str,
    bootstrap: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    full = common_metrics[common_metrics["model_name"] == full_model_name]
    zero = common_metrics[
        common_metrics["model_name"].astype(str).str.startswith("chronos2_")
        & ~common_metrics["model_name"].astype(str).str.startswith("chronos2_lora_")
        & ~common_metrics["model_name"].astype(str).str.startswith("chronos2_full_")
    ]
    if len(full) != 1 or len(zero) != 1:
        raise ValueError("Retention decision requires one full and one zero-shot row.")
    full_row = full.iloc[0]
    zero_row = zero.iloc[0]
    thresholds = config["retention"]
    zero_wape = float(zero_row["wape"])
    full_wape = float(full_row["wape"])
    relative_wape_improvement = (zero_wape - full_wape) / zero_wape
    pv_active_degradation = (
        float(full_row["pv_active_mae"]) - float(zero_row["pv_active_mae"])
    ) / float(zero_row["pv_active_mae"])
    pinball_degradation = (
        float(full_row["mean_pinball_loss"])
        - float(zero_row["mean_pinball_loss"])
    ) / float(zero_row["mean_pinball_loss"])
    full_coverage_error = abs(float(full_row["p10_p90_coverage"]) - 0.8)
    zero_coverage_error = abs(float(zero_row["p10_p90_coverage"]) - 0.8)
    coverage_error_degradation = full_coverage_error - zero_coverage_error
    checks = {
        "beats_zero_shot_wape": full_wape < zero_wape,
        "at_least_one_percent_relative_wape_improvement": (
            relative_wape_improvement
            >= float(thresholds["minimum_relative_wape_improvement"])
        ),
        "daily_wape_bootstrap_ci_excludes_zero_on_improving_side": (
            float(bootstrap["daily_wape"]["ci_upper_95"]) < 0
        ),
        "pv_active_mae_not_materially_degraded": (
            pv_active_degradation
            <= float(thresholds["maximum_pv_active_mae_relative_degradation"])
        ),
        "mean_pinball_not_materially_degraded": (
            pinball_degradation
            <= float(thresholds["maximum_mean_pinball_relative_degradation"])
        ),
        "coverage_calibration_not_materially_degraded": (
            coverage_error_degradation
            <= float(
                thresholds["maximum_coverage_error_absolute_degradation"]
            )
        ),
    }
    retain_full = all(checks.values())
    return {
        "engineering_test_only": True,
        "june_is_not_a_pristine_untouched_test": True,
        "next_pristine_evaluation": "July 2026 or later newly acquired data",
        "retain_full_model_as_forecasting_challenger": retain_full,
        "recommended_forecasting_model": (
            full_model_name if retain_full else str(zero_row["model_name"])
        ),
        "relative_wape_improvement": relative_wape_improvement,
        "pv_active_mae_relative_degradation": pv_active_degradation,
        "mean_pinball_relative_degradation": pinball_degradation,
        "coverage_error_absolute_degradation": coverage_error_degradation,
        "checks": checks,
        "no_revenue_claim": True,
    }


def _candidate_contract(
    candidate: dict[str, Any],
    batch_size: int | None = None,
) -> dict[str, Any]:
    result = {
        "candidate": str(candidate["name"]),
        "fine_tune_mode": "full",
        "target": "pv_kw",
        "learning_rate": float(candidate["learning_rate"]),
        "steps": int(candidate["steps"]),
        "seed": int(candidate["seed"]),
    }
    if batch_size is not None:
        result["batch_size"] = int(batch_size)
    return result


def _next_attempt_dir(candidate_dir: Path, batch_size: int) -> Path:
    base = candidate_dir / f"attempt_batch{batch_size}"
    if not base.exists():
        return base
    index = 1
    while (candidate_dir / f"attempt_batch{batch_size}_restart{index}").exists():
        index += 1
    return candidate_dir / f"attempt_batch{batch_size}_restart{index}"


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _cuda_peak_memory() -> tuple[int | None, int | None]:
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(0)),
        int(torch.cuda.max_memory_reserved(0)),
    )


def run_candidate_with_oom_fallback(
    frames: FoshanFineTuneFrames,
    table: pd.DataFrame,
    config: dict[str, Any],
    candidate: dict[str, Any],
    model_source: str,
    candidate_dir: Path,
    origins: list[pd.Timestamp],
    split_name: str,
    run_id: str,
    dataloader_num_workers: int,
    autogluon_classes: tuple[Any, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result_path = candidate_dir / "candidate_result.json"
    predictions_path = candidate_dir / f"{split_name}_predictions.csv"
    if result_path.is_file():
        result = _read_json(result_path)
        expected = _candidate_contract(candidate)
        for key, value in expected.items():
            if result.get(key) != value:
                raise ValueError(
                    f"Resumed candidate contract differs for {key}: "
                    f"{result.get(key)!r} != {value!r}."
                )
        if result.get("status") == "completed":
            predictor_path = Path(result["predictor_path"])
            if not predictor_path.is_dir() or not predictions_path.is_file():
                raise FileNotFoundError(
                    f"Completed candidate artifacts are incomplete: {candidate_dir}"
                )
            return result, pd.read_csv(predictions_path)
        return result, pd.DataFrame(columns=PREDICTION_COLUMNS)

    candidate_dir.mkdir(parents=True, exist_ok=True)
    _write_json(candidate_dir / "candidate_config.json", _candidate_contract(candidate))
    dataframe_class = (
        autogluon_classes[0] if autogluon_classes is not None else _load_autogluon()[0]
    )
    failures: list[dict[str, Any]] = []
    for batch_size in [int(value) for value in config["oom_batch_sizes"]]:
        attempt_dir = _next_attempt_dir(candidate_dir, batch_size)
        attempt_dir.mkdir(parents=True)
        attempt_contract = _candidate_contract(candidate, batch_size)
        _write_json(attempt_dir / "attempt_config.json", attempt_contract)
        started = time.monotonic()
        try:
            predictor, model_name, stats = fit_full_candidate(
                frames,
                config,
                candidate,
                batch_size,
                model_source,
                attempt_dir,
                dataloader_num_workers=dataloader_num_workers,
                autogluon_classes=autogluon_classes,
            )
            predictions, skipped, inference_runtime = run_full_origins(
                predictor,
                dataframe_class,
                model_name,
                table,
                config,
                str(candidate["name"]),
                origins,
                split_name,
                run_id,
                model_source,
            )
            if skipped or len(predictions["issue_time"].unique()) != len(origins):
                raise RuntimeError(
                    f"Candidate did not cover every requested origin: {skipped}"
                )
            result = {
                **attempt_contract,
                **stats,
                "model_name": str(predictions["model_name"].iloc[0]),
                "inference_runtime_seconds": inference_runtime,
                "origin_count": len(origins),
                "forecast_key_count": len(forecast_key_set(predictions)),
                "failures_before_success": failures,
                "status": "completed",
            }
            predictions.to_csv(predictions_path, index=False)
            metrics = evaluate_foshan_predictions(
                predictions,
                pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
            )
            metrics.to_csv(candidate_dir / f"{split_name}_metrics.csv", index=False)
            _write_json(attempt_dir / "attempt_result.json", result)
            _write_json(result_path, result)
            return result, predictions
        except Exception as error:
            oom = _is_cuda_oom(error)
            peak_allocated, peak_reserved = _cuda_peak_memory()
            failure = {
                **attempt_contract,
                "status": "oom" if oom else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "attempt_runtime_seconds": time.monotonic() - started,
                "peak_allocated_gpu_bytes": peak_allocated,
                "peak_reserved_gpu_bytes": peak_reserved,
            }
            failures.append(failure)
            _write_json(attempt_dir / "attempt_result.json", failure)
            _clear_cuda()
            if not oom:
                result = {
                    **_candidate_contract(candidate),
                    "status": "failed",
                    "failures": failures,
                }
                _write_json(result_path, result)
                return result, pd.DataFrame(columns=PREDICTION_COLUMNS)
        finally:
            _clear_cuda()
    result = {
        **_candidate_contract(candidate),
        "status": "failed_after_oom_retries",
        "failures": failures,
    }
    _write_json(result_path, result)
    return result, pd.DataFrame(columns=PREDICTION_COLUMNS)


def _metric_dict(row: pd.Series) -> dict[str, float]:
    return {column: float(row[column]) for column in METRIC_COLUMNS}


def _runtime_metadata(
    model_source: str,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    zero_shot_dir: Path,
    lora_run_dir: Path,
) -> dict[str, Any]:
    return {
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "model_id_or_path": model_source,
        "config": str(config_path),
        "input": str(input_path.resolve()),
        "output_dir": str(output_dir),
        "zero_shot_dir_read_only": str(zero_shot_dir.resolve()),
        "lora_run_dir_read_only": str(lora_run_dir.resolve()),
        "packages": {
            name: _package_version(name)
            for name in (
                "autogluon.timeseries",
                "chronos-forecasting",
                "torch",
                "pandas",
            )
        },
        "forecast_accuracy_only": True,
        "revenue_evaluation_performed": False,
    }


def _run_dry(
    table: pd.DataFrame,
    config: dict[str, Any],
    frames: FoshanFineTuneFrames,
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    may_origins = period_origins(
        table,
        config["selection_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=str(config["frequency"]),
        stride_steps=96,
        timezone=str(config["timezone"]),
    )
    june_origins = period_origins(
        table,
        config["engineering_test_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=str(config["frequency"]),
        stride_steps=96,
        timezone=str(config["timezone"]),
    )
    if len(may_origins) != 31 or len(june_origins) != 30:
        raise ValueError(
            f"Expected 31 May and 30 June origins, found {len(may_origins)} and "
            f"{len(june_origins)}."
        )
    train_end = pd.Timestamp(config["train_period"]["end_exclusive"])
    june_start = pd.Timestamp(config["engineering_test_period"]["start"])
    if frames.train["timestamp"].max() >= train_end:
        raise AssertionError("May entered the full-tuning training frame.")
    if frames.tuning["timestamp"].max() >= june_start:
        raise AssertionError("June entered the full-tuning selection frame.")
    result = {
        **metadata,
        "stage": "dry-run",
        "fine_tune": True,
        "fine_tune_mode": "full",
        "target": "pv_kw",
        "known_future_covariates": config["calendar_covariates"],
        "future_measured_covariates": [],
        "net_grid_kw_used_for_training": False,
        "train_period": config["train_period"],
        "selection_period": config["selection_period"],
        "engineering_test_period": config["engineering_test_period"],
        "june_used_for_candidate_selection": False,
        "june_is_pristine_untouched_test": False,
        "next_pristine_evaluation": "July 2026 or later newly acquired data",
        "train_rows": len(frames.train),
        "tuning_rows": len(frames.tuning),
        "masked_train_targets": frames.n_masked_train,
        "masked_tuning_targets": frames.n_masked_tuning,
        "may_origin_count": len(may_origins),
        "june_origin_count": len(june_origins),
        "search_candidates": config["search_candidates"],
        "oom_batch_sizes": config["oom_batch_sizes"],
    }
    _write_json(output_dir / "dry_run.json", result)
    return result


def _run_smoke(
    args: argparse.Namespace,
    table: pd.DataFrame,
    config: dict[str, Any],
    frames: FoshanFineTuneFrames,
    model_source: str,
    autogluon_classes: tuple[Any, Any] | None,
) -> dict[str, Any]:
    smoke_dir = Path(args.output_dir).resolve() / "smoke"
    may_origins = period_origins(
        table,
        config["selection_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=str(config["frequency"]),
        stride_steps=96,
        timezone=str(config["timezone"]),
    )[:1]
    result, predictions = run_candidate_with_oom_fallback(
        frames,
        table,
        config,
        config["smoke_candidate"],
        model_source,
        smoke_dir,
        may_origins,
        "may_2026_selection",
        "foshan_full_smoke",
        args.dataloader_num_workers,
        autogluon_classes=autogluon_classes,
    )
    if result["status"] != "completed" or predictions.empty:
        raise RuntimeError(f"Full-fine-tuning smoke failed: {result}")
    return result


def _run_search(
    args: argparse.Namespace,
    table: pd.DataFrame,
    config: dict[str, Any],
    frames: FoshanFineTuneFrames,
    model_source: str,
    protected_manifest: dict[str, str],
    autogluon_classes: tuple[Any, Any] | None,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    search_dir = output_dir / "search"
    summary_path = search_dir / "summary.json"
    if summary_path.is_file():
        assert_protected_artifacts_unchanged(protected_manifest)
        return _read_json(summary_path)
    search_dir.mkdir(parents=True, exist_ok=True)
    may_origins = period_origins(
        table,
        config["selection_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=str(config["frequency"]),
        stride_steps=96,
        timezone=str(config["timezone"]),
    )
    june_origins = period_origins(
        table,
        config["engineering_test_period"],
        prediction_length=int(config["prediction_length"]),
        frequency=str(config["frequency"]),
        stride_steps=96,
        timezone=str(config["timezone"]),
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    may_predictions_by_candidate: dict[str, pd.DataFrame] = {}
    for candidate in config["search_candidates"]:
        candidate_dir = search_dir / "candidates" / str(candidate["name"])
        result, predictions = run_candidate_with_oom_fallback(
            frames,
            table,
            config,
            candidate,
            model_source,
            candidate_dir,
            may_origins,
            "may_2026_selection",
            "foshan_full_search",
            args.dataloader_num_workers,
            autogluon_classes=autogluon_classes,
        )
        if result["status"] == "completed":
            physical_metrics = evaluate_foshan_predictions(
                predictions,
                pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
            )
            physical = physical_metrics[
                physical_metrics["postprocessing"] == "physical_clip_0_1700"
            ].iloc[0]
            result.update(
                {
                    "may_wape": float(physical["wape"]),
                    "may_pv_active_mae": float(physical["pv_active_mae"]),
                }
            )
            records.append(result)
            may_predictions_by_candidate[str(candidate["name"])] = predictions
        else:
            failures.append(result)
        _write_json(candidate_dir / "candidate_result.json", result)

    if not may_predictions_by_candidate:
        raise RuntimeError("Every full-fine-tuning candidate failed.")
    key_validation = validate_identical_candidate_keys(may_predictions_by_candidate)
    _write_json(search_dir / "may_forecast_key_validation.json", key_validation)
    may_predictions = pd.concat(
        may_predictions_by_candidate.values(),
        ignore_index=True,
    )
    may_metrics = evaluate_foshan_predictions(
        may_predictions,
        pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
    )
    selection = select_full_candidate(may_metrics, records)
    may_predictions.to_csv(search_dir / "may_predictions.csv", index=False)
    may_metrics.to_csv(search_dir / "may_metrics.csv", index=False)
    pd.DataFrame([*records, *failures]).to_csv(
        search_dir / "search_log.csv",
        index=False,
    )
    _write_json(search_dir / "failures.json", failures)
    _write_json(search_dir / "selected_configuration.json", selection)

    june_state_path = search_dir / "june_evaluation_state.json"
    june_predictions_path = search_dir / "june_predictions.csv"
    if june_predictions_path.is_file():
        state = _read_json(june_state_path)
        if state.get("status") != "completed" or state.get("evaluation_count") != 1:
            raise RuntimeError("Existing June outputs have an invalid evaluation state.")
        june_predictions = pd.read_csv(june_predictions_path)
        june_runtime = float(state["inference_runtime_seconds"])
    else:
        if june_state_path.exists():
            raise RuntimeError(
                "A prior June evaluation started without producing predictions; "
                "use a new run directory to preserve the one-evaluation contract."
            )
        _write_json(
            june_state_path,
            {
                "status": "started",
                "evaluation_count": 1,
                "selected_on": "may_2026_selection",
                "candidate": selection["candidate"],
            },
        )
        dataframe_class, predictor_class = autogluon_classes or _load_autogluon()
        predictor, inspection = load_full_predictor_for_evaluation(
            predictor_class,
            Path(selection["predictor_path"]),
            str(selection["trained_model_name"]),
        )
        if inspection["fine_tune_mode"] != "full":
            raise RuntimeError("Frozen June evaluator did not load a full checkpoint.")
        june_predictions, skipped, june_runtime = run_full_origins(
            predictor,
            dataframe_class,
            str(selection["trained_model_name"]),
            table,
            config,
            str(selection["candidate"]),
            june_origins,
            str(config["engineering_test_period"]["split_name"]),
            "foshan_full_frozen_june_engineering_test",
            model_source,
        )
        if skipped or len(june_predictions["issue_time"].unique()) != len(june_origins):
            raise RuntimeError(
                f"Frozen full model did not cover every June origin: {skipped}"
            )
        june_predictions.to_csv(june_predictions_path, index=False)
        _write_json(
            june_state_path,
            {
                "status": "completed",
                "evaluation_count": 1,
                "selected_on": "may_2026_selection",
                "candidate": selection["candidate"],
                "inference_runtime_seconds": june_runtime,
                "model_inspection": inspection,
            },
        )

    june_metrics = evaluate_foshan_predictions(
        june_predictions,
        pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
    )
    june_metrics.to_csv(search_dir / "june_metrics.csv", index=False)
    full_physical = june_predictions[
        june_predictions["postprocessing"] == "physical_clip_0_1700"
    ]
    frozen_rows = load_frozen_reference_rows(
        Path(args.zero_shot_dir),
        Path(args.lora_run_dir),
    )
    comparison_input = pd.concat(
        [full_physical[PREDICTION_COLUMNS], frozen_rows],
        ignore_index=True,
    )
    common_predictions, common_metrics = evaluate_common_scored_timestamps(
        comparison_input,
        target="pv_kw",
        split="june_2026_test",
        pv_active_threshold_kw=float(config["pv_active_threshold_kw"]),
    )
    common_predictions.to_csv(
        search_dir / "common_scored_predictions_june.csv",
        index=False,
    )
    common_metrics.to_csv(
        search_dir / "common_scored_metrics_june.csv",
        index=False,
    )
    full_model_name = str(selection["model_name"])
    improvements = comparison_improvements(common_metrics, full_model_name)
    _write_json(search_dir / "comparison_improvements.json", improvements)
    daily, bootstrap = paired_daily_errors_vs_zero_shot(
        common_predictions,
        full_model_name,
        timezone=str(config["timezone"]),
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["seed"]),
    )
    daily.to_csv(search_dir / "paired_daily_errors_vs_zero_shot.csv", index=False)
    _write_json(search_dir / "paired_daily_bootstrap.json", bootstrap)
    decision = retention_decision(
        common_metrics,
        full_model_name,
        bootstrap,
        config,
    )
    _write_json(search_dir / "retention_decision.json", decision)
    assert_protected_artifacts_unchanged(protected_manifest)

    selected_may = may_metrics[
        (may_metrics["model_name"] == full_model_name)
        & (may_metrics["postprocessing"] == "physical_clip_0_1700")
    ].iloc[0]
    selected_june = june_metrics[
        (june_metrics["model_name"] == full_model_name)
        & (june_metrics["postprocessing"] == "physical_clip_0_1700")
    ].iloc[0]
    successful = [record for record in records if record["status"] == "completed"]
    failed_attempts = [
        attempt
        for record in successful
        for attempt in record.get("failures_before_success", [])
    ] + [
        attempt
        for failed_candidate in failures
        for attempt in failed_candidate.get("failures", [])
    ]
    successful_peak_allocated = [
        int(record["peak_allocated_gpu_bytes"])
        for record in successful
        if record["peak_allocated_gpu_bytes"] is not None
    ]
    failed_peak_allocated = [
        int(attempt["peak_allocated_gpu_bytes"])
        for attempt in failed_attempts
        if attempt.get("peak_allocated_gpu_bytes") is not None
    ]
    successful_peak_reserved = [
        int(record["peak_reserved_gpu_bytes"])
        for record in successful
        if record["peak_reserved_gpu_bytes"] is not None
    ]
    failed_peak_reserved = [
        int(attempt["peak_reserved_gpu_bytes"])
        for attempt in failed_attempts
        if attempt.get("peak_reserved_gpu_bytes") is not None
    ]
    successful_training_runtime = sum(
        float(record["training_runtime_seconds"]) for record in successful
    )
    failed_attempt_runtime = sum(
        float(attempt["attempt_runtime_seconds"]) for attempt in failed_attempts
    )
    summary = {
        "selected_configuration": selection,
        "selected_may_metrics": _metric_dict(selected_may),
        "selected_june_engineering_metrics": _metric_dict(selected_june),
        "may_origin_count": len(may_origins),
        "june_origin_count": len(june_origins),
        "june_evaluation_count": 1,
        "june_is_pristine_untouched_test": False,
        "next_pristine_evaluation": "July 2026 or later newly acquired data",
        "total_candidate_training_runtime_seconds": successful_training_runtime,
        "total_failed_attempt_runtime_seconds": failed_attempt_runtime,
        "total_training_attempt_runtime_seconds": (
            successful_training_runtime + failed_attempt_runtime
        ),
        "june_inference_runtime_seconds": june_runtime,
        "peak_training_allocated_gpu_bytes": max(
            [*successful_peak_allocated, *failed_peak_allocated],
            default=None,
        ),
        "peak_training_reserved_gpu_bytes": max(
            [*successful_peak_reserved, *failed_peak_reserved],
            default=None,
        ),
        "trainable_parameters": int(selection["trainable_parameters"]),
        "total_parameters": int(selection["total_parameters"]),
        "checkpoint_path": selection["checkpoint_path"],
        "checkpoint_size_bytes": int(selection["checkpoint_size_bytes"]),
        "failures_and_oom_retries": failed_attempts,
        "failed_candidates": failures,
        "comparison": improvements,
        "paired_daily_bootstrap": bootstrap,
        "retention_decision": decision,
        "revenue_results_claimed": False,
    }
    _write_json(summary_path, summary)
    return summary


def run(
    args: argparse.Namespace,
    autogluon_classes: tuple[Any, Any] | None = None,
    hardware_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(args.config)
    config = load_full_config(config_path)
    input_path = Path(args.input)
    table = validate_processed_table(pd.read_parquet(input_path), config)
    frames = prepare_full_finetune_frames(table, config)
    zero_shot_dir = Path(args.zero_shot_dir)
    lora_run_dir = Path(args.lora_run_dir)
    output_dir = validate_output_isolation(
        Path(args.output_dir),
        zero_shot_dir,
        lora_run_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_source = resolve_model_source(args.model_id, args.model_path)
    manifest = protected_artifact_manifest(zero_shot_dir, lora_run_dir)
    metadata = _runtime_metadata(
        model_source,
        config_path,
        input_path,
        output_dir,
        zero_shot_dir,
        lora_run_dir,
    )
    contract_path = output_dir / "run_contract.json"
    contract = {
        "git_commit": metadata["git_commit"],
        "model_id_or_path": model_source,
        "input": metadata["input"],
        "config": str(config_path),
        "target": "pv_kw",
        "fine_tune_mode": "full",
        "zero_shot_dir_read_only": metadata["zero_shot_dir_read_only"],
        "lora_run_dir_read_only": metadata["lora_run_dir_read_only"],
    }
    if contract_path.is_file() and _read_json(contract_path) != contract:
        raise ValueError("Existing full-tuning run contract differs from this invocation.")
    _write_json(contract_path, contract)
    _write_json(output_dir / "protected_artifacts_before.json", manifest)

    if args.stage == "dry-run":
        return _run_dry(table, config, frames, metadata, output_dir)

    validate_model_source_for_training(model_source)
    hardware = hardware_override or full_gpu_preflight()
    if hardware.get("precision") != "bf16":
        raise RuntimeError("Foshan full fine-tuning requires BF16.")
    _write_json(
        output_dir / f"{args.stage}_run_metadata.json",
        {**metadata, "stage": args.stage, "hardware": hardware},
    )
    if args.stage == "smoke":
        result = _run_smoke(
            args,
            table,
            config,
            frames,
            model_source,
            autogluon_classes,
        )
    else:
        result = _run_search(
            args,
            table,
            config,
            frames,
            model_source,
            manifest,
            autogluon_classes,
        )
    _write_json(
        output_dir / f"{args.stage}_run_metadata.json",
        {
            **metadata,
            "stage": args.stage,
            "hardware": hardware,
            "result": result,
        },
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--zero-shot-dir", type=Path, default=DEFAULT_ZERO_SHOT_DIR)
    parser.add_argument("--lora-run-dir", type=Path, default=DEFAULT_LORA_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--stage",
        choices=["dry-run", "smoke", "search"],
        required=True,
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
