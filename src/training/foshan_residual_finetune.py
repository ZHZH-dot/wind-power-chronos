"""Chronos-2 LoRA/full challengers for the Foshan signed residual target."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from src.data.reconstruct_foshan_residual import (
    CALENDAR_COLUMNS,
    FREQUENCY,
    TARGET_COLUMN,
    TARGET_LABEL,
    TIMEZONE,
    sha256_file,
)
from src.evaluation.foshan_benchmark import normalize_chronos_quantiles
from src.models.foshan_residual_zero_shot import (
    MODEL_ID,
    MODEL_REVISION,
    PREDICTION_COLUMNS,
    PREDICTION_LENGTH,
    QUANTILES,
    build_inference_frames,
    forecast_issue_times,
    load_residual_config,
    package_version,
    resolve_pinned_snapshot,
)
from src.training.chronos_finetune import build_chronos2_hyperparameters
from src.utils.runtime import git_commit, git_is_dirty


MIN_GPU_VRAM_GIB = 23.0


@dataclass(frozen=True)
class ResidualTrainingFrames:
    train: pd.DataFrame
    tuning: pd.DataFrame
    train_start: pd.Timestamp
    train_end_exclusive: pd.Timestamp
    selection_start: pd.Timestamp
    selection_end_exclusive: pd.Timestamp
    n_missing_train_targets: int
    n_missing_tuning_targets: int


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _local_naive(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if isinstance(timestamps, pd.DatetimeIndex):
        series = pd.Series(timestamps)
    else:
        series = timestamps
    if series.dt.tz is None:
        return series.astype("datetime64[ns]")
    return series.dt.tz_convert(TIMEZONE).dt.tz_localize(None).astype("datetime64[ns]")


def _site_aware(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="raise")
    if isinstance(timestamps, pd.DatetimeIndex):
        series = pd.Series(timestamps)
    else:
        series = timestamps
    if series.dt.tz is None:
        return series.dt.tz_localize(TIMEZONE)
    return series.dt.tz_convert(TIMEZONE)


def prepare_residual_training_frames(
    residual_table: pd.DataFrame,
    config: dict[str, Any],
) -> ResidualTrainingFrames:
    """Use March targets for fitting and March-April only for validation context."""
    table = residual_table.copy()
    table["timestamp"] = _site_aware(table["timestamp"]).to_numpy()
    required = {"timestamp", TARGET_COLUMN, *CALENDAR_COLUMNS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Residual training data is missing columns: {missing}")
    train_start = pd.Timestamp(config["train_period"]["start"])
    train_end = pd.Timestamp(config["train_period"]["end_exclusive"])
    selection_start = pd.Timestamp(config["selection_period"]["start"])
    selection_end = pd.Timestamp(config["selection_period"]["end_exclusive"])
    if train_end != selection_start:
        raise ValueError("March training must end exactly when April selection starts.")
    if selection_end != pd.Timestamp(config["test_period"]["start"]):
        raise ValueError("April selection must end exactly when May test starts.")
    train_source = table.loc[
        (table["timestamp"] >= train_start) & (table["timestamp"] < train_end)
    ].copy()
    tuning_source = table.loc[
        (table["timestamp"] >= train_start) & (table["timestamp"] < selection_end)
    ].copy()
    if train_source.empty or tuning_source.empty:
        raise ValueError("March training or cumulative April validation context is empty.")

    def boundary_frame(source: pd.DataFrame) -> pd.DataFrame:
        result = source[["timestamp", TARGET_COLUMN, *CALENDAR_COLUMNS]].copy()
        result.insert(0, "id", "foshan_signed_residual")
        result = result.rename(columns={TARGET_COLUMN: "target"})
        result["timestamp"] = _local_naive(result["timestamp"]).to_numpy()
        return result.sort_values(["id", "timestamp"]).reset_index(drop=True)

    train = boundary_frame(train_source)
    tuning = boundary_frame(tuning_source)
    if train["timestamp"].max() >= train_end.tz_localize(None):
        raise AssertionError("April targets entered train_data.")
    if tuning["timestamp"].max() >= selection_end.tz_localize(None):
        raise AssertionError("May targets entered tuning_data or model selection.")
    expected_train = pd.date_range(
        train_start.tz_localize(None), train_end.tz_localize(None), freq=FREQUENCY, inclusive="left"
    )
    if train["timestamp"].tolist() != expected_train.tolist():
        raise ValueError("March residual training timestamps are not a complete 15-minute grid.")
    return ResidualTrainingFrames(
        train=train,
        tuning=tuning,
        train_start=train_start,
        train_end_exclusive=train_end,
        selection_start=selection_start,
        selection_end_exclusive=selection_end,
        n_missing_train_targets=int(train["target"].isna().sum()),
        n_missing_tuning_targets=int(tuning["target"].isna().sum()),
    )


def _float_name(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-0", "-")


def training_candidates(
    config: dict[str, Any],
    fine_tune_mode: str,
    *,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    if fine_tune_mode == "lora":
        search = config["lora_search"]
        pairs = [
            (int(pair["rank"]), int(pair["alpha"]))
            for pair in search["rank_alpha_pairs"]
        ]
        if pairs != [(8, 16), (16, 32)]:
            raise ValueError(
                "Residual LoRA rank/alpha pairs must be exactly (8, 16) and (16, 32)."
            )
        if smoke:
            return [
                {
                    "name": "residual_lora_smoke_5s",
                    "rank": pairs[0][0],
                    "lora_alpha": pairs[0][1],
                    "learning_rate": float(search["learning_rates"][0]),
                    "steps": int(search["smoke_steps"]),
                    "batch_size": int(search["smoke_batch_size"]),
                    "seed": int(config["seed"]),
                }
            ]
        candidates = []
        for (rank, alpha), learning_rate, steps in itertools.product(
            pairs,
            search["learning_rates"],
            search["steps"],
        ):
            candidates.append(
                {
                    "name": (
                        f"residual_lora_r{rank}_a{alpha}_lr{_float_name(float(learning_rate))}_"
                        f"s{steps}_seed{config['seed']}"
                    ),
                    "rank": int(rank),
                    "lora_alpha": int(alpha),
                    "learning_rate": float(learning_rate),
                    "steps": int(steps),
                    "batch_size": int(search["batch_size"]),
                    "seed": int(config["seed"]),
                }
            )
        return candidates
    if fine_tune_mode == "full":
        search = config["full_search"]
        steps_values = [int(search["smoke_steps"])] if smoke else [int(value) for value in search["steps"]]
        learning_rates = [float(search["learning_rates"][0])] if smoke else [
            float(value) for value in search["learning_rates"]
        ]
        return [
            {
                "name": (
                    "residual_full_smoke_5s"
                    if smoke
                    else f"residual_full_lr{_float_name(lr)}_s{steps}_seed{config['seed']}"
                ),
                "learning_rate": lr,
                "steps": steps,
                "batch_size": int(search["batch_size"]),
                "gradient_accumulation_steps": int(search["gradient_accumulation_steps"]),
                "effective_batch_size": int(search["effective_batch_size"]),
                "seed": int(config["seed"]),
            }
            for lr, steps in itertools.product(learning_rates, steps_values)
        ]
    raise ValueError("fine_tune_mode must be 'lora' or 'full'.")


def build_residual_hyperparameters(
    snapshot: Path,
    config: dict[str, Any],
    candidate: dict[str, Any],
    fine_tune_mode: str,
    *,
    dataloader_num_workers: int = 0,
) -> dict[str, dict[str, Any]]:
    hyperparameters = build_chronos2_hyperparameters(
        model_id=str(snapshot),
        mode="univariate",
        prediction_length=int(config["prediction_length"]),
        context_length=int(config["training_context_length"]),
        steps=int(candidate["steps"]),
        learning_rate=float(candidate["learning_rate"]),
        batch_size=int(candidate["batch_size"]),
        inference_batch_size=int(config["inference_batch_size"]),
        seed=int(candidate["seed"]),
        dataloader_num_workers=dataloader_num_workers,
        bf16=True,
        fp16=False,
        lora_rank=(int(candidate["rank"]) if fine_tune_mode == "lora" else None),
        lora_alpha=(
            int(candidate["lora_alpha"]) if fine_tune_mode == "lora" else None
        ),
        disable_known_covariates=False,
        disable_past_covariates=True,
        fine_tune_mode=fine_tune_mode,
        model_name_suffix="ResidualLoRA" if fine_tune_mode == "lora" else "ResidualFull",
    )
    if fine_tune_mode == "full":
        hyperparameters["Chronos2"]["fine_tune_trainer_kwargs"][
            "gradient_accumulation_steps"
        ] = int(candidate["gradient_accumulation_steps"])
    return hyperparameters


def gpu_preflight(
    torch_module: Any | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Require one explicit BF16-capable CUDA GPU without locking a model name."""
    if torch_module is None:
        import torch as torch_module

    env = os.environ if environment is None else environment
    if env.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 for residual fine-tuning.")
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise RuntimeError("Residual fine-tuning requires exactly one visible CUDA GPU.")
    name = str(torch_module.cuda.get_device_name(0))
    if not torch_module.cuda.is_bf16_supported():
        raise RuntimeError(f"The visible CUDA GPU must support BF16; detected {name}.")
    total_memory = int(torch_module.cuda.get_device_properties(0).total_memory)
    if total_memory < int(MIN_GPU_VRAM_GIB * 1024**3):
        raise RuntimeError(
            f"Residual fine-tuning requires at least {MIN_GPU_VRAM_GIB:.0f} GiB VRAM; "
            f"detected {total_memory / 1024**3:.2f} GiB on {name}."
        )
    return {
        "gpu_name": name,
        "gpu_total_vram_bytes": total_memory,
        "bf16_supported": True,
        "dtype": "bfloat16",
        "torch_version": str(torch_module.__version__),
        "cuda_version": str(torch_module.version.cuda),
        "chronos_forecasting_version": package_version("chronos-forecasting"),
        "autogluon_timeseries_version": package_version("autogluon.timeseries"),
        "peft_version": package_version("peft"),
        "accelerate_version": package_version("accelerate"),
    }

def _load_autogluon() -> tuple[Any, Any]:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    return TimeSeriesDataFrame, TimeSeriesPredictor


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _parameter_counts(model: Any) -> dict[str, int]:
    parameters = list(model.parameters())
    total = sum(int(value.numel()) for value in parameters)
    trainable = sum(int(value.numel()) for value in parameters if value.requires_grad)
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "frozen_parameters": total - trainable,
    }


def _adapter_names(value: Any, field: str) -> list[str]:
    if value == "irregular":
        raise RuntimeError(f"PEFT reports irregular {field} state.")
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    return [str(name) for name in value]


def _peft_type_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).upper().split(".")[-1]


def _loaded_adapter_tensor_stats(model: Any, adapter_names: list[str]) -> tuple[int, int]:
    try:
        from peft import get_peft_model_state_dict
    except ImportError as error:
        raise RuntimeError(
            "PEFT does not expose an adapter state-dict API for reload verification."
        ) from error

    tensor_count = 0
    parameter_count = 0
    for adapter_name in adapter_names:
        state = get_peft_model_state_dict(model, adapter_name=adapter_name)
        if not isinstance(state, dict) or not state:
            raise RuntimeError(
                f"Active PEFT adapter {adapter_name!r} has no loaded tensors."
            )
        for tensor in state.values():
            if not hasattr(tensor, "numel"):
                raise RuntimeError(
                    f"Active PEFT adapter {adapter_name!r} contains a non-tensor value."
                )
            tensor_count += 1
            parameter_count += int(tensor.numel())
    if tensor_count <= 0 or parameter_count <= 0:
        raise RuntimeError("No active LoRA adapter tensors were loaded.")
    return tensor_count, parameter_count


def inspect_lora_model_state(
    model: Any,
    *,
    require_trainable: bool,
) -> dict[str, Any]:
    """Validate public PEFT state and count parameters without importing it in tests."""
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict) or not peft_config:
        raise RuntimeError("Loaded Chronos-2 model is base-only; no PEFT adapter is registered.")
    status_getter = getattr(model, "get_model_status", None)
    if not callable(status_getter):
        try:
            from peft import get_model_status
        except ImportError as error:
            raise RuntimeError(
                "Loaded model does not expose PEFT model-status validation."
            ) from error
        status = get_model_status(model)
    else:
        status = status_getter()

    enabled = getattr(status, "enabled", None)
    if enabled is not True:
        raise RuntimeError(f"Loaded PEFT adapter is not enabled: {enabled!r}.")
    active = _adapter_names(getattr(status, "active_adapters", None), "active adapter")
    available = _adapter_names(
        getattr(status, "available_adapters", None), "available adapter"
    )
    if not active:
        raise RuntimeError("Loaded PEFT model has no active adapter.")
    if not set(active).issubset(set(available)):
        raise RuntimeError(
            f"Active adapters {active} are not registered in available adapters {available}."
        )
    if not set(active).issubset(set(str(name) for name in peft_config)):
        raise RuntimeError(
            f"Active adapters {active} are absent from PEFT configuration {list(peft_config)}."
        )
    peft_types = getattr(status, "peft_types", {})
    if not isinstance(peft_types, dict) or any(
        _peft_type_name(peft_types.get(name)) != "LORA" for name in active
    ):
        raise RuntimeError(f"Active adapter is not LoRA: {peft_types!r}.")
    adapter_layers = int(getattr(status, "num_adapter_layers", 0))
    if adapter_layers <= 0:
        raise RuntimeError("Loaded PEFT model has no adapter layers.")

    counts = _parameter_counts(model)
    status_trainable = getattr(status, "trainable_params", None)
    status_total = getattr(status, "total_params", None)
    if status_trainable is not None and int(status_trainable) != counts["trainable_parameters"]:
        raise RuntimeError("PEFT and model trainable-parameter counts disagree.")
    if status_total is not None and int(status_total) != counts["total_parameters"]:
        raise RuntimeError("PEFT and model total-parameter counts disagree.")
    if require_trainable:
        if counts["trainable_parameters"] <= 0 or counts["frozen_parameters"] <= 0:
            raise RuntimeError(
                "LoRA training state must contain both trainable adapter and frozen base parameters."
            )
    tensor_count, adapter_parameter_count = _loaded_adapter_tensor_stats(model, active)
    return {
        **counts,
        "adapter_attached": True,
        "adapter_active": True,
        "active_adapter_names": active,
        "available_adapter_names": available,
        "adapter_layer_count": adapter_layers,
        "adapter_tensor_count": tensor_count,
        "adapter_parameter_count": adapter_parameter_count,
    }


def _training_parameter_metadata(model: Any, fine_tune_mode: str) -> dict[str, Any]:
    if fine_tune_mode == "lora":
        state = inspect_lora_model_state(model, require_trainable=True)
        return {
            "trainable_parameters": state["trainable_parameters"],
            "frozen_parameters": state["frozen_parameters"],
            "total_parameters": state["total_parameters"],
            "training_adapter_attached": state["adapter_attached"],
            "training_adapter_active": state["adapter_active"],
            "training_active_adapter_names": state["active_adapter_names"],
            "training_adapter_layer_count": state["adapter_layer_count"],
            "training_adapter_tensor_count": state["adapter_tensor_count"],
            "training_adapter_parameter_count": state["adapter_parameter_count"],
            "training_adapter_validation_status": "passed",
            "training_parameter_capture_point": "chronos2_pipeline_fit_return",
        }
    counts = _parameter_counts(model)
    if counts["trainable_parameters"] != counts["total_parameters"]:
        raise RuntimeError(
            "Full fine-tuning returned a model with frozen parameters: "
            f"{counts['trainable_parameters']}/{counts['total_parameters']}."
        )
    return {
        **counts,
        "training_parameter_capture_point": "chronos2_pipeline_fit_return",
    }


def _validate_training_parameter_metadata(
    metadata: dict[str, Any], fine_tune_mode: str
) -> None:
    try:
        trainable = int(metadata["trainable_parameters"])
        frozen = int(metadata["frozen_parameters"])
        total = int(metadata["total_parameters"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Training-time parameter metadata is missing or invalid.") from error
    if total != trainable + frozen:
        raise RuntimeError(
            "Training-time total parameters do not equal trainable plus frozen parameters."
        )
    if fine_tune_mode == "lora":
        if trainable <= 0 or frozen <= 0:
            raise RuntimeError(
                "LoRA training metadata must report positive trainable and frozen counts."
            )
        for field in (
            "training_adapter_attached",
            "training_adapter_active",
        ):
            if metadata.get(field) is not True:
                raise RuntimeError(f"LoRA training metadata failed {field} validation.")
        if metadata.get("training_adapter_validation_status") != "passed":
            raise RuntimeError("LoRA training adapter validation did not pass.")
        if metadata.get("training_parameter_capture_point") != "chronos2_pipeline_fit_return":
            raise RuntimeError("LoRA parameter counts were not captured at the Chronos fit return.")
        if not metadata.get("training_active_adapter_names") or int(
            metadata.get("training_adapter_tensor_count", 0)
        ) <= 0:
            raise RuntimeError("LoRA training metadata has no active adapter tensors.")
    elif trainable != total:
        raise RuntimeError("Full fine-tuning metadata must report every parameter trainable.")


@contextmanager
def capture_chronos2_training_state(
    fine_tune_mode: str,
    *,
    pipeline_class: Any | None = None,
) -> Iterator[dict[str, Any]]:
    """Capture the model returned by Chronos fit before AutoGluon serializes it."""
    if pipeline_class is None:
        from chronos import Chronos2Pipeline

        pipeline_class = Chronos2Pipeline
    original_fit = pipeline_class.fit
    captured: dict[str, Any] = {}
    completed = False

    @wraps(original_fit)
    def monitored_fit(pipeline: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_fit(pipeline, *args, **kwargs)
        actual_mode = str(kwargs.get("finetune_mode", "full"))
        if actual_mode != fine_tune_mode:
            raise RuntimeError(
                f"Chronos trained in {actual_mode!r}, expected {fine_tune_mode!r}."
            )
        if captured:
            raise RuntimeError("Expected exactly one Chronos-2 fine-tuning lifecycle.")
        model = getattr(result, "model", None)
        if model is None:
            raise RuntimeError("Chronos-2 fit returned no inspectable model.")
        captured.update(_training_parameter_metadata(model, fine_tune_mode))
        return result

    pipeline_class.fit = monitored_fit
    try:
        yield captured
        completed = True
    finally:
        pipeline_class.fit = original_fit
    if completed and not captured:
        raise RuntimeError("Chronos-2 fit completed without training-state capture.")


def _validate_lora_artifacts(
    checkpoint: Path,
    *,
    snapshot: Path | None = None,
    expected_lora_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapter_configs = sorted(checkpoint.rglob("adapter_config.json"))
    adapter_models = sorted(checkpoint.rglob("adapter_model.safetensors"))
    if len(adapter_configs) != 1:
        raise RuntimeError(
            "LoRA checkpoint must contain exactly one adapter_config.json; "
            f"found {adapter_configs}."
        )
    if len(adapter_models) != 1:
        raise RuntimeError(
            "LoRA checkpoint must contain exactly one adapter_model.safetensors; "
            f"found {adapter_models}."
        )
    config_path = adapter_configs[0]
    model_path = adapter_models[0]
    if config_path.stat().st_size <= 0 or model_path.stat().st_size <= 0:
        raise RuntimeError("LoRA adapter artifacts must be nonempty.")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid LoRA adapter configuration: {config_path}.") from error
    if not isinstance(config, dict) or _peft_type_name(config.get("peft_type")) != "LORA":
        raise RuntimeError("Adapter configuration is not a valid LoRA configuration.")
    auto_mapping = config.get("auto_mapping")
    if isinstance(auto_mapping, dict) and auto_mapping.get("base_model_class") not in (
        None,
        "Chronos2Model",
    ):
        raise RuntimeError("LoRA adapter configuration targets a non-Chronos-2 base model.")
    base_source = str(config.get("base_model_name_or_path", ""))
    if snapshot is not None:
        snapshot_matches = False
        if base_source:
            try:
                snapshot_matches = Path(base_source).expanduser().resolve() == snapshot.resolve()
            except OSError:
                snapshot_matches = False
        if base_source != MODEL_ID and not snapshot_matches:
            raise RuntimeError(
                f"LoRA adapter base source {base_source!r} does not match the pinned Chronos-2 base."
            )
    if expected_lora_config is not None:
        for field in ("r", "lora_alpha"):
            expected = expected_lora_config.get(field)
            actual = config.get(field)
            try:
                matches = expected is None or int(actual) == int(expected)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise RuntimeError(
                    f"LoRA adapter {field}={actual!r} does not match saved "
                    f"AutoGluon configuration {expected!r}."
                )
    return {
        "adapter_config_path": config_path,
        "adapter_model_path": model_path,
        "adapter_config": config,
    }


def find_fine_tuned_checkpoint(
    predictor_path: Path,
    fine_tune_mode: str,
) -> tuple[Path, list[Path]]:
    checkpoints = sorted(
        path for path in predictor_path.rglob("fine-tuned-ckpt") if path.is_dir()
    )
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected one fine-tuned-ckpt under {predictor_path}, found {checkpoints}."
        )
    checkpoint = checkpoints[0]
    adapters = sorted(checkpoint.rglob("adapter_model.safetensors"))
    adapter_configs = sorted(checkpoint.rglob("adapter_config.json"))
    weights = sorted(checkpoint.rglob("*.safetensors")) + sorted(
        checkpoint.rglob("pytorch_model*.bin")
    )
    if fine_tune_mode == "lora":
        _validate_lora_artifacts(checkpoint)
    if fine_tune_mode == "full" and (adapters or adapter_configs or not weights):
        raise RuntimeError("Full checkpoint must contain full weights and no LoRA adapter.")
    return checkpoint, weights


def verify_loaded_predictor(
    predictor: Any,
    predictor_path: Path,
    model_name: str,
    fine_tune_mode: str,
    snapshot: Path,
) -> dict[str, Any]:
    names = [str(value) for value in predictor.model_names()]
    if model_name not in names or len([value for value in names if value.startswith("Chronos2")]) != 1:
        raise RuntimeError(f"Expected one saved Chronos2 model after reload, found {names}.")
    trainer = getattr(predictor, "_trainer", None)
    if trainer is None or not hasattr(trainer, "load_model"):
        raise RuntimeError("Reloaded predictor does not expose its trained model.")
    ag_model = trainer.load_model(model_name)
    hyperparameters = ag_model.get_hyperparameters()
    if hyperparameters.get("fine_tune") is not True:
        raise RuntimeError("Reloaded Chronos2 model is not marked fine_tune=true.")
    if hyperparameters.get("fine_tune_mode") != fine_tune_mode:
        raise RuntimeError(
            f"Reloaded model mode is {hyperparameters.get('fine_tune_mode')!r}, "
            f"expected {fine_tune_mode!r}."
        )
    saved_source = Path(str(hyperparameters.get("model_path", ""))).expanduser().resolve()
    if saved_source != snapshot.resolve():
        raise RuntimeError(
            f"Reloaded model silently changed base checkpoint: {saved_source} != {snapshot}."
        )
    checkpoint, weight_files = find_fine_tuned_checkpoint(predictor_path, fine_tune_mode)
    artifact_info: dict[str, Any] = {}
    if fine_tune_mode == "lora":
        artifact_info = _validate_lora_artifacts(
            checkpoint,
            snapshot=snapshot,
            expected_lora_config=hyperparameters.get("fine_tune_lora_config"),
        )
    try:
        ag_model.load_model_pipeline()
        pipeline = getattr(ag_model, "_model_pipeline", None)
        core = getattr(pipeline, "model", None)
        if core is None:
            core = getattr(pipeline, "_model", None)
        if core is None:
            raise RuntimeError("Could not inspect the reloaded Chronos-2 model parameters.")
        if fine_tune_mode == "lora":
            reload_state = inspect_lora_model_state(core, require_trainable=False)
            expected_lora_config = hyperparameters.get("fine_tune_lora_config") or {}
            for adapter_name in reload_state["active_adapter_names"]:
                loaded_config = core.peft_config[adapter_name]
                for field in ("r", "lora_alpha"):
                    expected = expected_lora_config.get(field)
                    actual = getattr(loaded_config, field, None)
                    try:
                        matches = expected is None or int(actual) == int(expected)
                    except (TypeError, ValueError):
                        matches = False
                    if not matches:
                        raise RuntimeError(
                            f"Loaded LoRA adapter {field}={actual!r} does not match "
                            f"saved AutoGluon configuration {expected!r}."
                        )
            adapter_reload = {
                "adapter_attached_after_reload": reload_state["adapter_attached"],
                "adapter_active_after_reload": reload_state["adapter_active"],
                "active_adapter_names": reload_state["active_adapter_names"],
                "available_adapter_names": reload_state["available_adapter_names"],
                "reload_adapter_layer_count": reload_state["adapter_layer_count"],
                "reload_adapter_tensor_count": reload_state["adapter_tensor_count"],
                "reload_adapter_parameter_count": reload_state["adapter_parameter_count"],
                "base_model_fallback_rejected": True,
                "adapter_validation_status": "passed",
            }
        else:
            reload_state = _parameter_counts(core)
            adapter_reload = {
                "adapter_attached_after_reload": False,
                "adapter_active_after_reload": False,
                "active_adapter_names": [],
                "base_model_fallback_rejected": True,
                "adapter_validation_status": "not_applicable",
            }
        reload_counts = {
            "reload_trainable_parameters": reload_state["trainable_parameters"],
            "reload_total_parameters": reload_state["total_parameters"],
            "reload_frozen_parameters": reload_state["frozen_parameters"],
        }
    finally:
        ag_model._model_pipeline = None
        gc.collect()
    adapter_model = artifact_info.get("adapter_model_path")
    adapter_config = artifact_info.get("adapter_config_path")
    return {
        **reload_counts,
        **adapter_reload,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _directory_sha256(checkpoint),
        "checkpoint_size_bytes": sum(
            value.stat().st_size for value in checkpoint.rglob("*") if value.is_file()
        ),
        "checkpoint_weight_files": [str(value.resolve()) for value in weight_files],
        "adapter_model_path": (
            str(adapter_model.resolve()) if adapter_model is not None else None
        ),
        "adapter_model_sha256": (
            sha256_file(adapter_model) if adapter_model is not None else None
        ),
        "adapter_model_size_bytes": (
            int(adapter_model.stat().st_size) if adapter_model is not None else None
        ),
        "adapter_config_path": (
            str(adapter_config.resolve()) if adapter_config is not None else None
        ),
        "adapter_config_sha256": (
            sha256_file(adapter_config) if adapter_config is not None else None
        ),
        "adapter_config_size_bytes": (
            int(adapter_config.stat().st_size) if adapter_config is not None else None
        ),
        "base_model_source_verified": str(snapshot.resolve()),
        "fine_tune_mode_verified": fine_tune_mode,
    }


def fit_residual_candidate(
    frames: ResidualTrainingFrames,
    config: dict[str, Any],
    candidate: dict[str, Any],
    fine_tune_mode: str,
    snapshot: Path,
    candidate_dir: Path,
    *,
    dataloader_num_workers: int = 0,
) -> tuple[Any, Any, str, dict[str, Any]]:
    predictor_path = candidate_dir / "predictor"
    if predictor_path.exists():
        raise FileExistsError(f"Refusing to overwrite predictor: {predictor_path}")
    dataframe_class, predictor_class = _load_autogluon()
    train_data = dataframe_class.from_data_frame(
        frames.train, id_column="id", timestamp_column="timestamp"
    )
    tuning_data = dataframe_class.from_data_frame(
        frames.tuning, id_column="id", timestamp_column="timestamp"
    )
    hyperparameters = build_residual_hyperparameters(
        snapshot,
        config,
        candidate,
        fine_tune_mode,
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
    import torch

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    fit_kwargs = {
        "train_data": train_data,
        "tuning_data": tuning_data,
        "hyperparameters": hyperparameters,
        "enable_ensemble": False,
        "random_seed": int(candidate["seed"]),
        "refit_full": False,
        "skip_model_selection": False,
    }
    started = time.monotonic()
    with capture_chronos2_training_state(fine_tune_mode) as training_state:
        predictor.fit(**fit_kwargs)
    runtime = time.monotonic() - started
    _validate_training_parameter_metadata(training_state, fine_tune_mode)
    names = [str(value) for value in predictor.model_names()]
    trained = [value for value in names if value.startswith("Chronos2")]
    if len(trained) != 1:
        raise RuntimeError(f"AutoGluon trained no usable Chronos2 model: {names}")
    reloaded = predictor_class.load(str(predictor_path))
    inspection = verify_loaded_predictor(
        reloaded,
        predictor_path,
        trained[0],
        fine_tune_mode,
        snapshot,
    )
    stats = {
        "training_runtime_seconds": runtime,
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(0)),
        "trained_model_name": trained[0],
        "predictor_path": str(predictor_path.resolve()),
        "autogluon_hyperparameters": hyperparameters,
        **training_state,
        **inspection,
    }
    return reloaded, dataframe_class, trained[0], stats


def _normalize_autogluon_forecast(forecast: Any) -> pd.DataFrame:
    frame = pd.DataFrame(forecast).reset_index()
    if "item_id" in frame and "id" not in frame:
        frame = frame.rename(columns={"item_id": "id"})
    return frame


def predict_saved_period(
    predictor: Any,
    dataframe_class: Any,
    model_name: str,
    residual_table: pd.DataFrame,
    issues: Iterable[pd.Timestamp],
    candidate_name: str,
    fine_tune_mode: str,
    checkpoint_path: Path,
    context_length: int,
    split: str,
    *,
    max_issues: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = list(issues)
    if max_issues is not None:
        selected = selected[:max_issues]
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for issue in selected:
        context, future, metadata = build_inference_frames(
            residual_table, issue, context_length
        )
        context_data = dataframe_class.from_data_frame(
            context, id_column="id", timestamp_column="timestamp"
        )
        future_data = dataframe_class.from_data_frame(
            future, id_column="id", timestamp_column="timestamp"
        )
        started = time.monotonic()
        forecast = predictor.predict(
            context_data,
            known_covariates=future_data,
            model=model_name,
        )
        runtime = time.monotonic() - started
        issue_ts = pd.Timestamp(metadata["issue_time"])
        target_times = pd.date_range(issue_ts, periods=PREDICTION_LENGTH, freq=FREQUENCY)
        normalized = normalize_chronos_quantiles(
            _normalize_autogluon_forecast(forecast),
            targets=["target"],
            expected_times=target_times,
            site_id="foshan_signed_residual",
        )
        ordered = np.sort(
            normalized[["p10", "p50", "p90"]].to_numpy(dtype=float), axis=1
        )
        truth = residual_table.copy()
        truth["timestamp"] = _site_aware(truth["timestamp"]).to_numpy()
        truth = truth.set_index("timestamp").reindex(target_times)
        call_id = f"{candidate_name}:{issue_ts.isoformat()}"
        rows = pd.DataFrame(
            {
                "split": split,
                "candidate": candidate_name,
                "candidate_kind": f"chronos2_{fine_tune_mode}_residual",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "context_length": context_length,
                "refresh_cadence_minutes": 60,
                "issue_time": issue_ts,
                "context_start": metadata["context_start"],
                "context_end": metadata["context_end"],
                "target_time": target_times,
                "horizon_step": np.arange(1, PREDICTION_LENGTH + 1),
                "p10": ordered[:, 0],
                "p50": ordered[:, 1],
                "p90": ordered[:, 2],
                "y_pred": ordered[:, 1],
                "y_true_kw": truth[TARGET_COLUMN].to_numpy(dtype=float),
                "is_missing_target": truth[TARGET_COLUMN].isna().to_numpy(dtype=bool),
                "target_label": TARGET_LABEL,
                "known_future_covariates": "|".join(CALENDAR_COLUMNS),
                "used_future_realized_data": False,
                "inference_call_id": call_id,
            },
            columns=PREDICTION_COLUMNS,
        )
        frames.append(rows)
        audits.append(
            {
                **metadata,
                "candidate": candidate_name,
                "fine_tune_mode": fine_tune_mode,
                "inference_call_id": call_id,
                "runtime_seconds": runtime,
                "forecast_rows": len(rows),
                "different_issue_times_batched": False,
                "used_future_realized_data": False,
            }
        )
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.groupby("inference_call_id")["issue_time"].nunique().gt(1).any():
        raise AssertionError("A trained Chronos inference call batched issue times.")
    if predictions.groupby("inference_call_id").size().ne(PREDICTION_LENGTH).any():
        raise AssertionError("Every trained Chronos call must contain exactly 96 horizons.")
    expected_horizons = list(range(1, PREDICTION_LENGTH + 1))
    for _, group in predictions.groupby("inference_call_id", sort=False):
        if sorted(group["horizon_step"].astype(int).tolist()) != expected_horizons:
            raise AssertionError("A trained Chronos call has missing or duplicate horizons.")
    if not (
        pd.to_datetime(predictions["context_end"], utc=True)
        < pd.to_datetime(predictions["issue_time"], utc=True)
    ).all():
        raise AssertionError("A trained Chronos context does not end before issue_time.")
    expected_covariates = "|".join(CALENDAR_COLUMNS)
    if not predictions["known_future_covariates"].eq(expected_covariates).all():
        raise AssertionError("Trained inference used an unexpected known-future covariate set.")
    if predictions["used_future_realized_data"].astype(bool).any():
        raise AssertionError("Trained inference used future realized site data.")
    return predictions, pd.DataFrame(audits)


def _manifest_base(
    config_path: Path,
    input_path: Path,
    snapshot: Path,
    candidate: dict[str, Any],
    fine_tune_mode: str,
    frames: ResidualTrainingFrames,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "configured",
        "target_label": TARGET_LABEL,
        "verified_gross_factory_load": False,
        "base_model_id": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "base_model_snapshot": str(snapshot.resolve()),
        "fine_tune_mode": fine_tune_mode,
        "candidate_name": candidate["name"],
        "hyperparameters": candidate,
        "seed": int(candidate["seed"]),
        "train_period": {
            "start": frames.train_start.isoformat(),
            "end_exclusive": frames.train_end_exclusive.isoformat(),
            "weights_updated_from_this_period_only": True,
        },
        "validation_period": {
            "start": frames.selection_start.isoformat(),
            "end_exclusive": frames.selection_end_exclusive.isoformat(),
            "passed_as_autogluon_tuning_data": True,
            "updates_model_weights": False,
        },
        "autogluon_internal_validation_metric": "WQL",
        "autogluon_internal_checkpoint_selection": (
            "AutoGluon receives cumulative March-April tuning_data and may use WQL "
            "for validation, early stopping, and internal checkpoint selection."
        ),
        "outer_candidate_selection": (
            "April equal-SOC controller revenue descending, then residual WAPE and "
            "absolute bias ascending."
        ),
        "may_targets_passed_to_fit_or_selection": False,
        "may_used_for_selection": False,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "chronos_forecasting_version": package_version("chronos-forecasting"),
        "autogluon_version": package_version("autogluon.timeseries"),
        "python_version": sys.version,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "runtime_environment": preflight,
        "april_selection_results": None,
        "may_inference_completed": False,
        "may_evaluation_count": 0,
    }


def _manifest_compatibility_errors(
    manifest: dict[str, Any],
    *,
    input_sha256: str,
    config_sha256: str,
    snapshot: Path,
    fine_tune_mode: str,
    candidate: dict[str, Any],
) -> list[str]:
    expected = {
        "input_sha256": input_sha256,
        "config_sha256": config_sha256,
        "base_model_revision": MODEL_REVISION,
        "base_model_snapshot": str(snapshot.resolve()),
        "fine_tune_mode": fine_tune_mode,
        "hyperparameters": candidate,
    }
    return [
        key
        for key, value in expected.items()
        if manifest.get(key) != value
    ]


def _validate_completed_candidate(candidate_dir: Path, manifest: dict[str, Any]) -> None:
    required_files = [
        candidate_dir / "candidate_config.json",
        candidate_dir / "april_predictions.csv",
        candidate_dir / "april_inference_audit.csv",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    predictor_path = Path(str(manifest.get("predictor_path", "")))
    if not predictor_path.is_dir():
        missing.append(str(predictor_path))
    if missing:
        raise RuntimeError(
            f"Completed candidate has missing artifacts in {candidate_dir}: {missing}"
        )
    checkpoint, _ = find_fine_tuned_checkpoint(
        predictor_path, str(manifest["fine_tune_mode"])
    )
    if str(checkpoint.resolve()) != str(manifest.get("checkpoint_path")):
        raise RuntimeError(
            f"Completed candidate checkpoint path changed in {candidate_dir}."
        )
    if _directory_sha256(checkpoint) != manifest.get("checkpoint_sha256"):
        raise RuntimeError(
            f"Completed candidate checkpoint hash changed in {candidate_dir}."
        )
    fine_tune_mode = str(manifest["fine_tune_mode"])
    _validate_training_parameter_metadata(manifest, fine_tune_mode)
    if fine_tune_mode == "lora":
        required_reload = {
            "adapter_attached_after_reload": True,
            "adapter_active_after_reload": True,
            "base_model_fallback_rejected": True,
            "adapter_validation_status": "passed",
        }
        failures = [
            field
            for field, expected in required_reload.items()
            if manifest.get(field) != expected
        ]
        if failures or not manifest.get("active_adapter_names"):
            raise RuntimeError(
                "Completed LoRA candidate lacks verified active-adapter reload metadata: "
                f"{failures}."
            )
        try:
            reload_trainable = int(manifest["reload_trainable_parameters"])
            reload_frozen = int(manifest["reload_frozen_parameters"])
            reload_total = int(manifest["reload_total_parameters"])
            reload_tensors = int(manifest["reload_adapter_tensor_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Completed LoRA candidate lacks valid reload parameter metadata."
            ) from error
        if reload_total != reload_trainable + reload_frozen or reload_tensors <= 0:
            raise RuntimeError("Completed LoRA candidate reload metadata is inconsistent.")


def resolve_candidate_attempt(
    output_dir: Path,
    candidate: dict[str, Any],
    *,
    input_sha256: str,
    config_sha256: str,
    snapshot: Path,
    fine_tune_mode: str,
    resume: bool,
) -> tuple[Path, bool]:
    """Return a fresh attempt or one hash-compatible completed candidate."""
    base = output_dir / str(candidate["name"])
    attempts = [base, *sorted(output_dir.glob(f"{base.name}__attempt_*"))]
    existing = [path for path in attempts if path.exists()]
    if not existing:
        return base, False
    if not resume:
        raise FileExistsError(f"Candidate output already exists: {existing[0]}")
    for path in existing:
        manifest_path = path / "training_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Cannot resume incompatible partial candidate without manifest: {path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        errors = _manifest_compatibility_errors(
            manifest,
            input_sha256=input_sha256,
            config_sha256=config_sha256,
            snapshot=snapshot,
            fine_tune_mode=fine_tune_mode,
            candidate=candidate,
        )
        if errors:
            raise RuntimeError(
                f"Cannot resume incompatible candidate {path}; mismatched fields: {errors}"
            )
        if manifest.get("status") == "trained_checkpoint_reloaded":
            _validate_completed_candidate(path, manifest)
            return path, True
        if manifest.get("status") != "failed":
            raise RuntimeError(
                f"Cannot resume partial candidate {path} with status "
                f"{manifest.get('status')!r}."
            )
    attempt_number = 1
    while (output_dir / f"{base.name}__attempt_{attempt_number:02d}").exists():
        attempt_number += 1
    return output_dir / f"{base.name}__attempt_{attempt_number:02d}", False


def validate_may_prediction_gate(
    manifest: dict[str, Any],
    candidate_dir: Path,
) -> None:
    if manifest.get("status") != "trained_checkpoint_reloaded":
        raise RuntimeError("Selected candidate has no verified trained checkpoint.")
    selection = manifest.get("april_selection_results")
    if not isinstance(selection, dict):
        raise RuntimeError("May prediction requires completed April controller evaluation.")
    if selection.get("physical_requirements_satisfied") is not True:
        raise RuntimeError("May prediction requires a physically valid April controller result.")
    if selection.get("selected_within_mode") is not True:
        raise RuntimeError("May prediction requires selected_within_mode=true.")
    if selection.get("may_used_for_selection") is not False:
        raise RuntimeError("May data must not participate in candidate selection.")
    if manifest.get("may_inference_completed") is True or int(
        manifest.get("may_evaluation_count", 0)
    ) != 0:
        raise RuntimeError("The selected candidate has already been evaluated on May.")
    if (candidate_dir / "may_predictions.csv").exists() or (
        candidate_dir / "may_inference_audit.csv"
    ).exists():
        raise RuntimeError("May prediction artifacts already exist for this candidate.")


def run_training(args: argparse.Namespace) -> Path:
    if getattr(args, "allow_download", False):
        raise ValueError(
            "Residual fine-tuning is offline-only; --allow-download is not permitted."
        )
    config = load_residual_config(args.config)
    residual = pd.read_parquet(args.input)
    frames = prepare_residual_training_frames(residual, config)
    snapshot = resolve_pinned_snapshot(
        model_path=args.model_path,
        hf_home=args.hf_home,
        allow_download=False,
    )
    candidates = training_candidates(
        config,
        args.fine_tune_mode,
        smoke=args.stage == "smoke",
    )
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    batch_size_override = getattr(args, "batch_size_override", None)
    if batch_size_override is not None:
        if batch_size_override <= 0:
            raise ValueError("--batch-size-override must be positive.")
        candidates = [
            {
                **candidate,
                "name": f"{candidate['name']}_b{batch_size_override}",
                "batch_size": int(batch_size_override),
                "batch_size_was_explicitly_overridden": True,
            }
            for candidate in candidates
        ]
    dry_summary = {
        "status": "dry_run_validated",
        "fine_tune_mode": args.fine_tune_mode,
        "candidate_count": len(candidates),
        "candidate_grid": candidates,
        "train_rows": len(frames.train),
        "tuning_rows": len(frames.tuning),
        "train_max_timestamp": frames.train["timestamp"].max(),
        "tuning_max_timestamp": frames.tuning["timestamp"].max(),
        "train_end_exclusive": frames.train_end_exclusive,
        "selection_end_exclusive": frames.selection_end_exclusive,
        "target": TARGET_COLUMN,
        "known_future_covariates": list(CALENDAR_COLUMNS),
        "autogluon_validation_metric": "WQL",
        "april_updates_model_weights": False,
        "may_targets_passed_to_fit_or_selection": False,
        "may_used_for_selection": False,
        "snapshot": str(snapshot.resolve()),
    }
    if args.stage == "dry-run":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(args.output_dir / "dry_run.json", dry_summary)
        print(json.dumps(dry_summary, indent=2, default=str))
        return args.output_dir

    preflight = gpu_preflight()
    resume = bool(getattr(args, "resume", False))
    if args.output_dir.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=resume)
    _load_autogluon()
    input_sha256 = sha256_file(args.input)
    config_sha256 = sha256_file(args.config)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_dir, reused = resolve_candidate_attempt(
            args.output_dir,
            candidate,
            input_sha256=input_sha256,
            config_sha256=config_sha256,
            snapshot=snapshot,
            fine_tune_mode=args.fine_tune_mode,
            resume=resume,
        )
        if reused:
            completed.append(
                {
                    "candidate": candidate["name"],
                    "candidate_dir": str(candidate_dir.resolve()),
                    "reused": True,
                }
            )
            print(f"RESUME_REUSED: {candidate_dir}", flush=True)
            continue
        candidate_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "configured",
            "candidate_name": candidate["name"],
            "hyperparameters": candidate,
            "fine_tune_mode": args.fine_tune_mode,
            "base_model_revision": MODEL_REVISION,
            "base_model_snapshot": str(snapshot.resolve()),
            "input_sha256": input_sha256,
            "config_sha256": config_sha256,
            "stage": args.stage,
            "attempt_directory": str(candidate_dir.resolve()),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(candidate_dir / "candidate_config.json", candidate)
        try:
            manifest.update(
                _manifest_base(
                    args.config,
                    args.input,
                    snapshot,
                    candidate,
                    args.fine_tune_mode,
                    frames,
                    preflight,
                )
            )
            predictor, candidate_dataframe_class, model_name, stats = fit_residual_candidate(
                frames,
                config,
                candidate,
                args.fine_tune_mode,
                snapshot,
                candidate_dir,
                dataloader_num_workers=args.dataloader_num_workers,
            )
            april_issues = forecast_issue_times(
                config["selection_period"]["start"],
                config["selection_period"]["end_exclusive"],
                int(config["refresh_cadence_minutes"]),
            )
            max_issues = 1 if args.stage == "smoke" else None
            predictions, inference_audit = predict_saved_period(
                predictor,
                candidate_dataframe_class,
                model_name,
                residual,
                april_issues,
                str(candidate["name"]),
                args.fine_tune_mode,
                Path(stats["checkpoint_path"]),
                int(config["training_context_length"]),
                str(config["selection_period"]["name"]),
                max_issues=max_issues,
            )
            predictions.to_csv(
                candidate_dir / "april_predictions.csv", index=False, float_format="%.15g"
            )
            inference_audit.to_csv(
                candidate_dir / "april_inference_audit.csv",
                index=False,
                float_format="%.15g",
            )
            manifest.update(
                {
                    "status": "trained_checkpoint_reloaded",
                    "trained_model_name": model_name,
                    "april_issue_count": int(predictions["issue_time"].nunique()),
                    "april_inference_runtime_seconds": float(
                        inference_audit["runtime_seconds"].sum()
                    ),
                    "april_horizons_verified": list(range(1, PREDICTION_LENGTH + 1)),
                    "april_context_strictly_before_issue_time": True,
                    "known_future_covariates_verified": list(CALENDAR_COLUMNS),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "may_inference_completed": False,
                    "may_evaluation_count": 0,
                    **stats,
                }
            )
            try:
                import torch

                manifest["peak_allocated_gpu_bytes"] = max(
                    int(manifest.get("peak_allocated_gpu_bytes") or 0),
                    int(torch.cuda.max_memory_allocated(0)),
                )
                manifest["peak_reserved_gpu_bytes"] = max(
                    int(manifest.get("peak_reserved_gpu_bytes") or 0),
                    int(torch.cuda.max_memory_reserved(0)),
                )
            except (ImportError, RuntimeError):
                pass
        except Exception as error:
            manifest.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "oom_failure": "out of memory" in str(error).lower(),
                    "automatic_batch_size_retry": False,
                }
            )
            _write_json(candidate_dir / "training_manifest.json", manifest)
            failures.append(
                {
                    "candidate": candidate["name"],
                    "candidate_dir": str(candidate_dir.resolve()),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                f"CANDIDATE_FAILED: {candidate['name']}: {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        _write_json(candidate_dir / "training_manifest.json", manifest)
        completed.append(
            {
                "candidate": candidate["name"],
                "candidate_dir": str(candidate_dir.resolve()),
                "reused": False,
            }
        )
        print(f"CANDIDATE_COMPLETED: {candidate_dir}", flush=True)
    summary = {
        **dry_summary,
        "status": "completed" if completed else "failed",
        "completed_candidate_count": len(completed),
        "failed_candidate_count": len(failures),
        "completed_candidates": completed,
        "failed_candidates": failures,
        "resume": resume,
    }
    _write_json(args.output_dir / "run_summary.json", summary)
    if not completed:
        raise RuntimeError(
            f"All {len(candidates)} residual {args.fine_tune_mode} candidates failed."
        )
    return args.output_dir


def run_selected_may_inference(args: argparse.Namespace) -> Path:
    if args.candidate_dir is None:
        raise ValueError("--candidate-dir is required for --stage may-predict.")
    config = load_residual_config(args.config)
    if getattr(args, "allow_download", False):
        raise ValueError(
            "Residual fine-tuning is offline-only; --allow-download is not permitted."
        )
    snapshot = resolve_pinned_snapshot(
        model_path=args.model_path,
        hf_home=args.hf_home,
        allow_download=False,
    )
    gpu_preflight()
    manifest_path = args.candidate_dir / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_model_revision") != MODEL_REVISION:
        raise RuntimeError("Selected candidate uses a different base revision.")
    if manifest.get("input_sha256") != sha256_file(args.input):
        raise RuntimeError("Selected candidate input hash does not match --input.")
    if manifest.get("config_sha256") != sha256_file(args.config):
        raise RuntimeError("Selected candidate config hash does not match --config.")
    if manifest.get("base_model_snapshot") != str(snapshot.resolve()):
        raise RuntimeError("Selected candidate uses a different pinned snapshot path.")
    if manifest.get("fine_tune_mode") != args.fine_tune_mode:
        raise RuntimeError("Selected candidate mode does not match --fine-tune-mode.")
    validate_may_prediction_gate(manifest, args.candidate_dir)
    dataframe_class, predictor_class = _load_autogluon()
    predictor_path = Path(manifest["predictor_path"])
    predictor = predictor_class.load(str(predictor_path))
    inspection = verify_loaded_predictor(
        predictor,
        predictor_path,
        str(manifest["trained_model_name"]),
        str(manifest["fine_tune_mode"]),
        snapshot,
    )
    for field in (
        "checkpoint_path",
        "checkpoint_sha256",
        "adapter_model_sha256",
    ):
        if inspection.get(field) != manifest.get(field):
            raise RuntimeError(
                f"Selected candidate {field} changed after training: "
                f"{inspection.get(field)!r} != {manifest.get(field)!r}."
            )
    residual = pd.read_parquet(args.input)
    issues = forecast_issue_times(
        config["test_period"]["start"],
        config["test_period"]["end_exclusive"],
        int(config["refresh_cadence_minutes"]),
    )
    predictions, audit = predict_saved_period(
        predictor,
        dataframe_class,
        str(manifest["trained_model_name"]),
        residual,
        issues,
        str(manifest["candidate_name"]),
        str(manifest["fine_tune_mode"]),
        Path(manifest["checkpoint_path"]),
        int(config["training_context_length"]),
        str(config["test_period"]["name"]),
    )
    predictions.to_csv(
        args.candidate_dir / "may_predictions.csv", index=False, float_format="%.15g"
    )
    audit.to_csv(
        args.candidate_dir / "may_inference_audit.csv", index=False, float_format="%.15g"
    )
    manifest["may_inference_completed"] = True
    manifest["may_evaluation_count"] = 1
    manifest["may_inference_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["may_issue_count"] = int(predictions["issue_time"].nunique())
    manifest["may_inference_runtime_seconds"] = float(audit["runtime_seconds"].sum())
    _write_json(manifest_path, manifest)
    return args.candidate_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--config", default=Path("configs/foshan_chronos2_residual.json"), type=Path
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fine-tune-mode", choices=("lora", "full"), required=True)
    parser.add_argument(
        "--stage", choices=("dry-run", "smoke", "search", "may-predict"), required=True
    )
    parser.add_argument("--candidate-dir", default=None, type=Path)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--hf-home", default=None, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dataloader-num-workers", default=0, type=int)
    parser.add_argument("--max-candidates", default=None, type=int)
    parser.add_argument("--batch-size-override", default=None, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "may-predict":
        output = run_selected_may_inference(args)
    else:
        output = run_training(args)
    print(f"Residual {args.fine_tune_mode} stage {args.stage} completed at {output.resolve()}")


if __name__ == "__main__":
    main()
