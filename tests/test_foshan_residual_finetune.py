from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.reconstruct_foshan_residual import (
    CALENDAR_COLUMNS,
    TARGET_COLUMN,
    add_residual_calendar_covariates,
)
from src.evaluation.foshan_residual_revenue import (
    build_parser as build_revenue_parser,
    prepare_revenue_output_dir,
    select_by_april_revenue,
)
from src.models.foshan_residual_zero_shot import (
    MODEL_ID,
    MODEL_REVISION,
    load_residual_config,
    resolve_pinned_snapshot,
)
import src.training.foshan_residual_finetune as residual_finetune
from src.training.foshan_residual_finetune import (
    build_residual_hyperparameters,
    capture_chronos2_training_state,
    find_fine_tuned_checkpoint,
    gpu_preflight,
    inspect_lora_model_state,
    prepare_residual_training_frames,
    predict_saved_period,
    resolve_candidate_attempt,
    run_training,
    training_candidates,
    validate_may_prediction_gate,
    verify_loaded_predictor,
)


def _table() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-03-01", "2026-06-01", freq="15min", inclusive="left", tz="Asia/Shanghai"
    )
    target = np.arange(len(timestamps), dtype=float)
    target[0] = -125.0
    table = pd.DataFrame({"timestamp": timestamps, TARGET_COLUMN: target})
    tariff = {minute: "shoulder" for minute in range(0, 1440, 15)}
    return add_residual_calendar_covariates(table, tariff)


def test_residual_training_split_is_march_only() -> None:
    config = load_residual_config(Path("configs/foshan_chronos2_residual.json"))
    frames = prepare_residual_training_frames(_table(), config)
    assert len(frames.train) == 31 * 96
    assert frames.train["timestamp"].max() == pd.Timestamp("2026-03-31 23:45")
    assert frames.tuning["timestamp"].max() == pd.Timestamp("2026-04-30 23:45")
    assert not (frames.train["timestamp"] >= pd.Timestamp("2026-04-01")).any()
    assert not (frames.tuning["timestamp"] >= pd.Timestamp("2026-05-01")).any()
    assert frames.train["target"].iloc[0] == -125.0
    assert frames.train.columns.tolist() == ["id", "timestamp", "target", *CALENDAR_COLUMNS]
    assert frames.tuning.columns.tolist() == ["id", "timestamp", "target", *CALENDAR_COLUMNS]


def test_lora_and_full_grids_and_hyperparameters_use_exact_snapshot(tmp_path: Path) -> None:
    config = load_residual_config(Path("configs/foshan_chronos2_residual.json"))
    snapshot = tmp_path / MODEL_REVISION
    lora = training_candidates(config, "lora")
    full = training_candidates(config, "full")
    assert len(lora) == 8
    assert len(full) == 4
    assert {(row["rank"], row["lora_alpha"]) for row in lora} == {(8, 16), (16, 32)}
    smoke = training_candidates(config, "lora", smoke=True)[0]
    assert smoke["steps"] == 5
    assert smoke["batch_size"] == 1
    assert training_candidates(config, "full", smoke=True)[0]["steps"] == 5

    lora_hparams = build_residual_hyperparameters(snapshot, config, lora[0], "lora")[
        "Chronos2"
    ]
    assert lora_hparams["model_path"] == str(snapshot)
    assert lora_hparams["fine_tune_mode"] == "lora"
    assert lora_hparams["fine_tune_lora_config"]["r"] in {8, 16}
    full_hparams = build_residual_hyperparameters(snapshot, config, full[0], "full")[
        "Chronos2"
    ]
    assert full_hparams["fine_tune_mode"] == "full"
    assert "fine_tune_lora_config" not in full_hparams
    assert full_hparams["fine_tune_trainer_kwargs"]["gradient_accumulation_steps"] == 4
    assert full[0]["batch_size"] * full[0]["gradient_accumulation_steps"] == 16


def _write_snapshot(root: Path) -> Path:
    snapshot = root / MODEL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["Chronos2Model"]}), encoding="utf-8"
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def test_local_snapshot_validation_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Pinned Chronos-2 snapshot is absent"):
        resolve_pinned_snapshot(hf_home=tmp_path, allow_download=False)
    incomplete = tmp_path / MODEL_REVISION
    incomplete.mkdir()
    (incomplete / "config.json").write_text(
        json.dumps({"architectures": ["Chronos2Model"]}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="incomplete"):
        resolve_pinned_snapshot(model_path=incomplete, allow_download=False)


class FakeTensor:
    def __init__(self, size: int, *, requires_grad: bool = False) -> None:
        self.size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.size


class FakeMergedModel:
    def __init__(self, parameter_count: int = 1_000) -> None:
        self._parameters = [FakeTensor(parameter_count)]

    def parameters(self) -> list[FakeTensor]:
        return self._parameters


class FakePeftModel:
    def __init__(
        self,
        *,
        trainable: bool = False,
        enabled: bool = True,
        active: list[str] | None = None,
        rank: int = 8,
        alpha: int = 16,
        base_source: str = MODEL_ID,
        merged_model: object | None = None,
    ) -> None:
        self.active = ["default"] if active is None else active
        self.enabled = enabled
        self.peft_config = {
            "default": SimpleNamespace(
                peft_type="LORA",
                r=rank,
                lora_alpha=alpha,
                base_model_name_or_path=base_source,
            )
        }
        self._parameters = [
            FakeTensor(1_000),
            FakeTensor(32, requires_grad=trainable),
            FakeTensor(32, requires_grad=trainable),
        ]
        self.adapter_state = {
            "default": {
                "layer.lora_A.weight": FakeTensor(32),
                "layer.lora_B.weight": FakeTensor(32),
            }
        }
        self.merged_model = merged_model or FakeMergedModel()
        self.merge_count = 0

    def parameters(self) -> list[FakeTensor]:
        return self._parameters

    def get_model_status(self) -> SimpleNamespace:
        trainable = sum(value.numel() for value in self._parameters if value.requires_grad)
        total = sum(value.numel() for value in self._parameters)
        return SimpleNamespace(
            enabled=self.enabled,
            active_adapters=self.active,
            available_adapters=["default"],
            peft_types={"default": "LORA"},
            num_adapter_layers=2,
            trainable_params=trainable,
            total_params=total,
        )

    def merge_and_unload(self) -> object:
        self.merge_count += 1
        return self.merged_model


@pytest.fixture
def fake_peft_state_api(monkeypatch: pytest.MonkeyPatch) -> None:
    peft_module = ModuleType("peft")

    def get_peft_model_state_dict(
        model: FakePeftModel, *, adapter_name: str
    ) -> dict[str, FakeTensor]:
        return model.adapter_state.get(adapter_name, {})

    peft_module.get_peft_model_state_dict = get_peft_model_state_dict  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "peft", peft_module)


class FakeBaseModel:
    def parameters(self) -> list[FakeTensor]:
        return [FakeTensor(1_000)]


class FakeAutoPeftModel:
    next_model: object | None = None

    @classmethod
    def from_pretrained(cls, _path: Path | str, **_kwargs: object) -> object:
        if cls.next_model is None:
            raise RuntimeError("Fake AutoPeftModel has no configured model.")
        return cls.next_model


class FakeAutoGluonModel:
    def __init__(
        self,
        model_path: Path,
        mode: str,
        core: object,
        *,
        reload_path: Path | None = None,
        load_adapter: bool = True,
        pipeline_core_override: object | None = None,
    ) -> None:
        self.model_path = model_path
        self.mode = mode
        self.core = core
        self.reload_path = reload_path
        self.load_adapter = load_adapter
        self.pipeline_core_override = pipeline_core_override
        self._model_pipeline = None

    def get_hyperparameters(self) -> dict[str, object]:
        result: dict[str, object] = {
            "fine_tune": True,
            "fine_tune_mode": self.mode,
            "model_path": str(self.model_path),
        }
        if self.mode == "lora":
            result["fine_tune_lora_config"] = {"r": 8, "lora_alpha": 16}
        return result

    def load_model_pipeline(self) -> None:
        if self.mode == "lora" and self.load_adapter:
            FakeAutoPeftModel.next_model = self.core
            loaded = FakeAutoPeftModel.from_pretrained(self.reload_path)
            merged = loaded.merge_and_unload()
            core = self.pipeline_core_override or merged
        else:
            core = self.core
        self._model_pipeline = SimpleNamespace(model=core)


class FakeTrainer:
    def __init__(self, model: FakeAutoGluonModel) -> None:
        self.model = model

    def load_model(self, _name: str) -> FakeAutoGluonModel:
        return self.model


class FakePredictor:
    def __init__(self, model: FakeAutoGluonModel) -> None:
        self._trainer = FakeTrainer(model)

    def model_names(self) -> list[str]:
        return ["Chronos2ResidualLoRA"]


def _verify_fake_loaded_predictor(
    predictor: FakePredictor,
    predictor_path: Path,
    snapshot: Path,
) -> dict[str, object]:
    return verify_loaded_predictor(
        predictor,
        predictor_path,
        "Chronos2ResidualLoRA",
        "lora",
        snapshot,
        auto_peft_model_class=FakeAutoPeftModel,
        peft_model_class=FakePeftModel,
    )


def _write_adapter_artifacts(
    checkpoint: Path,
    *,
    rank: int = 8,
    alpha: int = 16,
    base_source: str = MODEL_ID,
) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": base_source,
                "inference_mode": True,
                "r": rank,
                "lora_alpha": alpha,
                "auto_mapping": {"base_model_class": "Chronos2Model"},
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")


def test_training_parameter_capture_records_real_lora_state(
    fake_peft_state_api: None,
) -> None:
    class FakeChronosPipeline:
        def fit(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(model=FakePeftModel(trainable=True))

    original_fit = FakeChronosPipeline.fit
    with capture_chronos2_training_state(
        "lora", pipeline_class=FakeChronosPipeline
    ) as metadata:
        FakeChronosPipeline().fit(finetune_mode="lora")

    assert FakeChronosPipeline.fit is original_fit
    assert metadata["trainable_parameters"] == 64
    assert metadata["frozen_parameters"] == 1_000
    assert metadata["total_parameters"] == 1_064
    assert metadata["total_parameters"] == (
        metadata["trainable_parameters"] + metadata["frozen_parameters"]
    )
    assert metadata["training_adapter_active"] is True
    assert metadata["training_parameter_capture_point"] == "chronos2_pipeline_fit_return"


def test_checkpoint_verification_separates_reload_state_and_rejects_base_fallback(
    tmp_path: Path,
    fake_peft_state_api: None,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    predictor_path = tmp_path / "predictor"
    checkpoint = predictor_path / "models" / "Chronos2ResidualLoRA" / "fine-tuned-ckpt"
    _write_adapter_artifacts(checkpoint, base_source=str(snapshot.resolve()))
    premerge = FakePeftModel(
        trainable=False,
        base_source=str(snapshot.resolve()),
    )
    predictor = FakePredictor(
        FakeAutoGluonModel(
            snapshot,
            "lora",
            premerge,
            reload_path=checkpoint,
        )
    )
    result = _verify_fake_loaded_predictor(predictor, predictor_path, snapshot)
    assert result["base_model_source_verified"] == str(snapshot.resolve())
    assert result["fine_tune_mode_verified"] == "lora"
    assert result["reload_trainable_parameters"] == 0
    assert result["reload_frozen_parameters"] == 1_000
    assert result["reload_total_parameters"] == 1_000
    assert result["reload_adapter_premerge_attached"] is True
    assert result["reload_adapter_premerge_active"] is True
    assert result["reload_premerge_active_adapter_names"] == ["default"]
    assert result["reload_adapter_tensor_count"] == 2
    assert result["adapter_application_mode_after_reload"] == "merged_and_unloaded"
    assert result["adapter_merged_after_reload"] is True
    assert result["adapter_effect_applied_after_reload"] is True
    assert result["merged_inference_model_identity_verified"] is True
    assert result["adapter_attached_after_reload"] is False
    assert result["adapter_active_after_reload"] is False
    assert result["active_adapter_names"] == []
    assert result["adapter_validation_status"] == "passed"
    assert result["base_model_fallback_rejected"] is True
    assert result["reload_adapter_checkpoint_loaded"] == str(checkpoint.resolve())
    assert premerge.merge_count == 1

    wrong_source = FakePredictor(
        FakeAutoGluonModel(
            Path("amazon/chronos-2"),
            "lora",
            premerge,
            reload_path=checkpoint,
        )
    )
    with pytest.raises(RuntimeError, match="silently changed base checkpoint"):
        _verify_fake_loaded_predictor(wrong_source, predictor_path, snapshot)

    fallback = FakePredictor(
        FakeAutoGluonModel(
            snapshot,
            "lora",
            FakeBaseModel(),
            reload_path=checkpoint,
        )
    )
    with pytest.raises(RuntimeError, match="base-only"):
        _verify_fake_loaded_predictor(fallback, predictor_path, snapshot)


def test_reload_requires_exact_checkpoint_snapshot_and_merged_pipeline_core(
    tmp_path: Path,
    fake_peft_state_api: None,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    predictor_path = tmp_path / "predictor"
    checkpoint = predictor_path / "models" / "Chronos2ResidualLoRA" / "fine-tuned-ckpt"
    _write_adapter_artifacts(checkpoint, base_source=str(snapshot.resolve()))

    def predictor_for(
        core: object,
        *,
        reload_path: Path = checkpoint,
        load_adapter: bool = True,
        pipeline_core_override: object | None = None,
    ) -> FakePredictor:
        return FakePredictor(
            FakeAutoGluonModel(
                snapshot,
                "lora",
                core,
                reload_path=reload_path,
                load_adapter=load_adapter,
                pipeline_core_override=pipeline_core_override,
            )
        )

    wrong_checkpoint = tmp_path / "adapter-elsewhere"
    wrong_checkpoint.mkdir()
    with pytest.raises(RuntimeError, match="did not use the verified LoRA checkpoint"):
        _verify_fake_loaded_predictor(
            predictor_for(
                FakePeftModel(base_source=str(snapshot.resolve())),
                reload_path=wrong_checkpoint,
            ),
            predictor_path,
            snapshot,
        )

    with pytest.raises(RuntimeError, match="skipped the verified LoRA checkpoint"):
        _verify_fake_loaded_predictor(
            predictor_for(FakeBaseModel(), load_adapter=False),
            predictor_path,
            snapshot,
        )

    different_snapshot = tmp_path / "different-snapshot"
    different_snapshot.mkdir()
    with pytest.raises(RuntimeError, match="does not match the pinned Chronos-2 snapshot"):
        _verify_fake_loaded_predictor(
            predictor_for(
                FakePeftModel(base_source=str(different_snapshot.resolve()))
            ),
            predictor_path,
            snapshot,
        )

    with pytest.raises(RuntimeError, match="does not use the model returned"):
        _verify_fake_loaded_predictor(
            predictor_for(
                FakePeftModel(base_source=str(snapshot.resolve())),
                pipeline_core_override=FakeMergedModel(),
            ),
            predictor_path,
            snapshot,
        )


def test_disabled_or_mismatched_lora_adapter_is_rejected(
    tmp_path: Path,
    fake_peft_state_api: None,
) -> None:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    predictor_path = tmp_path / "predictor"
    checkpoint = predictor_path / "models" / "Chronos2ResidualLoRA" / "fine-tuned-ckpt"
    _write_adapter_artifacts(checkpoint, base_source=str(snapshot.resolve()))
    disabled = FakePredictor(
        FakeAutoGluonModel(
            snapshot,
            "lora",
            FakePeftModel(
                trainable=False,
                enabled=False,
                base_source=str(snapshot.resolve()),
            ),
            reload_path=checkpoint,
        )
    )
    with pytest.raises(RuntimeError, match="not enabled"):
        _verify_fake_loaded_predictor(disabled, predictor_path, snapshot)

    _write_adapter_artifacts(
        checkpoint,
        rank=16,
        alpha=32,
        base_source=str(snapshot.resolve()),
    )
    with pytest.raises(RuntimeError, match="does not match saved AutoGluon"):
        _verify_fake_loaded_predictor(
            FakePredictor(
                FakeAutoGluonModel(
                    snapshot,
                    "lora",
                    FakePeftModel(
                        trainable=False,
                        base_source=str(snapshot.resolve()),
                    ),
                    reload_path=checkpoint,
                )
            ),
            predictor_path,
            snapshot,
        )

    _write_adapter_artifacts(checkpoint, base_source=str(snapshot.resolve()))
    with pytest.raises(RuntimeError, match="Loaded LoRA adapter r"):
        _verify_fake_loaded_predictor(
            FakePredictor(
                FakeAutoGluonModel(
                    snapshot,
                    "lora",
                    FakePeftModel(
                        trainable=False,
                        rank=16,
                        alpha=32,
                        base_source=str(snapshot.resolve()),
                    ),
                    reload_path=checkpoint,
                )
            ),
            predictor_path,
            snapshot,
        )


def test_lora_checkpoint_requires_complete_unique_nonempty_artifacts(tmp_path: Path) -> None:
    predictor_path = tmp_path / "predictor"
    checkpoint = predictor_path / "models" / "Chronos2ResidualLoRA" / "fine-tuned-ckpt"
    checkpoint.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="adapter_config.json"):
        find_fine_tuned_checkpoint(predictor_path, "lora")
    _write_adapter_artifacts(checkpoint)
    (checkpoint / "adapter_model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="adapter_model.safetensors"):
        find_fine_tuned_checkpoint(predictor_path, "lora")
    _write_adapter_artifacts(checkpoint)
    (checkpoint / "adapter_config.json").unlink()
    with pytest.raises(RuntimeError, match="adapter_config.json"):
        find_fine_tuned_checkpoint(predictor_path, "lora")
    _write_adapter_artifacts(checkpoint)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(RuntimeError, match="nonempty"):
        find_fine_tuned_checkpoint(predictor_path, "lora")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"one")
    nested = checkpoint / "duplicate"
    nested.mkdir()
    (nested / "adapter_model.safetensors").write_bytes(b"two")
    with pytest.raises(RuntimeError, match="adapter_model.safetensors"):
        find_fine_tuned_checkpoint(predictor_path, "lora")


def test_lora_model_state_requires_loaded_adapter_tensors(
    fake_peft_state_api: None,
) -> None:
    model = FakePeftModel(trainable=False)
    model.adapter_state["default"] = {}
    with pytest.raises(RuntimeError, match="no loaded tensors"):
        inspect_lora_model_state(model, require_trainable=False)


def test_real_peft_wrapper_is_valid_during_training_and_after_freeze() -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class TinyLinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, inputs: object) -> object:
            return self.linear(inputs)

    model = peft.get_peft_model(
        TinyLinearModel(),
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["linear"],
            bias="none",
        ),
    )
    training_state = inspect_lora_model_state(model, require_trainable=True)
    assert training_state["trainable_parameters"] > 0
    assert training_state["frozen_parameters"] > 0
    assert training_state["total_parameters"] == (
        training_state["trainable_parameters"]
        + training_state["frozen_parameters"]
    )
    assert training_state["adapter_attached"] is True
    assert training_state["adapter_active"] is True
    assert training_state["active_adapter_names"] == ["default"]
    assert training_state["adapter_tensor_count"] > 0
    assert training_state["adapter_parameter_count"] > 0

    for parameter in model.parameters():
        parameter.requires_grad = False

    reload_state = inspect_lora_model_state(model, require_trainable=False)
    assert reload_state["trainable_parameters"] == 0
    assert reload_state["frozen_parameters"] == reload_state["total_parameters"]
    assert reload_state["adapter_attached"] is True
    assert reload_state["adapter_active"] is True
    assert reload_state["active_adapter_names"] == ["default"]
    assert reload_state["adapter_tensor_count"] > 0
    assert reload_state["adapter_parameter_count"] > 0

    merged = model.merge_and_unload()
    assert merged is not model
    assert not isinstance(merged, peft.PeftModel)
    with pytest.raises(ValueError, match="No adapter layers found"):
        peft.get_model_status(merged)


class FakeCuda:
    def __init__(
        self,
        name: str,
        *,
        available: bool = True,
        count: int = 1,
        bf16: bool = True,
        vram_gib: int = 24,
    ):
        self.name = name
        self.available = available
        self.count = count
        self.bf16 = bf16
        self.vram_gib = vram_gib

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_name(self, _index: int) -> str:
        return self.name

    def is_bf16_supported(self) -> bool:
        return self.bf16

    def get_device_properties(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=self.vram_gib * 1024**3)


def _fake_torch(cuda: FakeCuda) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=cuda,
        __version__="2.test",
        version=SimpleNamespace(cuda="12.test"),
    )


@pytest.mark.parametrize("name", ["NVIDIA GeForce RTX 4090", "NVIDIA A100", "NVIDIA RTX 5090"])
def test_generic_single_gpu_bf16_preflight_accepts_compatible_names(name: str) -> None:
    result = gpu_preflight(_fake_torch(FakeCuda(name)), {"CUDA_VISIBLE_DEVICES": "0"})
    assert result["gpu_name"] == name
    assert result["bf16_supported"] is True


@pytest.mark.parametrize(
    "cuda,match",
    [
        (FakeCuda("CPU", available=False, count=0), "exactly one visible"),
        (FakeCuda("two GPUs", count=2), "exactly one visible"),
        (FakeCuda("legacy GPU", bf16=False), "must support BF16"),
        (FakeCuda("small GPU", vram_gib=16), "at least 23 GiB VRAM"),
    ],
)
def test_gpu_preflight_rejects_invalid_runtime(cuda: FakeCuda, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        gpu_preflight(_fake_torch(cuda), {"CUDA_VISIBLE_DEVICES": "0"})


class FakeTimeSeriesFrame:
    @classmethod
    def from_data_frame(cls, frame: pd.DataFrame, **_: object) -> pd.DataFrame:
        return frame.copy()


class FakeInferencePredictor:
    def __init__(self) -> None:
        self.calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    def predict(
        self,
        context: pd.DataFrame,
        *,
        known_covariates: pd.DataFrame,
        model: str,
    ) -> pd.DataFrame:
        assert model == "Chronos2ResidualLoRA"
        self.calls.append((context.copy(), known_covariates.copy()))
        return pd.DataFrame(index=range(len(known_covariates)))


def test_trained_inference_is_causal_and_uses_only_calendar_covariates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = FakeInferencePredictor()
    monkeypatch.setattr(
        residual_finetune,
        "normalize_chronos_quantiles",
        lambda *_args, expected_times, **_kwargs: pd.DataFrame(
            {"timestamp": expected_times, "p10": -1.0, "p50": 0.0, "p90": 1.0}
        ),
    )
    issue = pd.Timestamp("2026-04-01 00:00", tz="Asia/Shanghai")
    predictions, audit = predict_saved_period(
        predictor,
        FakeTimeSeriesFrame,
        "Chronos2ResidualLoRA",
        _table(),
        [issue],
        "candidate",
        "lora",
        tmp_path / "checkpoint",
        672,
        "april_2026_selection",
    )
    context, future = predictor.calls[0]
    assert context["timestamp"].max() < issue.tz_localize(None)
    assert future.columns.tolist() == ["id", "timestamp", *CALENDAR_COLUMNS]
    assert len(predictions) == 96
    assert predictions["horizon_step"].tolist() == list(range(1, 97))
    assert predictions["known_future_covariates"].eq("|".join(CALENDAR_COLUMNS)).all()
    assert not predictions["used_future_realized_data"].any()
    assert audit["forecast_rows"].tolist() == [96]


def _candidate_artifacts(
    output_dir: Path,
    candidate: dict[str, object],
    snapshot: Path,
) -> Path:
    candidate_dir = output_dir / str(candidate["name"])
    checkpoint = candidate_dir / "predictor" / "models" / "Chronos2ResidualLoRA" / "fine-tuned-ckpt"
    _write_adapter_artifacts(checkpoint)
    (candidate_dir / "candidate_config.json").write_text("{}", encoding="utf-8")
    (candidate_dir / "april_predictions.csv").write_text("issue_time\n", encoding="utf-8")
    (candidate_dir / "april_inference_audit.csv").write_text(
        "runtime_seconds\n", encoding="utf-8"
    )
    manifest = {
        "status": "trained_checkpoint_reloaded",
        "input_sha256": "input",
        "config_sha256": "config",
        "base_model_revision": MODEL_REVISION,
        "base_model_snapshot": str(snapshot.resolve()),
        "fine_tune_mode": "lora",
        "hyperparameters": candidate,
        "predictor_path": str((candidate_dir / "predictor").resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": residual_finetune._directory_sha256(checkpoint),
        "trainable_parameters": 64,
        "frozen_parameters": 1_000,
        "total_parameters": 1_064,
        "training_adapter_attached": True,
        "training_adapter_active": True,
        "training_active_adapter_names": ["default"],
        "training_adapter_tensor_count": 2,
        "training_adapter_validation_status": "passed",
        "training_parameter_capture_point": "chronos2_pipeline_fit_return",
        "reload_trainable_parameters": 0,
        "reload_frozen_parameters": 1_000,
        "reload_total_parameters": 1_000,
        "reload_adapter_premerge_attached": True,
        "reload_adapter_premerge_active": True,
        "reload_premerge_active_adapter_names": ["default"],
        "reload_adapter_tensor_count": 2,
        "reload_adapter_parameter_count": 64,
        "reload_adapter_checkpoint_loaded": str(checkpoint.resolve()),
        "adapter_application_mode_after_reload": "merged_and_unloaded",
        "adapter_merged_after_reload": True,
        "adapter_effect_applied_after_reload": True,
        "merged_inference_model_identity_verified": True,
        "adapter_attached_after_reload": False,
        "adapter_active_after_reload": False,
        "active_adapter_names": [],
        "base_model_fallback_rejected": True,
        "adapter_validation_status": "passed",
    }
    (candidate_dir / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return candidate_dir


def test_candidate_resume_reuses_only_compatible_completed_output(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "model")
    output = tmp_path / "run"
    output.mkdir()
    candidate = {
        "name": "candidate",
        "rank": 8,
        "lora_alpha": 16,
        "learning_rate": 5e-5,
        "steps": 100,
        "batch_size": 8,
        "seed": 42,
    }
    candidate_dir = _candidate_artifacts(output, candidate, snapshot)
    resolved, reused = resolve_candidate_attempt(
        output,
        candidate,
        input_sha256="input",
        config_sha256="config",
        snapshot=snapshot,
        fine_tune_mode="lora",
        resume=True,
    )
    assert resolved == candidate_dir
    assert reused is True
    with pytest.raises(RuntimeError, match="mismatched fields"):
        resolve_candidate_attempt(
            output,
            candidate,
            input_sha256="different",
            config_sha256="config",
            snapshot=snapshot,
            fine_tune_mode="lora",
            resume=True,
        )


def test_candidate_resume_rejects_partial_output(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "model")
    output = tmp_path / "run"
    partial = output / "candidate"
    partial.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="partial candidate without manifest"):
        resolve_candidate_attempt(
            output,
            {"name": "candidate"},
            input_sha256="input",
            config_sha256="config",
            snapshot=snapshot,
            fine_tune_mode="lora",
            resume=True,
        )


def test_candidate_failure_is_recorded_and_later_candidate_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "residual.parquet"
    _table().to_parquet(input_path, index=False)
    config_path = Path("configs/foshan_chronos2_residual.json")
    snapshot = _write_snapshot(tmp_path / "model")
    candidates = [
        {"name": "fails", "rank": 8, "lora_alpha": 16, "learning_rate": 5e-5, "steps": 5, "batch_size": 1, "seed": 42},
        {"name": "succeeds", "rank": 8, "lora_alpha": 16, "learning_rate": 5e-5, "steps": 5, "batch_size": 1, "seed": 42},
    ]
    monkeypatch.setattr(residual_finetune, "training_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(residual_finetune, "gpu_preflight", lambda: {"gpu_name": "fake"})
    monkeypatch.setattr(residual_finetune, "_load_autogluon", lambda: (object, object))

    def fake_fit(*args: object, **kwargs: object) -> tuple[object, object, str, dict[str, object]]:
        candidate = args[2]
        candidate_dir = Path(args[5])
        if candidate["name"] == "fails":
            raise RuntimeError("synthetic CUDA failure")
        checkpoint = candidate_dir / "checkpoint"
        checkpoint.mkdir()
        return object(), object, "Chronos2ResidualLoRA", {
            "checkpoint_path": str(checkpoint.resolve()),
            "predictor_path": str((candidate_dir / "predictor").resolve()),
            "checkpoint_sha256": "checkpoint",
        }

    monkeypatch.setattr(residual_finetune, "fit_residual_candidate", fake_fit)
    monkeypatch.setattr(
        residual_finetune,
        "predict_saved_period",
        lambda *_args, **_kwargs: (
            pd.DataFrame({"issue_time": [pd.Timestamp("2026-04-01")] * 96}),
            pd.DataFrame({"runtime_seconds": [0.01]}),
        ),
    )
    output = tmp_path / "output"
    run_training(
        Namespace(
            input=input_path,
            config=config_path,
            output_dir=output,
            fine_tune_mode="lora",
            stage="search",
            model_path=snapshot,
            hf_home=None,
            allow_download=False,
            dataloader_num_workers=0,
            max_candidates=None,
            batch_size_override=None,
            resume=False,
        )
    )
    failed = json.loads((output / "fails" / "training_manifest.json").read_text())
    summary = json.loads((output / "run_summary.json").read_text())
    assert failed["status"] == "failed"
    assert failed["automatic_batch_size_retry"] is False
    assert summary["failed_candidate_count"] == 1
    assert summary["completed_candidate_count"] == 1
    completed = json.loads(
        (output / "succeeds" / "training_manifest.json").read_text()
    )
    assert completed["hyperparameters"] == candidates[1]


def test_cpu_dry_run_does_not_import_autogluon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "residual.parquet"
    _table().to_parquet(input_path, index=False)
    snapshot = _write_snapshot(tmp_path / "model")
    monkeypatch.setattr(
        residual_finetune,
        "_load_autogluon",
        lambda: pytest.fail("dry run imported AutoGluon"),
    )
    output = tmp_path / "dry"
    run_training(
        Namespace(
            input=input_path,
            config=Path("configs/foshan_chronos2_residual.json"),
            output_dir=output,
            fine_tune_mode="lora",
            stage="dry-run",
            model_path=snapshot,
            hf_home=None,
            allow_download=False,
            dataloader_num_workers=0,
            max_candidates=None,
            batch_size_override=None,
            resume=False,
        )
    )
    summary = json.loads((output / "dry_run.json").read_text())
    assert summary["target"] == TARGET_COLUMN
    assert summary["candidate_count"] == 8
    assert summary["may_used_for_selection"] is False


def test_may_prediction_gate_requires_april_selection_and_allows_only_once(tmp_path: Path) -> None:
    base = {
        "status": "trained_checkpoint_reloaded",
        "may_inference_completed": False,
        "may_evaluation_count": 0,
    }
    with pytest.raises(RuntimeError, match="April controller evaluation"):
        validate_may_prediction_gate(base, tmp_path)
    selected = {
        **base,
        "april_selection_results": {
            "physical_requirements_satisfied": True,
            "selected_within_mode": True,
            "may_used_for_selection": False,
        },
    }
    validate_may_prediction_gate(selected, tmp_path)
    selected["may_inference_completed"] = True
    selected["may_evaluation_count"] = 1
    with pytest.raises(RuntimeError, match="already been evaluated"):
        validate_may_prediction_gate(selected, tmp_path)


def test_revenue_selection_order_is_revenue_then_wape_then_bias() -> None:
    revenue = pd.DataFrame(
        {
            "candidate": ["low", "wape", "bias", "invalid"],
            "raw_revenue_yuan": [10.0, 20.0, 20.0, 30.0],
            "physical_requirements_satisfied": [True, True, True, False],
        }
    )
    metrics = pd.DataFrame(
        {
            "candidate": ["low", "wape", "bias", "invalid"],
            "wape": [1.0, 0.5, 0.5, 0.1],
            "absolute_bias": [0.1, 2.0, 1.0, 0.0],
        }
    )
    assert select_by_april_revenue(revenue, metrics, revenue["candidate"])["candidate"] == "bias"


def _revenue_args(tmp_path: Path) -> Namespace:
    paths = {}
    for name in (
        "site_workbook",
        "storage_workbook",
        "dispatch_input",
        "residual_predictions",
        "april_pv_predictions",
        "may_pv_predictions",
        "may_pv_selection",
        "gross_load_predictions",
        "residual_data",
        "training_config",
    ):
        path = tmp_path / f"{name}.input"
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    return Namespace(
        **paths,
        trained_run_dir=[],
        model_path=tmp_path / "snapshot",
        mip_relative_gap=1e-7,
        resume=False,
    )


def test_revenue_resume_requires_matching_run_identity(tmp_path: Path) -> None:
    args = _revenue_args(tmp_path)
    output = tmp_path / "revenue"
    prepare_revenue_output_dir(output, args)
    args.resume = True
    resumed = prepare_revenue_output_dir(output, args)
    assert resumed["resume_count"] == 1
    args.residual_data.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible inputs"):
        prepare_revenue_output_dir(output, args)


def test_canonical_zero_shot_defaults_and_stage_gated_launchers() -> None:
    parser = build_revenue_parser()
    args = parser.parse_args(
        [
            "--site-workbook", "site.xlsx",
            "--storage-workbook", "storage.xlsx",
            "--dispatch-input", "dispatch.csv",
            "--residual-predictions", "residual.csv",
            "--april-pv-predictions", "april.csv",
        ]
    )
    assert str(args.may_pv_predictions).replace("\\", "/") == (
        "results/zero_shot/foshan_chronos2/predictions_long.csv"
    )
    assert str(args.may_pv_selection).replace("\\", "/") == (
        "results/zero_shot/foshan_chronos2/selected_configuration.json"
    )
    lora_launcher = Path("scripts/run_foshan_residual_lora_4090.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "$#" -ne 2 ]]' in lora_launcher
    assert 'STAGE="$1"' in lora_launcher
    assert 'INPUT="$2"' in lora_launcher
    assert '${1:?' not in lora_launcher
    assert '${2:?' not in lora_launcher
    assert '--stage "${STAGE}"' in lora_launcher
    assert "--stage search" not in lora_launcher
    revenue_launcher = Path("scripts/run_foshan_residual_revenue_eval.sh").read_text(
        encoding="utf-8"
    )
    assert "results/zero_shot/foshan_chronos2/predictions_long.csv" in revenue_launcher
    assert "INPUT_OK:" in revenue_launcher
    assert "INPUT_MISSING:" in revenue_launcher


def _bash_executable() -> str | None:
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.is_file():
        return str(git_bash)
    return shutil.which("bash")


def _bash_path(path: Path) -> str:
    return path.resolve().as_posix()


@pytest.mark.parametrize("stage", ["dry-run", "smoke", "search"])
def test_lora_launcher_passes_exact_valid_stage_without_running_training(
    tmp_path: Path,
    stage: str,
) -> None:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("Bash is unavailable for the launcher regression test.")
    snapshot = _write_snapshot(tmp_path / "model with spaces")
    input_path = tmp_path / "signed residual with spaces.parquet"
    input_path.write_bytes(b"fixture")
    output_path = tmp_path / f"output {stage}"
    launcher = Path("scripts/run_foshan_residual_lora_4090.sh")
    wrapper = r'''
python() {
  printf 'PYTHON_CALL:'
  printf ' <%s>' "$@"
  printf '\n'
}
nvidia-smi() {
  printf 'NVIDIA_SMI_CALL\n'
}
export -f python nvidia-smi
bash "$1" "$2" "$3"
'''
    environment = {
        **dict(os.environ),
        "CHRONOS_MODEL_PATH": _bash_path(snapshot),
        "OUTPUT_DIR": _bash_path(output_path),
        "CONFIG": "configs/foshan_chronos2_residual.json",
    }
    completed = subprocess.run(
        [
            bash,
            "-c",
            wrapper,
            "launcher-test",
            _bash_path(launcher),
            stage,
            _bash_path(input_path),
        ],
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    module_calls = [
        line
        for line in completed.stdout.splitlines()
        if "src.training.foshan_residual_finetune" in line
    ]
    assert len(module_calls) == 1
    assert f" <--stage> <{stage}>" in module_calls[0]
    for other_stage in {"dry-run", "smoke", "search"} - {stage}:
        assert f" <--stage> <{other_stage}>" not in module_calls[0]
    if stage == "smoke":
        assert " <--stage> <search>" not in completed.stdout


def test_lora_launcher_rejects_missing_and_unknown_stages_before_work(tmp_path: Path) -> None:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("Bash is unavailable for the launcher regression test.")
    launcher = _bash_path(Path("scripts/run_foshan_residual_lora_4090.sh"))
    input_path = tmp_path / "input.parquet"
    input_path.write_bytes(b"fixture")
    for arguments in ([], ["dry-run"], ["unknown", _bash_path(input_path)]):
        completed = subprocess.run(
            [bash, launcher, *arguments],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 64
        assert "PYTHON_CALL" not in completed.stdout


def test_residual_launchers_pin_revision_and_do_not_use_device_map_auto() -> None:
    for filename in (
        "run_foshan_residual_zero_shot.sh",
        "run_foshan_residual_lora_4090.sh",
        "run_foshan_residual_full_4090.sh",
        "run_foshan_residual_revenue_eval.sh",
    ):
        text = (Path("scripts") / filename).read_text(encoding="utf-8")
        assert MODEL_REVISION in text
        assert "device_map=\"auto\"" not in text
        assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "--fine-tune-mode lora" in Path(
        "scripts/run_foshan_residual_lora_4090.sh"
    ).read_text(encoding="utf-8")
    assert "--fine-tune-mode full" in Path(
        "scripts/run_foshan_residual_full_4090.sh"
    ).read_text(encoding="utf-8")
