from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.reconstruct_foshan_residual import (
    TARGET_COLUMN,
    add_residual_calendar_covariates,
)
from src.models.foshan_residual_zero_shot import MODEL_REVISION, load_residual_config
from src.training.foshan_residual_finetune import (
    build_residual_hyperparameters,
    prepare_residual_training_frames,
    training_candidates,
    verify_loaded_predictor,
)


def _table() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-03-01", "2026-06-01", freq="15min", inclusive="left", tz="Asia/Shanghai"
    )
    table = pd.DataFrame({"timestamp": timestamps, TARGET_COLUMN: np.arange(len(timestamps))})
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


def test_lora_and_full_grids_and_hyperparameters_use_exact_snapshot(tmp_path: Path) -> None:
    config = load_residual_config(Path("configs/foshan_chronos2_residual.json"))
    snapshot = tmp_path / MODEL_REVISION
    lora = training_candidates(config, "lora")
    full = training_candidates(config, "full")
    assert len(lora) == 16
    assert len(full) == 4
    assert training_candidates(config, "lora", smoke=True)[0]["steps"] == 5
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


class FakeModel:
    def __init__(self, model_path: Path, mode: str) -> None:
        self.model_path = model_path
        self.mode = mode

    def get_hyperparameters(self) -> dict[str, object]:
        return {
            "fine_tune": True,
            "fine_tune_mode": self.mode,
            "model_path": str(self.model_path),
        }


class FakeTrainer:
    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def load_model(self, _name: str) -> FakeModel:
        return self.model


class FakePredictor:
    def __init__(self, model: FakeModel) -> None:
        self._trainer = FakeTrainer(model)

    def model_names(self) -> list[str]:
        return ["Chronos2ResidualLoRA"]


def test_checkpoint_verification_cannot_fall_back_to_base_model(tmp_path: Path) -> None:
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    predictor_path = tmp_path / "predictor"
    checkpoint = predictor_path / "models" / "Chronos2ResidualLoRA" / "fine-tuned-ckpt"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    predictor = FakePredictor(FakeModel(snapshot, "lora"))
    result = verify_loaded_predictor(
        predictor,
        predictor_path,
        "Chronos2ResidualLoRA",
        "lora",
        snapshot,
        inspect_parameters=False,
    )
    assert result["base_model_source_verified"] == str(snapshot.resolve())
    assert result["fine_tune_mode_verified"] == "lora"

    fallback = FakePredictor(FakeModel(Path("amazon/chronos-2"), "lora"))
    with pytest.raises(RuntimeError, match="silently changed base checkpoint"):
        verify_loaded_predictor(
            fallback,
            predictor_path,
            "Chronos2ResidualLoRA",
            "lora",
            snapshot,
            inspect_parameters=False,
        )


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
