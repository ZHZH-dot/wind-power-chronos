import subprocess
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepare_foshan import add_calendar_covariates
from src.evaluation.foshan_benchmark import (
    PREDICTION_COLUMNS,
    build_forecast_window,
    period_origins,
    run_causal_baselines,
)
from src.training.foshan_chronos_finetune import (
    _run_search,
    _window_context_frame,
    _window_future_frame,
    build_foshan_lora_hyperparameters,
    load_lora_config,
    prepare_foshan_finetune_frames,
    run_lora_origins,
    select_lora_candidate,
)
from src.utils import runtime


def _foshan_table() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-03-01",
        "2026-07-01",
        freq="15min",
        tz="Asia/Shanghai",
    )
    position = np.arange(len(timestamps), dtype=float)
    pv = np.maximum(0.0, 800.0 * np.sin(2 * np.pi * (position % 96) / 96))
    grid = 400.0 * np.sin(2 * np.pi * position / (96 * 7))
    table = pd.DataFrame(
        {
            "id": "foshan_site",
            "timestamp": timestamps,
            "pv_kw_raw": pv,
            "pv_kw": pv,
            "net_grid_kw_raw": grid,
            "net_grid_kw": grid,
            "is_missing_pv_kw": False,
            "is_missing_net_grid_kw": False,
        }
    )
    table.loc[100, ["pv_kw_raw", "pv_kw"]] = np.nan
    table.loc[200, ["net_grid_kw_raw", "net_grid_kw"]] = np.nan
    return add_calendar_covariates(table)


def test_git_commit_is_exact_and_failure_is_not_silently_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert runtime.git_commit() == expected

    def fail(*args: object, **kwargs: object) -> object:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(runtime.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Could not run git rev-parse HEAD"):
        runtime.git_commit()


def test_foshan_frames_use_march_april_and_preserve_signed_joint_target() -> None:
    config = load_lora_config(Path("configs/foshan_chronos2_lora.json"))
    table = _foshan_table()
    pv_variant, joint_variant = config["variants"]

    pv = prepare_foshan_finetune_frames(table, config, pv_variant)
    joint = prepare_foshan_finetune_frames(table, config, joint_variant)

    assert pv.train["timestamp"].min() == pd.Timestamp(
        "2026-03-01T00:00:00+08:00"
    )
    assert pv.train["timestamp"].max() == pd.Timestamp(
        "2026-04-30T23:45:00+08:00"
    )
    assert pv.tuning["timestamp"].max() == pd.Timestamp(
        "2026-05-31T23:45:00+08:00"
    )
    assert pv.tuning["timestamp"].max() < pd.Timestamp(
        config["test_period"]["start"]
    )
    assert set(joint.item_target_map.values()) == {"pv_kw", "net_grid_kw"}
    grid_id = "foshan_site::net_grid_kw"
    assert joint.train.loc[joint.train["id"] == grid_id, "target"].min() < 0
    assert str(joint.train["timestamp"].dt.tz) == "Asia/Shanghai"


def test_foshan_lora_hyperparameters_use_rank_lora_and_calendar_covariates() -> None:
    config = load_lora_config(Path("configs/foshan_chronos2_lora.json"))
    candidate = config["search_candidates"][0]

    hyperparameters = build_foshan_lora_hyperparameters(
        "amazon/chronos-2",
        config,
        candidate,
        bf16=True,
        fp16=False,
        dataloader_num_workers=0,
    )

    assert list(hyperparameters) == ["Chronos2"]
    chronos = hyperparameters["Chronos2"]
    assert chronos["fine_tune_mode"] == "lora"
    assert chronos["fine_tune_lora_config"] == {"r": 8, "lora_alpha": 16}
    assert chronos["disable_known_covariates"] is False
    assert chronos["disable_past_covariates"] is True
    assert chronos["fine_tune_trainer_kwargs"]["bf16"] is True
    assert chronos["fine_tune_trainer_kwargs"]["disable_data_parallel"] is True


@pytest.mark.parametrize("variant_name", ["pv_calendar", "joint_calendar"])
def test_lora_inference_uses_target_free_calendar_future_and_scores_pv(
    variant_name: str,
) -> None:
    class FakeTimeSeriesDataFrame:
        @classmethod
        def from_data_frame(
            cls, frame: pd.DataFrame, **kwargs: object
        ) -> pd.DataFrame:
            return frame.copy()

    class FakePredictor:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def predict(
            self,
            data: pd.DataFrame,
            known_covariates: pd.DataFrame,
            model: str,
        ) -> pd.DataFrame:
            self.calls.append(
                {
                    "data": data.copy(),
                    "future": known_covariates.copy(),
                    "model": model,
                }
            )
            assert "target" not in known_covariates.columns
            assert not {"pv_kw", "net_grid_kw"}.intersection(
                known_covariates.columns
            )
            return pd.DataFrame(
                {
                    "item_id": known_covariates["id"],
                    "timestamp": known_covariates["timestamp"],
                    "0.1": 1.0,
                    "0.5": 2.0,
                    "0.9": 3.0,
                }
            )

    config = load_lora_config(Path("configs/foshan_chronos2_lora.json"))
    variant = next(
        item for item in config["variants"] if item["name"] == variant_name
    )
    table = _foshan_table()
    issue = pd.Timestamp(config["selection_period"]["start"])
    predictor = FakePredictor()

    predictions, skipped, _ = run_lora_origins(
        predictor,
        FakeTimeSeriesDataFrame,
        "Chronos2LoRA",
        table,
        config,
        variant,
        "candidate",
        [issue],
        "may_2026_selection",
        "test",
        "amazon/chronos-2",
    )

    assert not skipped
    assert len(predictor.calls) == 1
    call = predictor.calls[0]
    future = call["future"]
    context = call["data"]
    assert isinstance(future, pd.DataFrame)
    assert isinstance(context, pd.DataFrame)
    assert future["timestamp"].dt.tz is None
    assert context["timestamp"].dt.tz is None
    assert set(predictions["target"]) == {"pv_kw"}
    assert set(predictions["postprocessing"]) == {
        "raw",
        "physical_clip_0_1700",
    }
    assert str(predictions["target_time"].dt.tz) == "Asia/Shanghai"
    if variant_name == "joint_calendar":
        assert set(context["id"]) == {
            "foshan_site::pv_kw",
            "foshan_site::net_grid_kw",
        }
        assert set(future["id"]) == set(context["id"])


def test_window_helpers_do_not_expose_measured_targets_in_joint_future() -> None:
    config = load_lora_config(Path("configs/foshan_chronos2_lora.json"))
    table = _foshan_table()
    issue = pd.Timestamp(config["selection_period"]["start"])
    targets = ["pv_kw", "net_grid_kw"]
    window, reason = build_forecast_window(
        table,
        issue,
        targets=targets,
        context_length=672,
        prediction_length=96,
        known_future_covariates=config["calendar_covariates"],
        causal_fill_limit=3,
        frequency="15min",
    )
    assert reason is None
    assert window is not None

    context = _window_context_frame(
        window, targets, config["calendar_covariates"], "foshan_site"
    )
    future = _window_future_frame(
        window, targets, config["calendar_covariates"], "foshan_site"
    )

    assert "target" in context.columns
    assert not {"target", "pv_kw", "net_grid_kw"}.intersection(future.columns)


def test_lora_selection_uses_may_pv_wape_then_active_mae() -> None:
    metrics = pd.DataFrame(
        {
            "split": ["may_2026_selection"] * 2,
            "target": ["pv_kw"] * 2,
            "model_name": [
                "chronos2_lora_pv_calendar_a",
                "chronos2_lora_joint_calendar_b",
            ],
            "model_id": ["a", "b"],
            "context_length": [672, 672],
            "postprocessing": ["physical_clip_0_1700"] * 2,
            "provisional_target": [False, False],
            "n_origins": [31, 31],
            "forecast_origin_set": ["same", "same"],
            "wape": [0.2, 0.2],
            "pv_active_mae": [10.0, 9.0],
            "mae": [8.0, 8.0],
        }
    )
    records = [
        {
            "model_name": "chronos2_lora_pv_calendar_a",
            "variant": "pv_calendar",
            "candidate": "a",
        },
        {
            "model_name": "chronos2_lora_joint_calendar_b",
            "variant": "joint_calendar",
            "candidate": "b",
        },
    ]

    selected = select_lora_candidate(metrics, records)

    assert selected["variant"] == "joint_calendar"
    assert selected["selection_metric"] == "postprocessed_pv_wape"
    assert selected["tie_break_metric"] == "pv_active_mae"


def test_4090_launcher_gates_search_on_tests_and_smoke() -> None:
    script = Path("scripts/run_foshan_finetune_4090.sh").read_text(
        encoding="utf-8"
    )

    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert "python scripts/preflight_finetune_4090.py" in script
    assert "python -m pytest tests" in script
    assert script.index("--stage dry-run") < script.index("--stage smoke")
    assert script.index("--stage smoke") < script.index("--stage search")
    assert "results/fine_tune/" in script


def test_mocked_search_selects_on_may_and_evaluates_june_once(
    tmp_path: Path,
) -> None:
    class FakeTimeSeriesDataFrame:
        @classmethod
        def from_data_frame(
            cls, frame: pd.DataFrame, **kwargs: object
        ) -> pd.DataFrame:
            return frame.copy()

    class FakePredictor:
        registry: dict[str, "FakePredictor"] = {}

        def __init__(self, path: str, **kwargs: object) -> None:
            self.path = path
            self.fit_kwargs: dict[str, object] | None = None
            self.registry[path] = self

        @classmethod
        def load(cls, path: str) -> "FakePredictor":
            return cls.registry[path]

        def fit(self, **kwargs: object) -> "FakePredictor":
            self.fit_kwargs = dict(kwargs)
            return self

        def model_names(self) -> list[str]:
            return ["Chronos2LoRA"]

        def predict(
            self,
            data: pd.DataFrame,
            known_covariates: pd.DataFrame,
            model: str,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "item_id": known_covariates["id"],
                    "timestamp": known_covariates["timestamp"],
                    "0.1": 1.0,
                    "0.5": 2.0,
                    "0.9": 3.0,
                }
            )

    table = _foshan_table()
    config = load_lora_config(Path("configs/foshan_chronos2_lora.json"))
    config["search_candidates"] = [config["search_candidates"][0]]
    zero_shot_dir = tmp_path / "zero_shot"
    zero_shot_dir.mkdir()
    june_origins = period_origins(
        table,
        config["test_period"],
        prediction_length=96,
        frequency="15min",
        stride_steps=96,
        timezone="Asia/Shanghai",
    )
    baselines = run_causal_baselines(
        table,
        june_origins,
        split_name="june_2026_test",
        run_id="frozen",
        causal_fill_limit=3,
    )
    zero_shot = baselines[
        (baselines["target"] == "pv_kw")
        & (baselines["model_name"] == "previous_day")
    ].copy()
    zero_shot["model_name"] = "chronos2_pv_calendar"
    zero_shot["model_id"] = "amazon/chronos-2"
    zero_shot["context_length"] = 672
    zero_shot["postprocessing"] = "physical_clip_0_1700"
    zero_shot["p10"] = zero_shot["p50"] - 1.0
    zero_shot["p90"] = zero_shot["p50"] + 1.0
    pd.concat([baselines, zero_shot], ignore_index=True)[PREDICTION_COLUMNS].to_csv(
        zero_shot_dir / "predictions_long.csv",
        index=False,
    )
    (zero_shot_dir / "selected_configuration.json").write_text(
        json.dumps(
            {
                "targets": {
                    "pv_kw": {
                        "model_name": "chronos2_pv_calendar",
                        "context_length": 672,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "lora"
    args = Namespace(
        output_dir=output_dir,
        zero_shot_dir=zero_shot_dir,
        dataloader_num_workers=0,
    )

    summary = _run_search(
        args,
        table,
        config,
        "amazon/chronos-2",
        {"bf16": True, "fp16": False},
        (FakeTimeSeriesDataFrame, FakePredictor),
    )

    assert summary["may_origin_count"] == 31
    assert summary["june_origin_count"] == 30
    assert summary["june_evaluation_count"] == 1
    assert summary["selected_configuration"]["variant"] in {
        "pv_calendar",
        "joint_calendar",
    }
    assert (output_dir / "search" / "common_scored_metrics_june.csv").is_file()
    log = pd.read_csv(output_dir / "search" / "search_log.csv")
    assert len(log) == 2
    assert log["status"].eq("completed").all()
