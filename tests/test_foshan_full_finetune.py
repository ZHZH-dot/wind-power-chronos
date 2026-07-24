from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.prepare_foshan import add_calendar_covariates
from src.evaluation.foshan_benchmark import PREDICTION_COLUMNS
from src.training.foshan_chronos_full_finetune import (
    build_full_hyperparameters,
    find_full_checkpoint,
    forecast_key_set,
    inspect_loaded_full_model,
    load_full_config,
    load_full_predictor_for_evaluation,
    paired_daily_errors_vs_zero_shot,
    parameter_counts,
    prepare_full_finetune_frames,
    protected_artifact_manifest,
    retention_decision,
    run_candidate_with_oom_fallback,
    select_full_candidate,
    assert_protected_artifacts_unchanged,
    validate_identical_candidate_keys,
    validate_output_isolation,
)


CONFIG_PATH = Path("configs/foshan_chronos2_full_finetune.json")


def _table() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-03-01",
        "2026-07-01",
        freq="15min",
        tz="Asia/Shanghai",
    )
    position = np.arange(len(timestamps), dtype=float)
    pv = np.maximum(0.0, 500.0 * np.sin(2 * np.pi * (position % 96) / 96))
    grid = 100.0 * np.sin(2 * np.pi * position / (96 * 7))
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
    table.loc[10, ["pv_kw", "pv_kw_raw"]] = np.nan
    return add_calendar_covariates(table)


def _prediction_keys(
    candidate: str,
    issue_times: list[str],
) -> pd.DataFrame:
    rows = []
    for issue in issue_times:
        issue_time = pd.Timestamp(issue)
        for horizon in range(1, 3):
            rows.append(
                {
                    "model_name": f"chronos2_full_pv_calendar_{candidate}",
                    "postprocessing": "physical_clip_0_1700",
                    "issue_time": issue_time,
                    "target_time": issue_time + pd.Timedelta(minutes=15 * (horizon - 1)),
                    "horizon_step": horizon,
                }
            )
    return pd.DataFrame(rows)


def _may_metric(model_name: str, wape: float, active_mae: float) -> dict[str, object]:
    return {
        "split": "may_2026_selection",
        "target": "pv_kw",
        "model_name": model_name,
        "context_length": 672,
        "postprocessing": "physical_clip_0_1700",
        "n_origins": 31,
        "forecast_origin_set": "same",
        "wape": wape,
        "pv_active_mae": active_mae,
    }


def _full_prediction(candidate: str) -> pd.DataFrame:
    issue = pd.Timestamp("2026-05-01T00:00:00+08:00")
    common = {
        "run_id": "test",
        "split": "may_2026_selection",
        "issue_time": issue,
        "target_time": issue,
        "horizon_step": 1,
        "target": "pv_kw",
        "model_name": f"chronos2_full_pv_calendar_{candidate}",
        "model_id": f"amazon/chronos-2+full:{candidate}",
        "context_length": 672,
        "y_true_raw": 10.0,
        "y_true": 10.0,
        "is_missing_target": False,
        "p10": 8.0,
        "p50": 9.0,
        "p90": 10.0,
        "y_pred": 9.0,
        "mase_scale": 2.0,
        "used_future_covariates": True,
        "future_covariate_columns": "minute_of_day_sin",
        "provisional_target": False,
    }
    return pd.DataFrame(
        [
            {**common, "postprocessing": "raw"},
            {**common, "postprocessing": "physical_clip_0_1700"},
        ]
    )[PREDICTION_COLUMNS]


def test_full_hyperparameters_use_full_mode_without_lora() -> None:
    config = load_full_config(CONFIG_PATH)
    candidate = config["search_candidates"][0]

    hyperparameters = build_full_hyperparameters(
        "amazon/chronos-2",
        config,
        candidate,
        batch_size=4,
    )

    chronos = hyperparameters["Chronos2"]
    assert chronos["fine_tune"] is True
    assert chronos["fine_tune_mode"] == "full"
    assert "fine_tune_lora_config" not in chronos
    assert chronos["fine_tune_batch_size"] == 4
    assert chronos["fine_tune_trainer_kwargs"]["bf16"] is True
    assert chronos["fine_tune_trainer_kwargs"]["fp16"] is False
    assert chronos["ag_args"]["name_suffix"] == "Full"


def test_full_frames_train_only_on_march_april_and_never_use_grid_target() -> None:
    config = load_full_config(CONFIG_PATH)
    frames = prepare_full_finetune_frames(_table(), config)

    assert frames.targets == ["pv_kw"]
    assert set(frames.item_target_map.values()) == {"pv_kw"}
    assert frames.train["timestamp"].max() == pd.Timestamp(
        "2026-04-30T23:45:00+08:00"
    )
    assert frames.tuning["timestamp"].max() == pd.Timestamp(
        "2026-05-31T23:45:00+08:00"
    )
    assert "net_grid_kw" not in frames.train.columns
    assert "net_grid_kw" not in frames.tuning.columns
    assert not (frames.train["timestamp"] >= pd.Timestamp("2026-05-01T00:00:00+08:00")).any()
    assert not (frames.tuning["timestamp"] >= pd.Timestamp("2026-06-01T00:00:00+08:00")).any()


def test_all_intended_full_model_parameters_must_be_unfrozen() -> None:
    class Parameter:
        def __init__(self, count: int, requires_grad: bool) -> None:
            self.count = count
            self.requires_grad = requires_grad

        def numel(self) -> int:
            return self.count

    class Model:
        def __init__(self, parameters: list[Parameter]) -> None:
            self._parameters = parameters

        def parameters(self) -> list[Parameter]:
            return self._parameters

    counts = parameter_counts(
        Model([Parameter(10, True), Parameter(20, True)])
    )
    assert counts == {
        "trainable_parameters": 30,
        "total_parameters": 30,
        "frozen_parameters": 0,
    }

    with pytest.raises(RuntimeError, match="frozen base-model parameters"):
        parameter_counts(Model([Parameter(10, True), Parameter(20, False)]))


def test_loaded_full_predictor_is_validated_before_evaluation(tmp_path: Path) -> None:
    class Parameter:
        requires_grad = True

        def numel(self) -> int:
            return 5

    class CoreModel:
        def parameters(self) -> list[Parameter]:
            return [Parameter(), Parameter()]

    class Pipeline:
        model = CoreModel()

    class AutoGluonModel:
        _model_pipeline = None

        def get_hyperparameters(self) -> dict[str, object]:
            return {"fine_tune": True, "fine_tune_mode": "full"}

        def load_model_pipeline(self) -> None:
            self._model_pipeline = Pipeline()

    class Trainer:
        def load_model(self, name: str) -> AutoGluonModel:
            assert name == "Chronos2Full"
            return AutoGluonModel()

    class Predictor:
        _trainer = Trainer()

        def model_names(self) -> list[str]:
            return ["Chronos2Full"]

    class PredictorClass:
        loaded_path: str | None = None

        @classmethod
        def load(cls, path: str) -> Predictor:
            cls.loaded_path = path
            return Predictor()

    predictor, inspection = load_full_predictor_for_evaluation(
        PredictorClass,
        tmp_path / "predictor",
        "Chronos2Full",
    )

    assert isinstance(predictor, Predictor)
    assert PredictorClass.loaded_path == str(tmp_path / "predictor")
    assert inspection["fine_tune_mode"] == "full"
    assert inspection["trainable_parameters"] == inspection["total_parameters"] == 10

    with pytest.raises(RuntimeError, match="Expected a Chronos2Full"):
        inspect_loaded_full_model(Predictor(), "Chronos2LoRA")


def test_full_checkpoint_contains_base_weights_not_lora_adapter(
    tmp_path: Path,
) -> None:
    checkpoint = (
        tmp_path
        / "predictor"
        / "models"
        / "Chronos2Full"
        / "fine-tuned-ckpt"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"full-model")
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")

    result = find_full_checkpoint(tmp_path / "predictor", "Chronos2Full")

    assert result["checkpoint_path"] == str(checkpoint)
    assert result["checkpoint_size_bytes"] > 0
    assert result["checkpoint_weight_files"] == [
        str(checkpoint / "model.safetensors")
    ]

    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    with pytest.raises(RuntimeError, match="LoRA adapters"):
        find_full_checkpoint(tmp_path / "predictor", "Chronos2Full")


def test_selection_uses_may_only_and_pv_active_mae_tie_breaker() -> None:
    first = "chronos2_full_pv_calendar_first"
    second = "chronos2_full_pv_calendar_second"
    may_metrics = pd.DataFrame(
        [
            _may_metric(first, 0.4, 200.0),
            _may_metric(second, 0.4, 190.0),
        ]
    )
    records = [
        {"model_name": first, "candidate": "first"},
        {"model_name": second, "candidate": "second"},
    ]

    selected = select_full_candidate(may_metrics, records)

    assert selected["candidate"] == "second"
    assert selected["selected_on"] == "may_2026_selection"
    assert selected["selection_metric"] == "postprocessed_pv_wape"
    assert selected["tie_break_metric"] == "pv_active_mae"

    june_metrics = may_metrics.copy()
    june_metrics["split"] = "june_2026_test"
    with pytest.raises(ValueError, match="May metrics only"):
        select_full_candidate(june_metrics, records)


def test_all_candidates_must_share_exact_may_forecast_keys() -> None:
    origins = [
        "2026-05-01T00:00:00+08:00",
        "2026-05-02T00:00:00+08:00",
    ]
    first = _prediction_keys("first", origins)
    second = _prediction_keys("second", origins)

    result = validate_identical_candidate_keys({"first": first, "second": second})

    assert result["candidate_count"] == 2
    assert result["common_key_count"] == 4
    assert forecast_key_set(first) == forecast_key_set(second)

    second = second.iloc[:-1].copy()
    with pytest.raises(ValueError, match="different May forecast keys"):
        validate_identical_candidate_keys({"first": first, "second": second})


def test_full_output_cannot_overlap_zero_shot_or_lora(tmp_path: Path) -> None:
    root = tmp_path / "results" / "full_fine_tune"
    zero = tmp_path / "results" / "zero_shot" / "foshan"
    lora = tmp_path / "results" / "fine_tune" / "lora"
    output = root / "full_run"

    assert validate_output_isolation(output, zero, lora, root) == output.resolve()

    with pytest.raises(ValueError, match="under"):
        validate_output_isolation(zero, zero, lora, root)
    with pytest.raises(ValueError, match="overlaps protected LoRA"):
        validate_output_isolation(lora / "nested", zero, lora, tmp_path / "results")


def test_protected_zero_shot_and_lora_artifacts_are_hash_checked(
    tmp_path: Path,
) -> None:
    zero = tmp_path / "zero"
    lora_search = tmp_path / "lora" / "search"
    zero.mkdir()
    lora_search.mkdir(parents=True)
    paths = [
        zero / "predictions_long.csv",
        zero / "selected_configuration.json",
        lora_search / "june_predictions.csv",
        lora_search / "selected_configuration.json",
        lora_search / "summary.json",
    ]
    for index, path in enumerate(paths):
        path.write_text(f"artifact-{index}\n", encoding="utf-8")
    manifest = protected_artifact_manifest(zero, lora_search.parent)

    assert_protected_artifacts_unchanged(manifest)

    paths[0].write_text("modified\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifacts changed"):
        assert_protected_artifacts_unchanged(manifest)


def test_full_launcher_gates_search_with_tests_dry_run_and_smoke() -> None:
    script = Path("scripts/run_foshan_full_finetune_4090.sh").read_text(
        encoding="utf-8"
    )

    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert "python -m pytest tests" in script
    assert script.index("python -m pytest tests") < script.index("--stage dry-run")
    assert script.index("--stage dry-run") < script.index("--stage smoke")
    assert script.index("--stage smoke") < script.index("--stage search")
    assert "results/full_fine_tune/" in script


def test_oom_retry_changes_only_batch_size_and_is_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_full_config(CONFIG_PATH)
    candidate = config["search_candidates"][0]
    calls: list[int] = []

    def fake_fit(
        frames: object,
        config_value: dict[str, object],
        candidate_value: dict[str, object],
        batch_size: int,
        model_source: str,
        attempt_dir: Path,
        **kwargs: object,
    ) -> tuple[object, str, dict[str, object]]:
        calls.append(batch_size)
        assert candidate_value["learning_rate"] == candidate["learning_rate"]
        assert candidate_value["steps"] == candidate["steps"]
        assert candidate_value["seed"] == candidate["seed"]
        if batch_size == 4:
            raise RuntimeError("CUDA out of memory")
        predictor_path = attempt_dir / "predictor"
        predictor_path.mkdir()
        return (
            object(),
            "Chronos2Full",
            {
                "training_runtime_seconds": 1.0,
                "peak_allocated_gpu_bytes": 100,
                "peak_reserved_gpu_bytes": 200,
                "trained_model_name": "Chronos2Full",
                "predictor_path": str(predictor_path),
                "trainable_parameters": 10,
                "total_parameters": 10,
                "frozen_parameters": 0,
                "fine_tune": True,
                "fine_tune_mode": "full",
                "model_name": "Chronos2Full",
                "checkpoint_path": str(attempt_dir / "fine-tuned-ckpt"),
                "checkpoint_size_bytes": 123,
                "checkpoint_weight_files": ["model.safetensors"],
                "hyperparameters": {},
            },
        )

    def fake_predict(*args: object, **kwargs: object) -> tuple[pd.DataFrame, list, float]:
        return _full_prediction(str(candidate["name"])), [], 0.1

    monkeypatch.setattr(
        "src.training.foshan_chronos_full_finetune.fit_full_candidate",
        fake_fit,
    )
    monkeypatch.setattr(
        "src.training.foshan_chronos_full_finetune.run_full_origins",
        fake_predict,
    )
    candidate_dir = tmp_path / "candidate"

    result, predictions = run_candidate_with_oom_fallback(
        frames=object(),  # type: ignore[arg-type]
        table=pd.DataFrame(),
        config=config,
        candidate=candidate,
        model_source="amazon/chronos-2",
        candidate_dir=candidate_dir,
        origins=[pd.Timestamp("2026-05-01T00:00:00+08:00")],
        split_name="may_2026_selection",
        run_id="test",
        dataloader_num_workers=0,
        autogluon_classes=(object, object),
    )

    assert calls == [4, 2]
    assert result["batch_size"] == 2
    assert result["learning_rate"] == candidate["learning_rate"]
    assert result["failures_before_success"][0]["status"] == "oom"
    assert not predictions.empty

    calls.clear()
    resumed, resumed_predictions = run_candidate_with_oom_fallback(
        frames=object(),  # type: ignore[arg-type]
        table=pd.DataFrame(),
        config=config,
        candidate=candidate,
        model_source="amazon/chronos-2",
        candidate_dir=candidate_dir,
        origins=[pd.Timestamp("2026-05-01T00:00:00+08:00")],
        split_name="may_2026_selection",
        run_id="test",
        dataloader_num_workers=0,
        autogluon_classes=(object, object),
    )
    assert calls == []
    assert resumed["status"] == "completed"
    assert len(resumed_predictions) == len(predictions)


def test_paired_daily_bootstrap_and_retention_are_conservative() -> None:
    full_name = "chronos2_full_pv_calendar_selected"
    zero_name = "chronos2_joint_calendar"
    rows = []
    for day in range(1, 5):
        issue = pd.Timestamp(f"2026-06-{day:02d}T00:00:00+08:00")
        for model_name, prediction in ((full_name, 9.0), (zero_name, 8.0)):
            rows.append(
                {
                    "model_name": model_name,
                    "postprocessing": "physical_clip_0_1700",
                    "issue_time": issue,
                    "target_time": issue,
                    "horizon_step": 1,
                    "y_true": 10.0,
                    "p50": prediction,
                }
            )
    daily, bootstrap = paired_daily_errors_vs_zero_shot(
        pd.DataFrame(rows),
        full_name,
        timezone="Asia/Shanghai",
        bootstrap_samples=500,
        seed=42,
    )

    assert len(daily) == 4
    assert bootstrap["daily_wape"]["ci_upper_95"] < 0

    metrics = pd.DataFrame(
        [
            {
                "model_name": full_name,
                "wape": 0.40,
                "pv_active_mae": 190.0,
                "mean_pinball_loss": 30.0,
                "p10_p90_coverage": 0.81,
            },
            {
                "model_name": zero_name,
                "wape": 0.45,
                "pv_active_mae": 200.0,
                "mean_pinball_loss": 35.0,
                "p10_p90_coverage": 0.82,
            },
        ]
    )
    config = load_full_config(CONFIG_PATH)

    decision = retention_decision(metrics, full_name, bootstrap, config)

    assert decision["retain_full_model_as_forecasting_challenger"] is True
    assert decision["no_revenue_claim"] is True

    inconclusive = dict(bootstrap)
    inconclusive["daily_wape"] = dict(bootstrap["daily_wape"], ci_upper_95=0.01)
    decision = retention_decision(metrics, full_name, inconclusive, config)
    assert decision["retain_full_model_as_forecasting_challenger"] is False
    assert decision["recommended_forecasting_model"] == zero_name
