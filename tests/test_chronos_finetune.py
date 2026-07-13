from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.splits import (
    build_split_manifest,
    load_benchmark_config,
    write_split_manifest,
)
from src.training.chronos_finetune import (
    DEFAULT_MODEL_ID,
    FineTuneFrames,
    build_arg_parser,
    build_chronos2_hyperparameters,
    fit_with_autogluon,
    prepare_finetune_frames,
    resolve_split_manifest,
    run,
    validate_and_normalize_data,
)


def _regular_data(periods: int = 10) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=periods, freq="1h")
    rows = []
    for turbine_id in ("1", "2"):
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "id": turbine_id,
                    "timestamp": timestamp,
                    "target": float(index),
                    "Wspd": float(index + 1),
                    "is_imputed_target": turbine_id == "1" and index == 2,
                }
            )
    return pd.DataFrame(rows)


def test_cli_defaults_are_chronos2_lora_defaults() -> None:
    args = build_arg_parser().parse_args(
        ["--input", "data.parquet", "--output-dir", "results/fine_tune/test"]
    )

    assert args.model_id == DEFAULT_MODEL_ID
    assert args.mode == "multivariate"
    assert args.prediction_length == 72
    assert args.context_length == 168
    assert args.steps == 1000
    assert args.learning_rate == pytest.approx(1e-5)
    assert args.batch_size == 32
    assert args.inference_batch_size == 64
    assert args.seed == 42


def test_training_and_cumulative_validation_stop_before_test() -> None:
    data = validate_and_normalize_data(
        _regular_data(),
        mode="multivariate",
        covariates=["Wspd"],
        frequency="1h",
    )
    config = load_benchmark_config(Path("configs/splits/sdwpf_70_10_20.json"))
    manifest = build_split_manifest(data["timestamp"], config)

    frames = prepare_finetune_frames(
        data,
        manifest,
        mode="multivariate",
        covariates=["Wspd"],
        prediction_length=1,
        context_length=1,
        max_turbines=1,
    )

    timestamps = pd.date_range("2020-01-01", periods=10, freq="1h")
    assert frames.turbine_ids == ["1"]
    assert frames.train["timestamp"].max() == timestamps[6]
    assert frames.validation_context["timestamp"].max() == timestamps[7]
    assert frames.validation_context["timestamp"].max() < timestamps[8]
    assert len(frames.train) == 7
    assert len(frames.validation_context) == 8
    assert pd.isna(frames.train.loc[2, "target"])
    assert frames.n_masked_imputed_train == 1


def test_finetune_reuses_evaluation_manifest_boundaries(tmp_path: Path) -> None:
    data = _regular_data()
    evaluation_config = load_benchmark_config(Path("configs/sdwpf_benchmark.json"))
    evaluation_manifest = build_split_manifest(data["timestamp"], evaluation_config)
    manifest_path = tmp_path / "evaluation_manifest.json"
    write_split_manifest(evaluation_manifest, manifest_path)

    fine_tune_config = load_benchmark_config(Path("configs/splits/sdwpf_70_10_20.json"))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    resolved = resolve_split_manifest(
        data,
        split_config=fine_tune_config,
        output_dir=output_dir,
        split_manifest_path=manifest_path,
    )

    assert resolved["benchmark"] == evaluation_manifest["benchmark"]
    assert resolved["splits"] == evaluation_manifest["splits"]
    assert (output_dir / "resolved_split_manifest.json").exists()


def test_hyperparameters_enable_only_chronos2_lora() -> None:
    hyperparameters = build_chronos2_hyperparameters(
        model_id="amazon/chronos-2",
        mode="multivariate",
        prediction_length=72,
        context_length=168,
        steps=1000,
        learning_rate=1e-5,
        batch_size=32,
        inference_batch_size=64,
        seed=42,
    )

    assert list(hyperparameters) == ["Chronos2"]
    chronos = hyperparameters["Chronos2"]
    assert chronos["model_path"] == "amazon/chronos-2"
    assert chronos["fine_tune"] is True
    assert chronos["fine_tune_mode"] == "lora"
    assert chronos["disable_known_covariates"] is True
    assert chronos["disable_past_covariates"] is False
    assert chronos["fine_tune_batch_size"] == 32
    assert chronos["batch_size"] == 64


def test_dry_run_does_not_create_or_load_predictor(tmp_path: Path) -> None:
    input_path = tmp_path / "sdwpf.csv"
    _regular_data().to_csv(input_path, index=False)
    output_dir = tmp_path / "fine_tune"
    args = build_arg_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "multivariate",
            "--covariates",
            "Wspd",
            "--prediction-length",
            "1",
            "--context-length",
            "1",
            "--max-turbines",
            "1",
            "--dry-run",
        ]
    )

    summary = run(args)

    assert summary["dry_run"] is True
    assert not (output_dir / "predictor").exists()
    assert (output_dir / "resolved_split_manifest.json").exists()
    assert (output_dir / "run_config.json").exists()


def test_autogluon_fit_receives_train_and_cumulative_validation_only(tmp_path: Path) -> None:
    class FakeTimeSeriesDataFrame:
        @classmethod
        def from_data_frame(cls, frame, **kwargs):
            return frame.copy()

    class FakePredictor:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.fit_kwargs = None

        def fit(self, **kwargs):
            self.fit_kwargs = kwargs
            return self

    frames = FineTuneFrames(
        train=pd.DataFrame(
            {"id": ["1"], "timestamp": [pd.Timestamp("2020-01-01")], "target": [1.0]}
        ),
        validation_context=pd.DataFrame(
            {
                "id": ["1", "1"],
                "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                "target": [1.0, 2.0],
            }
        ),
        turbine_ids=["1"],
        n_masked_imputed_train=0,
        n_masked_imputed_validation=0,
    )
    hyperparameters = build_chronos2_hyperparameters(
        DEFAULT_MODEL_ID,
        "univariate",
        72,
        168,
        10,
        1e-5,
        2,
        4,
        42,
    )

    predictor = fit_with_autogluon(
        frames,
        output_dir=tmp_path,
        frequency="1h",
        prediction_length=72,
        hyperparameters=hyperparameters,
        seed=42,
        autogluon_classes=(FakeTimeSeriesDataFrame, FakePredictor),
    )

    assert predictor.fit_kwargs["train_data"].equals(frames.train)
    assert predictor.fit_kwargs["tuning_data"].equals(frames.validation_context)
    assert predictor.fit_kwargs["enable_ensemble"] is False
    assert predictor.fit_kwargs["refit_full"] is False
    assert predictor.init_kwargs["quantile_levels"] == [0.1, 0.5, 0.9]


def test_autodl_script_has_cpu_dry_run_before_training() -> None:
    script = Path("scripts/run_finetune_autodl.sh").read_text(encoding="utf-8")

    assert "requirements-finetune-autodl.txt" in script
    assert "python -m pytest tests" in script
    assert '"${COMMON_ARGS[@]}" --dry-run' in script
    assert "CUDA_VISIBLE_DEVICES" in script
