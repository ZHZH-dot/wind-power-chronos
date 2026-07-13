import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.chronos_finetune_predict import (
    AutoGluonChronos2Adapter,
    build_arg_parser,
    load_verified_predictor,
    run,
)
from src.evaluation.splits import (
    build_split_manifest,
    load_benchmark_config,
    write_split_manifest,
)


class FakeTimeSeriesDataFrame:
    @classmethod
    def from_data_frame(cls, frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        return frame.copy()


class FakePredictor:
    prediction_length = 2
    loaded: "FakePredictor | None" = None

    def __init__(self, model_names: list[str] | None = None) -> None:
        self.names = model_names or ["Chronos2LoRA"]
        self.calls: list[dict[str, object]] = []

    @classmethod
    def load(cls, path: str) -> "FakePredictor":
        assert Path(path).is_dir()
        cls.loaded = cls()
        return cls.loaded

    def model_names(self) -> list[str]:
        return self.names

    def predict(self, data: pd.DataFrame, model: str) -> pd.DataFrame:
        cutoff = pd.Timestamp(data["timestamp"].max())
        self.calls.append({"data": data.copy(), "model": model, "cutoff": cutoff})
        timestamps = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=2, freq="1h")
        return pd.DataFrame(
            {
                "item_id": [str(data["id"].iloc[0])] * 2,
                "timestamp": timestamps,
                "0.1": [1.0, 2.0],
                "0.5": [2.0, 3.0],
                "0.9": [3.0, 4.0],
            }
        )


def _data(periods: int = 20) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=periods, freq="1h")
    rows: list[dict[str, object]] = []
    for turbine_id in ("2", "1", "5", "3", "4"):
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "id": turbine_id,
                    "timestamp": timestamp,
                    "target": float(index),
                    "Wspd": float(index + 1),
                    "is_imputed_target": index == 16 and turbine_id == "1",
                }
            )
    return pd.DataFrame(rows)


def _saved_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    predictor_path = tmp_path / "run" / "predictor"
    checkpoint = (
        predictor_path
        / "models"
        / "Chronos2LoRA"
        / "fine-tuned-ckpt"
        / "adapter_model.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")

    input_path = tmp_path / "sdwpf.csv"
    data = _data()
    data.to_csv(input_path, index=False)
    config = load_benchmark_config(Path("configs/splits/sdwpf_70_10_20.json"))
    manifest = build_split_manifest(data["timestamp"], config)
    manifest_path = predictor_path.parent / "resolved_split_manifest.json"
    write_split_manifest(manifest, manifest_path)

    run_config = {
        "mode": "multivariate",
        "prediction_length": 2,
        "context_length": 4,
        "n_turbines": 5,
        "covariates": ["Wspd"],
        "quantile_levels": [0.1, 0.5, 0.9],
        "hyperparameters": {
            "Chronos2": {
                "batch_size": 64,
                "fine_tune_mode": "lora",
                "disable_known_covariates": True,
            }
        },
    }
    (predictor_path.parent / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )
    return predictor_path, input_path, manifest_path


def test_saved_predictor_uses_existing_rolling_protocol(tmp_path: Path) -> None:
    predictor_path, input_path, manifest_path = _saved_run(tmp_path)
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.csv"
    metadata_path = tmp_path / "metadata.json"
    args = build_arg_parser().parse_args(
        [
            "--predictor-path",
            str(predictor_path),
            "--input",
            str(input_path),
            "--split-manifest",
            str(manifest_path),
            "--output",
            str(predictions_path),
            "--metrics-output",
            str(metrics_path),
            "--metadata-output",
            str(metadata_path),
            "--covariates",
            "Wspd",
            "--prediction-length",
            "2",
            "--context-length",
            "4",
            "--horizons",
            "1",
            "2",
            "--max-turbines",
            "5",
            "--max-windows-per-turbine",
            "1",
        ]
    )

    metadata = run(args, autogluon_classes=(FakeTimeSeriesDataFrame, FakePredictor))
    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)

    assert metadata["selected_turbine_ids"] == ["1", "2", "3", "4", "5"]
    assert metadata["known_future_covariates"] == []
    assert len(predictions) == 10
    assert list(predictions.columns) == [
        "id",
        "mode",
        "horizon",
        "cutoff_timestamp",
        "timestamp",
        "y_true",
        "is_imputed_target",
        "p10",
        "p50",
        "p90",
        "y_pred",
        "test_start",
        "test_end",
        "model_id",
        "used_future_covariates",
    ]
    assert (predictions["mode"] == "multivariate_lora").all()
    assert (predictions["y_pred"] == predictions["p50"]).all()
    assert not predictions["used_future_covariates"].any()
    assert set(metrics["horizon"]) == {1, 2}
    assert metrics.loc[metrics["horizon"] == 1, "n_excluded_imputed"].iloc[0] == 1
    assert FakePredictor.loaded is not None
    assert len(FakePredictor.loaded.calls) == 5
    for call in FakePredictor.loaded.calls:
        context = call["data"]
        assert list(context.columns) == ["id", "timestamp", "target", "Wspd"]
        assert pd.Timestamp(call["cutoff"]) == pd.Timestamp("2020-01-01 15:00")


def test_adapter_rejects_known_future_covariates() -> None:
    adapter = AutoGluonChronos2Adapter(FakePredictor(), FakeTimeSeriesDataFrame, "Chronos2LoRA")
    with pytest.raises(ValueError, match="known-future"):
        adapter.predict_df(
            _data().head(4),
            future_df=pd.DataFrame({"Wspd": [1.0]}),
            prediction_length=2,
        )


def test_checkpoint_and_model_name_are_required(tmp_path: Path) -> None:
    predictor_path = tmp_path / "predictor"
    predictor_path.mkdir()
    with pytest.raises(RuntimeError, match="adapter_model.safetensors"):
        load_verified_predictor(predictor_path, predictor_class=FakePredictor)

    checkpoint = predictor_path / "fine-tuned-ckpt" / "adapter_model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fake")

    class BrokenPredictor(FakePredictor):
        @classmethod
        def load(cls, path: str) -> "BrokenPredictor":
            raise OSError("corrupt predictor")

    with pytest.raises(RuntimeError, match="Could not reload AutoGluon predictor"):
        load_verified_predictor(predictor_path, predictor_class=BrokenPredictor)

    class WrongModelPredictor(FakePredictor):
        @classmethod
        def load(cls, path: str) -> "WrongModelPredictor":
            return cls(["OtherModel"])

    with pytest.raises(RuntimeError, match="does not contain Chronos2LoRA"):
        load_verified_predictor(predictor_path, predictor_class=WrongModelPredictor)
