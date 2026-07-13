import json

import pandas as pd

from src.evaluation.splits import ensure_split_manifest, test_period as get_test_period


def test_global_chronological_split_boundaries_are_persisted(tmp_path) -> None:
    timestamps = pd.date_range("2020-01-01", periods=10, freq="1h")
    data = pd.DataFrame(
        {
            "id": ["1"] * 10 + ["2"] * 10,
            "timestamp": list(timestamps) * 2,
            "target": range(20),
        }
    )
    config = {
        "name": "test_sdwpf",
        "frequency": "1h",
        "split_strategy": "global_chronological_timestamp",
        "train_fraction": 0.7,
        "validation_fraction": 0.1,
        "test_fraction": 0.2,
    }
    config_path = tmp_path / "benchmark.json"
    manifest_path = tmp_path / "split_manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    manifest = ensure_split_manifest(data, config_path, manifest_path)

    assert manifest_path.exists()
    assert manifest["splits"]["train"] == {
        "start": timestamps[0].isoformat(),
        "end": timestamps[6].isoformat(),
        "n_timestamps": 7,
    }
    assert manifest["splits"]["validation"] == {
        "start": timestamps[7].isoformat(),
        "end": timestamps[7].isoformat(),
        "n_timestamps": 1,
    }
    assert get_test_period(manifest) == (timestamps[8], timestamps[9])
