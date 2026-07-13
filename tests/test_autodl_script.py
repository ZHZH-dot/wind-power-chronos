from pathlib import Path


def test_autodl_script_uses_regularized_benchmark_dataset() -> None:
    script = Path("scripts/run_zero_shot_autodl.sh").read_text(encoding="utf-8")

    assert "data/processed/sdwpf_hourly_regularized.parquet" in script
    assert "--regularize-hourly" in script
    assert "python -m src.evaluation.splits" in script
    assert "--split-manifest" in script
