from pathlib import Path


def test_foshan_autodl_script_uses_single_gpu_exact_release_and_staged_benchmark() -> None:
    script = Path("scripts/run_foshan_zero_shot_autodl.sh").read_text(encoding="utf-8")
    requirements = Path("requirements-foshan-zero-shot.txt").read_text(encoding="utf-8")

    assert "chronos-forecasting==2.3.1" in requirements
    assert "openpyxl" in requirements
    assert "export CUDA_VISIBLE_DEVICES=0" in script
    assert "--stage baselines" in script
    assert "--stage chronos" in script
    assert "--processed-input" in script
    assert "--model-path" in script
    assert "device-map auto" not in script
