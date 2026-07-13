"""Model-free runtime preflight for single-GPU Chronos-2 fine-tuning."""

from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


MIN_CHRONOS_VERSION = Version("2.1.0")
TARGET_GPU_NAME = "RTX 4090"
MIN_VRAM_GIB = 23.0


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def autogluon_chronos_requirement() -> Requirement | None:
    for value in importlib.metadata.requires("autogluon.timeseries") or []:
        requirement = Requirement(value)
        if canonicalize_name(requirement.name) == "chronos-forecasting":
            return requirement
    return None


def collect_preflight() -> tuple[dict[str, Any], list[str]]:
    import torch

    autogluon_version = installed_version("autogluon.timeseries")
    chronos_version = installed_version("chronos-forecasting")
    cuda_available = torch.cuda.is_available()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    vram_gib = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if cuda_available
        else None
    )
    bf16_supported = bool(torch.cuda.is_bf16_supported()) if cuda_available else False

    report = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "autogluon.timeseries": autogluon_version,
        "chronos-forecasting": chronos_version,
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "torch.cuda.is_available": cuda_available,
        "visible_gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "vram_gib": vram_gib,
        "bf16_supported": bf16_supported,
        "training_precision": "bf16" if bf16_supported else "fp16",
    }

    failures: list[str] = []
    if not (Version("3.10") <= Version(platform.python_version()) < Version("3.14")):
        failures.append("AutoGluon 1.5 requires Python 3.10-3.13.")
    if autogluon_version is None:
        failures.append("autogluon.timeseries is not installed.")
    if chronos_version is None:
        failures.append("chronos-forecasting is not installed.")
    elif Version(chronos_version) < MIN_CHRONOS_VERSION:
        failures.append("chronos-forecasting 2.1.0 or newer is required for past covariates.")
    if autogluon_version is not None and chronos_version is not None:
        requirement = autogluon_chronos_requirement()
        report["autogluon_chronos_requirement"] = str(requirement) if requirement else None
        if requirement is not None and Version(chronos_version) not in requirement.specifier:
            failures.append(
                f"chronos-forecasting {chronos_version} does not satisfy {requirement}."
            )
    if visible_devices != "0":
        failures.append("Set CUDA_VISIBLE_DEVICES=0 to expose exactly one GPU.")
    if not cuda_available:
        failures.append("PyTorch cannot access CUDA.")
    elif gpu_count != 1:
        failures.append(f"Expected one visible GPU, detected {gpu_count}.")
    if gpu_name is None or TARGET_GPU_NAME not in gpu_name:
        failures.append(f"Expected an {TARGET_GPU_NAME}, detected {gpu_name or 'no GPU'}.")
    if vram_gib is None or vram_gib < MIN_VRAM_GIB:
        failures.append(f"Expected at least {MIN_VRAM_GIB:.0f} GiB VRAM, detected {vram_gib}.")
    return report, failures


def main() -> None:
    report, failures = collect_preflight()
    for key, value in report.items():
        print(f"{key}: {value}")
    if failures:
        print("Preflight: FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Preflight: PASSED")


if __name__ == "__main__":
    main()
