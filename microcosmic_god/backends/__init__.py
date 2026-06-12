from __future__ import annotations

from .contracts import ControllerLearningCase, ControllerRuntime
from .cpu import CpuBrainRuntime


def make_controller_runtime(backend: str, device: str = "auto") -> ControllerRuntime:
    if backend == "cpu":
        return CpuBrainRuntime()
    if backend == "torch":
        from .torch_gpu import TorchBrainRuntime

        return TorchBrainRuntime(device=device)
    raise ValueError(f"unknown compute backend: {backend}")


__all__ = ["ControllerLearningCase", "ControllerRuntime", "CpuBrainRuntime", "make_controller_runtime"]
