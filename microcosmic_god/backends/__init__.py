from __future__ import annotations

from .contracts import BrainLearningCase, BrainRuntime
from .cpu import CpuBrainRuntime


def make_brain_runtime(backend: str, device: str = "auto") -> BrainRuntime:
    if backend == "cpu":
        return CpuBrainRuntime()
    if backend == "torch":
        from .torch_gpu import TorchBrainRuntime

        return TorchBrainRuntime(device=device)
    raise ValueError(f"unknown compute backend: {backend}")


__all__ = ["BrainLearningCase", "BrainRuntime", "CpuBrainRuntime", "make_brain_runtime"]
