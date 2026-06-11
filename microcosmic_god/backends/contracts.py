from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from microcosmic_god.brain import TinyController


@dataclass(slots=True)
class BrainLearningCase:
    controller: TinyController
    action_index: int
    valence: float
    energy_delta: float
    learning_rate: float
    plasticity: float
    prediction_weight: float
    outcome_targets: dict[str, float]


class BrainRuntime(Protocol):
    name: str
    device: str

    def forward_many(self, brains: list[TinyController], observations: list[list[float]]) -> list[list[float]]:
        ...

    def learn_many(self, cases: list[BrainLearningCase]) -> list[float]:
        ...
