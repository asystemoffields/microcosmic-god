from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from microcosmic_god.controller import TinyController


@dataclass(slots=True)
class ControllerLearningCase:
    controller: TinyController
    action_index: int
    valence: float
    energy_delta: float
    learning_rate: float
    plasticity: float
    prediction_weight: float
    outcome_targets: dict[str, float]


class ControllerRuntime(Protocol):
    name: str
    device: str

    def forward_many(self, controllers: list[TinyController], observations: list[list[float]]) -> list[list[float]]:
        ...

    def learn_many(self, cases: list[ControllerLearningCase]) -> list[float]:
        ...
