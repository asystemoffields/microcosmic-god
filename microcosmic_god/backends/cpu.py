from __future__ import annotations

from .contracts import ControllerLearningCase
from microcosmic_god.controller import TinyController


class CpuBrainRuntime:
    name = "cpu"
    device = "cpu"

    def forward_many(self, controllers: list[TinyController], observations: list[list[float]]) -> list[list[float]]:
        return [controller.forward(observation) for controller, observation in zip(controllers, observations)]

    def learn_many(self, cases: list[ControllerLearningCase]) -> list[float]:
        errors: list[float] = []
        for case in cases:
            errors.append(
                case.controller.learn(
                    action_index=case.action_index,
                    valence=case.valence,
                    energy_delta=case.energy_delta,
                    learning_rate=case.learning_rate,
                    plasticity=case.plasticity,
                    prediction_weight=case.prediction_weight,
                    outcome_targets=case.outcome_targets,
                )
            )
        return errors
