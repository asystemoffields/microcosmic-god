from __future__ import annotations

from .contracts import BrainLearningCase
from microcosmic_god.brain import TinyBrain


class CpuBrainRuntime:
    name = "cpu"
    device = "cpu"

    def forward_many(self, brains: list[TinyBrain], observations: list[list[float]]) -> list[list[float]]:
        return [brain.forward(observation) for brain, observation in zip(brains, observations)]

    def learn_many(self, cases: list[BrainLearningCase]) -> list[float]:
        errors: list[float] = []
        for case in cases:
            errors.append(
                case.brain.learn(
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
