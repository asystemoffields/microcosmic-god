from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random
from typing import Any


def _rand_weight(rng: Random) -> float:
    return rng.gauss(0.0, 0.45)


@dataclass(slots=True)
class TinyBrain:
    input_size: int
    hidden_size: int
    output_size: int
    weights_in: list[float]
    weights_out: list[float]
    bias_h: list[float]
    bias_o: list[float]
    prediction_weights: list[float]
    hidden: list[float] = field(default_factory=list)
    last_outputs: list[float] = field(default_factory=list)

    @classmethod
    def random(cls, rng: Random, input_size: int, hidden_size: int, output_size: int) -> "TinyBrain":
        hidden_size = max(1, min(64, hidden_size))
        brain = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weights_in=[_rand_weight(rng) for _ in range(input_size * hidden_size)],
            weights_out=[_rand_weight(rng) for _ in range(hidden_size * output_size)],
            bias_h=[rng.gauss(0.0, 0.08) for _ in range(hidden_size)],
            bias_o=[rng.gauss(0.0, 0.08) for _ in range(output_size)],
            prediction_weights=[_rand_weight(rng) for _ in range(hidden_size)],
            hidden=[0.0 for _ in range(hidden_size)],
            last_outputs=[0.0 for _ in range(output_size)],
        )
        return brain

    def forward(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.input_size:
            raise ValueError(f"expected {self.input_size} inputs, got {len(inputs)}")
        inv = 1.0 / math.sqrt(max(1, self.input_size))
        new_hidden: list[float] = []
        for h in range(self.hidden_size):
            base = self.bias_h[h] + 0.62 * self.hidden[h]
            offset = h * self.input_size
            total = base
            for i, value in enumerate(inputs):
                total += self.weights_in[offset + i] * value * inv
            new_hidden.append(math.tanh(total))
        self.hidden = new_hidden

        inv_h = 1.0 / math.sqrt(max(1, self.hidden_size))
        outputs: list[float] = []
        for o in range(self.output_size):
            total = self.bias_o[o]
            offset = o * self.hidden_size
            for h, value in enumerate(self.hidden):
                total += self.weights_out[offset + h] * value * inv_h
            outputs.append(total)
        self.last_outputs = outputs
        return outputs

    def predict_next_energy(self) -> float:
        inv_h = 1.0 / math.sqrt(max(1, self.hidden_size))
        return sum(w * h for w, h in zip(self.prediction_weights, self.hidden)) * inv_h

    def learn(
        self,
        action_index: int,
        valence: float,
        energy_delta: float,
        learning_rate: float,
        plasticity: float,
        prediction_weight: float,
    ) -> float:
        if action_index < 0 or action_index >= self.output_size:
            return 0.0
        valence = max(-2.0, min(2.0, valence))
        lr = max(0.0, min(0.25, learning_rate)) * max(0.0, min(1.0, plasticity))
        offset = action_index * self.hidden_size
        for h, value in enumerate(self.hidden):
            self.weights_out[offset + h] += lr * valence * value * 0.035
        self.bias_o[action_index] += lr * valence * 0.015

        predicted = self.predict_next_energy()
        error = max(-2.0, min(2.0, energy_delta - predicted))
        pred_lr = lr * max(0.0, min(1.0, prediction_weight)) * 0.025
        for h, value in enumerate(self.hidden):
            self.prediction_weights[h] += pred_lr * error * value
        return error

    def clone_for_offspring(self, rng: Random, mutation_scale: float = 0.03) -> "TinyBrain":
        data = self.to_dict(include_state=False)

        def mutate_many(values: list[float]) -> list[float]:
            return [value + rng.gauss(0.0, mutation_scale) for value in values]

        data["weights_in"] = mutate_many(data["weights_in"])
        data["weights_out"] = mutate_many(data["weights_out"])
        data["bias_h"] = mutate_many(data["bias_h"])
        data["bias_o"] = mutate_many(data["bias_o"])
        data["prediction_weights"] = mutate_many(data["prediction_weights"])
        return TinyBrain.from_dict(data)

    def to_dict(self, include_state: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "weights_in": [round(v, 7) for v in self.weights_in],
            "weights_out": [round(v, 7) for v in self.weights_out],
            "bias_h": [round(v, 7) for v in self.bias_h],
            "bias_o": [round(v, 7) for v in self.bias_o],
            "prediction_weights": [round(v, 7) for v in self.prediction_weights],
        }
        if include_state:
            data["hidden"] = [round(v, 7) for v in self.hidden]
            data["last_outputs"] = [round(v, 7) for v in self.last_outputs]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyBrain":
        brain = cls(
            input_size=int(data["input_size"]),
            hidden_size=int(data["hidden_size"]),
            output_size=int(data["output_size"]),
            weights_in=[float(v) for v in data["weights_in"]],
            weights_out=[float(v) for v in data["weights_out"]],
            bias_h=[float(v) for v in data["bias_h"]],
            bias_o=[float(v) for v in data["bias_o"]],
            prediction_weights=[float(v) for v in data["prediction_weights"]],
            hidden=[0.0 for _ in range(int(data["hidden_size"]))],
            last_outputs=[0.0 for _ in range(int(data["output_size"]))],
        )
        if "hidden" in data:
            brain.hidden = [float(v) for v in data["hidden"]]
        if "last_outputs" in data:
            brain.last_outputs = [float(v) for v in data["last_outputs"]]
        return brain

