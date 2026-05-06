from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from random import Random
from typing import Any


def _rand_weight(rng: Random) -> float:
    return rng.gauss(0.0, 0.45)


PREDICTION_HEADS = ("energy", "damage", "reproduction", "social", "tool", "hazard")
AUXILIARY_PREDICTION_HEADS = tuple(head for head in PREDICTION_HEADS if head != "energy")

# Information-as-attention: brains learn (during life) where to look. Total
# fidelity is bounded so the brain must choose; what's not attended to gets
# noise instead of signal. The mechanism is general (no marks-specific or
# mg-specific structure baked in), so brains transfer cleanly to environments
# without writing/marks - the attention rule simply re-targets via the same
# error-modulated plasticity in the new environment's prediction errors.
#
# Init bias is set high enough that untrained brains pass nearly-full fidelity
# (sigmoid(2.5) ≈ 0.92). Budget then bounds total attention at 0.85 * N, so
# untrained avg fidelity ≈ 0.85 - learning bootstraps normally. Selection
# pressure activates once the brain learns to push some features toward 1.0
# (which requires pulling others down, since total is bounded).
ATTENTION_BUDGET_FRACTION = 0.85
ATTENTION_BIAS_INIT_MEAN = 2.5
ATTENTION_BIAS_INIT_SCALE = 0.10
ATTENTION_NOISE_SCALE = 0.30
ATTENTION_LEARNING_RATE_FACTOR = 0.012
ATTENTION_DECAY = 0.0008


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
    auxiliary_prediction_weights: dict[str, list[float]] = field(default_factory=dict)
    attention_weights: list[float] = field(default_factory=list)
    attention_bias: list[float] = field(default_factory=list)
    hidden: list[float] = field(default_factory=list)
    last_outputs: list[float] = field(default_factory=list)
    last_inputs: list[float] = field(default_factory=list)
    last_attention: list[float] = field(default_factory=list)
    last_prediction_errors: dict[str, float] = field(default_factory=dict)
    input_trace: list[float] = field(default_factory=list)
    hidden_trace: list[float] = field(default_factory=list)

    @classmethod
    def random(
        cls,
        rng: Random,
        input_size: int,
        hidden_size: int,
        output_size: int,
        with_attention: bool = True,
    ) -> "TinyBrain":
        hidden_size = max(1, min(128, hidden_size))
        if with_attention:
            attention_weights = [rng.gauss(0.0, 0.15) for _ in range(input_size * hidden_size)]
            attention_bias = [rng.gauss(ATTENTION_BIAS_INIT_MEAN, ATTENTION_BIAS_INIT_SCALE) for _ in range(input_size)]
        else:
            attention_weights = []
            attention_bias = []
        brain = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weights_in=[_rand_weight(rng) for _ in range(input_size * hidden_size)],
            weights_out=[_rand_weight(rng) for _ in range(hidden_size * output_size)],
            bias_h=[rng.gauss(0.0, 0.08) for _ in range(hidden_size)],
            bias_o=[rng.gauss(0.0, 0.08) for _ in range(output_size)],
            prediction_weights=[_rand_weight(rng) for _ in range(hidden_size)],
            auxiliary_prediction_weights={
                head: [_rand_weight(rng) for _ in range(hidden_size)]
                for head in AUXILIARY_PREDICTION_HEADS
            },
            attention_weights=attention_weights,
            attention_bias=attention_bias,
            hidden=[0.0 for _ in range(hidden_size)],
            last_outputs=[0.0 for _ in range(output_size)],
            last_inputs=[0.0 for _ in range(input_size)],
            last_attention=[0.0 for _ in range(input_size)],
            last_prediction_errors={head: 0.0 for head in PREDICTION_HEADS},
            input_trace=[0.0 for _ in range(input_size)],
            hidden_trace=[0.0 for _ in range(hidden_size)],
        )
        return brain

    def _ensure_auxiliary_prediction_heads(self) -> None:
        for head in AUXILIARY_PREDICTION_HEADS:
            weights = self.auxiliary_prediction_weights.get(head)
            if weights is None or len(weights) != self.hidden_size:
                self.auxiliary_prediction_weights[head] = [0.0 for _ in range(self.hidden_size)]

    def _has_attention(self) -> bool:
        return (
            len(self.attention_weights) == self.input_size * self.hidden_size
            and len(self.attention_bias) == self.input_size
        )

    def _attend(self, inputs: list[float]) -> list[float]:
        """Compute fidelity from current hidden state and apply it to inputs.

        Bounded by ATTENTION_BUDGET_FRACTION * input_size; what isn't attended
        to is replaced with gaussian noise. The brain's own hidden state shapes
        what it looks at, so attention is a function of context (neuroplastic
        in life via the learning rule below, and inheritable via clone_for_offspring).
        """
        if not self._has_attention():
            self.last_attention = [1.0 for _ in range(self.input_size)]
            return list(inputs)
        inv_h = 1.0 / math.sqrt(max(1, self.hidden_size))
        raw: list[float] = []
        for i in range(self.input_size):
            total = self.attention_bias[i]
            for h in range(self.hidden_size):
                total += self.attention_weights[h * self.input_size + i] * self.hidden[h] * inv_h
            # bounded sigmoid; clamp argument to avoid math.exp overflow
            arg = max(-30.0, min(30.0, total))
            raw.append(1.0 / (1.0 + math.exp(-arg)))
        budget = self.input_size * ATTENTION_BUDGET_FRACTION
        total_attention = sum(raw)
        if total_attention > budget and total_attention > 0.0:
            scale = budget / total_attention
            fidelity = [r * scale for r in raw]
        else:
            fidelity = raw
        self.last_attention = fidelity
        return [
            float(value) * f + _random.gauss(0.0, ATTENTION_NOISE_SCALE) * (1.0 - f)
            for value, f in zip(inputs, fidelity)
        ]

    def forward(self, inputs: list[float]) -> list[float]:
        if len(inputs) != self.input_size:
            raise ValueError(f"expected {self.input_size} inputs, got {len(inputs)}")
        if not self.input_trace or len(self.input_trace) != self.input_size:
            self.input_trace = [0.0 for _ in range(self.input_size)]
        if not self.hidden_trace or len(self.hidden_trace) != self.hidden_size:
            self.hidden_trace = [0.0 for _ in range(self.hidden_size)]
        # Apply attention: brain decides (from its hidden state) what to look
        # at this tick. Total fidelity bounded; unattended channels get noise.
        attended = self._attend(inputs)
        inputs = attended
        self.last_inputs = list(inputs)
        self.input_trace = [old * 0.92 + float(value) * 0.08 for old, value in zip(self.input_trace, inputs)]
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
        self.hidden_trace = [old * 0.90 + value * 0.10 for old, value in zip(self.hidden_trace, self.hidden)]

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

    def predict_outcomes(self) -> dict[str, float]:
        self._ensure_auxiliary_prediction_heads()
        inv_h = 1.0 / math.sqrt(max(1, self.hidden_size))
        predictions = {"energy": self.predict_next_energy()}
        for head, weights in self.auxiliary_prediction_weights.items():
            predictions[head] = sum(w * h for w, h in zip(weights, self.hidden)) * inv_h
        return predictions

    def _learn_prediction_heads(
        self,
        targets: dict[str, float],
        learning_rate: float,
        plasticity: float,
        prediction_weight: float,
    ) -> dict[str, float]:
        self._ensure_auxiliary_prediction_heads()
        lr = max(0.0, min(0.25, learning_rate)) * max(0.0, min(1.0, plasticity))
        pred_lr = lr * max(0.0, min(1.0, prediction_weight)) * 0.025
        predictions = self.predict_outcomes()
        errors: dict[str, float] = {}
        for head in PREDICTION_HEADS:
            if head not in targets:
                continue
            target = max(-2.0, min(2.0, targets[head]))
            error = max(-2.0, min(2.0, target - predictions.get(head, 0.0)))
            errors[head] = error
            weights = self.prediction_weights if head == "energy" else self.auxiliary_prediction_weights[head]
            for h, value in enumerate(self.hidden):
                updated = weights[h] + pred_lr * error * value
                weights[h] = max(-4.0, min(4.0, updated))
        self.last_prediction_errors = {head: errors.get(head, 0.0) for head in PREDICTION_HEADS}
        return errors

    def learn(
        self,
        action_index: int,
        valence: float,
        energy_delta: float,
        learning_rate: float,
        plasticity: float,
        prediction_weight: float,
        outcome_targets: dict[str, float] | None = None,
    ) -> float:
        if action_index < 0 or action_index >= self.output_size:
            return 0.0
        valence = max(-2.0, min(2.0, valence))
        lr = max(0.0, min(0.25, learning_rate)) * max(0.0, min(1.0, plasticity))
        offset = action_index * self.hidden_size
        targets = {"energy": energy_delta}
        if outcome_targets:
            targets.update(outcome_targets)
        errors = self._learn_prediction_heads(targets, learning_rate, plasticity, prediction_weight)
        error = errors.get("energy", 0.0)
        for h, value in enumerate(self.hidden_trace or self.hidden):
            updated = self.weights_out[offset + h] + lr * valence * value * 0.035
            self.weights_out[offset + h] = max(-4.0, min(4.0, updated))
        self.bias_o[action_index] = max(-4.0, min(4.0, self.bias_o[action_index] + lr * valence * 0.015))

        representation_lr = lr * (0.15 + max(0.0, min(1.0, prediction_weight)) * 0.35) * 0.010
        if representation_lr > 0.0 and self.last_inputs:
            surprise = sum(abs(value) for value in errors.values()) / max(1, len(errors))
            modulation = max(-2.0, min(2.0, valence * 0.55 + error * prediction_weight * 0.35 + surprise * prediction_weight * 0.10))
            for h, hidden_value in enumerate(self.hidden_trace or self.hidden):
                if abs(hidden_value) < 0.015:
                    continue
                offset_in = h * self.input_size
                hidden_gate = max(-1.0, min(1.0, hidden_value))
                for i, input_value in enumerate(self.input_trace or self.last_inputs):
                    if abs(input_value) < 0.010:
                        continue
                    index = offset_in + i
                    updated = self.weights_in[index] + representation_lr * modulation * hidden_gate * input_value
                    self.weights_in[index] = max(-4.0, min(4.0, updated))

        # Neuroplastic attention update. Surprise (sum of prediction errors)
        # signals "I should be looking at things I'm not predicting well";
        # valence sign tells us whether this tick was good or bad. Hebbian-ish
        # at the hidden-to-input boundary: features active alongside hidden
        # units that fired during a surprising/valenced moment get strengthened
        # attention; everything decays slowly so stale attention patterns fade.
        if self._has_attention():
            attention_lr = lr * ATTENTION_LEARNING_RATE_FACTOR
            if attention_lr > 0.0:
                surprise_signal = sum(abs(value) for value in errors.values()) / max(1, len(errors))
                valence_sign = 1.0 if valence >= 0.0 else -1.0
                signal_strength = surprise_signal * valence_sign
                for h, hidden_value in enumerate(self.hidden_trace or self.hidden):
                    if abs(hidden_value) < 0.015:
                        continue
                    offset_a = h * self.input_size
                    hidden_gate = max(-1.0, min(1.0, hidden_value))
                    for i, input_value in enumerate(self.input_trace or self.last_inputs):
                        if abs(input_value) < 0.010:
                            continue
                        index = offset_a + i
                        delta = attention_lr * signal_strength * hidden_gate * input_value
                        decayed = self.attention_weights[index] * (1.0 - ATTENTION_DECAY * attention_lr)
                        self.attention_weights[index] = max(-3.0, min(3.0, decayed + delta))
                # Bias also slowly tracks chronic feature value (independent of hidden state).
                bias_lr = attention_lr * 0.4
                for i, input_value in enumerate(self.input_trace or self.last_inputs):
                    if abs(input_value) < 0.010:
                        continue
                    delta = bias_lr * signal_strength * input_value
                    decayed = self.attention_bias[i] * (1.0 - ATTENTION_DECAY * attention_lr)
                    self.attention_bias[i] = max(-2.0, min(2.0, decayed + delta))
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
        data["auxiliary_prediction_weights"] = {
            head: mutate_many(weights)
            for head, weights in data.get("auxiliary_prediction_weights", {}).items()
        }
        if data.get("attention_weights"):
            data["attention_weights"] = mutate_many(data["attention_weights"])
        if data.get("attention_bias"):
            data["attention_bias"] = mutate_many(data["attention_bias"])
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
            "auxiliary_prediction_weights": {
                head: [round(v, 7) for v in weights]
                for head, weights in sorted(self.auxiliary_prediction_weights.items())
            },
            "attention_weights": [round(v, 7) for v in self.attention_weights],
            "attention_bias": [round(v, 7) for v in self.attention_bias],
        }
        if include_state:
            data["hidden"] = [round(v, 7) for v in self.hidden]
            data["last_outputs"] = [round(v, 7) for v in self.last_outputs]
            data["last_inputs"] = [round(v, 7) for v in self.last_inputs]
            data["last_attention"] = [round(v, 7) for v in self.last_attention]
            data["last_prediction_errors"] = {head: round(value, 7) for head, value in sorted(self.last_prediction_errors.items())}
            data["input_trace"] = [round(v, 7) for v in self.input_trace]
            data["hidden_trace"] = [round(v, 7) for v in self.hidden_trace]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyBrain":
        input_size = int(data["input_size"])
        hidden_size = int(data["hidden_size"])
        output_size = int(data["output_size"])
        # Backward compat: pre-attention checkpoints have neither field. Default
        # to empty (no attention head -> brain receives unfiltered observations
        # via _attend's pass-through path).
        attention_weights_raw = data.get("attention_weights")
        attention_bias_raw = data.get("attention_bias")
        if attention_weights_raw is not None and len(attention_weights_raw) == input_size * hidden_size:
            attention_weights = [float(v) for v in attention_weights_raw]
        else:
            attention_weights = []
        if attention_bias_raw is not None and len(attention_bias_raw) == input_size:
            attention_bias = [float(v) for v in attention_bias_raw]
        else:
            attention_bias = []
        brain = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weights_in=[float(v) for v in data["weights_in"]],
            weights_out=[float(v) for v in data["weights_out"]],
            bias_h=[float(v) for v in data["bias_h"]],
            bias_o=[float(v) for v in data["bias_o"]],
            prediction_weights=[float(v) for v in data["prediction_weights"]],
            auxiliary_prediction_weights={
                head: [float(v) for v in weights]
                for head, weights in data.get("auxiliary_prediction_weights", {}).items()
            },
            attention_weights=attention_weights,
            attention_bias=attention_bias,
            hidden=[0.0 for _ in range(hidden_size)],
            last_outputs=[0.0 for _ in range(output_size)],
            last_inputs=[0.0 for _ in range(input_size)],
            last_attention=[0.0 for _ in range(input_size)],
            last_prediction_errors={head: 0.0 for head in PREDICTION_HEADS},
            input_trace=[0.0 for _ in range(input_size)],
            hidden_trace=[0.0 for _ in range(hidden_size)],
        )
        brain._ensure_auxiliary_prediction_heads()
        if "hidden" in data:
            brain.hidden = [float(v) for v in data["hidden"]]
        if "last_outputs" in data:
            brain.last_outputs = [float(v) for v in data["last_outputs"]]
        if "last_inputs" in data:
            brain.last_inputs = [float(v) for v in data["last_inputs"]]
        if "last_attention" in data:
            brain.last_attention = [float(v) for v in data["last_attention"]]
        if "last_prediction_errors" in data:
            brain.last_prediction_errors = {head: float(data["last_prediction_errors"].get(head, 0.0)) for head in PREDICTION_HEADS}
        if "input_trace" in data:
            brain.input_trace = [float(v) for v in data["input_trace"]]
        if "hidden_trace" in data:
            brain.hidden_trace = [float(v) for v in data["hidden_trace"]]
        return brain
