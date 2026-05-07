from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random
from typing import Any

import numpy as np


def _rand_weight(rng: Random) -> float:
    return rng.gauss(0.0, 0.45)


PREDICTION_HEADS = ("energy", "damage", "reproduction", "social", "tool", "hazard")
AUXILIARY_PREDICTION_HEADS = tuple(head for head in PREDICTION_HEADS if head != "energy")

# Brain hidden-layer size cap. Brains can grow up to this size if their
# genome.neural_budget evolves there; metabolic cost (in organisms.py) scales
# with neural_budget, so growth pays off only when cognition does.
BRAIN_HIDDEN_MAX = 512

# Information-as-attention: brains learn (during life) where to look. Total
# fidelity is bounded so the brain must choose; what's not attended to gets
# noise instead of signal. The mechanism is general (no marks-specific or
# mg-specific structure baked in), so brains transfer cleanly to environments
# without writing/marks - the attention rule simply re-targets via the same
# error-modulated plasticity in the new environment's prediction errors.
#
# Init bias is set high enough that untrained brains pass nearly-full fidelity
# (sigmoid(3.0) ≈ 0.95). Budget then bounds total attention at 0.95 * N, so
# untrained avg fidelity ≈ 0.95 - learning bootstraps normally and the
# information cost is gentle enough that early-life agents still survive long
# enough to reproduce. Selection pressure activates once the brain learns to
# push some features toward 1.0 (which requires pulling others down, since
# total is bounded). Earlier 0.85 budget + 0.30 noise crashed early lineages
# because untrained agents received too-noisy observations to bootstrap good
# action policies before starving.
ATTENTION_BUDGET_FRACTION = 0.95
ATTENTION_BIAS_INIT_MEAN = 3.0
ATTENTION_BIAS_INIT_SCALE = 0.10
ATTENTION_NOISE_SCALE = 0.18
ATTENTION_LEARNING_RATE_FACTOR = 0.012
ATTENTION_DECAY = 0.0008

# Float dtype for all brain arrays. float64 matches Python float semantics so
# checkpoint round-trips are bit-exact within rounding tolerance and tests
# that assert almostEqual at 6+ places stay clean. Switching to float32 would
# halve memory and ~2x throughput on most CPUs but requires loosening tests.
_DTYPE = np.float64


def _empty_array(shape: tuple[int, ...] | int) -> np.ndarray:
    return np.zeros(shape, dtype=_DTYPE)


@dataclass(slots=True)
class TinyBrain:
    """Small recurrent network with prediction heads and a neuroplastic
    attention mechanism. Internally stores all parameters and state as numpy
    arrays for vectorized forward/learn; serializes as nested lists so
    checkpoints, the torch backend, and tests stay format-compatible.
    """

    input_size: int
    hidden_size: int
    output_size: int
    # Weight matrices: weights_in is (hidden_size, input_size); weights_out is
    # (output_size, hidden_size); attention_weights is (hidden_size, input_size).
    weights_in: np.ndarray
    weights_out: np.ndarray
    bias_h: np.ndarray
    bias_o: np.ndarray
    prediction_weights: np.ndarray
    auxiliary_prediction_weights: dict[str, np.ndarray] = field(default_factory=dict)
    # Attention head: empty arrays mean "no attention" (legacy/disabled).
    attention_weights: np.ndarray = field(default_factory=lambda: _empty_array((0, 0)))
    attention_bias: np.ndarray = field(default_factory=lambda: _empty_array(0))
    # Per-tick state.
    hidden: np.ndarray = field(default_factory=lambda: _empty_array(0))
    last_outputs: np.ndarray = field(default_factory=lambda: _empty_array(0))
    last_inputs: np.ndarray = field(default_factory=lambda: _empty_array(0))
    last_attention: np.ndarray = field(default_factory=lambda: _empty_array(0))
    last_prediction_errors: dict[str, float] = field(default_factory=dict)
    input_trace: np.ndarray = field(default_factory=lambda: _empty_array(0))
    hidden_trace: np.ndarray = field(default_factory=lambda: _empty_array(0))

    @classmethod
    def random(
        cls,
        rng: Random,
        input_size: int,
        hidden_size: int,
        output_size: int,
        with_attention: bool = True,
    ) -> "TinyBrain":
        hidden_size = max(1, min(BRAIN_HIDDEN_MAX, hidden_size))

        def _rand_matrix(shape: tuple[int, int]) -> np.ndarray:
            return np.array(
                [[_rand_weight(rng) for _ in range(shape[1])] for _ in range(shape[0])],
                dtype=_DTYPE,
            )

        def _rand_vector(size: int, scale: float = 0.08) -> np.ndarray:
            return np.array([rng.gauss(0.0, scale) for _ in range(size)], dtype=_DTYPE)

        if with_attention:
            attention_weights = np.array(
                [[rng.gauss(0.0, 0.15) for _ in range(input_size)] for _ in range(hidden_size)],
                dtype=_DTYPE,
            )
            attention_bias = np.array(
                [rng.gauss(ATTENTION_BIAS_INIT_MEAN, ATTENTION_BIAS_INIT_SCALE) for _ in range(input_size)],
                dtype=_DTYPE,
            )
        else:
            attention_weights = _empty_array((0, 0))
            attention_bias = _empty_array(0)

        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weights_in=_rand_matrix((hidden_size, input_size)),
            weights_out=_rand_matrix((output_size, hidden_size)),
            bias_h=_rand_vector(hidden_size),
            bias_o=_rand_vector(output_size),
            prediction_weights=np.array([_rand_weight(rng) for _ in range(hidden_size)], dtype=_DTYPE),
            auxiliary_prediction_weights={
                head: np.array([_rand_weight(rng) for _ in range(hidden_size)], dtype=_DTYPE)
                for head in AUXILIARY_PREDICTION_HEADS
            },
            attention_weights=attention_weights,
            attention_bias=attention_bias,
            hidden=_empty_array(hidden_size),
            last_outputs=_empty_array(output_size),
            last_inputs=_empty_array(input_size),
            last_attention=_empty_array(input_size),
            last_prediction_errors={head: 0.0 for head in PREDICTION_HEADS},
            input_trace=_empty_array(input_size),
            hidden_trace=_empty_array(hidden_size),
        )

    def _ensure_auxiliary_prediction_heads(self) -> None:
        for head in AUXILIARY_PREDICTION_HEADS:
            weights = self.auxiliary_prediction_weights.get(head)
            if weights is None or weights.shape != (self.hidden_size,):
                self.auxiliary_prediction_weights[head] = _empty_array(self.hidden_size)

    def _has_attention(self) -> bool:
        return (
            self.attention_weights.shape == (self.hidden_size, self.input_size)
            and self.attention_bias.shape == (self.input_size,)
        )

    def _attend(self, inputs: np.ndarray) -> np.ndarray:
        """Compute fidelity from current hidden state and apply it to inputs.

        Bounded by ATTENTION_BUDGET_FRACTION * input_size; what isn't attended
        to is replaced with gaussian noise. The brain's own hidden state shapes
        what it looks at, so attention is a function of context (neuroplastic
        in life via the learning rule below, and inheritable via clone_for_offspring).
        """
        if not self._has_attention():
            self.last_attention = np.ones(self.input_size, dtype=_DTYPE)
            return inputs
        inv_h = 1.0 / math.sqrt(max(1, self.hidden_size))
        # Linear: (input_size,) = (input_size,) + (input_size,) * scalar
        logits = self.attention_bias + (self.hidden @ self.attention_weights) * inv_h
        np.clip(logits, -30.0, 30.0, out=logits)
        raw = 1.0 / (1.0 + np.exp(-logits))
        budget = self.input_size * ATTENTION_BUDGET_FRACTION
        total = raw.sum()
        if total > budget and total > 0.0:
            fidelity = raw * (budget / total)
        else:
            fidelity = raw
        self.last_attention = fidelity
        noise = np.random.normal(0.0, ATTENTION_NOISE_SCALE, size=self.input_size).astype(_DTYPE)
        return inputs * fidelity + noise * (1.0 - fidelity)

    def forward(self, inputs: list[float] | np.ndarray) -> list[float]:
        if len(inputs) != self.input_size:
            raise ValueError(f"expected {self.input_size} inputs, got {len(inputs)}")
        x = np.asarray(inputs, dtype=_DTYPE)
        # Apply attention (brain decides what to look at this tick). Uses the
        # hidden state from end of last tick, since this tick's hidden hasn't
        # been computed yet.
        x = self._attend(x)
        self.last_inputs = x
        self.input_trace = self.input_trace * 0.92 + x * 0.08
        inv = 1.0 / math.sqrt(max(1, self.input_size))
        new_hidden = np.tanh(self.bias_h + 0.62 * self.hidden + (self.weights_in @ x) * inv)
        self.hidden = new_hidden
        self.hidden_trace = self.hidden_trace * 0.90 + self.hidden * 0.10
        inv_h = 1.0 / math.sqrt(max(1, self.hidden_size))
        outputs = self.bias_o + (self.weights_out @ self.hidden) * inv_h
        self.last_outputs = outputs
        return outputs.tolist()

    def predict_next_energy(self) -> float:
        inv_h = 1.0 / math.sqrt(max(1, self.hidden_size))
        return float(self.prediction_weights @ self.hidden) * inv_h

    def predict_outcomes(self) -> dict[str, float]:
        self._ensure_auxiliary_prediction_heads()
        inv_h = 1.0 / math.sqrt(max(1, self.hidden_size))
        predictions: dict[str, float] = {"energy": float(self.prediction_weights @ self.hidden) * inv_h}
        for head, weights in self.auxiliary_prediction_weights.items():
            predictions[head] = float(weights @ self.hidden) * inv_h
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
            if head == "energy":
                self.prediction_weights = np.clip(
                    self.prediction_weights + pred_lr * error * self.hidden, -4.0, 4.0
                )
            else:
                weights = self.auxiliary_prediction_weights[head]
                self.auxiliary_prediction_weights[head] = np.clip(
                    weights + pred_lr * error * self.hidden, -4.0, 4.0
                )
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
        targets = {"energy": energy_delta}
        if outcome_targets:
            targets.update(outcome_targets)
        errors = self._learn_prediction_heads(targets, learning_rate, plasticity, prediction_weight)
        error = errors.get("energy", 0.0)

        # Action policy update: nudge weights_out for the chosen action toward
        # hidden_trace * valence. Vectorized: a single row of weights_out.
        hidden_for_policy = self.hidden_trace if self.hidden_trace.size else self.hidden
        self.weights_out[action_index] = np.clip(
            self.weights_out[action_index] + lr * valence * 0.035 * hidden_for_policy,
            -4.0,
            4.0,
        )
        self.bias_o[action_index] = max(
            -4.0, min(4.0, self.bias_o[action_index] + lr * valence * 0.015)
        )

        # Representation learning: error-modulated Hebbian on weights_in.
        representation_lr = lr * (0.15 + max(0.0, min(1.0, prediction_weight)) * 0.35) * 0.010
        if representation_lr > 0.0 and self.last_inputs.size:
            error_values = list(errors.values())
            surprise = sum(abs(value) for value in error_values) / max(1, len(error_values))
            modulation = max(
                -2.0,
                min(
                    2.0,
                    valence * 0.55
                    + error * prediction_weight * 0.35
                    + surprise * prediction_weight * 0.10,
                ),
            )
            hidden_active = self.hidden_trace if self.hidden_trace.size else self.hidden
            input_active = self.input_trace if self.input_trace.size else self.last_inputs
            hidden_mask = (np.abs(hidden_active) >= 0.015).astype(_DTYPE)
            input_mask = (np.abs(input_active) >= 0.010).astype(_DTYPE)
            hidden_gate = np.clip(hidden_active, -1.0, 1.0) * hidden_mask
            input_gate = input_active * input_mask
            # Outer product gives the (hidden, input) update matrix.
            delta_in = (representation_lr * modulation) * np.outer(hidden_gate, input_gate)
            self.weights_in = np.clip(self.weights_in + delta_in, -4.0, 4.0)

        # Neuroplastic attention update. Surprise (sum of prediction errors)
        # signals "I should be looking at things I'm not predicting well";
        # valence sign tells us whether this tick was good or bad. Same shape
        # as the representation update: outer product of hidden-trace * input-
        # trace, modulated by surprise * sign(valence). Slow decay keeps stale
        # attention patterns from persisting.
        if self._has_attention():
            attention_lr = lr * ATTENTION_LEARNING_RATE_FACTOR
            if attention_lr > 0.0:
                error_values = list(errors.values())
                surprise_signal = sum(abs(value) for value in error_values) / max(1, len(error_values))
                valence_sign = 1.0 if valence >= 0.0 else -1.0
                signal_strength = surprise_signal * valence_sign
                hidden_active = self.hidden_trace if self.hidden_trace.size else self.hidden
                input_active = self.input_trace if self.input_trace.size else self.last_inputs
                hidden_mask = (np.abs(hidden_active) >= 0.015).astype(_DTYPE)
                input_mask = (np.abs(input_active) >= 0.010).astype(_DTYPE)
                hidden_gate = np.clip(hidden_active, -1.0, 1.0) * hidden_mask
                input_gate = input_active * input_mask
                delta = attention_lr * signal_strength * np.outer(hidden_gate, input_gate)
                decay_factor = 1.0 - ATTENTION_DECAY * attention_lr
                self.attention_weights = np.clip(
                    self.attention_weights * decay_factor + delta, -3.0, 3.0
                )
                bias_lr = attention_lr * 0.4
                bias_delta = bias_lr * signal_strength * input_gate
                self.attention_bias = np.clip(
                    self.attention_bias * decay_factor + bias_delta, -2.0, 2.0
                )
        return error

    def resize_hidden(self, rng: Random, new_size: int) -> None:
        """Grow or shrink the hidden layer while preserving learned function.

        Grow: pad weights with small random values, biases with zeros, and the
        hidden state with zeros - the existing function is preserved (the new
        units start near-inert and can learn to contribute over time). Shrink:
        rank hidden units by total |incoming|+|outgoing| weight magnitude and
        keep the top new_size; the most heavily-used connections survive.

        This allows neural_budget mutations to actually change brain capacity
        across generations without losing the parent's learned representations.
        """
        new_size = max(1, min(BRAIN_HIDDEN_MAX, int(new_size)))
        old_size = self.hidden_size
        if new_size == old_size:
            return
        if new_size > old_size:
            extra = new_size - old_size
            # Grow weights_in: append rows of small random values.
            new_in_rows = np.array(
                [[_rand_weight(rng) * 0.30 for _ in range(self.input_size)] for _ in range(extra)],
                dtype=_DTYPE,
            )
            self.weights_in = np.vstack([self.weights_in, new_in_rows])
            # Grow weights_out: append columns to each row.
            new_out_cols = np.array(
                [[_rand_weight(rng) * 0.30 for _ in range(extra)] for _ in range(self.output_size)],
                dtype=_DTYPE,
            )
            self.weights_out = np.hstack([self.weights_out, new_out_cols])
            self.bias_h = np.concatenate(
                [self.bias_h, np.array([rng.gauss(0.0, 0.04) for _ in range(extra)], dtype=_DTYPE)]
            )
            self.prediction_weights = np.concatenate(
                [
                    self.prediction_weights,
                    np.array([_rand_weight(rng) * 0.30 for _ in range(extra)], dtype=_DTYPE),
                ]
            )
            for head in self.auxiliary_prediction_weights:
                self.auxiliary_prediction_weights[head] = np.concatenate(
                    [
                        self.auxiliary_prediction_weights[head],
                        np.array([_rand_weight(rng) * 0.30 for _ in range(extra)], dtype=_DTYPE),
                    ]
                )
            if self._has_attention():
                new_attn_rows = np.array(
                    [[rng.gauss(0.0, 0.15) for _ in range(self.input_size)] for _ in range(extra)],
                    dtype=_DTYPE,
                )
                self.attention_weights = np.vstack([self.attention_weights, new_attn_rows])
            self.hidden = np.concatenate([self.hidden, _empty_array(extra)])
            self.hidden_trace = np.concatenate([self.hidden_trace, _empty_array(extra)])
        else:
            # Rank old hidden units by total connection magnitude and keep top-k.
            in_mag = np.abs(self.weights_in).sum(axis=1)
            out_mag = np.abs(self.weights_out).sum(axis=0)
            scores = in_mag + out_mag
            keep_indices = np.argsort(-scores, kind="stable")[:new_size]
            keep = np.sort(keep_indices)
            self.weights_in = self.weights_in[keep, :]
            self.weights_out = self.weights_out[:, keep]
            self.bias_h = self.bias_h[keep]
            self.prediction_weights = self.prediction_weights[keep]
            for head in self.auxiliary_prediction_weights:
                self.auxiliary_prediction_weights[head] = self.auxiliary_prediction_weights[head][keep]
            if self._has_attention():
                self.attention_weights = self.attention_weights[keep, :]
            self.hidden = self.hidden[keep] if self.hidden.size == old_size else _empty_array(new_size)
            self.hidden_trace = (
                self.hidden_trace[keep] if self.hidden_trace.size == old_size else _empty_array(new_size)
            )
        self.hidden_size = new_size

    def clone_for_offspring(
        self,
        rng: Random,
        mutation_scale: float = 0.03,
        target_hidden_size: int | None = None,
    ) -> "TinyBrain":
        # Deep-copy state via the dict round-trip to avoid alias bugs, then
        # mutate the weights with gaussian noise of the requested scale.
        data = self.to_dict(include_state=False)

        # Each weight gets an independent gaussian perturbation. We loop the
        # rng element-wise to use the caller-provided Random for reproducibility
        # rather than numpy's separate global state.
        def _mutate_array(arr: np.ndarray) -> np.ndarray:
            shape = arr.shape
            flat = arr.flatten()
            for i in range(flat.size):
                flat[i] += rng.gauss(0.0, mutation_scale)
            return flat.reshape(shape)

        def _mutated(name: str, fallback_shape: tuple[int, ...] | None = None) -> np.ndarray:
            arr = np.array(data[name], dtype=_DTYPE)
            if fallback_shape is not None and arr.size and arr.shape != fallback_shape:
                arr = arr.reshape(fallback_shape)
            return _mutate_array(arr)

        weights_in = _mutated("weights_in", (self.hidden_size, self.input_size))
        weights_out = _mutated("weights_out", (self.output_size, self.hidden_size))
        bias_h = _mutated("bias_h")
        bias_o = _mutated("bias_o")
        prediction_weights = _mutated("prediction_weights")
        auxiliary_prediction_weights: dict[str, np.ndarray] = {}
        for head, weights_list in data.get("auxiliary_prediction_weights", {}).items():
            arr = np.array(weights_list, dtype=_DTYPE)
            auxiliary_prediction_weights[head] = _mutate_array(arr)
        if self._has_attention() and data.get("attention_weights"):
            attention_weights = _mutate_array(
                np.array(data["attention_weights"], dtype=_DTYPE).reshape(self.hidden_size, self.input_size)
            )
            attention_bias = _mutate_array(np.array(data["attention_bias"], dtype=_DTYPE))
        else:
            attention_weights = _empty_array((0, 0))
            attention_bias = _empty_array(0)

        child = TinyBrain(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            weights_in=weights_in,
            weights_out=weights_out,
            bias_h=bias_h,
            bias_o=bias_o,
            prediction_weights=prediction_weights,
            auxiliary_prediction_weights=auxiliary_prediction_weights,
            attention_weights=attention_weights,
            attention_bias=attention_bias,
            hidden=_empty_array(self.hidden_size),
            last_outputs=_empty_array(self.output_size),
            last_inputs=_empty_array(self.input_size),
            last_attention=_empty_array(self.input_size),
            last_prediction_errors={head: 0.0 for head in PREDICTION_HEADS},
            input_trace=_empty_array(self.input_size),
            hidden_trace=_empty_array(self.hidden_size),
        )
        child._ensure_auxiliary_prediction_heads()
        if target_hidden_size is not None and target_hidden_size != child.hidden_size:
            child.resize_hidden(rng, target_hidden_size)
        return child

    def to_dict(self, include_state: bool = True) -> dict[str, Any]:
        # Serialize as nested lists so checkpoints, the torch backend, and
        # tests stay format-compatible. Round to 7 decimals for stable diffs.
        def _round(arr: np.ndarray) -> list[float]:
            return [round(float(v), 7) for v in arr.flatten().tolist()]

        data: dict[str, Any] = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "weights_in": _round(self.weights_in),
            "weights_out": _round(self.weights_out),
            "bias_h": _round(self.bias_h),
            "bias_o": _round(self.bias_o),
            "prediction_weights": _round(self.prediction_weights),
            "auxiliary_prediction_weights": {
                head: _round(weights) for head, weights in sorted(self.auxiliary_prediction_weights.items())
            },
            "attention_weights": _round(self.attention_weights),
            "attention_bias": _round(self.attention_bias),
        }
        if include_state:
            data["hidden"] = _round(self.hidden)
            data["last_outputs"] = _round(self.last_outputs)
            data["last_inputs"] = _round(self.last_inputs)
            data["last_attention"] = _round(self.last_attention)
            data["last_prediction_errors"] = {
                head: round(float(value), 7)
                for head, value in sorted(self.last_prediction_errors.items())
            }
            data["input_trace"] = _round(self.input_trace)
            data["hidden_trace"] = _round(self.hidden_trace)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TinyBrain":
        input_size = int(data["input_size"])
        hidden_size = int(data["hidden_size"])
        output_size = int(data["output_size"])

        # Backward compat: pre-attention checkpoints have neither field. Default
        # to empty arrays (no attention head -> _attend pass-through path).
        attention_weights_raw = data.get("attention_weights")
        attention_bias_raw = data.get("attention_bias")
        if attention_weights_raw is not None and len(attention_weights_raw) == input_size * hidden_size:
            attention_weights = np.array(attention_weights_raw, dtype=_DTYPE).reshape(hidden_size, input_size)
        else:
            attention_weights = _empty_array((0, 0))
        if attention_bias_raw is not None and len(attention_bias_raw) == input_size:
            attention_bias = np.array(attention_bias_raw, dtype=_DTYPE)
        else:
            attention_bias = _empty_array(0)

        brain = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weights_in=np.array(data["weights_in"], dtype=_DTYPE).reshape(hidden_size, input_size),
            weights_out=np.array(data["weights_out"], dtype=_DTYPE).reshape(output_size, hidden_size),
            bias_h=np.array(data["bias_h"], dtype=_DTYPE),
            bias_o=np.array(data["bias_o"], dtype=_DTYPE),
            prediction_weights=np.array(data["prediction_weights"], dtype=_DTYPE),
            auxiliary_prediction_weights={
                head: np.array(weights, dtype=_DTYPE)
                for head, weights in data.get("auxiliary_prediction_weights", {}).items()
            },
            attention_weights=attention_weights,
            attention_bias=attention_bias,
            hidden=_empty_array(hidden_size),
            last_outputs=_empty_array(output_size),
            last_inputs=_empty_array(input_size),
            last_attention=_empty_array(input_size),
            last_prediction_errors={head: 0.0 for head in PREDICTION_HEADS},
            input_trace=_empty_array(input_size),
            hidden_trace=_empty_array(hidden_size),
        )
        brain._ensure_auxiliary_prediction_heads()
        if "hidden" in data:
            brain.hidden = np.array(data["hidden"], dtype=_DTYPE)
        if "last_outputs" in data:
            brain.last_outputs = np.array(data["last_outputs"], dtype=_DTYPE)
        if "last_inputs" in data:
            brain.last_inputs = np.array(data["last_inputs"], dtype=_DTYPE)
        if "last_attention" in data:
            brain.last_attention = np.array(data["last_attention"], dtype=_DTYPE)
        if "last_prediction_errors" in data:
            brain.last_prediction_errors = {
                head: float(data["last_prediction_errors"].get(head, 0.0)) for head in PREDICTION_HEADS
            }
        if "input_trace" in data:
            brain.input_trace = np.array(data["input_trace"], dtype=_DTYPE)
        if "hidden_trace" in data:
            brain.hidden_trace = np.array(data["hidden_trace"], dtype=_DTYPE)
        return brain
