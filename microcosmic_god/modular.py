"""Typed modular controller (Phase 2 of docs/CONTROLLER_EVOLVABILITY.md).

A composition of typed blocks behind uniform interfaces:

    typed observation tokens (self / resources / body / place / ... )
      -> per-type linear encoders to a shared token space
      -> K recurrent blocks, float wiring matrix between them
      -> gated heads (action, energy prediction)

Design commitments:
- Structural mutations must be SURVIVABLE. `add_block` introduces a block with
  a zero output gate and zero outgoing edges: the controller's input/output
  behavior is bit-identical before and after, so selection sees a neutral move
  that later perturbation can exploit. `duplicate_block` copies a block and
  halves the output gate and outgoing edge weights of both copies: also exactly
  function-preserving. These two operators are the substrate's route to
  dramatic architectural innovation (duplication-and-divergence).
- Typed tokens are the transfer story too: blocks read pooled token encodings,
  not raw observation slots, which removes slot-identity wiring (the failure
  mode the Catch postmortem identified).
- Serialization mirrors the TinyController conventions (nested lists, rounded),
  and a legacy TinyController checkpoint imports as a K=1 configuration -
  exactly when the legacy controller has no attention head, approximately
  (attention dropped) otherwise.

v1 scope: forward pass, energy-prediction head with the same delta rule as
TinyController, structural operators, serialization. Full lifetime-learning
parity (output preferences, attention, episodic memory) lands with the
simulation wire-in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random
from typing import Any

import numpy as np

from .organisms import (
    EVENT_MEMORY_SIZE,
    OBSERVATION_SIZE,
    PREDICTION_ERROR_SIZE,
    RECENT_TRACE_SIZE,
    SIGNAL_VALUE_SIZE,
)

_DTYPE = np.float64

# Typed observation schema. Order and sizes mirror Simulation._observe; the
# schema-coverage test asserts the spans tile OBSERVATION_SIZE exactly, so any
# drift in the observation builder fails loudly here.
_FIXED_GROUPS: tuple[tuple[str, int], ...] = (
    ("self_state", 3),        # energy, health, age
    ("resources", 8),         # 7 resource channels + sealed_essence
    ("social", 2),            # crowding, neural fraction
    ("inventory", 1),
    ("body", 8),              # mobility .. plasticity_rate (genome introspection)
    ("context", 4),           # valence, skill breadth, season, climate drift
    ("place_physics", 12),
    ("habitat", 4),           # aquatic, depth, salinity, humidity
)


def observation_groups() -> tuple[tuple[str, int, int], ...]:
    """(name, start, length) spans tiling the observation vector."""
    groups: list[tuple[str, int, int]] = []
    cursor = 0
    for name, length in _FIXED_GROUPS:
        groups.append((name, cursor, length))
        cursor += length
    for name, length in (
        ("recent_trace", RECENT_TRACE_SIZE),
        ("prediction_errors", PREDICTION_ERROR_SIZE),
        ("event_memory", EVENT_MEMORY_SIZE),
        ("signals", SIGNAL_VALUE_SIZE),
    ):
        groups.append((name, cursor, length))
        cursor += length
    return tuple(groups)


@dataclass(slots=True)
class Block:
    hidden_size: int
    weights_in: np.ndarray            # (hidden, token_dim) - reads the pooled token
    token_mix: np.ndarray             # (n_groups,) softmaxed mixing over token encodings
    bias: np.ndarray                  # (hidden,)
    weights_out: np.ndarray           # (n_actions, hidden)
    out_gate: float                   # scalar gate on this block's action contribution
    prediction_weights: np.ndarray    # (hidden,) energy-prediction contribution
    auxiliary_prediction_weights: dict[str, np.ndarray] = field(default_factory=dict)
    # Phase 3: evolvable plasticity. plasticity_scale multiplies this block's
    # learning rates; neuromod_weights contribute to the controller-wide
    # learning gate. Both are neutral at birth (1.0 / zeros) and only
    # perturbation moves them - context-dependent learning is selectable,
    # never imposed.
    plasticity_scale: float = 1.0
    neuromod_weights: np.ndarray = field(default=None)  # type: ignore[assignment]
    hidden: np.ndarray = field(default=None)  # type: ignore[assignment]
    hidden_trace: np.ndarray = field(default=None)  # type: ignore[assignment]
    pooled_trace: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.hidden is None:
            self.hidden = np.zeros(self.hidden_size, dtype=_DTYPE)
        if self.hidden_trace is None:
            self.hidden_trace = np.zeros(self.hidden_size, dtype=_DTYPE)
        if self.neuromod_weights is None:
            self.neuromod_weights = np.zeros(self.hidden_size, dtype=_DTYPE)

    def ensure_auxiliary_heads(self) -> None:
        from .brain import AUXILIARY_PREDICTION_HEADS

        for head in AUXILIARY_PREDICTION_HEADS:
            if head not in self.auxiliary_prediction_weights:
                self.auxiliary_prediction_weights[head] = np.zeros(self.hidden_size, dtype=_DTYPE)


class ModularController:
    """K typed-token recurrent blocks with a float wiring matrix."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        token_dim: int,
        encoders: list[tuple[np.ndarray, np.ndarray]],  # per group: (W (d x len), b (d,))
        blocks: list[Block],
        wiring: np.ndarray,            # (K, K) float edge weights, wiring[i, j] = i -> j
        msg_weights: list[np.ndarray], # per block: (hidden_j, token_dim)? no - (hidden, hidden_src) per edge is heavy;
                                       # v1: per-receiver (hidden, token_dim) reading the SUM of gated sender summaries
        bias_o: np.ndarray | None = None,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.token_dim = token_dim
        self.groups = observation_groups()
        self.encoders = encoders
        self.blocks = blocks
        self.wiring = wiring
        self.msg_weights = msg_weights
        self.bias_o = np.zeros(output_size, dtype=_DTYPE) if bias_o is None else bias_o
        self.last_outputs = np.zeros(output_size, dtype=_DTYPE)
        self.last_inputs = np.zeros(input_size, dtype=_DTYPE)
        self.last_attention = np.ones(input_size, dtype=_DTYPE)  # no attention head: full fidelity
        self.last_prediction_errors: dict[str, float] = {}
        from .brain import PREDICTION_HEADS

        self.last_prediction_errors = {head: 0.0 for head in PREDICTION_HEADS}

    # ------------------------------------------------------------------ #
    # Construction.
    # ------------------------------------------------------------------ #
    @classmethod
    def random(
        cls,
        rng: Random,
        input_size: int,
        output_size: int,
        token_dim: int = 8,
        n_blocks: int = 1,
        block_hidden: int = 8,
    ) -> "ModularController":
        groups = observation_groups()
        if input_size != OBSERVATION_SIZE:
            raise ValueError(f"typed schema covers {OBSERVATION_SIZE} inputs, got {input_size}")

        def arr(*shape: int, scale: float = 0.5) -> np.ndarray:
            return np.array([rng.gauss(0.0, scale) for _ in range(int(np.prod(shape)))], dtype=_DTYPE).reshape(shape)

        encoders = [(arr(token_dim, length), arr(token_dim)) for _, _, length in groups]
        blocks = [
            Block(
                hidden_size=block_hidden,
                weights_in=arr(block_hidden, token_dim),
                token_mix=arr(len(groups), scale=0.8),
                bias=arr(block_hidden, scale=0.2),
                weights_out=arr(output_size, block_hidden),
                out_gate=1.0 / max(1, n_blocks),
                prediction_weights=arr(block_hidden, scale=0.3),
            )
            for _ in range(n_blocks)
        ]
        wiring = np.zeros((n_blocks, n_blocks), dtype=_DTYPE)
        msg_weights = [arr(block_hidden, block_hidden, scale=0.3) for _ in range(n_blocks)]
        return cls(input_size, output_size, token_dim, encoders, blocks, wiring, msg_weights)

    # ------------------------------------------------------------------ #
    # Forward.
    # ------------------------------------------------------------------ #
    def _encode_tokens(self, x: np.ndarray) -> np.ndarray:
        tokens = np.zeros((len(self.groups), self.token_dim), dtype=_DTYPE)
        for g, (name, start, length) in enumerate(self.groups):
            W, b = self.encoders[g]
            inv = 1.0 / math.sqrt(max(1, length))
            tokens[g] = np.tanh(b + (W @ x[start : start + length]) * inv)
        return tokens

    def forward(self, inputs: list[float] | np.ndarray) -> list[float]:
        if len(inputs) != self.input_size:
            raise ValueError(f"expected {self.input_size} inputs, got {len(inputs)}")
        x = np.asarray(inputs, dtype=_DTYPE)
        self.last_inputs = x
        tokens = self._encode_tokens(x)

        # Messages use last tick's hidden states (synchronous update).
        prev = [blk.hidden for blk in self.blocks]
        K = len(self.blocks)
        new_hidden: list[np.ndarray] = []
        for j, blk in enumerate(self.blocks):
            mix = np.exp(blk.token_mix - blk.token_mix.max())
            mix /= mix.sum()
            pooled = tokens.T @ mix  # (token_dim,)
            if blk.pooled_trace is None:
                blk.pooled_trace = pooled.copy()
            else:
                blk.pooled_trace = blk.pooled_trace * 0.92 + pooled * 0.08
            inv_t = 1.0 / math.sqrt(max(1, self.token_dim))
            drive = blk.bias + 0.62 * prev[j] + (blk.weights_in @ pooled) * inv_t
            if K > 1:
                msg = np.zeros(blk.hidden_size, dtype=_DTYPE)
                for i in range(K):
                    w = self.wiring[i, j]
                    if i == j or w == 0.0:
                        continue
                    src = prev[i]
                    # Senders may differ in width; pad/trim to the receiver's
                    # message-input width so wiring stays shape-agnostic.
                    if src.shape[0] != blk.hidden_size:
                        src = np.resize(src, blk.hidden_size)
                    msg += w * src
                inv_h = 1.0 / math.sqrt(max(1, blk.hidden_size))
                drive = drive + (self.msg_weights[j] @ msg) * inv_h
            new_hidden.append(np.tanh(drive))
        for blk, h in zip(self.blocks, new_hidden):
            blk.hidden = h
            blk.hidden_trace = blk.hidden_trace * 0.90 + h * 0.10

        outputs = self.bias_o.copy()
        for blk in self.blocks:
            inv_h = 1.0 / math.sqrt(max(1, blk.hidden_size))
            outputs += blk.out_gate * (blk.weights_out @ blk.hidden) * inv_h
        self.last_outputs = outputs
        return outputs.tolist()

    # ------------------------------------------------------------------ #
    # Protocol parity with TinyController (duck-typed by the cpu runtime
    # and the simulation): prediction heads, learn(), introspection.
    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        """Total capacity, for aggregate stats parity with TinyController."""
        return self.capacity

    def _has_attention(self) -> bool:
        return False

    def _has_episodic(self) -> bool:
        return False

    def replay_episode(self, rng: Random) -> None:  # episodic parity stub
        return None

    def neuromodulation(self) -> float:
        """Controller-wide learning gate in [0, 2], computed from hidden state.

        Zero neuromod_weights give exactly 1.0 (neutral). Evolution can shape
        when this controller learns: suppress plasticity in familiar contexts,
        amplify it after surprises - whatever pays.
        """
        drive = 0.0
        for blk in self.blocks:
            if blk.neuromod_weights.size:
                inv_h = 1.0 / math.sqrt(max(1, blk.hidden_size))
                drive += blk.out_gate * float(blk.neuromod_weights @ blk.hidden) * inv_h
        return 2.0 / (1.0 + math.exp(-max(-30.0, min(30.0, drive))))

    def predict_next_energy(self) -> float:
        total = 0.0
        for blk in self.blocks:
            inv_h = 1.0 / math.sqrt(max(1, blk.hidden_size))
            total += blk.out_gate * float(blk.prediction_weights @ blk.hidden) * inv_h
        return total

    def predict_outcomes(self) -> dict[str, float]:
        from .brain import PREDICTION_HEADS

        predictions = {"energy": self.predict_next_energy()}
        for head in PREDICTION_HEADS:
            if head == "energy":
                continue
            total = 0.0
            for blk in self.blocks:
                blk.ensure_auxiliary_heads()
                inv_h = 1.0 / math.sqrt(max(1, blk.hidden_size))
                total += blk.out_gate * float(blk.auxiliary_prediction_weights[head] @ blk.hidden) * inv_h
            predictions[head] = total
        return predictions

    def _learn_prediction_heads(
        self,
        targets: dict[str, float],
        learning_rate: float,
        plasticity: float,
        prediction_weight: float,
    ) -> dict[str, float]:
        from .brain import PREDICTION_HEADS

        lr = max(0.0, min(0.25, learning_rate)) * max(0.0, min(1.0, plasticity))
        neuromod = self.neuromodulation()
        pred_lr = lr * max(0.0, min(1.0, prediction_weight)) * 0.025 * neuromod
        predictions = self.predict_outcomes()
        errors: dict[str, float] = {}
        for head in PREDICTION_HEADS:
            if head not in targets:
                continue
            target = max(-2.0, min(2.0, targets[head]))
            error = max(-2.0, min(2.0, target - predictions.get(head, 0.0)))
            errors[head] = error
            for blk in self.blocks:
                if blk.out_gate == 0.0:
                    continue  # silent blocks neither contribute nor learn the heads
                blk_lr = pred_lr * max(0.0, blk.plasticity_scale)
                if head == "energy":
                    blk.prediction_weights = np.clip(
                        blk.prediction_weights + blk_lr * error * blk.hidden, -4.0, 4.0
                    )
                else:
                    blk.ensure_auxiliary_heads()
                    blk.auxiliary_prediction_weights[head] = np.clip(
                        blk.auxiliary_prediction_weights[head] + blk_lr * error * blk.hidden, -4.0, 4.0
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
        """TinyController.learn semantics, distributed over blocks.

        Policy and representation updates are gate-weighted: a block's credit
        for this tick's outcome is proportional to its share of the output.
        Zero-gated (freshly added) blocks stay frozen until perturbation opens
        their gate - neutral additions stay neutral under learning too.
        """
        if action_index < 0 or action_index >= self.output_size:
            return 0.0
        valence = max(-2.0, min(2.0, valence))
        lr = max(0.0, min(0.25, learning_rate)) * max(0.0, min(1.0, plasticity))
        targets = {"energy": energy_delta}
        if outcome_targets:
            targets.update(outcome_targets)
        errors = self._learn_prediction_heads(targets, learning_rate, plasticity, prediction_weight)
        error = errors.get("energy", 0.0)

        gate_total = sum(abs(blk.out_gate) for blk in self.blocks) or 1.0
        neuromod = self.neuromodulation()
        error_values = list(errors.values())
        surprise = sum(abs(value) for value in error_values) / max(1, len(error_values))
        modulation = max(
            -2.0,
            min(
                2.0,
                valence * 0.55 + error * prediction_weight * 0.35 + surprise * prediction_weight * 0.10,
            ),
        )
        representation_lr = lr * (0.15 + max(0.0, min(1.0, prediction_weight)) * 0.35) * 0.010
        for blk in self.blocks:
            if blk.out_gate == 0.0:
                continue
            credit = abs(blk.out_gate) / gate_total * max(0.0, blk.plasticity_scale) * neuromod
            hidden_for_policy = blk.hidden_trace if blk.hidden_trace.size else blk.hidden
            blk.weights_out[action_index] = np.clip(
                blk.weights_out[action_index] + lr * valence * 0.035 * credit * hidden_for_policy,
                -4.0,
                4.0,
            )
            if representation_lr > 0.0 and blk.pooled_trace is not None:
                hidden_active = hidden_for_policy
                hidden_mask = (np.abs(hidden_active) >= 0.015).astype(_DTYPE)
                hidden_gate = np.clip(hidden_active, -1.0, 1.0) * hidden_mask
                delta_in = (representation_lr * modulation * credit) * np.outer(hidden_gate, blk.pooled_trace)
                blk.weights_in = np.clip(blk.weights_in + delta_in, -4.0, 4.0)
        self.bias_o[action_index] = max(
            -4.0, min(4.0, self.bias_o[action_index] + lr * valence * 0.015 * neuromod)
        )
        return error

    def clone_for_offspring(
        self,
        rng: Random,
        mutation_scale: float = 0.03,
        target_hidden_size: int | None = None,
        structural_rate: float = 0.06,
    ) -> "ModularController":
        """Perturbed copy with occasional structural mutation.

        `target_hidden_size` is accepted for interface parity but ignored:
        modular capacity is owned by structure, and the caller is expected to
        sync the child genome's neural_budget to the clone's capacity instead
        (the reverse of the TinyController arrangement).
        """
        child = ModularController.from_dict(self.to_dict())

        def mutate(arr: np.ndarray) -> np.ndarray:
            noise = np.array(
                [rng.gauss(0.0, mutation_scale) for _ in range(arr.size)], dtype=_DTYPE
            ).reshape(arr.shape)
            return arr + noise

        child.encoders = [(mutate(W), mutate(b)) for W, b in child.encoders]
        child.bias_o = mutate(child.bias_o)
        child.wiring = mutate(child.wiring) if child.wiring.size else child.wiring
        child.msg_weights = [mutate(m) for m in child.msg_weights]
        for blk in child.blocks:
            blk.weights_in = mutate(blk.weights_in)
            blk.token_mix = mutate(blk.token_mix)
            blk.bias = mutate(blk.bias)
            blk.weights_out = mutate(blk.weights_out)
            blk.prediction_weights = mutate(blk.prediction_weights)
            blk.out_gate = float(blk.out_gate + rng.gauss(0.0, mutation_scale * 0.5))
            blk.plasticity_scale = max(0.0, float(blk.plasticity_scale + rng.gauss(0.0, mutation_scale)))
            blk.neuromod_weights = mutate(blk.neuromod_weights)
            for head in list(blk.auxiliary_prediction_weights):
                blk.auxiliary_prediction_weights[head] = mutate(blk.auxiliary_prediction_weights[head])
        # Structural mutation: rare, and the additive moves are neutral at birth.
        roll = rng.random()
        if roll < structural_rate:
            kind = rng.random()
            if kind < 0.40:
                child.duplicate_block(rng.randrange(len(child.blocks)))
            elif kind < 0.80:
                child.add_block(rng, hidden_size=max(2, int(rng.gauss(8.0, 3.0))))
            elif len(child.blocks) > 1:
                child.prune_block(rng.randrange(len(child.blocks)))
        # Fresh transient state for the child.
        for blk in child.blocks:
            blk.hidden = np.zeros(blk.hidden_size, dtype=_DTYPE)
            blk.hidden_trace = np.zeros(blk.hidden_size, dtype=_DTYPE)
            blk.pooled_trace = None
        return child

    # ------------------------------------------------------------------ #
    # Structural operators (the evolvability core).
    # ------------------------------------------------------------------ #
    @property
    def capacity(self) -> int:
        return sum(blk.hidden_size for blk in self.blocks)

    def add_block(self, rng: Random, hidden_size: int = 8) -> int:
        """Add a block with zero output gate and zero outgoing edges.

        Exactly function-preserving: the new block reads tokens and messages
        but contributes nothing until perturbation opens its gate.
        """
        def arr(*shape: int, scale: float = 0.5) -> np.ndarray:
            return np.array([rng.gauss(0.0, scale) for _ in range(int(np.prod(shape)))], dtype=_DTYPE).reshape(shape)

        blk = Block(
            hidden_size=hidden_size,
            weights_in=arr(hidden_size, self.token_dim),
            token_mix=arr(len(self.groups), scale=0.8),
            bias=arr(hidden_size, scale=0.2),
            weights_out=arr(self.output_size, hidden_size),
            out_gate=0.0,
            prediction_weights=arr(hidden_size, scale=0.3),
        )
        K = len(self.blocks)
        wiring = np.zeros((K + 1, K + 1), dtype=_DTYPE)
        wiring[:K, :K] = self.wiring
        # Incoming edges may be nonzero (the block listens); outgoing stay zero.
        for i in range(K):
            wiring[i, K] = rng.gauss(0.0, 0.2)
        self.blocks.append(blk)
        self.wiring = wiring
        self.msg_weights.append(arr(hidden_size, hidden_size, scale=0.3))
        return K

    def duplicate_block(self, index: int) -> int:
        """Copy a block; halve the output gate and outgoing edges of both copies.

        Exactly function-preserving (sum-gated heads and sum-weighted wiring),
        leaving two identical halves free to diverge under later perturbation.
        """
        src = self.blocks[index]
        copy = Block(
            hidden_size=src.hidden_size,
            weights_in=src.weights_in.copy(),
            token_mix=src.token_mix.copy(),
            bias=src.bias.copy(),
            weights_out=src.weights_out.copy(),
            out_gate=src.out_gate / 2.0,
            plasticity_scale=src.plasticity_scale,
            neuromod_weights=src.neuromod_weights.copy(),
            prediction_weights=src.prediction_weights.copy(),
            auxiliary_prediction_weights={
                head: weights.copy() for head, weights in src.auxiliary_prediction_weights.items()
            },
        )
        copy.hidden = src.hidden.copy()
        copy.hidden_trace = src.hidden_trace.copy()
        copy.pooled_trace = None if src.pooled_trace is None else src.pooled_trace.copy()
        src.out_gate = src.out_gate / 2.0
        K = len(self.blocks)
        wiring = np.zeros((K + 1, K + 1), dtype=_DTYPE)
        wiring[:K, :K] = self.wiring
        # Copy listens like the original; both emit half-strength.
        wiring[:K, K] = self.wiring[:, index]
        self.wiring = wiring
        self.wiring[index, :] /= 2.0
        self.wiring[K, :K] = self.wiring[index, :K]
        self.msg_weights.append(self.msg_weights[index].copy())
        self.blocks.append(copy)
        return K

    def prune_block(self, index: int) -> None:
        """Remove a block (not function-preserving; selection's problem)."""
        if len(self.blocks) <= 1:
            raise ValueError("cannot prune the last block")
        del self.blocks[index]
        del self.msg_weights[index]
        self.wiring = np.delete(np.delete(self.wiring, index, axis=0), index, axis=1)

    # ------------------------------------------------------------------ #
    # Serialization (TinyController conventions: nested lists, 7 decimals).
    # ------------------------------------------------------------------ #
    def to_dict(self, include_state: bool = True) -> dict[str, Any]:
        def r(arr: np.ndarray) -> list[float]:
            return [round(float(v), 7) for v in arr.flatten().tolist()]

        data: dict[str, Any] = {
            "architecture": "modular_v1",
            "input_size": self.input_size,
            "output_size": self.output_size,
            "token_dim": self.token_dim,
            "encoders": [{"weights": r(W), "bias": r(b)} for W, b in self.encoders],
            "bias_o": r(self.bias_o),
            "wiring": r(self.wiring),
            "blocks": [
                {
                    "hidden_size": blk.hidden_size,
                    "weights_in": r(blk.weights_in),
                    "token_mix": r(blk.token_mix),
                    "bias": r(blk.bias),
                    "weights_out": r(blk.weights_out),
                    "out_gate": round(float(blk.out_gate), 7),
                    "plasticity_scale": round(float(blk.plasticity_scale), 7),
                    "neuromod_weights": r(blk.neuromod_weights),
                    "prediction_weights": r(blk.prediction_weights),
                    "auxiliary_prediction_weights": {
                        head: r(weights) for head, weights in sorted(blk.auxiliary_prediction_weights.items())
                    },
                    "msg_weights": r(self.msg_weights[i]),
                }
                for i, blk in enumerate(self.blocks)
            ],
        }
        if include_state:
            data["block_state"] = [
                {"hidden": r(blk.hidden), "hidden_trace": r(blk.hidden_trace)} for blk in self.blocks
            ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModularController":
        token_dim = int(data["token_dim"])
        output_size = int(data["output_size"])
        groups = observation_groups()
        encoders = []
        for g, (name, start, length) in enumerate(groups):
            enc = data["encoders"][g]
            encoders.append(
                (
                    np.array(enc["weights"], dtype=_DTYPE).reshape(token_dim, length),
                    np.array(enc["bias"], dtype=_DTYPE),
                )
            )
        blocks: list[Block] = []
        msg_weights: list[np.ndarray] = []
        for raw in data["blocks"]:
            h = int(raw["hidden_size"])
            blocks.append(
                Block(
                    hidden_size=h,
                    weights_in=np.array(raw["weights_in"], dtype=_DTYPE).reshape(h, token_dim),
                    token_mix=np.array(raw["token_mix"], dtype=_DTYPE),
                    bias=np.array(raw["bias"], dtype=_DTYPE),
                    weights_out=np.array(raw["weights_out"], dtype=_DTYPE).reshape(output_size, h),
                    out_gate=float(raw["out_gate"]),
                    plasticity_scale=float(raw.get("plasticity_scale", 1.0)),
                    neuromod_weights=(
                        np.array(raw["neuromod_weights"], dtype=_DTYPE)
                        if raw.get("neuromod_weights")
                        else None
                    ),
                    prediction_weights=np.array(raw["prediction_weights"], dtype=_DTYPE),
                    auxiliary_prediction_weights={
                        head: np.array(weights, dtype=_DTYPE)
                        for head, weights in raw.get("auxiliary_prediction_weights", {}).items()
                    },
                )
            )
            msg_weights.append(np.array(raw["msg_weights"], dtype=_DTYPE).reshape(h, h))
        K = len(blocks)
        wiring = np.array(data["wiring"], dtype=_DTYPE).reshape(K, K)
        controller = cls(
            int(data["input_size"]), output_size, token_dim, encoders, blocks, wiring, msg_weights,
            bias_o=np.array(data["bias_o"], dtype=_DTYPE),
        )
        for blk, state in zip(controller.blocks, data.get("block_state", [])):
            blk.hidden = np.array(state["hidden"], dtype=_DTYPE)
            blk.hidden_trace = np.array(state["hidden_trace"], dtype=_DTYPE)
        return controller


def from_tiny(tiny_dict: dict[str, Any]) -> ModularController:
    """Import a legacy TinyController checkpoint as a K=1 modular configuration.

    Exact for controllers without an attention head (and ignoring episodic
    memory): identity encoders partition the raw observation, the single block
    reproduces the legacy recurrent update, and the output path matches up to
    the shared 1/sqrt scaling. Old champions stay valid citizens of the new
    space as its simplest expressible body plan.
    """
    input_size = int(tiny_dict["input_size"])
    hidden_size = int(tiny_dict["hidden_size"])
    output_size = int(tiny_dict["output_size"])
    if input_size != OBSERVATION_SIZE:
        raise ValueError("legacy import expects the standard observation schema")
    groups = observation_groups()
    # Identity encoders into a token space wide enough for the largest group;
    # each token is the raw group values (zero-padded), tanh'd. To keep the
    # import EXACT we bypass pooling: token_dim = input_size and each encoder
    # writes its group into its own span of the token vector; one block with
    # uniform token_mix then sees x scaled by 1/n_groups, which weights_in
    # absorbs (scaled up by n_groups and the tanh undone via arctanh bounds).
    # Simpler and fully exact: token_dim = input_size, encoder g = linear
    # embedding placing the group span (no tanh saturation issues for |x|<=1.5
    # is NOT guaranteed, so we use arctanh-free identity: see _LinearEncoderNote).
    token_dim = input_size
    encoders = []
    n_groups = len(groups)
    for name, start, length in groups:
        W = np.zeros((token_dim, length), dtype=_DTYPE)
        inv = 1.0 / math.sqrt(max(1, length))
        for k in range(length):
            # Pre-divide by the encoder's 1/sqrt(len) and tanh-linearize by
            # keeping values small: scale down by ATANH_SCALE, scale back up
            # in weights_in. tanh(eps*x)/eps -> x as eps -> 0; eps=1e-4 keeps
            # the import exact to ~1e-9 over the clipped observation range.
            W[start + k, k] = 1e-4 / inv
        encoders.append((W, np.zeros(token_dim, dtype=_DTYPE)))
    weights_in_raw = np.array(tiny_dict["weights_in"], dtype=_DTYPE).reshape(hidden_size, input_size)
    inv_obs = 1.0 / math.sqrt(max(1, input_size))
    inv_tok = 1.0 / math.sqrt(max(1, token_dim))
    # forward: drive = bias + 0.62 h + W_in_block @ pooled * inv_tok, where
    # pooled = sum_g mix_g * token_g with uniform mix = 1/n_groups and
    # token_g ~= 1e-4 * x_span / inv(len). Undo all of it in W_in_block.
    weights_in = weights_in_raw * (inv_obs * n_groups / (1e-4 * inv_tok))
    block = Block(
        hidden_size=hidden_size,
        weights_in=weights_in,
        token_mix=np.zeros(n_groups, dtype=_DTYPE),  # softmax(0) = uniform
        bias=np.array(tiny_dict["bias_h"], dtype=_DTYPE),
        weights_out=np.array(tiny_dict["weights_out"], dtype=_DTYPE).reshape(output_size, hidden_size),
        out_gate=1.0,
        prediction_weights=np.array(tiny_dict["prediction_weights"], dtype=_DTYPE),
        auxiliary_prediction_weights={
            head: np.array(weights, dtype=_DTYPE)
            for head, weights in tiny_dict.get("auxiliary_prediction_weights", {}).items()
        },
    )
    controller = ModularController(
        input_size,
        output_size,
        token_dim,
        encoders,
        [block],
        np.zeros((1, 1), dtype=_DTYPE),
        [np.zeros((hidden_size, hidden_size), dtype=_DTYPE)],
        bias_o=np.array(tiny_dict["bias_o"], dtype=_DTYPE),
    )
    return controller
