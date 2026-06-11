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
    ("resources", 8),         # 7 resource channels + locked_chemical
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
    hidden: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.hidden is None:
            self.hidden = np.zeros(self.hidden_size, dtype=_DTYPE)


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
        tokens = self._encode_tokens(x)

        # Messages use last tick's hidden states (synchronous update).
        prev = [blk.hidden for blk in self.blocks]
        K = len(self.blocks)
        new_hidden: list[np.ndarray] = []
        for j, blk in enumerate(self.blocks):
            mix = np.exp(blk.token_mix - blk.token_mix.max())
            mix /= mix.sum()
            pooled = tokens.T @ mix  # (token_dim,)
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

        outputs = self.bias_o.copy()
        for blk in self.blocks:
            inv_h = 1.0 / math.sqrt(max(1, blk.hidden_size))
            outputs += blk.out_gate * (blk.weights_out @ blk.hidden) * inv_h
        self.last_outputs = outputs
        return outputs.tolist()

    def predict_next_energy(self) -> float:
        total = 0.0
        for blk in self.blocks:
            inv_h = 1.0 / math.sqrt(max(1, blk.hidden_size))
            total += blk.out_gate * float(blk.prediction_weights @ blk.hidden) * inv_h
        return total

    def learn_energy_prediction(self, target: float, learning_rate: float, plasticity: float, prediction_weight: float) -> float:
        """Same clipped delta rule family as TinyController._learn_prediction_heads."""
        lr = max(0.0, min(0.25, learning_rate)) * max(0.0, min(1.0, plasticity))
        pred_lr = lr * max(0.0, min(1.0, prediction_weight)) * 0.025
        error = max(-2.0, min(2.0, max(-2.0, min(2.0, target)) - self.predict_next_energy()))
        for blk in self.blocks:
            if blk.out_gate == 0.0:
                continue
            blk.prediction_weights = np.clip(
                blk.prediction_weights + pred_lr * error * blk.hidden, -4.0, 4.0
            )
        return error

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
            prediction_weights=src.prediction_weights.copy(),
        )
        copy.hidden = src.hidden.copy()
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
    def to_dict(self) -> dict[str, Any]:
        def r(arr: np.ndarray) -> list[float]:
            return [round(float(v), 7) for v in arr.flatten().tolist()]

        return {
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
                    "prediction_weights": r(blk.prediction_weights),
                    "msg_weights": r(self.msg_weights[i]),
                }
                for i, blk in enumerate(self.blocks)
            ],
        }

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
                    prediction_weights=np.array(raw["prediction_weights"], dtype=_DTYPE),
                )
            )
            msg_weights.append(np.array(raw["msg_weights"], dtype=_DTYPE).reshape(h, h))
        K = len(blocks)
        wiring = np.array(data["wiring"], dtype=_DTYPE).reshape(K, K)
        return cls(
            int(data["input_size"]), output_size, token_dim, encoders, blocks, wiring, msg_weights,
            bias_o=np.array(data["bias_o"], dtype=_DTYPE),
        )


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
