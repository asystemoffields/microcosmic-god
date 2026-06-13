"""Coupling measure: how strongly a controller's action ranking depends on
its observations — the dependent variable for the perception-must-pay
program. A blind attractor program shows few distinct rankings, per-action
output std far below the inter-action gaps (coupling ratio << 1), and a
zero-input head identical to the streamed head. A perceiving controller
shows many rankings and a coupling ratio approaching or exceeding 1.

Forward passes only, no world. Shared by analysis/coupling_probe.py
(checkpoint files) and Simulation._log_aggregate (live periodic sample, so
coupling-over-cycles is captured in the aggregates regardless of which
checkpoints the score-based retention happens to keep).
"""

from __future__ import annotations

from random import Random
from typing import Any, Callable

from .individuals import ACTIONS


def _ranking(outputs: list[float]) -> tuple[int, ...]:
    head = outputs[: len(ACTIONS)]
    return tuple(sorted(range(len(head)), key=lambda i: head[i], reverse=True))


def measure_coupling(
    controller_factory: Callable[[], Any],
    steps: int = 200,
    settle: int = 48,
    seed: int = 11,
) -> dict[str, Any]:
    """Stream seeded uniform inputs through a fresh controller and report the
    coupling statistics. `controller_factory` must return a fresh built
    controller each call (two are used: one for the zero-input attractor, one
    for the stream) so the caller's live state is never touched."""
    rng = Random(seed)
    n_actions = len(ACTIONS)

    zero_ctrl = controller_factory()
    zero_vector = [0.0] * zero_ctrl.input_size
    zero_outputs = zero_ctrl.forward(zero_vector)
    for _ in range(settle - 1):
        zero_outputs = zero_ctrl.forward(zero_vector)
    zero_rank = _ranking(zero_outputs)

    ctrl = controller_factory()
    n_in = ctrl.input_size
    for _ in range(settle):
        ctrl.forward([rng.random() for _ in range(n_in)])

    sums = [0.0] * n_actions
    sq_sums = [0.0] * n_actions
    rankings: set[tuple[int, ...]] = set()
    heads: dict[int, int] = {}
    for _ in range(steps):
        outputs = ctrl.forward([rng.random() for _ in range(n_in)])
        rank = _ranking(outputs)
        rankings.add(rank)
        heads[rank[0]] = heads.get(rank[0], 0) + 1
        for i in range(n_actions):
            sums[i] += outputs[i]
            sq_sums[i] += outputs[i] * outputs[i]

    means = [s / steps for s in sums]
    stds = [max(0.0, sq_sums[i] / steps - means[i] ** 2) ** 0.5 for i in range(n_actions)]
    ordered_means = sorted(means, reverse=True)
    gaps = [ordered_means[i] - ordered_means[i + 1] for i in range(n_actions - 1)]
    mean_std = sum(stds) / n_actions
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    modal_head = max(heads, key=heads.get)

    return {
        "distinct_rankings": len(rankings),
        "distinct_heads": len(heads),
        "output_std_mean": round(mean_std, 6),
        "adjacent_gap_mean": round(mean_gap, 6),
        "coupling_ratio": round(mean_std / mean_gap, 6) if mean_gap > 0 else None,
        "stream_head": [ACTIONS[i] for i in _ranking(means)[:3]],
        "zero_input_head": [ACTIONS[i] for i in zero_rank[:3]],
        "zero_head_matches_stream": zero_rank[0] == modal_head,
    }
