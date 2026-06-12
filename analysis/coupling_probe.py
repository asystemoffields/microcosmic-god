"""Measure how strongly a checkpointed controller's action ranking is coupled
to its observations — the dependent variable for the perception-must-pay
program (docs/TRANSFER_BARRIER.md).

A blind priority program shows: one or two distinct rankings over a long input
stream, per-action output std far below the inter-action gaps, and a zero-input
head identical to the streamed head. A perceiving controller shows many
rankings and a coupling ratio approaching or exceeding 1.

Usage:
  python analysis/coupling_probe.py --checkpoint runs/.../controller_t..._founder.json
  python analysis/coupling_probe.py --checkpoints-dir runs/.../checkpoints --json out.jsonl

With --checkpoints-dir, every controller_*.json is probed and rows are sorted by the
tick parsed from the filename, so the series reads as coupling-over-cycles.
Cheap: forward passes only, no world.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from random import Random

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from microcosmic_god.controller import TinyController  # noqa: E402
from microcosmic_god.modular import ModularController  # noqa: E402
from microcosmic_god.individuals import ACTIONS  # noqa: E402


def _load_controller(path: Path):
    data = json.loads(path.read_text())
    controller_dict = data.get("controller", data)
    if controller_dict.get("architecture") == "modular_v1":
        return controller_dict, ModularController.from_dict
    return controller_dict, TinyController.from_dict


def _ranking(outputs: list[float]) -> tuple[int, ...]:
    head = outputs[: len(ACTIONS)]
    return tuple(sorted(range(len(head)), key=lambda i: head[i], reverse=True))


def probe(path: Path, steps: int, settle: int, seed: int) -> dict:
    controller_dict, from_dict = _load_controller(path)
    rng = Random(seed)

    # Zero-input attractor: what the network does with nothing to see.
    zero_ctrl = from_dict(controller_dict)
    zero_vector = [0.0] * zero_ctrl.input_size
    for _ in range(settle):
        zero_outputs = zero_ctrl.forward(zero_vector)
    zero_rank = _ranking(zero_outputs)

    # Streamed inputs: observation features live in [0, 1] for the most part
    # (clipped to [-1, 1.5] at the source), so uniform [0, 1] draws are a
    # reasonable stand-in for a live stream without needing a world.
    ctrl = from_dict(controller_dict)
    n_in = ctrl.input_size
    for _ in range(settle):
        ctrl.forward([rng.random() for _ in range(n_in)])
    n_actions = len(ACTIONS)
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
    mean_gap = sum(gaps) / len(gaps)
    modal_head = max(heads, key=heads.get)

    tick_match = re.search(r"_t(\d+)_", path.name)
    return {
        "checkpoint": str(path),
        "tick": int(tick_match.group(1)) if tick_match else None,
        "distinct_rankings": len(rankings),
        "distinct_heads": len(heads),
        "stream_head": [ACTIONS[i] for i in _ranking(means)[:3]],
        "zero_input_head": [ACTIONS[i] for i in zero_rank[:3]],
        "zero_head_matches_stream": zero_rank[0] == modal_head,
        "output_std_mean": round(mean_std, 6),
        "output_std_max": round(max(stds), 6),
        "adjacent_gap_mean": round(mean_gap, 6),
        "coupling_ratio": round(mean_std / mean_gap, 6) if mean_gap > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--settle", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--json", type=Path, default=None, help="also write rows as JSONL")
    args = parser.parse_args()

    if args.checkpoint is None and args.checkpoints_dir is None:
        parser.error("need --checkpoint or --checkpoints-dir")
    paths = [args.checkpoint] if args.checkpoint else sorted(args.checkpoints_dir.glob("controller_*.json"))
    rows = [probe(path, args.steps, args.settle, args.seed) for path in paths]
    rows.sort(key=lambda r: (r["tick"] is None, r["tick"]))

    for row in rows:
        print(
            f"t={row['tick']}: rankings={row['distinct_rankings']} heads={row['distinct_heads']} "
            f"coupling={row['coupling_ratio']} std={row['output_std_mean']} gap={row['adjacent_gap_mean']} "
            f"head={'>'.join(row['stream_head'])} zero_match={row['zero_head_matches_stream']}"
        )
    if args.json:
        with args.json.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
