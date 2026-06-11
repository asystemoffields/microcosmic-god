"""Print the architecture-census trajectory of one or more runs.

Usage: python analysis/architecture_trajectory.py runs/e2_rates runs/e1_headstart ...
Each row: tick, modular/legacy counts, block stats, capacity, neuromod gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def show(run_root: Path) -> None:
    for events in sorted(run_root.glob("**/events.jsonl")):
        label = events.parent.name
        rows = []
        for line in events.open():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("kind") == "aggregate" and "architecture" in e:
                a = e["architecture"]
                bc = e.get("brain_capacity", {})
                rows.append((e["tick"], a, bc))
        if not rows:
            continue
        print(f"\n== {run_root.name}/{label} ==")
        print(f"{'tick':>6} {'mod':>5} {'leg':>5} {'blk_mean':>9} {'blk_max':>8} {'cap_mean':>9} {'cap_max':>8} {'nm_mean':>8} {'nm_span':>13} {'plast':>6}")
        step = max(1, len(rows) // 12)
        for tick, a, bc in rows[::step] + ([rows[-1]] if len(rows) % step != 1 else []):
            nm_span = f"{a.get('neuromod_min', '-')}-{a.get('neuromod_max', '-')}" if "neuromod_mean" in a else "-"
            print(
                f"{tick:>6} {a.get('modular', 0):>5} {a.get('legacy', 0):>5} "
                f"{a.get('blocks_mean', '-'):>9} {a.get('blocks_max', '-'):>8} "
                f"{a.get('capacity_mean', '-'):>9} {a.get('capacity_max', '-'):>8} "
                f"{a.get('neuromod_mean', '-'):>8} {nm_span:>13} {a.get('plasticity_scale_mean', '-'):>6}"
            )


def main() -> None:
    for arg in sys.argv[1:]:
        show(Path(arg))


if __name__ == "__main__":
    main()
