"""The 12-h tier's two pre-registered questions, per run directory.

Usage: python analysis/retention_report.py <run_dir_or_parent> ...

Q1 (held vs delayed erosion): blocks_mean among modulars at t≈11k vs the last
aggregate (t18k+ for full 12-h runs). "Held" = late value within 15% of the
t11k value or above it.
Q2 (dynasty competition): count of structurally active lineages (≥100 steps in
structure_events.jsonl), to correlate with Q1 across worlds.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def report(run_dir: Path) -> None:
    aggs = []
    events = run_dir / "events.jsonl"
    if not events.exists():
        return
    for line in events.open():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") == "aggregate" and e.get("architecture", {}).get("modular"):
            aggs.append((e["tick"], e["architecture"]))
    if not aggs:
        print(f"{run_dir.parent.name}: no modular census")
        return

    def at(t: int):
        best = None
        for tick, arch in aggs:
            if tick <= t:
                best = (tick, arch)
            else:
                break
        return best

    early = at(11_000) or aggs[-1]
    late = aggs[-1]
    eb, lb = early[1]["blocks_mean"], late[1]["blocks_mean"]
    held = lb >= eb * 0.85
    by_lineage: Counter[int] = Counter()
    sf = run_dir / "structure_events.jsonl"
    if sf.exists():
        for line in sf.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("lineage_root_id") is not None:
                by_lineage[r["lineage_root_id"]] += 1
    dynasties = sum(1 for _, n in by_lineage.items() if n >= 100)
    steps = sum(by_lineage.values())
    verdict = "HELD" if held else "ERODED"
    print(
        f"{run_dir.parent.name:<14} t{early[0]:>6} blocks {eb:>5.2f} -> t{late[0]:>6} blocks {lb:>5.2f}  "
        f"[{verdict}]  dynasties(>=100 steps): {dynasties}  total steps: {steps}  "
        f"mod/leg at end: {late[1]['modular']}/{late[1].get('legacy', 0)}"
    )


def main() -> None:
    for arg in sys.argv[1:]:
        root = Path(arg)
        if (root / "events.jsonl").exists():
            report(root)
        else:
            for config in sorted(root.glob("**/config.json")):
                report(config.parent)


if __name__ == "__main__":
    main()
