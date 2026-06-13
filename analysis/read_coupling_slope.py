"""Read the era-2.1 P-2.1-B signal from a run's aggregates: the live
coupling_sample median over ticks (the population perception trajectory) next
to the neural-pool boot curve. Usage:

  python analysis/read_coupling_slope.py kaggle/results/mg-e21-f50s341-c1
  python analysis/read_coupling_slope.py <dir1> <dir2> ...   # compare arms

Prints, per run, a coarse trajectory and an early-vs-late slope verdict: the
median over the first quarter of the run vs the last quarter. A floor-0.5
(treatment) arm should show late > early; a floor-1.0 (control) arm should
stay flat. Robust to checkpoint eviction — reads only the aggregates.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def load_aggregates(run_dir: str) -> list[dict]:
    hits = glob.glob(f"{run_dir}/**/events.jsonl", recursive=True) + glob.glob(f"{run_dir}/events.jsonl")
    if not hits:
        return []
    rows = []
    for line in open(hits[0]):
        e = json.loads(line)
        if e.get("kind") == "aggregate":
            rows.append(e)
    return rows


def summarize(run_dir: str) -> None:
    rows = load_aggregates(run_dir)
    name = Path(run_dir).name
    if not rows:
        print(f"{name}: no aggregates found")
        return
    pts = []
    for e in rows:
        cs = e.get("coupling_sample") or {}
        pts.append((e["tick"], e["pool"].get("neural", 0), e["pool"].get("total", 0),
                    cs.get("median"), cs.get("max"), cs.get("decoupled_head_frac")))
    end = rows[-1]
    floor = None
    # staple floor isn't in the aggregate; infer from config_used if present
    cfg_hits = glob.glob(f"{run_dir}/**/config_used.json", recursive=True) + glob.glob(f"{run_dir}/config_used.json")
    if cfg_hits:
        try:
            floor = json.load(open(cfg_hits[0])).get("staple_cue_floor")
        except Exception:
            pass
    last_tick = end["tick"]
    quarter = last_tick / 4 if last_tick else 0
    early = [p[3] for p in pts if p[3] is not None and p[0] <= quarter]
    late = [p[3] for p in pts if p[3] is not None and p[0] >= 3 * quarter]

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return round(xs[len(xs) // 2], 4) if xs else None

    print(f"=== {name}  (floor={floor}, end t{last_tick}, neural {end['pool'].get('neural',0)}/{end['pool'].get('total',0)}) ===")
    # coarse trajectory: ~10 evenly spaced points
    step = max(1, len(pts) // 10)
    traj = [(p[0], p[3]) for p in pts[::step] if p[3] is not None]
    print("  coupling_median traj:", traj)
    e_med, l_med = med(early), med(late)
    if e_med is not None and l_med is not None:
        delta = round(l_med - e_med, 4)
        verdict = "RISE" if delta > 0.03 else "flat" if abs(delta) <= 0.03 else "fall"
        print(f"  early-q median {e_med} -> late-q median {l_med}  (Δ {delta:+}, {verdict})")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summarize(d)
        print()
