"""Analyze the growth-economics A/B: does the developmental subsidy let the
pool explore neural capacity that legacy pricing strangles?

Reads brain_capacity aggregates {min, mean, p90, max} from each run's
events.jsonl and reports per-arm trajectories plus the Phase 1 gate verdict
(docs/CONTROLLER_EVOLVABILITY.md): the subsidy arm must EXPLORE budget > 20
within a standard run. Exploration, not retention, is the gate - we are
testing that the valley is crossable.

Usage: python analysis/ab_econ_capacity.py [runs/ab_econ]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

GATE_CAPACITY = 20


def trajectory(run_dir: Path) -> list[tuple[int, dict[str, float]]]:
    points = []
    for events in run_dir.glob("*/events.jsonl"):
        for line in events.open():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue  # tail line of a still-running run
            if e.get("kind") == "aggregate" and "brain_capacity" in e:
                points.append((int(e["tick"]), e["brain_capacity"]))
    return sorted(points)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/ab_econ")
    arms: dict[str, dict[str, list]] = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        arm, _, seed = run_dir.name.partition("_seed")
        traj = trajectory(run_dir)
        if not traj:
            continue
        peak_max = max(bc["max"] for _, bc in traj)
        peak_p90 = max(bc["p90"] for _, bc in traj)
        end_tick, end_bc = traj[-1]
        arms.setdefault(arm, {"peak_max": [], "peak_p90": [], "end_mean": [], "rows": []})
        arms[arm]["peak_max"].append(peak_max)
        arms[arm]["peak_p90"].append(peak_p90)
        arms[arm]["end_mean"].append(end_bc["mean"])
        arms[arm]["rows"].append(
            f"  seed {seed}: ticks={end_tick} peak_max={peak_max} peak_p90={peak_p90} "
            f"end mean={end_bc['mean']:.2f} p90={end_bc['p90']} max={end_bc['max']}"
        )

    for arm, data in sorted(arms.items()):
        print(f"== {arm} ==")
        for row in data["rows"]:
            print(row)
        print(
            f"  arm peaks: max={max(data['peak_max'])} "
            f"mean-of-peak-max={statistics.fmean(data['peak_max']):.1f} "
            f"end-mean={statistics.fmean(data['end_mean']):.2f}"
        )

    if "subsidy" in arms and "control" in arms:
        explored = max(arms["subsidy"]["peak_max"])
        control_peak = max(arms["control"]["peak_max"])
        verdict = "PASS" if explored > GATE_CAPACITY else "PARK (raise grace/floor or escalate run length before judging)"
        print(f"\nGATE (subsidy explores capacity > {GATE_CAPACITY}): peak={explored} vs control peak={control_peak} -> {verdict}")


if __name__ == "__main__":
    main()
