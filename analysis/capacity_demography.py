"""Capacity demography: do large-capacity controllers ESTABLISH, or only appear?

The A/B aggregate analysis showed both arms *explore* large capacity (rare-reset
perturbations propose giants everywhere), so exploration does not discriminate. The
Phase 1 question is establishment: conditioned on being born big, does an
individual persist longer and spawn more under the developmental subsidy
than under legacy pricing?

This driver steps a Simulation directly and records, for every neural agent
ever alive: capacity (hidden size), birth tick, death tick (or censored at end),
child count, and death cause. Output: one JSONL per run + a summary table
of persistence and spawning by capacity bin per arm.

Usage:
  python analysis/capacity_demography.py --seeds 201 202 203 --ticks 3000 \
      --out runs/capacity_demo
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from microcosmic_god.config import RunConfig
from microcosmic_god.simulation import Simulation

BINS = ((0, 16, "small(<=16)"), (17, 48, "mid(17-48)"), (49, 128, "large(49-128)"), (129, 10_000, "giant(>128)"))


def bin_label(capacity: int) -> str:
    for low, high, label in BINS:
        if low <= capacity <= high:
            return label
    return "?"


def run_arm(arm: str, seed: int, ticks: int, grace: int, out_dir: Path) -> Path:
    config = RunConfig.from_profile(
        "minute",
        seed=seed,
        max_ticks=ticks,
        max_wall_seconds=0.0,
        neural_upkeep_grace_ticks=grace,
        log_every=10**9,
        checkpoint_every=10**9,
        neural_checkpoint_limit=0,
        event_detail=False,
        stop_on_neural_washout=False,
        stop_on_full_washout=False,
        output_dir=str(out_dir / f"_scratch_{arm}_{seed}"),
    )
    sim = Simulation(config)
    ledger: dict[int, dict] = {}
    for tick in range(1, ticks + 1):
        sim.step()
        for org in sim.individuals.values():
            if org.kind != "agent" or not org.neural:
                continue
            rec = ledger.get(org.id)
            if rec is None:
                ledger[org.id] = rec = {
                    "id": org.id,
                    "capacity": org.controller.hidden_size if org.controller else org.hidden_size(),
                    "born": tick - org.age,
                    "last_seen": tick,
                    "child": org.child_count,
                    "alive_at_end": True,
                }
            else:
                rec["last_seen"] = tick
                rec["child"] = org.child_count
        if tick % 500 == 0:
            active = sum(1 for o in sim.individuals.values() if o.kind == "agent" and o.alive)
            print(f"  [{arm} seed {seed}] tick {tick} active_agents={active} ledger={len(ledger)}", flush=True)
    for rec in ledger.values():
        org = sim.individuals.get(rec["id"])
        rec["alive_at_end"] = bool(org is not None and org.alive)
        rec["lifespan"] = rec["last_seen"] - rec["born"]
    out = out_dir / f"{arm}_seed{seed}.jsonl"
    with out.open("w") as fh:
        for rec in ledger.values():
            fh.write(json.dumps(rec) + "\n")
    return out


def summarize(out_dir: Path) -> None:
    by_arm_bin: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(out_dir.glob("*_seed*.jsonl")):
        arm = path.name.split("_seed")[0]
        for line in path.open():
            rec = json.loads(line)
            by_arm_bin[(arm, bin_label(rec["capacity"]))].append(rec)
    print(f"\n{'arm':<9}{'capacity bin':<15}{'n':>6}{'med life':>10}{'mean life':>11}{'% reach 150':>12}{'mean child':>15}")
    for (arm, label), recs in sorted(by_arm_bin.items()):
        lives = [r["lifespan"] for r in recs]
        reach = sum(1 for r in recs if r["lifespan"] >= 150) / len(recs)
        child = statistics.fmean(r["child"] for r in recs)
        print(f"{arm:<9}{label:<15}{len(recs):>6}{statistics.median(lives):>10.0f}{statistics.fmean(lives):>11.1f}{reach*100:>11.1f}%{child:>15.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203])
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--grace", type=int, default=150)
    parser.add_argument("--out", default="runs/capacity_demo")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--arm", choices=["control", "subsidy"], default=None, help="run a single arm (for parallel launching)")
    parser.add_argument("--seed", type=int, default=None, help="run a single seed (for parallel launching)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        arms = [args.arm] if args.arm else ["control", "subsidy"]
        seeds = [args.seed] if args.seed is not None else args.seeds
        for arm in arms:
            grace = 0 if arm == "control" else args.grace
            for seed in seeds:
                print(f"running {arm} seed {seed} (grace={grace})", flush=True)
                run_arm(arm, seed, args.ticks, grace, out_dir)
    summarize(out_dir)


if __name__ == "__main__":
    main()
