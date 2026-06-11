"""Probe-world transfer evaluation for microcosmic-god controllers.

The question (from docs/TRANSFER_RUNWAY.md, "Probe Worlds Before Video Games"):

    Does a controller evolved in Microcosmic God carry transferable competence into
    HELD-OUT world variants it never saw - changed physics, resources, and
    environment - measured against controls of the same architecture?

This is the step the project skipped when it jumped straight to Catch. Catch
forced an observation/action adapter problem that dominated the result. Probe
worlds avoid that entirely: the controller's native 72-input / 15-action schema is
preserved, and ONLY the world changes. So any advantage is attributable to the
evolved core, not to adapter training.

Design (isolation):
  - One saved champion controller is the object of study.
  - Controls share its architecture and its GENOME (learning rate, plasticity,
    valences, mobility, episodic capacity). Only the controller WEIGHTS differ:
        trained   - the saved champion weights
        random    - TinyController.random, same shape
        permuted  - champion weights, each array's entries shuffled
                    (preserves the per-array weight distribution, destroys
                    structure - the key "is it structure or just conditioning"
                    control that the Catch test established passes)
  - A cohort of K identical-controller agents is dropped into each held-out world.
    Cohort averaging cuts per-agent variance.
  - Agent reproduction is FROZEN (successors blocked) so we measure the founders'
    own lifetime competence, not a multi-generation evolutionary race that would
    wash out the init signal. The non-policy producer/consumer network keeps reproducing.
  - Lifetime learning stays ON for every condition - the learning machinery is
    part of what may transfer, and random gets exactly the same machinery.
  - The same set of world seeds is used for every condition => paired comparison.

No selection or training is done for transfer. We take already-saved champions
and measure them. (Anti-hidden-objective, per the runway doc.)
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from microcosmic_god.brain import PREDICTION_HEADS, TinyController
from microcosmic_god.config import RunConfig
from microcosmic_god.genome import Genome
from microcosmic_god.organisms import OBSERVATION_SIZE
from microcosmic_god.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = REPO_ROOT / "transfer" / "_probe_scratch"  # overridable via --scratch


# --------------------------------------------------------------------------- #
# Frozen simulation: founders only, resource–consumer network still active.
# --------------------------------------------------------------------------- #
class FrozenAgentSim(Simulation):
    """Simulation in which neural agents cannot reproduce.

    All creation events flow through `_instantiate_offspring` -> `add_individual`. We refuse
    to instantiate any `agent`-kind successors, so the neural cohort stays fixed
    to the injected founders while non-policy producers and consumers keep reproducing as a resource supply.
    """

    def _instantiate_offspring(self, plan: Any):  # type: ignore[override]
        if getattr(plan, "child_kind", None) == "agent":
            return None
        return super()._instantiate_offspring(plan)


# --------------------------------------------------------------------------- #
# Controller condition construction.
# --------------------------------------------------------------------------- #
def _reset_transient_and_memory(controller: TinyController) -> TinyController:
    """Clear hidden state, traces, and episodic contents (keep capacity).

    A fair "drop into a new world" test measures the learned WEIGHTS, not stale
    activations or memories carried over from the world the controller evolved in.
    """
    hs = controller.hidden_size
    controller.hidden = np.zeros(hs, dtype=controller.weights_in.dtype)
    controller.hidden_trace = np.zeros(hs, dtype=controller.weights_in.dtype)
    controller.input_trace = np.zeros(controller.input_size, dtype=controller.weights_in.dtype)
    controller.last_inputs = np.zeros(controller.input_size, dtype=controller.weights_in.dtype)
    controller.last_outputs = np.zeros(controller.output_size, dtype=controller.weights_in.dtype)
    if controller.episodic_slots.ndim == 2 and controller.episodic_slots.shape[0] > 0:
        cap = controller.episodic_slots.shape[0]
        controller.episodic_slots = np.zeros((cap, hs), dtype=controller.weights_in.dtype)
        controller.episodic_age = np.full(cap, -1.0, dtype=controller.weights_in.dtype)
    return controller


def _trained_template(controller_dict: dict[str, Any]) -> TinyController:
    return _reset_transient_and_memory(TinyController.from_dict(controller_dict))


def _random_template(controller_dict: dict[str, Any], seed: int) -> TinyController:
    rng = Random(seed)
    has_attention = bool(controller_dict.get("attention_weights"))
    capacity = int(controller_dict.get("episodic_capacity") or 0)
    controller = TinyController.random(
        rng,
        input_size=int(controller_dict["input_size"]),
        hidden_size=int(controller_dict["hidden_size"]),
        output_size=int(controller_dict["output_size"]),
        with_attention=has_attention,
        episodic_capacity=capacity,
    )
    return _reset_transient_and_memory(controller)


def _permuted_template(controller_dict: dict[str, Any], seed: int) -> TinyController:
    """Shuffle every weight array's entries in place (distribution preserved,
    structure destroyed)."""
    controller = TinyController.from_dict(controller_dict)
    rs = np.random.RandomState(seed)

    def shuffle(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        flat = arr.flatten()
        rs.shuffle(flat)
        return flat.reshape(arr.shape)

    controller.weights_in = shuffle(controller.weights_in)
    controller.weights_out = shuffle(controller.weights_out)
    controller.bias_h = shuffle(controller.bias_h)
    controller.bias_o = shuffle(controller.bias_o)
    controller.prediction_weights = shuffle(controller.prediction_weights)
    for head in controller.auxiliary_prediction_weights:
        controller.auxiliary_prediction_weights[head] = shuffle(controller.auxiliary_prediction_weights[head])
    if controller.attention_weights.size:
        controller.attention_weights = shuffle(controller.attention_weights)
        controller.attention_bias = shuffle(controller.attention_bias)
    return _reset_transient_and_memory(controller)


@dataclass
class BrainInstance:
    condition: str
    label: str
    template: TinyController


def build_brain_instances(
    checkpoint: dict[str, Any], n_random: int, n_permuted: int
) -> list[BrainInstance]:
    controller_dict = checkpoint["brain"]
    instances = [BrainInstance("trained", "trained", _trained_template(controller_dict))]
    for i in range(n_random):
        instances.append(
            BrainInstance("random", f"random_{i}", _random_template(controller_dict, seed=1000 + i))
        )
    for i in range(n_permuted):
        instances.append(
            BrainInstance("permuted", f"permuted_{i}", _permuted_template(controller_dict, seed=2000 + i))
        )
    return instances


# --------------------------------------------------------------------------- #
# One evaluation run: a cohort of one controller in one held-out world.
# --------------------------------------------------------------------------- #
@dataclass
class RunMetrics:
    cohort_size: int
    ticks: int
    alive_fraction: float
    mean_lifespan: float
    mean_energy: float          # mean over founders of their mean-while-active energy
    peak_energy: float          # mean over founders of their peak energy
    tool_successes: float       # mean per founder
    causal_steps: float
    causal_unlocks: float
    structures: float
    energy_gain: float          # success_profile["energy_gain"], mean per founder
    pred_err_early: float       # mean |prediction error| over first third of active span
    pred_err_late: float        # mean |prediction error| over last third of active span
    pred_err_delta: float       # late - early (negative => lifetime learning)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def evaluate_run(
    template: TinyController,
    genome_dict: dict[str, Any],
    world_seed: int,
    cohort_size: int,
    ticks: int,
    places: int,
    harshness: float,
    start_energy: float,
) -> RunMetrics:
    config = RunConfig.from_profile(
        "smoke",
        seed=world_seed,
        places=places,
        initial_plants=max(80, places * 6),
        initial_fungi=max(20, places * 2),
        initial_agents=0,           # we inject the cohort ourselves
        max_population=4000,
        max_ticks=ticks,
        max_wall_seconds=0.0,       # tick-bounded, not wall-bounded
        environment_harshness=harshness,
        log_every=10**9,
        checkpoint_every=10**9,
        neural_checkpoint_limit=0,  # no checkpoint files
        event_detail=False,
        stop_on_neural_extinction=False,
        stop_on_full_extinction=False,
        output_dir=str(SCRATCH_DIR),
    )
    sim = FrozenAgentSim(config)

    # Inject the founding cohort: identical controller, identical genome, spread
    # deterministically across places so every condition starts the same way.
    founder_ids: list[int] = []
    place_rng = Random(world_seed ^ 0x5EED)
    for _ in range(cohort_size):
        genome = Genome.from_dict(genome_dict)
        location = place_rng.randrange(len(sim.world.places))
        individual = sim.add_individual(
            "agent", genome, location, start_energy, controller_template=template
        )
        if individual is not None:
            founder_ids.append(individual.id)

    # Per-founder accumulators.
    energy_sum = {fid: 0.0 for fid in founder_ids}
    energy_ticks = {fid: 0 for fid in founder_ids}
    energy_peak = {fid: 0.0 for fid in founder_ids}
    # Prediction-error trajectory: (life_tick_index, mean_abs_pred_err) per founder.
    pred_traj: dict[int, list[float]] = {fid: [] for fid in founder_ids}

    for _ in range(ticks):
        sim.step()
        for fid in founder_ids:
            org = sim.organisms.get(fid)
            if org is None or not org.alive:
                continue
            energy_sum[fid] += org.energy
            energy_ticks[fid] += 1
            if org.energy > energy_peak[fid]:
                energy_peak[fid] = org.energy
            pe = org.prediction_error_profile
            pred_traj[fid].append(_mean([abs(float(v)) for v in pe]))

    # Aggregate.
    alive = [1.0 if (sim.organisms.get(fid) and sim.organisms[fid].alive) else 0.0 for fid in founder_ids]
    lifespans = [float(sim.organisms[fid].age) for fid in founder_ids if fid in sim.organisms]
    mean_energy = _mean(
        [energy_sum[fid] / energy_ticks[fid] for fid in founder_ids if energy_ticks[fid] > 0]
    )
    peak_energy = _mean([energy_peak[fid] for fid in founder_ids])

    def profile(fid: int, key: str) -> float:
        org = sim.organisms.get(fid)
        if org is None:
            return 0.0
        return float(org.success_profile.get(key, 0.0))

    tool_successes = _mean([float(sim.organisms[fid].successful_tools) for fid in founder_ids if fid in sim.organisms])
    causal_steps = _mean([profile(fid, "causal_step") for fid in founder_ids])
    causal_unlocks = _mean([profile(fid, "causal_unlock") for fid in founder_ids])
    structures = _mean([profile(fid, "structure") for fid in founder_ids])
    energy_gain = _mean([profile(fid, "energy_gain") for fid in founder_ids])

    # Prediction-error early vs late (only founders that stayed active long enough).
    early_vals, late_vals = [], []
    for fid in founder_ids:
        traj = pred_traj[fid]
        if len(traj) < 6:
            continue
        third = max(1, len(traj) // 3)
        early_vals.append(_mean(traj[:third]))
        late_vals.append(_mean(traj[-third:]))
    pred_err_early = _mean(early_vals)
    pred_err_late = _mean(late_vals)

    return RunMetrics(
        cohort_size=len(founder_ids),
        ticks=ticks,
        alive_fraction=_mean(alive),
        mean_lifespan=_mean(lifespans),
        mean_energy=mean_energy,
        peak_energy=peak_energy,
        tool_successes=tool_successes,
        causal_steps=causal_steps,
        causal_unlocks=causal_unlocks,
        structures=structures,
        energy_gain=energy_gain,
        pred_err_early=pred_err_early,
        pred_err_late=pred_err_late,
        pred_err_delta=pred_err_late - pred_err_early,
    )


# --------------------------------------------------------------------------- #
# Sweep + reporting.
# --------------------------------------------------------------------------- #
METRIC_FIELDS = (
    "alive_fraction",
    "mean_lifespan",
    "mean_energy",
    "peak_energy",
    "tool_successes",
    "causal_steps",
    "causal_unlocks",
    "structures",
    "energy_gain",
    "pred_err_delta",
)


def main() -> None:
    global SCRATCH_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "transfer" / "sample_brains" / "learner_champion_hidden14.json"))
    parser.add_argument("--worlds", type=int, default=10, help="number of held-out world seeds")
    parser.add_argument("--world-seed-base", type=int, default=70_000)
    parser.add_argument("--cohort", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--places", type=int, default=24)
    parser.add_argument("--harshness", type=float, default=1.2)
    parser.add_argument("--start-energy", type=float, default=40.0)
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--n-permuted", type=int, default=3)
    parser.add_argument("--out", default=str(REPO_ROOT / "transfer" / "probe_worlds_results.json"))
    parser.add_argument("--scratch", default=str(SCRATCH_DIR), help="per-process scratch dir (parallel-safe)")
    args = parser.parse_args()

    SCRATCH_DIR = Path(args.scratch)

    checkpoint = json.loads(Path(args.checkpoint).read_text())
    controller_dict = checkpoint["brain"]
    genome_dict = checkpoint["genome"]
    if int(controller_dict["input_size"]) != OBSERVATION_SIZE:
        raise SystemExit(
            f"controller input_size {controller_dict['input_size']} != current OBSERVATION_SIZE {OBSERVATION_SIZE}"
        )

    instances = build_brain_instances(checkpoint, args.n_random, args.n_permuted)
    world_seeds = [args.world_seed_base + i for i in range(args.worlds)]

    print(f"checkpoint : {args.checkpoint}")
    print(f"controller      : in={controller_dict['input_size']} hidden={controller_dict['hidden_size']} "
          f"out={controller_dict['output_size']} episodic={controller_dict.get('episodic_capacity')}")
    print(f"conditions : {[inst.label for inst in instances]}")
    print(f"worlds     : {len(world_seeds)} held-out seeds  cohort={args.cohort}  ticks={args.ticks}")
    print()

    # results[label][seed] = RunMetrics
    results: dict[str, dict[int, RunMetrics]] = {inst.label: {} for inst in instances}
    total = len(instances) * len(world_seeds)
    done = 0
    for inst in instances:
        for seed in world_seeds:
            m = evaluate_run(
                inst.template, genome_dict, seed,
                cohort_size=args.cohort, ticks=args.ticks, places=args.places,
                harshness=args.harshness, start_energy=args.start_energy,
            )
            results[inst.label][seed] = m
            done += 1
            print(f"  [{done:>3}/{total}] {inst.label:<12} seed={seed}  "
                  f"alive={m.alive_fraction:.2f} life={m.mean_lifespan:6.1f} "
                  f"E={m.mean_energy:6.2f} tools={m.tool_successes:5.2f} "
                  f"causal_unlock={m.causal_unlocks:.2f} dPE={m.pred_err_delta:+.4f}", flush=True)

    # Collapse instances into conditions (trained / random / permuted), paired by seed.
    def condition_of(label: str) -> str:
        return label.split("_")[0]

    conditions = ["trained", "random", "permuted"]
    # per condition, per metric: list of per-seed means (averaged over instances of that condition)
    summary: dict[str, dict[str, float]] = {}
    per_seed_condition: dict[str, dict[int, dict[str, float]]] = {c: {} for c in conditions}
    for c in conditions:
        labels = [inst.label for inst in instances if condition_of(inst.label) == c]
        for seed in world_seeds:
            for field_name in METRIC_FIELDS:
                vals = [getattr(results[lab][seed], field_name) for lab in labels]
                per_seed_condition[c].setdefault(seed, {})[field_name] = _mean(vals)
        summary[c] = {
            field_name: _mean([per_seed_condition[c][seed][field_name] for seed in world_seeds])
            for field_name in METRIC_FIELDS
        }

    # Paired contrasts vs trained, per metric.
    print("\n" + "=" * 78)
    print("SUMMARY (mean over held-out worlds; paired by seed)")
    print("=" * 78)
    header = f"{'metric':<16}{'trained':>11}{'random':>11}{'permuted':>11}{'tr-rand':>11}{'tr-perm':>11}"
    print(header)
    print("-" * len(header))
    def paired(field_name: str, other: str) -> dict[str, float]:
        diffs = [
            per_seed_condition["trained"][s][field_name] - per_seed_condition[other][s][field_name]
            for s in world_seeds
        ]
        mean = _mean(diffs)
        sd = float(statistics.pstdev(diffs)) if len(diffs) > 1 else 0.0
        n = len(diffs)
        # paired t-like statistic (mean / standard error); robust enough as a
        # magnitude indicator for n~10-12 seeds.
        se = sd / math.sqrt(n) if n > 0 and sd > 0 else 0.0
        tstat = mean / se if se > 0 else (math.inf if mean != 0 else 0.0)
        wins = sum(1 for d in diffs if d > 0)
        return {"mean": mean, "sd": sd, "n": n, "t": tstat, "win_rate": wins / n if n else 0.0}

    contrasts: dict[str, dict[str, Any]] = {}
    for field_name in METRIC_FIELDS:
        t = summary["trained"][field_name]
        r = summary["random"][field_name]
        p = summary["permuted"][field_name]
        vr = paired(field_name, "random")
        vp = paired(field_name, "permuted")
        contrasts[field_name] = {"trained_minus_random": vr, "trained_minus_permuted": vp}
        print(f"{field_name:<16}{t:>11.3f}{r:>11.3f}{p:>11.3f}"
              f"{vr['mean']:>+10.3f}[t{vr['t']:>+5.1f} w{vr['win_rate']*100:>3.0f}]"
              f"{vp['mean']:>+10.3f}[t{vp['t']:>+5.1f} w{vp['win_rate']*100:>3.0f}]")

    payload = {
        "checkpoint": args.checkpoint,
        "brain_shape": {k: controller_dict[k] for k in ("input_size", "hidden_size", "output_size")},
        "episodic_capacity": controller_dict.get("episodic_capacity"),
        "config": {
            "worlds": world_seeds, "cohort": args.cohort, "ticks": args.ticks,
            "places": args.places, "harshness": args.harshness,
            "start_energy": args.start_energy,
            "n_random": args.n_random, "n_permuted": args.n_permuted,
        },
        "summary": summary,
        "contrasts": contrasts,
        "per_seed": {
            c: {str(s): per_seed_condition[c][s] for s in world_seeds} for c in conditions
        },
        "raw": {
            lab: {str(s): vars(results[lab][s]) for s in world_seeds} for lab in results
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")

    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
