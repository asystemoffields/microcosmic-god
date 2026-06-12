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
    own lifetime competence, not a multi-cycle evolutionary race that would
    wash out the init signal. The non-policy producer/consumer network keeps reproducing.
  - Lifetime learning stays ON for every condition - the learning machinery is
    part of what may transfer, and random gets exactly the same machinery.
  - The same set of world seeds is used for every condition => paired comparison.

No ranking or training is done for transfer. We take already-saved champions
and measure them. (Anti-hidden-objective, per the runway doc.)
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

import numpy as np

from microcosmic_god.brain import PREDICTION_HEADS, TinyController
from microcosmic_god.config import RunConfig
from microcosmic_god.modular import ModularController
from microcosmic_god.params import ParamVector
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


# ---- modular (typed multi-block) variants of the three builders ---------- #
def _is_modular(controller_dict: dict[str, Any]) -> bool:
    return controller_dict.get("architecture") == "modular_v1"


def _reset_transient_modular(controller: ModularController) -> ModularController:
    for blk in controller.blocks:
        blk.hidden = np.zeros(blk.hidden_size, dtype=blk.weights_in.dtype)
        blk.hidden_trace = np.zeros(blk.hidden_size, dtype=blk.weights_in.dtype)
        blk.pooled_trace = None
    return controller


def _trained_template_modular(controller_dict: dict[str, Any]) -> ModularController:
    return _reset_transient_modular(ModularController.from_dict(controller_dict))


def _random_template_modular(controller_dict: dict[str, Any], seed: int) -> ModularController:
    """Fresh-init control with the exact trained block structure.

    Same block count and per-block hidden sizes; every array re-drawn at the
    init scales used by ModularController.random/add_block; gates back to the
    uniform 1/K init convention; plasticity/neuromod/wiring init-neutral.
    """
    rng = Random(seed)
    controller = ModularController.from_dict(controller_dict)
    n_blocks = len(controller.blocks)

    def draw(arr: np.ndarray, scale: float) -> np.ndarray:
        return np.array(
            [rng.gauss(0.0, scale) for _ in range(arr.size)], dtype=arr.dtype
        ).reshape(arr.shape)

    controller.encoders = [(draw(W, 0.5), draw(b, 0.5)) for W, b in controller.encoders]
    controller.bias_o = np.zeros_like(controller.bias_o)
    controller.wiring = np.zeros_like(controller.wiring)
    controller.msg_weights = [draw(m, 0.3) for m in controller.msg_weights]
    for blk in controller.blocks:
        blk.weights_in = draw(blk.weights_in, 0.5)
        blk.token_mix = draw(blk.token_mix, 0.8)
        blk.bias = draw(blk.bias, 0.2)
        blk.weights_out = draw(blk.weights_out, 0.5)
        blk.prediction_weights = draw(blk.prediction_weights, 0.3)
        blk.out_gate = 1.0 / max(1, n_blocks)
        blk.plasticity_scale = 1.0
        blk.neuromod_weights = np.zeros_like(blk.neuromod_weights)
        for head in blk.auxiliary_prediction_weights:
            blk.auxiliary_prediction_weights[head] = np.zeros_like(blk.auxiliary_prediction_weights[head])
    return _reset_transient_modular(controller)


def _permuted_template_modular(controller_dict: dict[str, Any], seed: int) -> ModularController:
    """Shuffle every weight array's entries in place (per-array distribution
    preserved, arrangement destroyed). Scalar gates/plasticity stay — they are
    per-block scalars with no arrangement to destroy."""
    controller = ModularController.from_dict(controller_dict)
    rs = np.random.RandomState(seed)

    def shuffle(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        flat = arr.flatten()
        rs.shuffle(flat)
        return flat.reshape(arr.shape)

    controller.encoders = [(shuffle(W), shuffle(b)) for W, b in controller.encoders]
    controller.bias_o = shuffle(controller.bias_o)
    controller.wiring = shuffle(controller.wiring) if controller.wiring.size else controller.wiring
    controller.msg_weights = [shuffle(m) for m in controller.msg_weights]
    for blk in controller.blocks:
        blk.weights_in = shuffle(blk.weights_in)
        blk.token_mix = shuffle(blk.token_mix)
        blk.bias = shuffle(blk.bias)
        blk.weights_out = shuffle(blk.weights_out)
        blk.prediction_weights = shuffle(blk.prediction_weights)
        blk.neuromod_weights = shuffle(blk.neuromod_weights)
        for head in blk.auxiliary_prediction_weights:
            blk.auxiliary_prediction_weights[head] = shuffle(blk.auxiliary_prediction_weights[head])
    return _reset_transient_modular(controller)


def _remapped_template(controller_dict: dict[str, Any], seed: int) -> TinyController:
    """Trained weights with input columns permuted: a controlled interface
    remap. Measures how indexical the competence is — the first rung of the
    interface-distance ladder (the Catch question in controlled form)."""
    controller = TinyController.from_dict(controller_dict)
    rs = np.random.RandomState(seed)
    perm = rs.permutation(controller.input_size)
    controller.weights_in = controller.weights_in[:, perm]
    if controller.attention_weights.size:
        controller.attention_weights = controller.attention_weights[:, perm]
    return _reset_transient_and_memory(controller)


def _remapped_template_modular(controller_dict: dict[str, Any], seed: int) -> ModularController:
    """Modular variant: permute each typed encoder's input columns within its
    group span. Group identity (which span is resource-like, self-like, ...)
    is preserved; the wiring inside each type is scrambled. A pure
    within-type re-mapping challenge."""
    controller = ModularController.from_dict(controller_dict)
    rs = np.random.RandomState(seed)
    controller.encoders = [
        (W[:, rs.permutation(W.shape[1])], b) for W, b in controller.encoders
    ]
    return _reset_transient_modular(controller)


@dataclass
class BrainInstance:
    condition: str
    label: str
    template: Any  # TinyController or ModularController


def build_brain_instances(
    checkpoint: dict[str, Any], n_random: int, n_permuted: int, n_frozen: int = 0, n_remapped: int = 0
) -> list[BrainInstance]:
    controller_dict = checkpoint["brain"]
    modular = _is_modular(controller_dict)
    trained = _trained_template_modular if modular else _trained_template
    random_b = _random_template_modular if modular else _random_template
    permuted_b = _permuted_template_modular if modular else _permuted_template
    remapped_b = _remapped_template_modular if modular else _remapped_template

    instances = [BrainInstance("trained", "trained", trained(controller_dict))]
    for i in range(n_random):
        instances.append(BrainInstance("random", f"random_{i}", random_b(controller_dict, seed=1000 + i)))
    for i in range(n_permuted):
        instances.append(BrainInstance("permuted", f"permuted_{i}", permuted_b(controller_dict, seed=2000 + i)))
    for i in range(n_remapped):
        instances.append(BrainInstance("remapped", f"remapped_{i}", remapped_b(controller_dict, seed=3000 + i)))
    # Frozen arm: the trained weights with lifetime learning disabled — the
    # "is its merit what it knows at init, or what it keeps re-learning?"
    # control. The flag survives the per-individual serialization round-trip.
    for i in range(n_frozen):
        template = trained(controller_dict)
        template.learning_frozen = True
        instances.append(BrainInstance("frozen", f"frozen_{i}", template))
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
    # Behavioral fingerprint (defaults keep pre-fingerprint .runs.jsonl loadable;
    # don't mix resumed pre-fingerprint records into a fingerprint analysis).
    # Distinguishes strategic conservation from stasis, and skill from spam.
    action_entropy: float = 0.0     # mean per-founder entropy (nats) of action distribution
    move_rate: float = 0.0          # fraction of active ticks spent on each action family
    observe_rate: float = 0.0
    rest_rate: float = 0.0
    tool_attempt_rate: float = 0.0  # use_tool/craft/build attempts (successes are tool_successes)
    places_visited: float = 0.0     # mean unique places per founder


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def evaluate_run(
    template: Any,
    genome_dict: dict[str, Any],
    world_seed: int,
    cohort_size: int,
    ticks: int,
    places: int,
    harshness: float,
    start_energy: float,
    world_refresh_every: int = 0,
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
        world_refresh_every=world_refresh_every,
        log_every=10**9,
        checkpoint_every=10**9,
        neural_checkpoint_limit=0,  # no checkpoint files
        event_detail=False,
        stop_on_neural_extinction=False,
        stop_on_full_extinction=False,
        output_dir=str(SCRATCH_DIR),
    )
    sim = FrozenAgentSim(config)

    # Inject the founding cohort: identical controller, identical params, spread
    # deterministically across places so every condition starts the same way.
    founder_ids: list[int] = []
    place_rng = Random(world_seed ^ 0x5EED)
    for _ in range(cohort_size):
        params = ParamVector.from_dict(genome_dict)
        location = place_rng.randrange(len(sim.world.places))
        individual = sim.add_individual(
            "agent", params, location, start_energy, controller_template=template
        )
        if individual is not None:
            founder_ids.append(individual.id)

    # Per-founder accumulators.
    energy_sum = {fid: 0.0 for fid in founder_ids}
    energy_ticks = {fid: 0 for fid in founder_ids}
    energy_peak = {fid: 0.0 for fid in founder_ids}
    # Prediction-error trajectory: (life_tick_index, mean_abs_pred_err) per founder.
    pred_traj: dict[int, list[float]] = {fid: [] for fid in founder_ids}
    # Behavioral fingerprint accumulators.
    action_counts: dict[int, Counter[str]] = {fid: Counter() for fid in founder_ids}
    visited: dict[int, set[int]] = {fid: set() for fid in founder_ids}

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
            action_counts[fid][org.last_action] += 1
            visited[fid].add(org.location)

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

    # Behavioral fingerprint aggregation.
    def _entropy(counter: Counter[str]) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log(c / total) for c in counter.values() if c)

    def _rate(names: set[str]) -> float:
        vals = []
        for fid in founder_ids:
            total = sum(action_counts[fid].values())
            if total:
                vals.append(sum(v for k, v in action_counts[fid].items() if k in names) / total)
        return _mean(vals)

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
        action_entropy=_mean([_entropy(action_counts[fid]) for fid in founder_ids]),
        move_rate=_rate({"move"}),
        observe_rate=_rate({"observe"}),
        rest_rate=_rate({"rest"}),
        tool_attempt_rate=_rate({"use_tool", "craft", "build"}),
        places_visited=_mean([float(len(visited[fid])) for fid in founder_ids]),
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
    "action_entropy",
    "move_rate",
    "observe_rate",
    "rest_rate",
    "tool_attempt_rate",
    "places_visited",
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
    parser.add_argument("--n-frozen", type=int, default=0,
                        help="trained weights with lifetime learning disabled (re-learning control)")
    parser.add_argument("--n-remapped", type=int, default=0,
                        help="trained weights with input columns permuted (interface-remap arm)")
    parser.add_argument("--world-refresh-every", type=int, default=0,
                        help="rewrite world physics every N ticks inside each probe (0 = never)")
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

    instances = build_brain_instances(checkpoint, args.n_random, args.n_permuted, args.n_frozen, args.n_remapped)
    world_seeds = [args.world_seed_base + i for i in range(args.worlds)]

    # Incremental per-run results, so a crash mid-sweep never loses finished
    # evaluations. Each (label, seed) run is independent and fully seeded, so
    # skipping completed pairs on resume reproduces the uninterrupted sweep.
    runs_path = Path(args.out).with_suffix(".runs.jsonl")
    completed: dict[tuple[str, int], RunMetrics] = {}
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            completed[(rec["label"], int(rec["seed"]))] = RunMetrics(**rec["metrics"])
        if completed:
            print(f"resuming   : {len(completed)} completed runs found in {runs_path}")

    print(f"checkpoint : {args.checkpoint}")
    if _is_modular(controller_dict):
        sizes = [int(b["hidden_size"]) for b in controller_dict["blocks"]]
        print(f"controller : modular  in={controller_dict['input_size']} blocks={sizes} "
              f"out={controller_dict['output_size']}")
    else:
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
            key = (inst.label, seed)
            cached = key in completed
            if cached:
                m = completed[key]
            else:
                m = evaluate_run(
                    inst.template, genome_dict, seed,
                    cohort_size=args.cohort, ticks=args.ticks, places=args.places,
                    harshness=args.harshness, start_energy=args.start_energy,
                    world_refresh_every=args.world_refresh_every,
                )
                with runs_path.open("a") as fh:
                    fh.write(json.dumps({"label": inst.label, "seed": seed, "metrics": vars(m)}) + "\n")
            results[inst.label][seed] = m
            done += 1
            print(f"  [{done:>3}/{total}]{'*' if cached else ' '}{inst.label:<12} seed={seed}  "
                  f"alive={m.alive_fraction:.2f} life={m.mean_lifespan:6.1f} "
                  f"E={m.mean_energy:6.2f} tools={m.tool_successes:5.2f} "
                  f"causal_unlock={m.causal_unlocks:.2f} dPE={m.pred_err_delta:+.4f}", flush=True)

    # Collapse instances into conditions (trained / random / permuted / frozen), paired by seed.
    def condition_of(label: str) -> str:
        return label.split("_")[0]

    conditions = list(dict.fromkeys(condition_of(inst.label) for inst in instances))
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
    others = [c for c in conditions if c != "trained"]
    print("\n" + "=" * 78)
    print("SUMMARY (mean over held-out worlds; paired by seed)")
    print("=" * 78)
    header = f"{'metric':<16}{'trained':>11}" + "".join(f"{c:>11}" for c in others) \
        + "".join(f"{'tr-' + c[:4]:>11}" for c in others)
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
        row = f"{field_name:<16}{summary['trained'][field_name]:>11.3f}"
        row += "".join(f"{summary[c][field_name]:>11.3f}" for c in others)
        contrasts[field_name] = {}
        for c in others:
            v = paired(field_name, c)
            contrasts[field_name][f"trained_minus_{c}"] = v
            row += f"{v['mean']:>+10.3f}[t{v['t']:>+5.1f} w{v['win_rate']*100:>3.0f}]"
        print(row)

    payload = {
        "checkpoint": args.checkpoint,
        "brain_shape": {k: controller_dict.get(k) for k in ("input_size", "hidden_size", "output_size", "architecture")},
        "blocks": [int(b["hidden_size"]) for b in controller_dict["blocks"]] if _is_modular(controller_dict) else None,
        "episodic_capacity": controller_dict.get("episodic_capacity"),
        "config": {
            "worlds": world_seeds, "cohort": args.cohort, "ticks": args.ticks,
            "places": args.places, "harshness": args.harshness,
            "start_energy": args.start_energy,
            "n_random": args.n_random, "n_permuted": args.n_permuted,
            "n_frozen": args.n_frozen, "n_remapped": args.n_remapped,
            "world_refresh_every": args.world_refresh_every,
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
