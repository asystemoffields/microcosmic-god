from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from random import Random
from typing import Any

from .backends import BrainLearningCase, make_brain_runtime
from .brain import TinyController
from .checkpoints import CheckpointManager
from .config import RunConfig
from .debrief import build_debrief, population_counts, success_profile_summary, world_energy_summary, world_physics_summary
from .energy import (
    AFFORDANCES,
    Artifact,
    MATERIALS,
    artifact_capability,
    best_affordance,
    build_artifact,
    build_structure,
    component_properties,
    derive_artifact_capabilities,
    derive_affordances,
    extend_structure,
    structure_capability,
)
from .optimization import Optimizer, OffspringPlan
from .params import MEMORY_BUDGET_MAX, NEURAL_BUDGET_MAX, ParamVector
from .interventions import Intervention, load_interventions
from .observer import EventObserver
from .organisms import (
    ACTIONS,
    ACTION_INDEX,
    OBSERVATION_SIZE,
    Individual,
    individual_from_genome,
    make_modular_brain_for_genome,
)
from .runlog import RunLogger
from .world import Place, World


SKILL_TRANSFER: dict[str, dict[str, float]] = {
    "lash": {"craft": 0.35, "build": 0.18, "support": 0.12, "encase": 0.08, "carry": 0.08, "record": 0.05},
    "craft": {"lash": 0.18, "build": 0.12},
    "build": {"support": 0.24, "anchor": 0.14, "shelter": 0.08, "lash": 0.08},
    "cleave": {"hoist": 0.14, "shear": 0.06},
    "shear": {"cleave": 0.06, "winnow": 0.05},
    "encase": {"winnow": 0.14, "insulate": 0.08, "energy_storage": 0.08, "carry": 0.05},
    "winnow": {"encase": 0.10, "permeable": 0.08, "reaction_surface": 0.05},
    "kindle": {"ferry": 0.12, "insulate": 0.06, "energy_storage": 0.04},
    "ferry": {"kindle": 0.08, "energy_storage": 0.08, "gradient_harvest": 0.05},
    "hoist": {"traverse": 0.10, "support": 0.08, "cleave": 0.05},
    "traverse": {"anchor": 0.10, "float": 0.08, "support": 0.06},
    "protect": {"insulate": 0.08, "shelter": 0.05},
    "record": {"inscribe": 0.12, "carry": 0.06},
    "inscribe": {"record": 0.08, "interpret_mark": 0.04},
    "interpret_mark": {"inscribe": 0.03},
}


class Simulation:
    def __init__(self, config: RunConfig):
        self.config = config
        self.rng = Random(config.seed)
        self.world = World.generate(self.rng, config)
        self.logger = RunLogger(config)
        self.observer = EventObserver(self.logger)
        self.checkpoints = CheckpointManager(self.logger.checkpoint_dir, config.neural_checkpoint_limit)
        self.brain_runtime = make_brain_runtime(config.compute_backend, config.device)
        self.optimization = Optimizer(self.rng, config)
        self.interventions = load_interventions(config.interventions_path) if config.run_mode == "garden" else {}
        self.organisms: dict[int, Individual] = {}
        self.next_id = 1
        self.tick = 0
        self.living_total = 0
        self.living_by_kind: Counter[str] = Counter()
        self.living_neural = 0
        self.births_by_mode: Counter[str] = Counter()
        self.deaths_by_cause: Counter[str] = Counter()
        self.deaths_by_kind_cause: Counter[str] = Counter()
        self.tool_successes: Counter[str] = Counter()
        self.causal_steps: Counter[str] = Counter()
        self.causal_unlocks: Counter[str] = Counter()
        self.mark_lessons: Counter[str] = Counter()
        self.marks_created: Counter[str] = Counter()
        self.mark_lesson_packets: Counter[str] = Counter()
        self.mark_read_value: Counter[str] = Counter()
        self.mark_author_feedbacks: Counter[str] = Counter()
        self.portable_marks_created: Counter[str] = Counter()
        self.portable_mark_reads: Counter[str] = Counter()
        self.collaboration_events: Counter[str] = Counter()
        self.patch_recovery_triggers: int = 0
        self.movement_events: Counter[str] = Counter()
        self.movement_costs: Counter[str] = Counter()
        self.movement_motives: Counter[str] = Counter()
        self.movement_routes: Counter[str] = Counter()
        self.artifacts_created: Counter[str] = Counter()
        self.artifacts_broken: Counter[str] = Counter()
        self.structures_built: Counter[str] = Counter()
        self.structures_extended: Counter[str] = Counter()
        self.physics_events: Counter[str] = Counter()
        self.reproduction_attempts: Counter[str] = Counter()
        self.reproduction_failures: Counter[str] = Counter()
        self.action_counts: Counter[str] = Counter()
        self.action_energy_delta: Counter[str] = Counter()
        self.aggregate_history: list[dict[str, Any]] = []
        self.interventions_applied: list[dict[str, Any]] = []
        self.demonstrations: dict[int, list[tuple[int, str, bool]]] = defaultdict(list)
        self._seed_initial_population()

    def _environment_harshness(self) -> float:
        return max(0.2, float(getattr(self.config, "environment_harshness", 1.0)))

    def _refresh_world(self) -> None:
        """Replace physics + causal challenges with fresh ones, but keep
        accumulated resource state.

        The point of multi-world is to invalidate controllers' memorized solutions
        (place 7 needs kindle first, etc.) so that only controllers with
        abstracted causal rules survive. We do NOT want to also reset resource
        depletion, because that's a free buff to the pool — it lets
        energy-depleted lineages get re-supplied every refresh cycle, which masks the
        cognitive ranking pressure with a basic survival-pressure release.

        So the refresh swaps:
          - place.physics (temperature, fluid_level, abrasion, pressure, ...)
          - place.obstacles (water/thorn/heat/etc. barriers)
          - place.causal_challenge (the prep-step rules that controllers memorize)
          - place.archetype, sun_exposure, water_flow, geothermal, mineral_richness, volatility
        and KEEPS:
          - place.resources (current essence / organic_store / thermal / etc.)
          - place.sealed_essence (the unlocked-energy reserve)
          - place.materials (currently accumulated materials)
          - place.structures, signals, marks (built artifacts persist across refreshes)
          - place.id, name, neighbors (graph topology preserved so movement stays valid)
        """
        from random import Random as _Random
        new_rng = _Random((self.config.seed * 1_000_003) ^ (self.tick * 1_009))
        new_world = self.world.__class__.generate(new_rng, self.config)
        new_world.tick = self.world.tick

        # Carry over the resource/material/artifact state place-by-place. The
        # new world has the same place count and matching ids (both are 0..N-1
        # by construction), but freshly generated physics/obstacles/challenges.
        # We perturb the new world's places to inherit the carried state.
        n_places = min(len(self.world.places), len(new_world.places))
        for i in range(n_places):
            old = self.world.places[i]
            new = new_world.places[i]
            new.resources = dict(old.resources)
            new.regen_recovery_until = old.regen_recovery_until
            new.sealed_essence = old.sealed_essence
            new.materials = dict(old.materials)
            new.structures = list(old.structures)
            new.signals = list(old.signals)
            new.marks = list(old.marks)

        # Edges/topology stay attached to the new world (which was just
        # generated, so its edges already match its place ids). We swap the
        # whole world object.
        for individual in self.organisms.values():
            if not individual.alive:
                continue
            if individual.location >= n_places:
                individual.location = self.rng.randrange(n_places)
            # Place memories refer to the old physics; clear them so controllers
            # have to re-learn what each place is good for.
            individual.place_memory.clear()
        self.world = new_world
        self.logger.event(
            self.tick,
            "world_refreshed",
            {"new_world_seed_basis": self.tick, "resources_preserved": True},
        )

    def _seed_initial_population(self) -> None:
        for _ in range(self.config.initial_plants):
            self.add_individual("plant", ParamVector.plant(self.rng), self.rng.randrange(len(self.world.places)), self.rng.uniform(10.0, 35.0))
        for _ in range(self.config.initial_fungi):
            self.add_individual("fungus", ParamVector.fungus(self.rng), self.rng.randrange(len(self.world.places)), self.rng.uniform(8.0, 28.0))
        for _ in range(self.config.initial_agents):
            params = ParamVector.neural(self.rng)
            template = None
            if self.rng.random() < self.config.initial_modular_fraction:
                n_blocks = self.rng.randint(1, max(1, self.config.initial_modular_max_blocks))
                _, template = make_modular_brain_for_genome(self.rng, params, n_blocks=n_blocks)
            self.add_individual(
                "agent", params, self.rng.randrange(len(self.world.places)), self.rng.uniform(22.0, 55.0),
                controller_template=template,
            )
        self.logger.event(0, "seeded", {"population": population_counts(self.organisms)})

    def add_individual(
        self,
        kind: str,
        params: ParamVector,
        location: int,
        energy: float,
        cycle: int = 0,
        parent_ids: tuple[int, ...] = (),
        controller_template: TinyController | None = None,
    ) -> Individual | None:
        if self.living_total >= self.config.max_population:
            return None
        if kind != "agent":
            params.neural_budget = 0.0
            params.memory_budget = 0.0
            controller_template = None
        individual = individual_from_genome(
            self.rng,
            id_=self.next_id,
            kind=kind,
            params=params,
            location=location % len(self.world.places),
            energy=max(0.1, energy),
            cycle=cycle,
            parent_ids=parent_ids,
            controller_template=controller_template,
        )
        parent_lineages = tuple(
            self.organisms[parent_id].lineage_root_id or parent_id
            for parent_id in parent_ids
            if parent_id in self.organisms
        )
        if parent_ids:
            individual.parent_lineage_ids = parent_lineages or parent_ids
            individual.lineage_root_id = individual.parent_lineage_ids[0]
        else:
            individual.lineage_root_id = individual.id
        individual.neural_upkeep_grace_ticks = self.config.neural_upkeep_grace_ticks
        individual.neural_upkeep_grace_floor = self.config.neural_upkeep_grace_floor
        self.organisms[individual.id] = individual
        self.next_id += 1
        self.living_total += 1
        self.living_by_kind[individual.kind] += 1
        if individual.neural:
            self.living_neural += 1
        return individual

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        reason = "max_ticks"
        try:
            while True:
                elapsed = time.monotonic() - started
                if self.config.max_wall_seconds > 0 and elapsed >= self.config.max_wall_seconds:
                    reason = "max_wall_seconds"
                    break
                if self.tick >= self.config.max_ticks:
                    reason = "max_ticks"
                    break
                counts = self._fast_counts()
                if self.config.stop_on_full_extinction and counts.get("total", 0) == 0:
                    reason = "full_extinction"
                    break
                if self.config.stop_on_neural_extinction and self.tick > 10 and counts.get("neural", 0) == 0:
                    reason = "neural_extinction"
                    break
                self.step()
        finally:
            elapsed = time.monotonic() - started
            self._checkpoint_champions("final")
            debrief = build_debrief(self, reason, elapsed)
            self.logger.write_json("summary.json", debrief)
            self.logger.write_json("world_final.json", self.world.to_summary())
            self.logger.close()
        return debrief

    def step(self) -> None:
        self.tick += 1
        # Multi-world ranking: regenerate the world periodically so the
        # pool faces shifting physics. Controllers that memorized one
        # specific world are removed when it changes; controllers that abstracted the
        # underlying causal rules survive. Selects FOR generalization.
        refresh_every = int(getattr(self.config, "world_refresh_every", 0) or 0)
        if refresh_every > 0 and self.tick > 1 and self.tick % refresh_every == 0:
            self._refresh_world()
        physics_events = self.world.update_environment(self.rng)
        self.physics_events.update(physics_events)
        self._apply_physics_transport()
        self._apply_interventions()
        self.demonstrations.clear()

        # Perception is a tick-start snapshot; action effects below resolve sequentially
        # against current individual locations, spawns, and removals.
        rosters = self._rosters()
        context_bases: dict[int, tuple[list[float], float, float, list[int]]] = {}
        contexts: dict[int, tuple[list[float], int, float, float, list[int]]] = {}
        intents: dict[int, str] = {}
        intent_slots: list[tuple[int, str | None]] = []
        neural_choice_rows: list[tuple[Individual, list[float]]] = []
        neural_actions: dict[int, str] = {}
        feedback: dict[int, dict[str, float]] = defaultdict(lambda: {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        for individual in list(self.organisms.values()):
            if not individual.alive:
                continue
            self._apply_upkeep(individual)
            self._habitat_stress(individual)
            if not individual.alive:
                continue
            observed_tokens = self._observed_tokens(individual)
            observation = self._observe(individual, rosters)
            context_bases[individual.id] = (observation, individual.energy, individual.health, observed_tokens)
            if individual.controller is not None and self.config.compute_backend == "torch":
                neural_choice_rows.append((individual, observation))
                intent_slots.append((individual.id, None))
            else:
                action = self._choose_action(individual, observation)
                intent_slots.append((individual.id, action))

        if neural_choice_rows:
            neural_brains: list[TinyController] = []
            neural_observations: list[list[float]] = []
            for individual, observation in neural_choice_rows:
                assert individual.controller is not None
                neural_brains.append(individual.controller)
                neural_observations.append(observation)
            outputs = self.brain_runtime.forward_many(neural_brains, neural_observations)
            for (individual, _observation), action_outputs in zip(neural_choice_rows, outputs):
                neural_actions[individual.id] = self._choose_action_from_outputs(individual, action_outputs)

        for organism_id, action in intent_slots:
            resolved_action = action if action is not None else neural_actions.get(organism_id, "rest")
            intents[organism_id] = resolved_action
            observation, before_energy, before_health, observed_tokens = context_bases[organism_id]
            contexts[organism_id] = (observation, ACTION_INDEX[resolved_action], before_energy, before_health, observed_tokens)

        active_recombine_places: set[int] = set()
        for organism_id, action in list(intents.items()):
            individual = self.organisms.get(organism_id)
            if individual is None or not individual.alive:
                continue
            if action == "coordinate":
                self._coordinate_recombine(individual, feedback[organism_id])
                active_recombine_places.add(individual.location)
                continue
            self._resolve_action(individual, action, feedback[organism_id])

        for individual in self.organisms.values():
            if individual.alive and individual.recombine_intent_until >= self.tick:
                active_recombine_places.add(individual.location)
        self._resolve_recombine(active_recombine_places, feedback)

        learning_rows: list[tuple[Individual, int, float, float, float, float, dict[str, float], list[int]]] = []
        learning_cases: list[BrainLearningCase] = []
        for organism_id, context in contexts.items():
            individual = self.organisms.get(organism_id)
            if individual is None:
                continue
            _observation, action_index, before_energy, before_health, observed_tokens = context
            action_name = ACTIONS[action_index]
            energy_delta = individual.energy - before_energy
            self.action_counts[action_name] += 1
            self.action_energy_delta[action_name] += energy_delta
            if not individual.alive or individual.controller is None:
                continue
            health_delta = individual.health - before_health
            damage = max(0.0, -health_delta)
            extra = feedback[organism_id]
            movement_hazard = damage * 4.0 if action_name == "move" else 0.0
            valence = (
                individual.params.valence_energy * (energy_delta / 10.0)
                + individual.params.valence_health * (health_delta * 4.0)
                - individual.params.valence_damage * (damage * 4.0)
                + individual.params.valence_reproduction * extra.get("reproduction", 0.0)
                + individual.params.valence_social * extra.get("social", 0.0)
            )
            outcome_targets = {
                "damage": damage * 4.0,
                "reproduction": extra.get("reproduction", 0.0),
                "social": extra.get("social", 0.0),
                "tool": extra.get("tool", 0.0),
                "hazard": movement_hazard,
            }
            learning_rows.append((individual, action_index, energy_delta, health_delta, damage, valence, extra, observed_tokens))
            learning_cases.append(
                BrainLearningCase(
                    controller=individual.controller,
                    action_index=action_index,
                    valence=valence,
                    energy_delta=energy_delta / 10.0,
                    learning_rate=individual.params.learning_rate,
                    plasticity=individual.params.plasticity_rate,
                    prediction_weight=individual.params.prediction_weight,
                    outcome_targets=outcome_targets,
                )
            )

        prediction_errors = self.brain_runtime.learn_many(learning_cases)
        for row, prediction_error in zip(learning_rows, prediction_errors):
            individual, action_index, energy_delta, health_delta, damage, valence, extra, observed_tokens = row
            individual.last_valence = valence
            individual.record_action_result(
                action_index=action_index,
                energy_delta=energy_delta,
                health_delta=health_delta,
                damage=damage,
                prediction_error=prediction_error,
                reproduction_feedback=extra.get("reproduction", 0.0),
                social_feedback=extra.get("social", 0.0),
                tool_feedback=extra.get("tool", 0.0),
                prediction_errors=individual.controller.last_prediction_errors,
            )
            for token in observed_tokens:
                individual.learn_signal_value(token, valence + prediction_error * 0.05)
            self._remember_place(individual)

        for individual in list(self.organisms.values()):
            if individual.alive:
                individual.repair_or_decay()

        if self.tick % self.config.log_every == 0:
            self._log_aggregate()
        if self.tick % self.config.checkpoint_every == 0:
            self._checkpoint_champions("interval")

    def _rosters(self) -> dict[int, list[int]]:
        rosters: dict[int, list[int]] = {place.id: [] for place in self.world.places}
        for individual in self.organisms.values():
            if individual.alive:
                rosters[individual.location].append(individual.id)
        return rosters

    def _subjects(self, individual: Individual | None = None, place_id: int | None = None, extra: list[str] | None = None) -> list[str]:
        subjects: list[str] = []
        if individual is not None:
            subjects.append(f"organism:{individual.id}")
            subjects.append(f"place:{individual.location}")
            lineage_root_id = getattr(individual, "lineage_root_id", 0)
            if lineage_root_id:
                subjects.append(f"lineage:{lineage_root_id}")
        if place_id is not None:
            subject = f"place:{place_id}"
            if subject not in subjects:
                subjects.append(subject)
        if extra:
            subjects.extend(extra)
        return subjects

    def _living_ids_at(self, place_id: int) -> list[int]:
        return [individual.id for individual in self.organisms.values() if individual.alive and individual.location == place_id]

    def _fast_counts(self) -> dict[str, int]:
        counts = {kind: count for kind, count in self.living_by_kind.items() if count > 0}
        counts["neural"] = self.living_neural
        counts["total"] = self.living_total
        return counts

    def _interaction_control(self, individual: Individual) -> float:
        if individual.controller is None:
            return 0.0
        has_history = (
            abs(individual.last_energy_delta)
            + abs(individual.recent_health_delta)
            + sum(abs(value) for value in individual.event_memory)
        ) > 0.0001
        average_error = sum(abs(value) for value in individual.prediction_error_profile) / max(1, len(individual.prediction_error_profile))
        prediction_fit = max(0.0, 1.0 - average_error / 1.5) if has_history else 0.0
        memory_signal = min(1.0, sum(abs(value) for value in individual.event_memory) / max(1.0, len(individual.event_memory) * 0.35))
        learned_skill = self._skill_breadth(individual)
        place_knowledge = min(1.0, len(individual.place_memory) / max(1.0, individual.params.memory_budget * 2.0))
        return max(0.0, min(1.0, prediction_fit * 0.38 + memory_signal * 0.24 + learned_skill * 0.24 + place_knowledge * 0.14))

    def _skill_breadth(self, individual: Individual) -> float:
        tracked = (*AFFORDANCES, "craft", "build", "traverse", "protect", "record", "inscribe", "interpret_mark")
        values = [max(0.0, min(1.0, individual.tool_skill.get(name, 0.0))) for name in tracked]
        if not values:
            return 0.0
        top = sorted(values, reverse=True)[:4]
        mastery = max(values)
        active = sum(1 for value in values if value >= 0.18) / len(values)
        diversity = min(1.0, len([name for name, count in individual.tool_use_counts.items() if count > 0]) / 5.0)
        specialist_practice = 1.0 - math.exp(-max(individual.tool_use_counts.values(), default=0) / 24.0)
        return max(
            0.0,
            min(
                1.0,
                mastery * 0.28
                + sum(top) / max(1, len(top)) * 0.24
                + active * 0.20
                + diversity * 0.14
                + specialist_practice * 0.14,
            ),
        )

    def _increase_skill(self, individual: Individual, skill: str, amount: float, *, transfer: float = 0.0) -> None:
        if amount <= 0.0 or skill not in individual.tool_skill:
            return
        current = individual.tool_skill.get(skill, 0.0)
        individual.tool_skill[skill] = min(1.0, current + amount)
        if transfer <= 0.0:
            return
        for related, weight in SKILL_TRANSFER.get(skill, {}).items():
            if related not in individual.tool_skill:
                continue
            related_current = individual.tool_skill.get(related, 0.0)
            related_gain = amount * transfer * weight * (0.35 + max(0.0, 1.0 - related_current) * 0.65)
            individual.tool_skill[related] = min(1.0, related_current + related_gain)

    def _signal_intensity_from(self, place: Place, source_id: int) -> float:
        return max((signal.intensity for signal in place.signals if signal.source_id == source_id), default=0.0)

    def _active_helper_candidates(self, individual: Individual, focus: str) -> list[tuple[Individual, float]]:
        place = self.world.places[individual.location]
        helpers: list[tuple[Individual, float]] = []
        for helper_id in self._living_ids_at(place.id):
            if helper_id == individual.id:
                continue
            helper = self.organisms.get(helper_id)
            if helper is None or not helper.alive or helper.kind != "agent":
                continue
            signal = self._signal_intensity_from(place, helper.id)
            active = (
                helper.recombine_intent_until >= self.tick
                or helper.last_action in {"coordinate", "signal", "observe"}
                or signal > 0.025
            )
            if not active:
                continue
            energy_readiness = min(1.0, helper.energy / max(1.0, helper.storage_limit()))
            focus_skill = max(
                helper.tool_skill.get(focus, 0.0),
                helper.tool_skill.get("support", 0.0) * 0.45,
                helper.tool_skill.get("protect", 0.0) * 0.30 if focus in {"traverse", "build"} else 0.0,
            )
            body = helper.params.manipulator * 0.22 + helper.params.mobility * 0.12 + helper.params.sensor_range * 0.16
            alignment = 0.32 + signal * 0.34 + (0.18 if helper.recombine_intent_until >= self.tick else 0.0)
            alignment += 0.12 if helper.last_action in {"coordinate", "signal", "observe"} else 0.0
            contribution = max(0.0, min(1.0, (body + focus_skill * 0.32 + helper.params.signal_strength * 0.10 + energy_readiness * 0.14) * alignment))
            if contribution > 0.035:
                helpers.append((helper, contribution))
        helpers.sort(key=lambda item: item[1], reverse=True)
        return helpers[:8]

    def _collective_support(self, individual: Individual, focus: str, difficulty: float = 0.0) -> dict[str, Any]:
        helpers = self._active_helper_candidates(individual, focus)
        if not helpers:
            return {"support": 0.0, "helpers": [], "raw": 0.0}
        raw = sum(value for _, value in helpers)
        support = (1.0 - math.exp(-raw)) * (0.28 + min(0.62, max(0.0, difficulty) * 0.85))
        return {
            "support": max(0.0, min(0.82, support)),
            "helpers": [helper.id for helper, _ in helpers],
            "raw": raw,
        }

    def _agent_defense_context(self, target: Individual, attack_power: float) -> dict[str, Any]:
        if target.kind != "agent":
            return {"bonus": 0.0, "support": 0.0, "structure": 0.0, "shelter": 0.0, "collaboration": None}
        place = self.world.places[target.location]
        shelter = max(place.physics.get("shelter", 0.0), structure_capability(place.structures, "shelter"))
        structure = max(
            structure_capability(place.structures, "protect"),
            structure_capability(place.structures, "support") * 0.35,
            structure_capability(place.structures, "enclose") * 0.25,
            structure_capability(place.structures, "anchor") * 0.18,
        )
        guarded = 0.05 if target.last_action in {"observe", "signal", "coordinate", "build"} else 0.0
        collaboration = self._collective_support(target, "protect", attack_power)
        support = float(collaboration.get("support", 0.0))
        bonus = shelter * 0.16 + structure * 0.24 + support * 0.30 + guarded
        return {
            "bonus": bonus,
            "support": support,
            "structure": structure,
            "shelter": shelter,
            "collaboration": collaboration,
        }

    def _apply_collaboration_effects(self, individual: Individual, focus: str, context: str, collaboration: dict[str, Any], score: float) -> None:
        support = float(collaboration.get("support", 0.0))
        helper_ids = list(collaboration.get("helpers", []))
        if support <= 0.01 or not helper_ids:
            return
        self.collaboration_events[context] += 1
        individual.record_success("collaboration", support * (0.25 + min(1.0, score) * 0.40))
        for helper_id in helper_ids:
            helper = self.organisms.get(helper_id)
            if helper is None or not helper.alive:
                continue
            helper.energy -= 0.004 + support * 0.004
            helper.recent_social_feedback = max(helper.recent_social_feedback, min(1.0, support * 0.35))
            helper.record_success("collaboration", support / max(1, len(helper_ids)))
            self._increase_skill(helper, focus, 0.0015 + support * 0.003, transfer=0.08)
        self.observer.observe(
            self.tick,
            "collaboration",
            {
                "organism_id": individual.id,
                "place": individual.location,
                "focus": focus,
                "context": context,
                "support": support,
                "helpers": helper_ids[:8],
            },
            subjects=self._subjects(individual, extra=[f"affordance:{focus}", f"collaboration:{context}", *[f"organism:{helper_id}" for helper_id in helper_ids[:4]]]),
            score=support + min(1.0, score) * 0.45 + min(0.35, len(helper_ids) * 0.04),
            rarity_key=f"collaboration:{context}:{focus}",
        )

    def _place_exposure_pressure(self, place: Place) -> dict[str, Any]:
        physics = place.physics
        temperature = physics.get("temperature", 0.5)
        humidity = physics.get("humidity", place.habitat.get("humidity", 0.5))
        fluid = physics.get("fluid_level", place.habitat.get("aquatic", 0.0))
        current = physics.get("current_exposure", 0.0)
        elevation = physics.get("elevation", 0.5)
        abrasion = physics.get("abrasion", 0.0)
        wet_dry = physics.get("wet_dry_cycle", 0.0)
        shelter = physics.get("shelter", 0.0)
        cold = max(
            0.0,
            0.34
            - temperature
            + humidity * 0.07
            + current * 0.05
            + max(0.0, elevation - 0.45) * 0.05
            + place.volatility * 0.16
            - place.geothermal * 0.08
            - place.resources.get("thermal", 0.0) / 460.0,
        )
        heat = max(0.0, temperature - 0.72 + place.obstacles.get("heat", 0.0) * 0.12 + place.volatility * 0.06)
        wet = max(0.0, fluid * 0.18 + humidity * 0.08 + wet_dry * 0.20 + current * 0.18 - shelter * 0.05)
        abrasion_pressure = max(0.0, abrasion * 0.20 + max(0.0, elevation - 0.55) * 0.07 + place.volatility * 0.05)
        components = {
            "cold": cold,
            "heat": heat,
            "wet": wet,
            "abrasion": abrasion_pressure,
        }
        primary = max(components, key=components.get)
        if primary == "cold":
            required_affordance = "kindle"
            required_capability = "insulate"
        elif primary == "heat":
            required_affordance = "kindle"
            required_capability = "shelter"
        elif primary == "wet":
            required_affordance = "encase"
            required_capability = "enclose"
        else:
            required_affordance = "lash"
            required_capability = "protect"
        score = max(0.0, min(1.5, cold * 0.72 + heat * 0.54 + wet * 0.42 + abrasion_pressure * 0.36))
        return {
            "kind": "exposure",
            "place": place.id,
            "severity": round(score, 6),
            "primary": primary,
            "required_affordance": required_affordance,
            "required_capability": required_capability,
            "components": {key: round(value, 6) for key, value in components.items() if value > 0.0},
        }

    def _place_hazard_pressure(self, place: Place) -> float:
        physics = place.physics
        obstacles = place.obstacles
        return max(
            0.0,
            min(
                1.5,
                obstacles.get("water", 0.0) * 0.18
                + obstacles.get("height", 0.0) * 0.18
                + obstacles.get("thorn", 0.0) * 0.14
                + obstacles.get("heat", 0.0) * 0.16
                + physics.get("pressure", 0.0) * 0.12
                + physics.get("salinity", 0.0) * 0.08
                + physics.get("current_exposure", 0.0) * 0.10
                + self._place_exposure_pressure(place)["severity"] * 0.16
                + max(0.0, physics.get("interiority", 0.0) - physics.get("boundary_permeability", 0.0)) * 0.10,
            ),
        )

    def _movement_motivation(self, individual: Individual, origin: Place, destination: Place, planning: float) -> dict[str, float]:
        origin_energy = min(1.0, origin.total_accessible_energy() / 420.0)
        destination_energy = min(1.0, destination.total_accessible_energy() / 420.0)
        origin_memory = individual.place_memory.get(origin.id, origin_energy)
        destination_memory = individual.place_memory.get(destination.id, destination_energy)
        origin_crowd = len(self._living_ids_at(origin.id)) / max(1.0, origin.capacity)
        destination_crowd = len(self._living_ids_at(destination.id)) / max(1.0, destination.capacity)
        origin_hazard = self._place_hazard_pressure(origin)
        destination_hazard = self._place_hazard_pressure(destination)
        locked_pull = max(0.0, destination.sealed_essence - origin.sealed_essence) / 520.0
        material_pull = max(0.0, sum(destination.materials.values()) - sum(origin.materials.values())) / 120.0
        motivation = {
            "energy_gradient": max(0.0, destination_energy - origin_energy),
            "remembered_value": max(0.0, destination_memory - origin_memory * 0.72),
            "crowding_escape": max(0.0, origin_crowd - destination_crowd),
            "hazard_escape": max(0.0, origin_hazard - destination_hazard),
            "locked_resource_pull": min(1.0, locked_pull + material_pull * 0.20),
            "novelty": 0.16 if destination.id not in individual.place_memory else 0.0,
            "drift": max(0.0, 0.10 - planning * 0.05),
        }
        total = sum(motivation.values())
        if total <= 0.0001:
            motivation["drift"] = 0.10
        return motivation

    def _relocation_shock(self, individual: Individual, origin: Place, destination: Place, planning: float) -> float:
        origin_physics = origin.physics
        destination_physics = destination.physics
        physical_delta = (
            abs(destination_physics.get("temperature", 0.5) - origin_physics.get("temperature", 0.5)) * 0.30
            + abs(destination_physics.get("fluid_level", 0.0) - origin_physics.get("fluid_level", 0.0)) * 0.30
            + abs(destination_physics.get("pressure", 0.0) - origin_physics.get("pressure", 0.0)) * 0.26
            + abs(destination_physics.get("humidity", 0.5) - origin_physics.get("humidity", 0.5)) * 0.16
            + abs(destination_physics.get("salinity", 0.0) - origin_physics.get("salinity", 0.0)) * 0.18
            + abs(destination_physics.get("elevation", 0.5) - origin_physics.get("elevation", 0.5)) * 0.18
            + abs(destination_physics.get("oxygen", 0.35) - origin_physics.get("oxygen", 0.35)) * 0.14
        )
        destination_mismatch = (
            max(0.0, destination_physics.get("fluid_level", 0.0) - individual.params.aquatic_affinity * 0.88) * 0.28
            + max(0.0, individual.params.aquatic_affinity * (1.0 - destination_physics.get("humidity", 0.5)) - individual.params.desiccation_tolerance * 0.58) * 0.24
            + max(0.0, destination_physics.get("pressure", 0.0) - individual.params.pressure_tolerance * 1.05) * 0.22
            + max(0.0, abs(destination_physics.get("salinity", 0.0) - individual.params.salinity_tolerance) - 0.55) * 0.14
            + max(0.0, destination_physics.get("temperature", 0.5) - (0.60 + individual.params.thermal_tolerance * 0.42)) * 0.18
            + max(0.0, 0.18 - destination_physics.get("temperature", 0.5) - individual.params.thermal_tolerance * 0.12) * 0.10
        )
        hazard_delta = max(0.0, self._place_hazard_pressure(destination) - self._place_hazard_pressure(origin) * 0.55)
        raw = physical_delta + destination_mismatch + hazard_delta
        protection = artifact_capability(individual.artifacts, "protect")
        traverse = artifact_capability(individual.artifacts, "traverse")
        insulate = artifact_capability(individual.artifacts, "insulate")
        encase = artifact_capability(individual.artifacts, "encase")
        memory = min(1.0, individual.place_memory.get(destination.id, 0.0) * 2.0)
        skill = max(
            individual.tool_skill.get("traverse", 0.0) * 0.20,
            individual.tool_skill.get("protect", 0.0) * 0.18,
            individual.tool_skill.get("encase", 0.0) * 0.12,
        )
        mitigation = min(
            0.72,
            planning * 0.14
            + memory * 0.16
            + protection * 0.20
            + traverse * 0.12
            + insulate * 0.10
            + encase * 0.08
            + skill,
        )
        return max(0.0, raw * (1.0 - mitigation))

    def _record_movement(
        self,
        individual: Individual,
        origin: Place,
        destination: Place,
        *,
        success: bool,
        before_energy: float,
        before_health: float,
        barrier: float,
        effective_barrier: float,
        distance: float,
        support: float,
        relocation_shock: float,
        planning: float,
        motivation: dict[str, float],
        solo_success: float,
        success_score: float,
    ) -> None:
        self.movement_events["attempt"] += 1
        self.movement_events["success" if success else "failure"] += 1
        if support > 0.04:
            self.movement_events["assisted_success" if success else "assisted_failure"] += 1
        dominant = max(motivation, key=motivation.get) if motivation else "unknown"
        self.movement_motives[dominant] += 1
        for key, value in motivation.items():
            self.movement_costs[f"motivation_{key}"] += max(0.0, float(value))
        self.movement_costs["energy"] += max(0.0, before_energy - individual.energy)
        self.movement_costs["health"] += max(0.0, before_health - individual.health)
        self.movement_costs["barrier"] += max(0.0, barrier)
        self.movement_costs["effective_barrier"] += max(0.0, effective_barrier)
        self.movement_costs["distance"] += max(0.0, distance)
        self.movement_costs["support"] += max(0.0, support)
        self.movement_costs["relocation_shock"] += max(0.0, relocation_shock)
        self.movement_costs["planning"] += max(0.0, planning)
        route = f"{origin.id}->{destination.id}"
        self.movement_routes[route] += 1
        if relocation_shock > 0.08:
            self.movement_events["major_relocation"] += 1
        notable = support > 0.06 or barrier > 0.14 or relocation_shock > 0.08 or before_health - individual.health > 0.004 or (success and destination.id not in individual.place_memory)
        if notable:
            self.observer.observe(
                self.tick,
                "movement_attempt",
                {
                    "organism_id": individual.id,
                    "origin": origin.id,
                    "destination": destination.id,
                    "success": success,
                    "dominant_motive": dominant,
                    "barrier": barrier,
                    "effective_barrier": effective_barrier,
                    "distance": distance,
                    "support": support,
                    "relocation_shock": relocation_shock,
                    "energy_cost": max(0.0, before_energy - individual.energy),
                    "health_cost": max(0.0, before_health - individual.health),
                    "solo_success": solo_success,
                    "success_score": success_score,
                    "motivation": {key: round(value, 5) for key, value in motivation.items()},
                },
                subjects=self._subjects(individual, origin.id, [f"place:{destination.id}", f"movement_motive:{dominant}"]),
                score=min(3.0, barrier * 2.0 + relocation_shock * 2.2 + support * 2.4 + max(0.0, before_health - individual.health) * 40.0 + max(motivation.values(), default=0.0)),
                rarity_key=f"movement:{dominant}",
            )

    def _movement_summary(self) -> dict[str, Any]:
        attempts = max(1, int(self.movement_events.get("attempt", 0)))
        motive_signal = {
            key.replace("motivation_", ""): round(value / attempts, 6)
            for key, value in sorted(self.movement_costs.items())
            if key.startswith("motivation_")
        }
        return {
            "events": dict(self.movement_events),
            "avg_energy_cost": round(self.movement_costs.get("energy", 0.0) / attempts, 6),
            "avg_health_cost": round(self.movement_costs.get("health", 0.0) / attempts, 6),
            "avg_barrier": round(self.movement_costs.get("barrier", 0.0) / attempts, 6),
            "avg_effective_barrier": round(self.movement_costs.get("effective_barrier", 0.0) / attempts, 6),
            "avg_distance": round(self.movement_costs.get("distance", 0.0) / attempts, 6),
            "avg_support": round(self.movement_costs.get("support", 0.0) / attempts, 6),
            "avg_relocation_shock": round(self.movement_costs.get("relocation_shock", 0.0) / attempts, 6),
            "avg_planning": round(self.movement_costs.get("planning", 0.0) / attempts, 6),
            "motives": dict(self.movement_motives),
            "motive_signal": motive_signal,
            "top_routes": dict(self.movement_routes.most_common(12)),
        }

    def _salient_problem(self, individual: Individual, place: Place, target_affordance: str = "") -> dict[str, Any]:
        target_affordance = target_affordance if target_affordance in AFFORDANCES else ""
        challenge = place.causal_challenge
        if challenge is not None and challenge.payoff_remaining > 0.0:
            expected = challenge.expected_affordance()
            return {
                "kind": "causal_challenge",
                "place": place.id,
                "required_affordance": expected or target_affordance,
                "sequence": list(challenge.sequence),
                "payoff_energy": challenge.payoff_energy,
                "remaining": challenge.payoff_remaining,
                "difficulty": challenge.difficulty,
                "progress": challenge.progress,
            }

        obstacle_options = [
            (place.obstacles.get("height", 0.0), "height", "hoist", "traverse"),
            (place.obstacles.get("water", 0.0) + place.physics.get("current_exposure", 0.0) * 0.35, "water", "encase", "float"),
            (place.obstacles.get("thorn", 0.0), "thorn", "shear", "shear"),
            (place.obstacles.get("heat", 0.0), "heat", "kindle", "insulate"),
            (place.physics.get("salinity", 0.0), "salinity", "winnow", "winnow"),
            (place.physics.get("pressure", 0.0), "pressure", "encase", "anchor"),
        ]
        exposure = self._place_exposure_pressure(place)
        obstacle_options.append(
            (
                float(exposure["severity"]),
                str(exposure["primary"]),
                str(exposure["required_affordance"]),
                str(exposure["required_capability"]),
            )
        )
        severity, obstacle, required_affordance, required_capability = max(
            obstacle_options,
            key=lambda item: item[0],
        )
        if target_affordance:
            required_affordance = target_affordance
        if severity > 0.16:
            return {
                "kind": "obstacle",
                "place": place.id,
                "obstacle": obstacle,
                "severity": severity,
                "required_affordance": required_affordance,
                "required_capability": required_capability,
            }

        resource_options = [
            (place.sealed_essence / 180.0, "sealed_essence", "cleave", "essence"),
            (place.resources.get("mechanical", 0.0) / 180.0 + place.physics.get("current_exposure", 0.0) * 0.18, "mechanical_gradient", "encase", "mechanical"),
            (place.resources.get("electrical", 0.0) / 140.0 + place.resources.get("dense_node", 0.0) / 80.0, "electrical_source", "ferry", "electrical"),
            (place.resources.get("solar", 0.0) / 220.0 + place.resources.get("thermal", 0.0) / 240.0, "heat_source", "kindle", "thermal"),
            (place.resources.get("essence", 0.0) / 220.0 + place.resources.get("organic_store", 0.0) / 220.0, "surface_food", "winnow", "essence"),
        ]
        value, resource, required_affordance, energy = max(resource_options, key=lambda item: item[0])
        if target_affordance:
            required_affordance = target_affordance
        return {
            "kind": "resource",
            "place": place.id,
            "resource": resource,
            "value": value,
            "required_affordance": required_affordance,
            "payoff_energy": energy,
        }

    def _record_tool_lesson(
        self,
        individual: Individual,
        place: Place,
        *,
        kind: str,
        affordance: str,
        success: bool,
        gain: float = 0.0,
        score: float = 0.0,
        method_quality: float = 0.0,
        components: dict[str, int] | None = None,
        sequence: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if affordance not in individual.tool_skill and affordance not in {"build", "craft"}:
            return
        lesson: dict[str, Any] = {
            "kind": kind,
            "place": place.id,
            "affordance": affordance,
            "success": bool(success),
            "gain": gain,
            "score": score,
            "method_quality": method_quality,
            "skill": individual.tool_skill.get(affordance, 0.0),
            "problem": self._salient_problem(individual, place),
            "attempted_affordance": affordance,
        }
        if components:
            lesson["components"] = {name: int(qty) for name, qty in components.items() if qty > 0}
        if sequence:
            lesson["sequence"] = list(sequence)
        individual.record_lesson(lesson)

    def _affordance_score(self, individual: Individual, affordance: str) -> float:
        if affordance not in AFFORDANCES:
            return 0.0
        potentials = derive_affordances(individual.inventory)
        potential = max(potentials.get(affordance, 0.0), artifact_capability(individual.artifacts, affordance))
        return potential * (0.65 + 0.35 * individual.tool_skill.get(affordance, 0.0))

    def _problem_affordance_fit(self, problem: dict[str, Any], affordance: str) -> float:
        if affordance not in AFFORDANCES:
            return 0.0
        required = str(problem.get("required_affordance") or "")
        fit = 0.0
        if affordance == required:
            fit += 0.46

        kind = str(problem.get("kind") or "")
        if kind == "causal_challenge":
            sequence = [str(item) for item in problem.get("sequence", []) if str(item) in AFFORDANCES]
            if affordance == required:
                progress = float(problem.get("progress", 0.0) or 0.0)
                fit += 0.20 + min(0.18, progress * 0.05)
            elif affordance in sequence:
                fit += 0.16
        elif kind == "obstacle":
            obstacle = str(problem.get("obstacle") or "")
            alternatives = {
                "height": {"hoist": 0.24, "cleave": 0.08, "lash": 0.06},
                "water": {"encase": 0.25, "winnow": 0.16, "lash": 0.06},
                "thorn": {"shear": 0.26, "lash": 0.08},
                "heat": {"kindle": 0.20, "encase": 0.07, "winnow": 0.05},
                "salinity": {"winnow": 0.28, "encase": 0.10},
                "pressure": {"encase": 0.20, "lash": 0.07, "hoist": 0.06},
            }
            fit += alternatives.get(obstacle, {}).get(affordance, 0.0)
            fit *= 0.78 + min(0.40, float(problem.get("severity", 0.0) or 0.0))
        elif kind == "resource":
            resource = str(problem.get("resource") or "")
            alternatives = {
                "sealed_essence": {"cleave": 0.26, "hoist": 0.17, "ferry": 0.04},
                "mechanical_gradient": {"encase": 0.20, "hoist": 0.12, "lash": 0.08},
                "electrical_source": {"ferry": 0.30, "encase": 0.05},
                "heat_source": {"kindle": 0.24, "ferry": 0.08},
                "surface_food": {"winnow": 0.24, "shear": 0.10, "encase": 0.06},
            }
            fit += alternatives.get(resource, {}).get(affordance, 0.0)
            fit *= 0.78 + min(0.36, float(problem.get("value", 0.0) or 0.0))
        return max(0.0, min(1.0, fit))

    def _problem_match(self, current: dict[str, Any], remembered: dict[str, Any]) -> float:
        if not current or not remembered:
            return 0.0
        match = 0.0
        if current.get("kind") == remembered.get("kind"):
            match += 0.24
        if current.get("required_affordance") and current.get("required_affordance") == remembered.get("required_affordance"):
            match += 0.24
        for key in ("obstacle", "resource", "payoff_energy"):
            if current.get(key) and current.get(key) == remembered.get(key):
                match += 0.18
        current_sequence = {str(item) for item in current.get("sequence", []) if str(item) in AFFORDANCES}
        remembered_sequence = {str(item) for item in remembered.get("sequence", []) if str(item) in AFFORDANCES}
        if current_sequence and remembered_sequence:
            overlap = len(current_sequence & remembered_sequence) / max(1, len(current_sequence | remembered_sequence))
            match += overlap * 0.16
        for key in ("severity", "value", "difficulty"):
            if key in current and key in remembered:
                distance = abs(float(current.get(key, 0.0) or 0.0) - float(remembered.get(key, 0.0) or 0.0))
                match += max(0.0, 0.08 - distance * 0.05)
        return max(0.0, min(1.0, match))

    def _memory_affordance_bias(self, individual: Individual, problem: dict[str, Any], affordance: str) -> tuple[float, float]:
        if not individual.lesson_memory:
            return 0.0, 0.0
        total = 0.0
        strongest_match = 0.0
        for lesson in individual.lesson_memory[-5:]:
            if str(lesson.get("affordance") or lesson.get("attempted_affordance") or "") != affordance:
                continue
            remembered = lesson.get("problem", {}) if isinstance(lesson.get("problem"), dict) else {}
            match = self._problem_match(problem, remembered)
            if match <= 0.0:
                continue
            strongest_match = max(strongest_match, match)
            gain = max(0.0, float(lesson.get("gain", 0.0) or 0.0))
            score = max(0.0, float(lesson.get("score", 0.0) or 0.0))
            method = max(0.0, float(lesson.get("method_quality", 0.0) or 0.0))
            value = (0.22 if lesson.get("success") else -0.12) + min(0.24, gain / 18.0) + score * 0.12 + method * 0.08
            total += match * value
        return max(-0.24, min(0.42, total)), strongest_match

    def _situation_affordance_choice(self, individual: Individual, place: Place, base_affordance: str, base_score: float) -> tuple[str, float, bool, dict[str, Any]]:
        problem = self._salient_problem(individual, place)
        planning = self._interaction_control(individual)
        capability_floor = 0.055
        best_affordance_name = base_affordance
        best_score = base_score
        best_value = base_score
        best_fit = self._problem_affordance_fit(problem, base_affordance)
        best_memory, best_recognition = self._memory_affordance_bias(individual, problem, base_affordance)
        best_reason = "capability"
        candidates = 0
        for affordance in AFFORDANCES:
            raw_score = self._affordance_score(individual, affordance)
            if raw_score < capability_floor and affordance != base_affordance:
                continue
            candidates += 1
            situation_fit = self._problem_affordance_fit(problem, affordance)
            memory_bias, recognition = self._memory_affordance_bias(individual, problem, affordance)
            skill = max(0.0, individual.tool_skill.get(affordance, 0.0))
            continuity = 0.030 if affordance == individual.last_tool_affordance else 0.0
            problem_weight = 0.27 + planning * 0.22
            memory_weight = 0.23 + planning * 0.20
            uncertainty = self.rng.random() * max(0.006, 0.028 - planning * 0.016)
            value = (
                raw_score
                + situation_fit * problem_weight * max(0.50, raw_score)
                + memory_bias * memory_weight * max(0.50, raw_score)
                + recognition * skill * 0.035
                + continuity
                + uncertainty
            )
            if value > best_value:
                best_affordance_name = affordance
                best_score = raw_score
                best_value = value
                best_fit = situation_fit
                best_memory = memory_bias
                best_recognition = recognition
                best_reason = "memory" if memory_bias > situation_fit * 0.45 else "situation" if situation_fit > 0.0 else "capability"

        context = {
            "problem": problem,
            "base_affordance": base_affordance,
            "base_score": round(base_score, 6),
            "choice_value": round(best_value, 6),
            "situation_fit": round(best_fit, 6),
            "memory_bias": round(best_memory, 6),
            "recognition": round(best_recognition, 6),
            "reason": best_reason,
            "candidates": candidates,
        }
        situation_directed = best_affordance_name != base_affordance or best_fit > 0.28 or best_memory > 0.08
        return best_affordance_name, best_score, situation_directed, context

    def _environmental_affordance_choice(self, individual: Individual, place: Place, base_affordance: str, base_score: float) -> tuple[str, float, bool]:
        affordance, score, directed, _context = self._situation_affordance_choice(individual, place, base_affordance, base_score)
        return affordance, score, directed

    def _apply_physics_transport(self) -> None:
        for individual in list(self.organisms.values()):
            if not individual.alive:
                continue
            place = self.world.places[individual.location]
            physics = place.physics
            fluid = physics.get("fluid_level", 0.0)
            current = physics.get("current_exposure", 0.0)
            downstream = self.world.downstream_neighbor(place.id)
            if downstream and fluid > 0.35 and current > 0.08:
                float_cap = artifact_capability(individual.artifacts, "float")
                anchor = artifact_capability(individual.artifacts, "anchor")
                traverse = artifact_capability(individual.artifacts, "traverse")
                structure_anchor = structure_capability(place.structures, "anchor")
                structure_support = structure_capability(place.structures, "support")
                structure_shelter = structure_capability(place.structures, "shelter")
                resistance = max(
                    individual.params.aquatic_affinity * 0.65 + individual.params.buoyancy * 0.35,
                    individual.params.mobility * 0.30,
                    float_cap * 0.75,
                    anchor * 0.80,
                    traverse * 0.55,
                    structure_anchor * 0.45,
                    structure_support * 0.30,
                    structure_shelter * 0.55,
                )
                drift_chance = max(0.0, downstream[1] * fluid * (1.0 - resistance)) * 0.020
                if self.rng.random() < drift_chance:
                    individual.location = downstream[0]
                    individual.energy -= 0.010 + current * 0.012
                    if individual.params.aquatic_affinity < fluid * 0.45 and float_cap < 0.20:
                        individual.health -= fluid * 0.006
                    self.physics_events["current_transport"] += 1
                    if individual.health <= 0.0:
                        self._deactivate(individual, "current_washout")
                        continue

            steep_edges = [
                edge
                for edge in self.world.edges_from(place.id)
                if edge.slope_from(place.id) < -0.35 and edge.danger > 0.10
            ]
            if steep_edges and individual.params.mobility < 0.65:
                edge = min(steep_edges, key=lambda item: item.slope_from(place.id))
                traverse = artifact_capability(individual.artifacts, "traverse")
                anchor = artifact_capability(individual.artifacts, "anchor")
                structure_support = structure_capability(place.structures, "support")
                structure_anchor = structure_capability(place.structures, "anchor")
                footing = max(individual.params.mobility, traverse * 0.65, anchor * 0.75, structure_support * 0.35, structure_anchor * 0.40)
                fall_chance = max(0.0, abs(edge.slope_from(place.id)) * edge.danger * (1.0 - footing)) * 0.004
                if self.rng.random() < fall_chance:
                    individual.location = edge.other(place.id)
                    damage = max(0.0, abs(edge.slope_from(place.id)) - footing) * 0.035
                    individual.health -= damage
                    individual.energy -= 0.025
                    self.physics_events["gravity_fall"] += 1
                    if individual.health <= 0.0:
                        self._deactivate(individual, "fall")

    def _apply_upkeep(self, individual: Individual) -> None:
        individual.age += 1
        self._age_portable_inscriptions(individual)
        hardship = max(0.0, self._environment_harshness() - 1.0)
        hardship_multiplier = 1.0 + hardship * (0.16 if individual.kind == "agent" else 0.08)
        individual.energy -= individual.upkeep_cost() * hardship_multiplier
        if individual.energy < 0.0:
            individual.health += individual.energy * 0.030
            individual.energy = 0.0
        if individual.health <= 0.0:
            self._deactivate(individual, "starvation")

    def _age_portable_inscriptions(self, individual: Individual) -> None:
        for artifact in individual.artifacts:
            if not artifact.inscriptions:
                continue
            record_cap = artifact.capabilities.get("record", 0.0) * max(0.0, min(1.0, artifact.durability / 100.0))
            kept: list[dict[str, Any]] = []
            for inscription in artifact.inscriptions:
                inscription["age"] = int(inscription.get("age", 0)) + 1
                inscription["intensity"] = float(inscription.get("intensity", 0.0)) * (0.998 + record_cap * 0.001)
                inscription["durability"] = float(inscription.get("durability", 0.0)) - (0.018 + max(0.0, 1.0 - record_cap) * 0.022)
                if float(inscription.get("durability", 0.0)) > 0.0 and float(inscription.get("intensity", 0.0)) > 0.020:
                    kept.append(inscription)
            artifact.inscriptions = kept[-8:]

    def _habitat_stress(self, individual: Individual) -> None:
        place = self.world.places[individual.location]
        aquatic = place.habitat.get("aquatic", 0.0)
        depth = place.habitat.get("depth", 0.0)
        physics = place.physics
        salinity = physics.get("salinity", place.habitat.get("salinity", 0.0))
        humidity = physics.get("humidity", place.habitat.get("humidity", 0.5))
        temperature = physics.get("temperature", 0.5)
        pressure = physics.get("pressure", depth)
        current = physics.get("current_exposure", 0.0)
        structure_shelter = structure_capability(place.structures, "shelter")
        shelter = max(physics.get("shelter", 0.0), structure_shelter)
        interiority = max(physics.get("interiority", 0.0), structure_capability(place.structures, "enclose"))
        permeability = max(physics.get("boundary_permeability", 0.0), structure_capability(place.structures, "permeable"))
        artifact_insulation = artifact_capability(individual.artifacts, "insulate")
        insulation = max(artifact_insulation, shelter * 0.55)
        protection = artifact_capability(individual.artifacts, "protect")
        float_cap = artifact_capability(individual.artifacts, "float")
        anchor = max(artifact_capability(individual.artifacts, "anchor"), structure_capability(place.structures, "anchor") * 0.35)
        traverse = max(artifact_capability(individual.artifacts, "traverse"), structure_capability(place.structures, "support") * 0.25)
        heat_control = max(
            artifact_capability(individual.artifacts, "kindle"),
            structure_capability(place.structures, "kindle") * 0.45,
            structure_capability(place.structures, "gradient_harvest") * 0.24,
            structure_capability(place.structures, "energy_storage") * 0.20,
        )
        containment = max(
            artifact_capability(individual.artifacts, "encase"),
            structure_capability(place.structures, "enclose") * 0.42,
            structure_capability(place.structures, "channel") * 0.22,
        )
        hardship = max(0.0, self._environment_harshness() - 1.0)
        exposure = self._place_exposure_pressure(place)
        exposure_severity = float(exposure["severity"]) * (1.0 + hardship * 0.50)
        exposure_collaboration = self._collective_support(individual, "protect", exposure_severity) if individual.kind == "agent" and exposure_severity > 0.12 else {"support": 0.0, "helpers": [], "raw": 0.0}
        exposure_support = float(exposure_collaboration.get("support", 0.0))
        exposure_damping = max(0.45, 1.0 - shelter * 0.35 - protection * 0.16)
        drowning = max(0.0, aquatic * depth - individual.params.aquatic_affinity * 0.85 - individual.params.mobility * 0.15 - float_cap * 0.20 - shelter * 0.08 - protection * 0.05)
        desiccation = max(0.0, individual.params.aquatic_affinity * (1.0 - humidity) - individual.params.desiccation_tolerance * 0.55 - shelter * 0.14 - protection * 0.08)
        salinity_stress = max(0.0, abs(salinity - individual.params.salinity_tolerance) - 0.55 - protection * 0.06)
        heat_stress = max(0.0, temperature - (0.58 + individual.params.thermal_tolerance * 0.42 + insulation * 0.25 + protection * 0.06))
        cold_stress = max(0.0, 0.24 - temperature - individual.params.thermal_tolerance * 0.14 - insulation * 0.20 - heat_control * 0.12 - protection * 0.05)
        pressure_stress = max(0.0, pressure - (individual.params.pressure_tolerance * 1.05 + individual.params.aquatic_affinity * 0.20 + individual.params.armor * 0.12 + protection * 0.14))
        current_stress = max(0.0, current * aquatic - max(individual.params.buoyancy, float_cap, anchor * 0.80, traverse * 0.55, individual.params.mobility * 0.25))
        stagnant_interior = max(0.0, interiority - shelter) * max(0.0, 1.0 - permeability) * max(0.0, pressure + temperature - 0.80)
        tool_exposure_buffer = artifact_insulation * 0.24 + heat_control * 0.22 + protection * 0.13 + containment * 0.09 + structure_shelter * 0.12
        tool_synergy = min(1.0, tool_exposure_buffer * 2.5)
        social_exposure_buffer = exposure_support * (0.14 + tool_synergy * 0.08)
        exposure_buffer = (
            insulation * 0.24
            + heat_control * 0.22
            + protection * 0.13
            + shelter * 0.16
            + containment * 0.09
            + social_exposure_buffer
            + individual.params.thermal_tolerance * 0.10
            + individual.params.armor * 0.05
        )
        exposure_stress = max(0.0, exposure_severity - exposure_buffer)
        buffered_exposure = max(0.0, exposure_severity - exposure_stress)
        agent_pressure = 1.0 if individual.kind == "agent" else 0.35
        stress = (
            drowning * 0.020 * exposure_damping
            + desiccation * 0.015 * exposure_damping
            + salinity_stress * 0.010
            + heat_stress * 0.018
            + cold_stress * 0.012
            + pressure_stress * (0.012 + hardship * 0.003)
            + current_stress * 0.009 * exposure_damping
            + stagnant_interior * 0.006
            + exposure_stress * (0.014 + hardship * 0.010) * agent_pressure
        )
        if exposure_severity > 0.12:
            self.physics_events["exposure_pressure"] += 1
            self.physics_events[f"exposure_{exposure['primary']}"] += 1
        if hardship > 0.0 and exposure_stress > 0.05:
            self.physics_events["harsh_exposure_pressure"] += 1
        if exposure_support > 0.01:
            self.physics_events["collaborative_exposure_buffer"] += 1
            self._apply_collaboration_effects(individual, "protect", "exposure", exposure_collaboration, exposure_severity)
        if protection > 0.0:
            self._wear_artifacts(individual, "protect", amount=stress * (0.35 + protection * 0.20))
            self._increase_skill(individual, "protect", stress * 0.018, transfer=0.06)
        if insulation > 0.0 and exposure_severity > 0.08:
            self._wear_artifacts(individual, "insulate", amount=exposure_severity * (0.08 + insulation * 0.10))
            self._increase_skill(individual, "insulate", exposure_severity * 0.006, transfer=0.05)
        if heat_control > 0.0 and exposure_severity > 0.08:
            self._wear_artifacts(individual, "kindle", amount=exposure_severity * (0.05 + heat_control * 0.08))
            self._increase_skill(individual, "kindle", exposure_severity * 0.005, transfer=0.05)
        if tool_exposure_buffer > 0.05 and buffered_exposure > 0.05:
            self.physics_events["tool_buffered_exposure"] += 1
            individual.record_success("tool_use", min(0.12, min(buffered_exposure, tool_exposure_buffer) * 0.03))
        if stress <= 0.0:
            return
        individual.energy -= stress * (2.0 + hardship * 0.70)
        individual.health -= stress
        if individual.health <= 0.0:
            if exposure_stress > max(drowning, desiccation, salinity_stress, heat_stress, cold_stress, pressure_stress, current_stress):
                self._deactivate(individual, "exposure_stress")
            elif pressure_stress > max(drowning, desiccation, salinity_stress, heat_stress, cold_stress, exposure_stress):
                self._deactivate(individual, "pressure_stress")
            elif heat_stress > max(drowning, desiccation, salinity_stress, pressure_stress, cold_stress, exposure_stress):
                self._deactivate(individual, "thermal_stress")
            elif current_stress > max(drowning, desiccation, salinity_stress, pressure_stress, heat_stress, exposure_stress):
                self._deactivate(individual, "current_exposure")
            else:
                self._deactivate(individual, "habitat_mismatch")

    def _observe(self, individual: Individual, rosters: dict[int, list[int]]) -> list[float]:
        place = self.world.places[individual.location]
        resources = [place.resources[kind] / 120.0 for kind in ("solar", "essence", "organic_store", "thermal", "mechanical", "electrical", "dense_node")]
        local_ids = rosters.get(place.id, [])
        local_neural = sum(1 for oid in local_ids if self.organisms[oid].neural)
        best_skill = self._skill_breadth(individual)
        season = math.sin(2.0 * math.pi * self.world.tick / max(2, self.world.season_length))
        features = [
            individual.energy / max(1.0, individual.storage_limit()),
            individual.health,
            min(1.0, individual.age / 2_000.0),
            *resources,
            place.sealed_essence / 160.0,
            len(local_ids) / max(1.0, place.capacity),
            local_neural / max(1.0, len(local_ids)),
            individual.inventory_count() / max(1.0, individual.inventory_limit()),
            individual.params.mobility,
            individual.params.manipulator,
            individual.params.armor,
            individual.params.sensor_range,
            individual.params.neural_budget / NEURAL_BUDGET_MAX,
            individual.params.memory_budget / MEMORY_BUDGET_MAX,
            individual.params.prediction_weight,
            individual.params.plasticity_rate,
            max(-1.0, min(1.0, individual.last_valence)),
            best_skill,
            season,
            self.world.climate_drift,
            place.physics.get("temperature", 0.5),
            place.physics.get("pressure", 0.0),
            place.physics.get("current_exposure", 0.0),
            place.physics.get("interiority", 0.0),
            place.physics.get("boundary_permeability", 0.0),
            place.physics.get("shelter", 0.0),
            place.physics.get("oxygen", 0.35),
            place.physics.get("acidity", 0.10),
            place.physics.get("organic_activity", 0.0),
            place.physics.get("abrasion", 0.0),
            place.physics.get("wet_dry_cycle", 0.0),
            place.physics.get("elevation", 0.5),
            place.habitat.get("aquatic", 0.0),
            place.habitat.get("depth", 0.0),
            place.habitat.get("salinity", 0.0),
            place.habitat.get("humidity", 0.5),
            *individual.recent_trace(),
            *individual.prediction_error_profile,
            *individual.event_memory,
            *individual.signal_values,
        ]
        if len(features) != OBSERVATION_SIZE:
            raise AssertionError(f"observation size drifted to {len(features)}")
        return [max(-1.0, min(1.5, float(value))) for value in features]

    def _observed_tokens(self, individual: Individual) -> list[int]:
        place = self.world.places[individual.location]
        tokens: list[int] = []
        for signal in place.signals:
            if signal.source_id != individual.id:
                tokens.append(signal.token)
        for mark in place.marks[-8:]:
            if mark.source_id != individual.id:
                tokens.append(mark.token)
        return tokens[:4]

    def _choose_action(self, individual: Individual, observation: list[float]) -> str:
        if individual.controller is None:
            non_neural_birth_rate = 0.025 if individual.kind == "plant" else 0.035
            if individual.energy > self.optimization.clone_mutate_reserve_threshold(individual) and self.rng.random() < non_neural_birth_rate:
                return "clone_mutate"
            if individual.kind == "plant":
                return "absorb_solar"
            if individual.kind == "fungus":
                return "eat" if self.rng.random() < 0.72 else "forage"
            return "rest"

        outputs = self.brain_runtime.forward_many([individual.controller], [observation])[0]
        return self._choose_action_from_outputs(individual, outputs)

    def _choose_action_from_outputs(self, individual: Individual, outputs: list[float]) -> str:
        exploration = 0.025 + individual.params.plasticity_rate * 0.055 + individual.params.perturbation_rate * 0.25
        if self.rng.random() < exploration:
            return self.rng.choice(ACTIONS)
        energy_ratio = individual.energy / max(1.0, individual.storage_limit())
        if individual.adult() and energy_ratio > 0.62:
            reproductive_drive = individual.params.valence_reproduction * (energy_ratio - 0.62)
            outputs[ACTION_INDEX["coordinate"]] += reproductive_drive * (0.9 + individual.params.pairing_selectivity)
            outputs[ACTION_INDEX["clone_mutate"]] += reproductive_drive * (0.7 + (1.0 - individual.params.pairing_selectivity) * 0.4)
        ranked = sorted(range(len(outputs)), key=lambda i: outputs[i], reverse=True)
        for index in ranked:
            action = ACTIONS[index]
            if self._action_feasible(individual, action):
                return action
        return "rest"

    def _action_feasible(self, individual: Individual, action: str) -> bool:
        if action in {"coordinate", "clone_mutate"} and not individual.adult():
            return False
        if action == "pickup" and individual.inventory_count() >= individual.inventory_limit():
            return False
        if action == "craft" and (individual.inventory_count() < 2 or len(individual.artifacts) >= individual.artifact_limit()):
            return False
        if action == "build" and individual.params.manipulator < 0.18:
            return False
        if action == "build" and individual.inventory_count() < 3 and self._collective_material_count(individual) < 3:
            return False
        if action == "move" and individual.params.mobility < 0.05:
            return False
        if action == "use_tool" and individual.inventory_count() == 0 and not individual.artifacts:
            return False
        if action == "mark" and individual.params.manipulator < 0.12:
            return False
        return True

    def _resolve_action(
        self,
        individual: Individual,
        action: str,
        feedback: dict[str, float],
    ) -> None:
        individual.last_action = action
        if action == "rest":
            individual.energy -= 0.004
            # Replay-during-rest: if the controller has episodic memory, sample two
            # stored episodes, average them, and push the result through the
            # recurrent core. This is the offline replay association /
            # consolidation mechanism — the substrate gives the controller a way
            # to associate distant experiences. Whether the controller develops a
            # useful replay strategy is up to ranking; the substrate just
            # provides the channel.
            if individual.controller is not None and individual.controller._has_episodic():
                individual.controller.replay_episode(self.rng)
            return
        if action == "move":
            self._move(individual, feedback)
        elif action == "eat":
            self._eat(individual)
        elif action == "absorb_solar":
            self._absorb_solar(individual)
        elif action == "forage":
            self._forage(individual)
        elif action == "pickup":
            self._pickup(individual)
        elif action == "craft":
            self._craft(individual, feedback)
        elif action == "build":
            self._build_structure(individual, feedback)
        elif action == "use_tool":
            self._use_tool(individual, feedback)
        elif action == "attack":
            self._attack(individual)
        elif action == "signal":
            self._signal(individual, feedback)
        elif action == "mark":
            self._mark(individual, feedback)
        elif action == "clone_mutate":
            self._clone_mutate(individual, feedback)
        elif action == "observe":
            self._observe_others(individual, feedback)

    def _move(self, individual: Individual, feedback: dict[str, float] | None = None) -> None:
        place = self.world.places[individual.location]
        if not place.neighbors:
            return
        before_energy = individual.energy
        before_health = individual.health
        planning = self._interaction_control(individual)
        efficiency = 1.0 - planning * 0.14
        load_ratio = min(
            1.0,
            individual.inventory_count() / max(1.0, individual.inventory_limit())
            + len(individual.artifacts) / max(2.0, individual.artifact_limit() * 2.0),
        )
        individual.energy -= (0.055 + individual.params.mobility * 0.055 + load_ratio * 0.035) * efficiency
        if individual.controller and individual.place_memory:
            scored = []
            for neighbor in place.neighbors:
                memory = individual.place_memory.get(neighbor, 0.0)
                neighbor_place = self.world.places[neighbor]
                crowd = len(self._living_ids_at(neighbor)) / max(1.0, neighbor_place.capacity)
                scored.append((memory - crowd * (0.20 + planning * 0.80) + self.rng.random() * 0.08, neighbor))
            destination_id = max(scored)[1]
        else:
            destination_id = self.rng.choice(place.neighbors)
        destination = self.world.places[destination_id]
        motivation = self._movement_motivation(individual, place, destination, planning)
        edge = self.world.edge_between(place.id, destination_id)
        structure_traverse = max(
            structure_capability(place.structures, "support") * 0.25,
            structure_capability(destination.structures, "support") * 0.18,
            structure_capability(place.structures, "channel") * 0.20,
            structure_capability(destination.structures, "channel") * 0.12,
        )
        traverse = max(artifact_capability(individual.artifacts, "traverse"), structure_traverse)
        insulation = max(artifact_capability(individual.artifacts, "insulate"), structure_capability(destination.structures, "shelter") * 0.35)
        float_cap = artifact_capability(individual.artifacts, "float")
        anchor = max(artifact_capability(individual.artifacts, "anchor"), structure_capability(place.structures, "anchor") * 0.25)
        shear = max(individual.tool_skill.get("shear", 0.0), artifact_capability(individual.artifacts, "shear"))
        aquatic_fit = individual.params.aquatic_affinity
        slope = edge.slope_from(place.id) if edge else 0.0
        current = edge.current_from(place.id) if edge else 0.0
        distance = edge.distance if edge else 1.0
        edge_required = edge.traversal_required if edge else 0.0
        uphill = max(0.0, slope)
        downhill = max(0.0, -slope)
        against_current = max(0.0, -current)
        with_current = max(0.0, current)
        boundary = max(0.0, destination.physics.get("interiority", 0.0) - destination.physics.get("boundary_permeability", 0.0))
        barrier = (
            destination.obstacles.get("water", 0.0) * (1.0 - max(traverse, aquatic_fit))
            + destination.obstacles.get("height", 0.0) * (1.0 - max(traverse, individual.params.mobility))
            + destination.obstacles.get("thorn", 0.0) * (1.0 - max(shear, individual.params.armor))
            + destination.obstacles.get("heat", 0.0) * (1.0 - max(insulation, individual.params.thermal_tolerance))
            + edge_required * (1.0 - max(traverse, float_cap, anchor * 0.65, individual.params.mobility))
            + uphill * (1.0 - max(traverse, individual.params.mobility))
            + against_current * (1.0 - max(traverse, float_cap, aquatic_fit, anchor * 0.55))
            + downhill * (1.0 - max(traverse, anchor, individual.params.mobility)) * 0.35
            + boundary * (1.0 - max(traverse, individual.params.manipulator * 0.35, shear * 0.25))
        ) / 6.10
        solo_success = (
            individual.params.mobility
            + traverse * 0.65
            + individual.params.sensor_range * 0.10
            + planning * 0.10
            + with_current * max(float_cap, aquatic_fit) * 0.10
            + anchor * 0.04
            + self.rng.gauss(0.0, 0.04)
        )
        edge_danger = edge.danger if edge else 0.0
        voyage_difficulty = barrier + edge_required * 0.45 + destination.physics.get("pressure", 0.0) * 0.12 + edge_danger * 0.35 + distance * 0.04
        collaboration = self._collective_support(individual, "traverse", voyage_difficulty)
        support = float(collaboration["support"])
        relocation_shock = self._relocation_shock(individual, place, destination, planning)
        success = solo_success + support * 0.36
        effective_barrier = max(0.0, barrier + relocation_shock * 0.22 + edge_danger * 0.12 - support * 0.12)
        if success >= effective_barrier:
            individual.location = destination_id
            individual.energy -= (
                effective_barrier * 0.125
                + distance * 0.026
                + uphill * 0.040
                + destination.physics.get("pressure", 0.0) * 0.016
                + edge_danger * 0.035
                + relocation_shock * 0.58
                + load_ratio * 0.025
            ) * efficiency
            individual.health -= relocation_shock * max(0.0, 0.030 + edge_danger * 0.012 - support * 0.006)
            if relocation_shock > 0.12:
                self._increase_skill(individual, "traverse", relocation_shock * 0.004, transfer=0.10)
                self._increase_skill(individual, "protect", relocation_shock * 0.002, transfer=0.05)
            if with_current > 0.1 and destination.obstacles.get("water", 0.0) > 0.3:
                self.physics_events["current_assisted_move"] += 1
            if support > 0.04:
                self._apply_collaboration_effects(individual, "traverse", "expedition", collaboration, voyage_difficulty)
                self._move_expedition_helpers(collaboration, from_place=place.id, to_place=destination_id, difficulty=voyage_difficulty)
                if feedback is not None:
                    feedback["social"] += support * 0.28
                    feedback["tool"] = feedback.get("tool", 0.0) + support * 0.10
            self._record_movement(
                individual,
                place,
                destination,
                success=True,
                before_energy=before_energy,
                before_health=before_health,
                barrier=barrier,
                effective_barrier=effective_barrier,
                distance=distance,
                support=support,
                relocation_shock=relocation_shock,
                planning=planning,
                motivation=motivation,
                solo_success=solo_success,
                success_score=success,
            )
            if individual.health <= 0.0:
                self._deactivate(individual, "relocation_shock")
        else:
            individual.energy -= (effective_barrier * 0.340 + distance * 0.032 + edge_danger * 0.060 + relocation_shock * 0.32 + load_ratio * 0.030) * efficiency
            individual.health -= (
                max(0.0, effective_barrier - success) * (0.052 + downhill * 0.018 + edge_danger * 0.020) * (1.0 - planning * 0.14) * max(0.42, 1.0 - support * 0.32)
                + relocation_shock * 0.012
            )
            if support > 0.04:
                self._apply_collaboration_effects(individual, "traverse", "failed_expedition", collaboration, voyage_difficulty * 0.4)
                if feedback is not None:
                    feedback["social"] += support * 0.12
            self._record_movement(
                individual,
                place,
                destination,
                success=False,
                before_energy=before_energy,
                before_health=before_health,
                barrier=barrier,
                effective_barrier=effective_barrier,
                distance=distance,
                support=support,
                relocation_shock=relocation_shock,
                planning=planning,
                motivation=motivation,
                solo_success=solo_success,
                success_score=success,
            )
            if individual.health <= 0.0:
                self._deactivate(individual, "movement_hazard")

    def _move_expedition_helpers(self, collaboration: dict[str, Any], *, from_place: int, to_place: int, difficulty: float) -> None:
        support = float(collaboration.get("support", 0.0))
        if support <= 0.06:
            return
        moved = 0
        for helper_id in list(collaboration.get("helpers", [])):
            if moved >= 3:
                break
            helper = self.organisms.get(helper_id)
            if helper is None or not helper.alive or helper.location != from_place:
                continue
            travel_chance = min(0.62, 0.14 + support * 0.48 + helper.params.mobility * 0.12)
            if helper.recombine_intent_until < self.tick and self.rng.random() > travel_chance:
                continue
            helper.location = to_place
            helper.energy -= 0.020 + difficulty * 0.030
            helper.recent_social_feedback = max(helper.recent_social_feedback, support * 0.35)
            moved += 1
        if moved:
            self.collaboration_events["helpers_moved"] += moved

    def _collective_material_count(self, individual: Individual) -> int:
        total = individual.inventory_count()
        for helper, _contribution in self._active_helper_candidates(individual, "build")[:4]:
            total += helper.inventory_count()
        return total

    def _collective_build_components(self, individual: Individual, components: dict[str, int], target_count: int, collaboration: dict[str, Any]) -> int:
        added = 0
        helper_ids = list(collaboration.get("helpers", []))
        if not helper_ids:
            return added
        while sum(components.values()) < target_count:
            changed = False
            for helper_id in helper_ids:
                helper = self.organisms.get(helper_id)
                if helper is None or not helper.alive or helper.location != individual.location:
                    continue
                choices = [name for name, qty in helper.inventory.items() if qty > 0]
                if not choices:
                    continue
                chosen = max(
                    choices,
                    key=lambda name: MATERIALS[name].properties.get("lashable", 0.0)
                    + MATERIALS[name].properties.get("hard", 0.0) * 0.55
                    + MATERIALS[name].properties.get("length", 0.0) * 0.22
                    + self.rng.random() * 0.03,
                )
                helper.inventory[chosen] -= 1
                if helper.inventory[chosen] <= 0:
                    del helper.inventory[chosen]
                components[chosen] = components.get(chosen, 0) + 1
                helper.energy -= 0.012
                helper.recent_social_feedback = max(helper.recent_social_feedback, 0.12)
                added += 1
                changed = True
                if sum(components.values()) >= target_count:
                    break
            if not changed:
                break
        return added

    def _eat(self, individual: Individual) -> None:
        place = self.world.places[individual.location]
        appetite = 2.0 + individual.params.essence_conversion * 7.0 + individual.params.essence_energy_gain * 3.0
        essence = min(place.resources["essence"], appetite * 0.55)
        place.resources["essence"] -= essence
        organic = min(place.resources["organic_store"], appetite - essence)
        place.resources["organic_store"] -= organic
        gain = essence * individual.params.essence_energy_gain + organic * (0.45 + individual.params.essence_conversion * 0.80)
        individual.energy += gain
        if essence + organic > 0.5 and self.world.note_patch_depletion(place.id, self.rng):
            self.patch_recovery_triggers += 1

    def _absorb_solar(self, individual: Individual) -> None:
        place = self.world.places[individual.location]
        gain = place.resources["solar"] * 0.018 * individual.params.solar_energy_gain * (0.2 + individual.params.solar_capture_area)
        thermal_stress = max(0.0, place.resources["thermal"] / 120.0 - individual.params.thermal_tolerance)
        individual.energy += gain
        individual.health -= thermal_stress * 0.003
        place.resources["organic_store"] += gain * 0.18
        if individual.health <= 0.0:
            self._deactivate(individual, "thermal_stress")

    def _forage(self, individual: Individual) -> None:
        place = self.world.places[individual.location]
        planning = self._interaction_control(individual)
        individual.energy -= (0.025 + individual.params.sensor_range * 0.020) * (1.0 - planning * 0.10)
        if self.rng.random() < 0.18 + individual.params.sensor_range * 0.45 + planning * 0.09:
            found = self.rng.choice(("essence", "organic_store", "mechanical"))
            amount = self.rng.uniform(0.2, 1.6) * (0.5 + individual.params.sensor_range) * (1.0 + planning * 0.25)
            place.resources[found] = min(180.0, place.resources[found] + amount)
        if self.rng.random() < 0.08 + individual.params.sensor_range * 0.12 + planning * 0.04:
            material = self.rng.choice(tuple(MATERIALS.keys()))
            place.materials[material] = min(99, place.materials.get(material, 0) + 1)

    def _pickup(self, individual: Individual) -> None:
        if individual.params.manipulator < 0.08 or individual.inventory_count() >= individual.inventory_limit():
            individual.energy -= 0.025
            return
        place = self.world.places[individual.location]
        available = [name for name, qty in place.materials.items() if qty > 0]
        if not available:
            individual.energy -= 0.015
            return
        material = self.rng.choice(available)
        place.materials[material] -= 1
        individual.inventory[material] = individual.inventory.get(material, 0) + 1
        individual.energy -= 0.035

    def _craft_target_affordance(self, individual: Individual, place: Place, planning: float) -> str:
        challenge = place.causal_challenge
        if challenge is not None:
            expected = challenge.expected_affordance()
            if expected in AFFORDANCES and self.rng.random() < 0.45 + planning * 0.45:
                return expected
        exposure = self._place_exposure_pressure(place)
        exposure_severity = float(exposure["severity"])
        cold_exposure = float(exposure["components"].get("cold", 0.0)) if isinstance(exposure.get("components"), dict) else 0.0
        scores = {
            "cleave": place.sealed_essence / 180.0 + place.mineral_richness * 0.25,
            "shear": place.obstacles.get("thorn", 0.0) * 0.62 + place.resources["organic_store"] / 220.0,
            "lash": individual.tool_skill.get("craft", 0.0) * 0.35 + individual.inventory_count() / max(1.0, individual.inventory_limit()) * 0.16,
            "encase": place.obstacles.get("water", 0.0) * 0.30 + place.physics.get("current_exposure", 0.0) * 0.34 + place.resources["mechanical"] / 240.0,
            "kindle": place.resources["solar"] / 220.0 + place.resources["thermal"] / 260.0 + place.obstacles.get("heat", 0.0) * 0.12 + cold_exposure * 0.42,
            "ferry": place.resources["electrical"] / 160.0 + place.resources["dense_node"] / 90.0 + place.mineral_richness * place.geothermal * 0.55,
            "hoist": place.sealed_essence / 230.0 + place.obstacles.get("height", 0.0) * 0.40,
            "winnow": place.physics.get("current_exposure", 0.0) * 0.36 + place.physics.get("salinity", 0.0) * 0.18 + place.resources["essence"] / 260.0,
            "carry": max(0.0, individual.inventory_count() / max(1.0, individual.inventory_limit()) - 0.55) + individual.tool_skill.get("pickup", 0.0) * 0.04,
            "insulate": exposure_severity * 0.56 + cold_exposure * 0.35 + max(0.0, 0.42 - place.physics.get("temperature", 0.5)) * 0.20,
            "protect": (
                place.physics.get("pressure", 0.0) * 0.16
                + place.physics.get("abrasion", 0.0) * 0.22
                + place.physics.get("temperature", 0.5) * 0.08
                + exposure_severity * 0.26
                + len(self._living_ids_at(place.id)) / max(1.0, place.capacity) * 0.14
            ),
            "record": (
                min(1.0, len(individual.lesson_memory) / 4.0) * 0.42
                + individual.tool_skill.get("inscribe", 0.0) * 0.28
                + individual.params.memory_budget / 45.0
            ),
        }
        if individual.last_craft_target in scores:
            scores[individual.last_craft_target] += planning * 0.16
        if individual.last_tool_affordance in scores:
            scores[individual.last_tool_affordance] += planning * 0.10
        return max(scores, key=lambda name: scores[name] + self.rng.random() * (0.18 - planning * 0.12))

    def _select_craft_components(self, individual: Individual, target: str, draws: int, planning: float) -> dict[str, int]:
        components: dict[str, int] = {}
        for _ in range(draws):
            choices = [name for name, qty in individual.inventory.items() if qty > components.get(name, 0)]
            if not choices:
                break
            if self.rng.random() > 0.24 + planning * 0.68 + individual.tool_skill.get("craft", 0.0) * 0.18:
                chosen = self.rng.choice(choices)
            else:
                chosen = max(
                    choices,
                    key=lambda name: self._component_craft_score(components, name, target, planning),
                )
            components[chosen] = components.get(chosen, 0) + 1
        return components

    def _component_craft_score(self, components: dict[str, int], candidate: str, target: str, planning: float) -> float:
        trial = dict(components)
        trial[candidate] = trial.get(candidate, 0) + 1
        properties = component_properties(trial)
        capabilities = derive_artifact_capabilities(properties)
        target_fit = capabilities.get(target, 0.0)
        lash_fit = capabilities.get("lash", 0.0)
        durability_fit = properties.get("hard", 0.0) * 0.20 + properties.get("lashable", 0.0) * 0.12 + properties.get("flexible", 0.0) * 0.05
        diversity = len(trial) / max(1.0, sum(trial.values()))
        return target_fit * 0.60 + lash_fit * 0.16 + durability_fit + diversity * planning * 0.08 + self.rng.random() * (0.10 - planning * 0.06)

    def _craft(self, individual: Individual, feedback: dict[str, float]) -> None:
        if individual.params.manipulator < 0.12 or individual.inventory_count() < 2 or len(individual.artifacts) >= individual.artifact_limit():
            individual.energy -= 0.025
            return
        available = [name for name, qty in individual.inventory.items() if qty > 0]
        if len(available) < 2:
            individual.energy -= 0.020
            return
        planning = self._interaction_control(individual)
        target = self._craft_target_affordance(individual, self.world.places[individual.location], planning)
        draws = min(3 + int((planning + individual.tool_skill.get("craft", 0.0)) > 1.15), individual.inventory_count())
        components = self._select_craft_components(individual, target, draws, planning)
        if sum(components.values()) < 2:
            individual.energy -= 0.020
            return
        component_affordances = derive_affordances(components)
        lash_help = component_affordances.get("lash", 0.0)
        target_fit = derive_artifact_capabilities(component_properties(components)).get(target, 0.0)
        best_skill = self._skill_breadth(individual)
        craft_skill = individual.tool_skill.get("craft", 0.0)
        target_skill = individual.tool_skill.get(target, 0.0)
        material_gate = max(0.0, min(1.0, target_fit * 0.78 + lash_help * 0.16 + min(1.0, len(components) / max(1, sum(components.values()))) * 0.06))
        method_quality = max(
            0.0,
            min(
                1.0,
                material_gate
                * (
                    planning * 0.30
                    + craft_skill * 0.24
                    + target_skill * 0.18
                    + lash_help * 0.10
                    + target_fit * 0.18
                ),
            ),
        )
        chance = min(
            0.96,
            individual.params.manipulator * 0.26
            + lash_help * 0.22
            + best_skill * 0.08
            + planning * 0.12
            + method_quality * 0.20
            + target_fit * 0.24,
        )
        individual.energy -= (0.12 + 0.04 * sum(components.values())) * (1.0 - method_quality * 0.18)
        if self.rng.random() > chance:
            lost = self._lose_failed_craft_components(individual, components, lash_help)
            skill_gain = 0.0015 + lost * 0.0035
            self._increase_skill(individual, "lash", skill_gain, transfer=0.18)
            self._increase_skill(individual, "craft", skill_gain * 0.80, transfer=0.10)
            feedback["tool"] = feedback.get("tool", 0.0) + skill_gain
            self._record_tool_lesson(
                individual,
                self.world.places[individual.location],
                kind="craft",
                affordance=target,
                success=False,
                score=target_fit,
                method_quality=method_quality,
                components=components,
            )
            return
        for name, qty in components.items():
            individual.inventory[name] -= qty
            if individual.inventory[name] <= 0:
                del individual.inventory[name]
        artifact = build_artifact(components, method_quality=method_quality, target_affordance=target)
        individual.artifacts.append(artifact)
        self.artifacts_created[artifact.name] += 1
        individual.record_tool_success("craft")
        individual.record_success("tool_make", 1.0 + method_quality)
        self.tool_successes["craft"] += 1
        feedback["social"] += 0.06
        feedback["tool"] = feedback.get("tool", 0.0) + 0.30 + method_quality * 0.30
        self._increase_skill(individual, "craft", 0.020 * (1.0 - craft_skill) + method_quality * 0.010, transfer=0.16)
        self._increase_skill(individual, target, artifact.capabilities.get(target, 0.0) * 0.012 + method_quality * 0.010, transfer=0.10)
        individual.last_craft_target = target
        individual.last_tool_affordance = target
        individual.last_artifact_method = method_quality
        self._record_tool_lesson(
            individual,
            self.world.places[individual.location],
            kind="craft",
            affordance=target,
            success=True,
            gain=feedback.get("tool", 0.0),
            score=artifact.capabilities.get(target, 0.0),
            method_quality=method_quality,
            components=components,
        )
        for capability, value in artifact.capabilities.items():
            if capability in individual.tool_skill:
                self._increase_skill(individual, capability, value * 0.010, transfer=0.04)
        self.observer.observe(
            self.tick,
            "crafted_tool",
            {
                "organism_id": individual.id,
                "place": individual.location,
                "artifact": artifact.name,
                "target": target,
                "method_quality": method_quality,
                "target_fit": artifact.capabilities.get(target, 0.0),
                "components": dict(components),
            },
            subjects=self._subjects(individual, extra=[f"affordance:{target}"]),
            score=method_quality * 1.1 + artifact.capabilities.get(target, 0.0) * 0.8,
            rarity_key=f"crafted_tool:{target}",
        )
        self.checkpoints.save_first_tool(self.tick, individual, "craft", {"place": self.world.places[individual.location].to_summary(), "artifact": artifact.to_dict()})

    def _build_structure(self, individual: Individual, feedback: dict[str, float]) -> None:
        if individual.params.manipulator < 0.18:
            individual.energy -= 0.035
            return
        planning = self._interaction_control(individual)
        collaboration = self._collective_support(individual, "build", 0.35)
        support = float(collaboration.get("support", 0.0))
        collective_materials = self._collective_material_count(individual) if support > 0.01 else individual.inventory_count()
        if collective_materials < 3:
            individual.energy -= 0.035
            return
        available = [name for name, qty in individual.inventory.items() if qty > 0]
        if not available and support <= 0.01:
            individual.energy -= 0.025
            return

        actor_components: dict[str, int] = {}
        draws = min(8, individual.inventory_count())
        for _ in range(draws):
            choices = [name for name, qty in individual.inventory.items() if qty > actor_components.get(name, 0)]
            if not choices:
                break
            chosen = self.rng.choice(choices)
            actor_components[chosen] = actor_components.get(chosen, 0) + 1
        components = dict(actor_components)
        target_count = min(8, max(3, sum(actor_components.values()) + min(4, len(collaboration.get("helpers", [])))))
        helper_components = self._collective_build_components(individual, components, target_count, collaboration) if support > 0.01 else 0
        material_count = sum(components.values())
        if material_count < 3:
            individual.energy -= 0.025
            return

        affordances = derive_affordances(components)
        lash_help = affordances.get("lash", 0.0)
        build_skill = individual.tool_skill.get("build", 0.0)
        general_skill = self._skill_breadth(individual)
        mass_bonus = min(1.0, material_count / 8.0) * 0.12
        chance = min(
            0.94,
            individual.params.manipulator * 0.30
            + lash_help * 0.26
            + build_skill * 0.22
            + general_skill * 0.08
            + planning * 0.16
            + mass_bonus
            + support * 0.24,
        )
        individual.energy -= (0.16 + 0.035 * material_count) * (1.0 - planning * 0.10) * max(0.80, 1.0 - support * 0.18)
        if self.rng.random() > chance:
            lost = self._lose_failed_craft_components(individual, actor_components, lash_help)
            self._increase_skill(individual, "build", 0.003 + lost * 0.003 + helper_components * 0.0008, transfer=0.14)
            self._increase_skill(individual, "lash", 0.0015 + lost * 0.002, transfer=0.10)
            feedback["tool"] = feedback.get("tool", 0.0) + 0.04 + lost * 0.01 + support * 0.04
            if support > 0.01:
                self._apply_collaboration_effects(individual, "build", "failed_build", collaboration, lash_help)
            self.demonstrations[individual.location].append((individual.id, "build", False))
            self._record_tool_lesson(
                individual,
                self.world.places[individual.location],
                kind="build",
                affordance="build",
                success=False,
                score=lash_help,
                components=components,
            )
            return

        for name, qty in actor_components.items():
            individual.inventory[name] -= qty
            if individual.inventory[name] <= 0:
                del individual.inventory[name]

        place = self.world.places[individual.location]
        if place.structures and (len(place.structures) >= 16 or self.rng.random() < 0.62):
            target = max(place.structures, key=lambda structure: (structure.durability, structure.scale))
            extend_structure(target, components)
            built_name = target.name
            self.structures_extended[built_name] += 1
            structure_summary = target.to_dict()
            extended_action = True
        else:
            structure = build_structure(components, builder_id=individual.id)
            place.structures.append(structure)
            built_name = structure.name
            self.structures_built[built_name] += 1
            structure_summary = structure.to_dict()
            extended_action = False

        individual.record_tool_success("build")
        individual.record_success("structure", 1.0 + min(1.5, structure_summary["scale"] / 8.0))
        self._increase_skill(individual, "build", 0.035 * (1.0 - build_skill) + 0.004 + support * 0.006, transfer=0.16)
        for capability, value in structure_summary["capabilities"].items():
            if capability in individual.tool_skill:
                self._increase_skill(individual, capability, float(value) * 0.006, transfer=0.04)
        self.tool_successes["build"] += 1
        feedback["social"] += 0.10
        feedback["tool"] = feedback.get("tool", 0.0) + 0.75 + support * 0.14
        if support > 0.01:
            feedback["social"] += support * 0.20
            self._apply_collaboration_effects(individual, "build", "build", collaboration, lash_help + mass_bonus)
        self.demonstrations[place.id].append((individual.id, "build", True))
        self._record_tool_lesson(
            individual,
            place,
            kind="build",
            affordance="build",
            success=True,
            gain=structure_summary["scale"] / 4.0,
            score=max((float(value) for value in structure_summary["capabilities"].values()), default=0.0),
            components=components,
        )
        best_structure_capability = max((float(value) for value in structure_summary["capabilities"].values()), default=0.0)
        self.observer.observe(
            self.tick,
            "structure_built",
            {
                "organism_id": individual.id,
                "place": place.id,
                "structure": built_name,
                "scale": structure_summary["scale"],
                "best_capability": best_structure_capability,
                "extended": extended_action,
                "helper_components": helper_components,
                "collaboration_support": support,
            },
            subjects=self._subjects(individual),
            score=min(2.5, structure_summary["scale"] / 6.0 + best_structure_capability + support),
            rarity_key=f"structure:{built_name}",
        )
        if self.config.event_detail:
            self.logger.event(
                self.tick,
                "structure_built",
                {"organism_id": individual.id, "structure": built_name, "place": place.id, "scale": structure_summary["scale"]},
            )
        self.checkpoints.save_first_tool(self.tick, individual, "build", {"place": place.to_summary(), "structure": structure_summary})

    def _use_tool(self, individual: Individual, feedback: dict[str, float]) -> None:
        place = self.world.places[individual.location]
        base_affordance, base_score = best_affordance(individual.inventory, individual.tool_skill, individual.artifacts)
        affordance, score, situation_directed, situation_context = self._situation_affordance_choice(individual, place, base_affordance, base_score)
        if score < 0.08:
            individual.energy -= 0.05
            return
        skill = individual.tool_skill.get(affordance, 0.0)
        resistance = self._affordance_resistance(place, affordance)
        planning = self._interaction_control(individual)
        collaboration = self._collective_support(individual, affordance, resistance)
        support = float(collaboration.get("support", 0.0))
        overmatch = score + skill * 0.25 + individual.params.manipulator * 0.10 + planning * 0.22 + support * 0.18 - resistance
        chance = min(0.96, max(0.02, score * 0.34 + skill * 0.30 + individual.params.manipulator * 0.16 + planning * 0.26 + support * 0.16 + overmatch * 0.24))
        individual.energy -= (0.12 + score * 0.10) * (1.0 - planning * 0.20) * max(0.82, 1.0 - support * 0.14)
        success = self.rng.random() < chance
        if success:
            gain = self._tool_effect(individual, place, affordance, score, skill)
            competence = min(1.35, 0.45 + score * 0.35 + skill * 0.20 + support * 0.18)
            gain += self._advance_causal_challenge(individual, place, affordance, competence, feedback, collaboration=collaboration)
            individual.energy += gain
            self._wear_artifacts(individual, affordance, amount=0.18 + score * 0.10)
            self._increase_skill(individual, affordance, 0.055 * (1.0 - skill) + 0.005, transfer=0.12)
            individual.record_tool_success(affordance)
            individual.last_tool_affordance = affordance
            individual.record_success("tool_use", 1.0 + min(2.0, max(0.0, gain) / 8.0))
            self.tool_successes[affordance] += 1
            feedback["social"] += 0.2
            feedback["tool"] = feedback.get("tool", 0.0) + 1.0 + support * 0.12
            if support > 0.01:
                feedback["social"] += support * 0.18
                self._apply_collaboration_effects(individual, affordance, "tool_use", collaboration, score)
            self.demonstrations[place.id].append((individual.id, affordance, True))
            self._record_tool_lesson(
                individual,
                place,
                kind="tool_use",
                affordance=affordance,
                success=True,
                gain=gain,
                score=score,
                method_quality=individual.last_artifact_method,
            )
            self.observer.observe(
                self.tick,
                "tool_success",
                {
                    "organism_id": individual.id,
                    "place": place.id,
                    "affordance": affordance,
                    "gain": gain,
                    "score": score,
                    "skill": individual.tool_skill[affordance],
                    "environment_directed": situation_directed,
                    "situation_kind": situation_context["problem"].get("kind"),
                    "situation_reason": situation_context["reason"],
                    "situation_fit": situation_context["situation_fit"],
                    "memory_bias": situation_context["memory_bias"],
                    "recognition": situation_context["recognition"],
                    "base_affordance": situation_context["base_affordance"],
                },
                subjects=self._subjects(individual, extra=[f"affordance:{affordance}"]),
                score=min(3.0, max(0.0, gain) / 6.0 + score * 0.45 + (0.12 if situation_directed else 0.0)),
                rarity_key=f"tool_success:{affordance}",
            )
            if self.config.event_detail:
                self.logger.event(
                    self.tick,
                    "tool_success",
                    {
                        "organism_id": individual.id,
                        "affordance": affordance,
                        "gain": round(gain, 5),
                        "place": place.id,
                        "support": round(support, 5),
                        "situation": situation_context["problem"].get("kind"),
                        "reason": situation_context["reason"],
                    },
                )
            self.checkpoints.save_first_tool(self.tick, individual, affordance, {"place": place.to_summary(), "gain": gain})
        else:
            self._increase_skill(individual, affordance, 0.010 * (1.0 - skill), transfer=0.04)
            individual.last_tool_affordance = affordance
            feedback["tool"] = feedback.get("tool", 0.0) + 0.05 + support * 0.02
            if self.rng.random() < 0.10:
                protection = artifact_capability(individual.artifacts, "protect")
                accident_damage = (0.015 + score * 0.030) * max(0.45, 1.0 - protection * 0.42)
                individual.health -= accident_damage
                if protection > 0.0:
                    self._increase_skill(individual, "protect", accident_damage * 0.030, transfer=0.06)
                    self._wear_artifacts(individual, "protect", amount=accident_damage * 0.55)
            overmatch_penalty = max(0.0, resistance - score)
            self._wear_artifacts(individual, affordance, amount=0.35 + score * 0.18 + overmatch_penalty * 2.2)
            if support > 0.01:
                self._apply_collaboration_effects(individual, affordance, "failed_tool_use", collaboration, score)
            self.demonstrations[place.id].append((individual.id, affordance, False))
            self._record_tool_lesson(
                individual,
                place,
                kind="tool_use",
                affordance=affordance,
                success=False,
                score=score,
                method_quality=individual.last_artifact_method,
            )
            if individual.health <= 0.0:
                self._deactivate(individual, "tool_accident")

    def _affordance_resistance(self, place: Place, affordance: str) -> float:
        physics = place.physics
        if affordance in {"cleave", "hoist"}:
            return min(1.0, 0.22 + place.mineral_richness * 0.45 + place.sealed_essence / 420.0 + physics.get("pressure", 0.0) * 0.05)
        if affordance == "shear":
            return min(1.0, 0.10 + place.obstacles.get("thorn", 0.0) * 0.60)
        if affordance == "encase":
            return min(1.0, 0.10 + place.obstacles.get("water", 0.0) * 0.22 + physics.get("pressure", 0.0) * 0.10 + physics.get("current_exposure", 0.0) * 0.12)
        if affordance == "kindle":
            return min(1.0, 0.15 + place.obstacles.get("heat", 0.0) * 0.25 + place.volatility * 0.20 + physics.get("humidity", 0.5) * 0.06)
        if affordance == "ferry":
            return min(1.0, 0.30 + place.mineral_richness * 0.25 + physics.get("fluid_level", 0.0) * 0.08)
        if affordance == "winnow":
            return min(1.0, 0.10 + physics.get("current_exposure", 0.0) * 0.18 + physics.get("salinity", 0.0) * 0.10)
        if affordance == "lash":
            return 0.18
        return 0.25

    def _tool_effect(self, individual: Individual, place: Place, affordance: str, score: float, skill: float) -> float:
        competence = 0.45 + score * 0.35 + skill * 0.20
        if affordance == "cleave":
            amount = min(place.sealed_essence, 2.0 + competence * 9.0)
            place.sealed_essence -= amount
            return amount * (0.25 + individual.params.essence_energy_gain * 0.65 + individual.params.essence_conversion * 0.30)
        if affordance == "shear":
            amount = min(place.resources["organic_store"], 1.5 + competence * 8.0)
            place.resources["organic_store"] -= amount
            return amount * (0.35 + individual.params.essence_conversion * 0.85)
        if affordance == "lash":
            self._increase_skill(individual, "lash", 0.004, transfer=0.55)
            return 0.5 + competence * 1.4
        if affordance == "encase":
            amount = min(place.resources["mechanical"], 0.8 + competence * 4.0 + place.physics.get("current_exposure", 0.0) * 2.5)
            place.resources["mechanical"] -= amount * 0.15
            place.physics["fluid_level"] = max(0.0, place.physics.get("fluid_level", 0.0) - amount * 0.0008)
            return amount * (0.15 + individual.params.storage_capacity * 0.25)
        if affordance == "kindle":
            solar = min(place.resources["solar"], 2.0 + competence * 10.0)
            place.resources["thermal"] = min(180.0, place.resources["thermal"] + solar * 0.15)
            place.physics["temperature"] = min(1.45, place.physics.get("temperature", 0.5) + solar * 0.0008)
            return solar * (0.04 + individual.params.thermal_tolerance * 0.09 + individual.params.solar_energy_gain * 0.06)
        if affordance == "ferry":
            amount = min(place.resources["electrical"], 0.5 + competence * 5.0 + place.mineral_richness * 1.5)
            place.resources["electrical"] -= amount
            structure_ferry = structure_capability(place.structures, "ferry")
            storage = structure_capability(place.structures, "energy_storage")
            gradient = max(structure_capability(place.structures, "gradient_harvest"), place.geothermal * place.mineral_richness)
            tap_pressure = max(0.0, competence + structure_ferry * 0.45 + storage * 0.25 + gradient * 0.35 - 0.92)
            dense_node = min(place.resources["dense_node"], tap_pressure * (1.0 + competence * 3.0))
            place.resources["dense_node"] -= dense_node
            place.resources["electrical"] = min(180.0, place.resources["electrical"] + dense_node * (0.25 + storage * 0.25))
            return amount * (0.1 + individual.params.electrical_use * 1.3) + dense_node * (0.35 + individual.params.electrical_use * 1.8)
        if affordance == "hoist":
            amount = min(place.sealed_essence, 1.0 + competence * 5.5)
            place.sealed_essence -= amount
            return amount * (0.15 + individual.params.mechanical_use * 0.55 + individual.params.essence_energy_gain * 0.25)
        if affordance == "winnow":
            flow_bonus = place.physics.get("current_exposure", 0.0) * 3.0 + place.physics.get("fluid_level", 0.0) * 1.5
            essence = min(place.resources["essence"], 0.5 + competence * 3.0 + flow_bonus)
            organic = min(place.resources["organic_store"], 0.3 + competence * 1.8 + flow_bonus * 0.40)
            place.resources["essence"] -= essence * 0.55
            place.resources["organic_store"] -= organic * 0.45
            return essence * (0.12 + individual.params.essence_energy_gain * 0.45) + organic * (0.15 + individual.params.essence_conversion * 0.38)
        return 0.0

    def _advance_causal_challenge(
        self,
        individual: Individual,
        place: Place,
        affordance: str,
        competence: float,
        feedback: dict[str, float],
        collaboration: dict[str, Any] | None = None,
    ) -> float:
        challenge = place.causal_challenge
        if challenge is None or challenge.payoff_remaining <= 0.0:
            return 0.0
        expected = challenge.expected_affordance()
        if expected is None:
            return 0.0
        challenge.attempts += 1
        planning = self._interaction_control(individual)
        support = float((collaboration or {}).get("support", 0.0))
        threshold = challenge.difficulty * (0.76 + challenge.progress * 0.14)
        margin = competence + individual.tool_skill.get(affordance, 0.0) * 0.12 + planning * 0.30 + support * 0.18 + self.rng.gauss(0.0, 0.025)
        if affordance != expected or margin < threshold:
            reset_chance = max(0.08, 0.26 + challenge.progress * 0.04 - planning * 0.10 - support * 0.06)
            if affordance in challenge.sequence and self.rng.random() < reset_chance:
                challenge.progress = 0
            return 0.0

        challenge.progress += 1
        signature = challenge.signature()
        self.causal_steps[signature] += 1
        individual.record_success("causal_step", 1.0)
        feedback["tool"] = feedback.get("tool", 0.0) + 0.10
        if support > 0.01 and collaboration is not None:
            self._apply_collaboration_effects(individual, affordance, "causal_step", collaboration, competence)
        if challenge.progress < len(challenge.sequence):
            self._record_tool_lesson(
                individual,
                place,
                kind="causal_step",
                affordance=affordance,
                success=True,
                gain=0.05 + planning * 0.08,
                score=competence,
                sequence=challenge.sequence[:challenge.progress],
            )
            return 0.05 + planning * 0.08

        challenge.progress = 0
        challenge.solved += 1
        sequence_bonus = max(0.0, len(challenge.sequence) - 1) * 2.0
        release = min(challenge.payoff_remaining, 3.0 + competence * 13.0 + planning * 6.0 + sequence_bonus + support * 3.0)
        challenge.payoff_remaining -= release
        place.resources[challenge.payoff_energy] = min(180.0, place.resources[challenge.payoff_energy] + release)
        self.causal_unlocks[signature] += 1
        individual.record_success("causal_unlock", 1.0 + min(2.0, release / 8.0))
        feedback["tool"] = feedback.get("tool", 0.0) + 0.70
        if support > 0.01 and collaboration is not None:
            self._apply_collaboration_effects(individual, affordance, "causal_unlock", collaboration, release / 8.0)
        self._record_tool_lesson(
            individual,
            place,
            kind="causal_unlock",
            affordance=affordance,
            success=True,
            gain=release,
            score=competence,
            sequence=challenge.sequence,
        )
        self.observer.force_promote(
            self.tick,
            "causal_unlock",
            {
                "organism_id": individual.id,
                "place": place.id,
                "sequence": list(challenge.sequence),
                "energy": challenge.payoff_energy,
                "released": release,
                "remaining": challenge.payoff_remaining,
            },
            subjects=self._subjects(individual, place.id, [f"challenge:{signature}"]),
            score=1.0 + min(4.0, release / 6.0),
            rarity_key=f"causal_unlock:{signature}",
        )
        if self.config.event_detail:
            self.logger.event(
                self.tick,
                "causal_unlock",
                {
                    "organism_id": individual.id,
                    "place": place.id,
                    "sequence": list(challenge.sequence),
                    "energy": challenge.payoff_energy,
                    "released": round(release, 5),
                },
            )
        return release * (0.30 + individual.params.sensor_range * 0.10 + individual.params.prediction_weight * 0.12 + planning * 0.10)

    def _wear_artifacts(self, individual: Individual, affordance: str, amount: float) -> None:
        kept = []
        for artifact in individual.artifacts:
            if artifact.capabilities.get(affordance, 0.0) > 0.05:
                artifact.durability -= amount
                artifact.age += 1
            if artifact.durability > 0.0:
                kept.append(artifact)
            else:
                self.artifacts_broken[artifact.name] += 1
                self.world.places[individual.location].resources["essence"] += 0.1
        individual.artifacts = kept

    def _lose_failed_craft_components(self, individual: Individual, components: dict[str, int], lash_help: float) -> int:
        place = self.world.places[individual.location]
        lost = 0
        break_chance = max(0.18, min(0.72, 0.48 - lash_help * 0.20 + (1.0 - individual.params.manipulator) * 0.12))
        for name, qty in components.items():
            for _ in range(qty):
                if individual.inventory.get(name, 0) <= 0 or self.rng.random() >= break_chance:
                    continue
                individual.inventory[name] -= 1
                if individual.inventory[name] <= 0:
                    del individual.inventory[name]
                lost += 1
                if self.rng.random() < 0.35:
                    place.materials[name] = min(99, place.materials.get(name, 0) + 1)
        return lost

    def _attack(self, individual: Individual) -> None:
        local = [self.organisms[oid] for oid in self._living_ids_at(individual.location) if oid != individual.id]
        if not local:
            individual.energy -= 0.04
            return
        target = min(local, key=lambda candidate: (candidate.health + candidate.params.armor * 0.7, -candidate.energy))
        attack_power = individual.params.mobility * 0.40 + individual.params.manipulator * 0.35 + individual.params.mechanical_use * 0.25
        protection = artifact_capability(target.artifacts, "protect")
        defense_context = self._agent_defense_context(target, attack_power)
        defense = target.params.armor * 0.45 + target.params.mobility * 0.25 + target.health * 0.20 + protection * 0.34 + float(defense_context["bonus"])
        damage = max(0.0, attack_power - defense + self.rng.gauss(0.0, 0.05))
        individual.energy -= 0.10 + attack_power * 0.08
        if target.kind == "agent" and float(defense_context["support"]) > 0.01 and defense_context["collaboration"] is not None:
            self._apply_collaboration_effects(target, "protect", "defense", defense_context["collaboration"], attack_power)
        if damage > 0.0:
            if protection > 0.0:
                damage *= max(0.35, 1.0 - protection * 0.38)
                self._increase_skill(target, "protect", damage * 0.025, transfer=0.06)
                self._wear_artifacts(target, "protect", amount=damage * (0.55 + protection * 0.35))
            target.health -= damage
        elif target.kind == "agent":
            self._increase_skill(target, "protect", 0.002 + max(0.0, defense - attack_power) * 0.006, transfer=0.05)
        if target.kind == "agent" and target.health > 0.0:
            counter_base = (
                target.params.armor * 0.16
                + target.params.mobility * 0.14
                + target.params.manipulator * 0.10
                + protection * 0.22
                + float(defense_context["structure"]) * 0.12
                + float(defense_context["support"]) * 0.18
            )
            counter_window = counter_base - attack_power * 0.38 + self.rng.gauss(0.0, 0.035)
            if counter_window > 0.12:
                counter_damage = min(0.32, (counter_window - 0.12) * 0.42)
                individual.health -= counter_damage
                target.energy -= 0.025 + counter_damage * 0.05
                target.record_success("collaboration", float(defense_context["support"]) * 0.25)
                self._increase_skill(target, "protect", 0.003 + counter_damage * 0.030, transfer=0.07)
                if individual.health <= 0.0:
                    self._deactivate(individual, "counterattack")
        if target.health <= 0.0 and individual.alive:
            gained = target.energy * (0.30 + individual.params.essence_conversion * 0.45)
            individual.energy += max(0.0, gained)
            self._deactivate(target, "predation")

    def _signal(self, individual: Individual, feedback: dict[str, float]) -> None:
        intensity = individual.params.signal_strength * (0.5 + individual.energy / max(1.0, individual.storage_limit()))
        if intensity <= 0.01:
            individual.energy -= 0.01
            return
        token = individual.choose_signal_token()
        individual.energy -= 0.025 + intensity * 0.045
        self.world.emit_signal(individual.location, individual.id, token, intensity)
        feedback["social"] += intensity * 0.1

    def _coordinate_recombine(self, individual: Individual, feedback: dict[str, float]) -> None:
        self.reproduction_attempts["coordinate"] += 1
        if not individual.adult():
            self.reproduction_failures["coordinate_not_adult"] += 1
            individual.energy -= 0.015
            return
        if individual.energy < self._recombine_reserve_threshold(individual) * 0.82:
            self.reproduction_failures["coordinate_low_energy"] += 1
            individual.energy -= 0.020
            return
        window = 6 + int(individual.params.signal_strength * 8.0 + individual.params.pairing_selectivity * 5.0)
        token = individual.choose_signal_token()
        intensity = 0.10 + individual.params.signal_strength * 0.45 + individual.params.pairing_selectivity * 0.10
        individual.recombine_intent_until = max(individual.recombine_intent_until, self.tick + window)
        individual.coordination_token = token
        individual.energy -= 0.035 + intensity * 0.040
        self.world.emit_signal(individual.location, individual.id, token, intensity)
        feedback["social"] += intensity * 0.12

    def _mark(self, individual: Individual, feedback: dict[str, float]) -> None:
        if individual.params.manipulator < 0.12:
            individual.energy -= 0.02
            return
        token = individual.choose_signal_token()
        affordances = derive_affordances(individual.inventory)
        inscription_help = max(affordances.get("shear", 0.0), affordances.get("lash", 0.0), affordances.get("kindle", 0.0))
        intensity = 0.20 + individual.params.signal_strength * 0.35 + individual.params.memory_budget / 40.0
        durability = 45.0 + individual.params.manipulator * 90.0 + individual.params.memory_budget * 12.0 + inscription_help * 180.0
        trace_intent = self._trace_inscription_intent(individual, inscription_help)
        trace = self._mark_trace(individual, inscription_help) if trace_intent else {}
        clarity = float(trace.get("inscription_quality", 0.0)) if trace else 0.0
        individual.energy -= 0.08 + intensity * 0.12 + durability * 0.0008 + clarity * 0.055
        portable_artifact = self._recording_artifact(individual)
        portable_written = False
        if trace and portable_artifact is not None:
            artifact, record_cap = portable_artifact
            planning = self._interaction_control(individual)
            portable_chance = min(0.72, record_cap * 0.48 + planning * 0.18 + individual.tool_skill.get("inscribe", 0.0) * 0.18)
            if self.rng.random() < portable_chance:
                self._inscribe_portable_mark(individual, artifact, token, intensity, durability, trace)
                portable_written = True
        if not portable_written:
            self.world.create_mark(individual.location, individual.id, token, intensity, durability, trace=trace)
            self.marks_created[str(token)] += 1
        if trace:
            affordance = str(trace.get("affordance", "unknown"))
            writing_quality = float(trace.get("writing_quality", clarity))
            self._increase_skill(individual, "inscribe", 0.003 + clarity * 0.006 + writing_quality * 0.008, transfer=0.10)
            self.mark_lesson_packets[affordance] += 1
            lesson = trace.get("lesson", {})
            problem = lesson.get("problem", {}) if isinstance(lesson, dict) else {}
            self.observer.observe(
                self.tick,
                "mark_lesson_written",
                {
                    "organism_id": individual.id,
                    "place": individual.location,
                    "token": token,
                    "affordance": affordance,
                    "clarity": clarity,
                    "writing_quality": writing_quality,
                    "coherence": trace.get("coherence"),
                    "portable": portable_written,
                    "problem_kind": problem.get("kind") if isinstance(problem, dict) else None,
                    "lesson_kind": lesson.get("kind") if isinstance(lesson, dict) else None,
                },
                subjects=self._subjects(individual, extra=[f"affordance:{affordance}", f"mark_token:{token}"]),
                score=writing_quality * 1.35,
                rarity_key=f"mark_lesson_written:{affordance}",
            )
        feedback["social"] += 0.04 + intensity * 0.08 + clarity * 0.04

    def _recording_artifact(self, individual: Individual) -> tuple[Artifact, float] | None:
        candidates: list[tuple[Artifact, float]] = []
        for artifact in individual.artifacts:
            durability_factor = max(0.0, min(1.0, artifact.durability / 100.0))
            record_cap = artifact.capabilities.get("record", 0.0) * durability_factor
            if record_cap > 0.08:
                candidates.append((artifact, record_cap))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1] + self.rng.random() * 0.02)

    def _portable_inscription_limit(self, artifact: Artifact) -> int:
        record_cap = artifact.capabilities.get("record", 0.0)
        carry_cap = artifact.capabilities.get("carry", 0.0)
        return max(1, min(8, int(1 + record_cap * 5.0 + carry_cap * 2.0)))

    def _inscribe_portable_mark(
        self,
        individual: Individual,
        artifact: Artifact,
        token: int,
        intensity: float,
        durability: float,
        trace: dict[str, Any],
    ) -> None:
        record_cap = artifact.capabilities.get("record", 0.0) * max(0.0, min(1.0, artifact.durability / 100.0))
        packet = {
            "source_id": individual.id,
            "token": token % 8,
            "intensity": max(0.0, intensity * (0.72 + record_cap * 0.24)),
            "durability": max(1.0, durability * (0.62 + record_cap * 0.38)),
            "age": 0,
            "trace": dict(trace),
            "reads": 0,
            "value_transmitted": 0.0,
            "last_read_tick": -1,
        }
        artifact.inscriptions.append(packet)
        artifact.inscriptions = artifact.inscriptions[-self._portable_inscription_limit(artifact) :]
        artifact.durability = max(0.0, artifact.durability - 0.08 - max(0.0, 1.0 - record_cap) * 0.12)
        affordance = str(trace.get("affordance", "unknown"))
        self.portable_marks_created[affordance] += 1
        self.observer.observe(
            self.tick,
            "portable_mark_written",
            {
                "organism_id": individual.id,
                "place": individual.location,
                "artifact": artifact.name,
                "token": token % 8,
                "affordance": affordance,
                "writing_quality": trace.get("writing_quality", trace.get("inscription_quality")),
                "record_cap": record_cap,
                "stored": len(artifact.inscriptions),
            },
            subjects=self._subjects(individual, extra=[f"affordance:{affordance}", f"mark_token:{token % 8}", f"artifact:{artifact.name}"]),
            score=float(trace.get("writing_quality", trace.get("inscription_quality", 0.0))) * (1.0 + record_cap * 0.50),
            rarity_key=f"portable_mark_written:{affordance}",
        )

    def _trace_inscription_intent(self, individual: Individual, inscription_help: float) -> bool:
        if not individual.lesson_memory:
            return False
        planning = self._interaction_control(individual)
        skill = individual.tool_skill.get("inscribe", 0.0)
        memory = min(1.0, individual.params.memory_budget / 14.0)
        lesson_value = self._lesson_value(self._select_mark_lesson(individual))
        capacity = (
            individual.params.manipulator * 0.20
            + individual.params.signal_strength * 0.10
            + individual.params.sensor_range * 0.10
            + memory * 0.18
            + min(1.0, inscription_help) * 0.14
        )
        chance = capacity + planning * 0.16 + skill * 0.42 + lesson_value * 0.08 - 0.24
        if skill < 0.04:
            chance *= 0.45
        return self.rng.random() < max(0.0, min(0.82, chance))

    def _select_mark_lesson(self, individual: Individual) -> dict[str, Any]:
        return max(individual.lesson_memory[-5:], key=self._lesson_value)

    def _lesson_value(self, lesson: dict[str, Any]) -> float:
        problem = lesson.get("problem", {}) if isinstance(lesson.get("problem"), dict) else {}
        problem_value = max(
            float(problem.get("severity", 0.0) or 0.0),
            float(problem.get("value", 0.0) or 0.0),
            min(1.0, float(problem.get("remaining", 0.0) or 0.0) / 30.0),
        )
        return max(
            0.0,
            min(
                2.5,
                (0.35 if lesson.get("success") else 0.08)
                + float(lesson.get("gain", 0.0) or 0.0) / 10.0
                + float(lesson.get("score", 0.0) or 0.0) * 0.35
                + float(lesson.get("method_quality", 0.0) or 0.0) * 0.35
                + problem_value * 0.30
                + len(lesson.get("sequence", []) or []) * 0.12,
            ),
        )

    def _lesson_coherence(self, lesson: dict[str, Any]) -> float:
        affordance = str(lesson.get("affordance", ""))
        problem = lesson.get("problem", {}) if isinstance(lesson.get("problem"), dict) else {}
        required = str(problem.get("required_affordance", affordance))
        sequence = lesson.get("sequence", []) if isinstance(lesson.get("sequence", []), list) else []
        components = lesson.get("components", {}) if isinstance(lesson.get("components"), dict) else {}
        score = float(lesson.get("score", 0.0) or 0.0)
        gain = float(lesson.get("gain", 0.0) or 0.0)
        method_quality = float(lesson.get("method_quality", 0.0) or 0.0)
        coherence = 0.24
        if affordance and required == affordance:
            coherence += 0.30
        if lesson.get("success"):
            coherence += 0.12
        if score > 0.0 or gain > 0.0:
            coherence += min(0.18, score * 0.10 + gain / 80.0)
        if method_quality > 0.0:
            coherence += min(0.12, method_quality * 0.12)
        if sequence:
            coherence += min(0.10, len(sequence) * 0.035)
        if components:
            coherence += min(0.10, len(components) * 0.035)
        return max(0.0, min(1.0, coherence))

    def _mark_trace(self, individual: Individual, inscription_help: float) -> dict[str, Any]:
        if not individual.lesson_memory:
            return {}
        lesson = self._select_mark_lesson(individual)
        affordance = str(lesson.get("affordance") or individual.last_tool_affordance or individual.last_craft_target)
        if affordance not in individual.tool_skill:
            return {}
        planning = self._interaction_control(individual)
        inscribe_skill = individual.tool_skill.get("inscribe", 0.0)
        clarity = max(
            0.0,
            min(
                1.0,
                individual.params.memory_budget / 18.0 * 0.22
                + individual.params.sensor_range * 0.14
                + individual.params.manipulator * 0.15
                + individual.params.signal_strength * 0.08
                + min(1.0, inscription_help) * 0.16
                + planning * 0.12
                + inscribe_skill * 0.33
                + self.rng.gauss(0.0, 0.025),
            ),
        )
        if clarity < 0.12:
            return {}
        lesson_value = self._lesson_value(lesson)
        coherence = self._lesson_coherence(lesson)
        writing_quality = max(0.0, min(1.0, clarity * (0.52 + coherence * 0.28 + min(1.0, lesson_value / 1.6) * 0.20)))
        encoded = self._encoded_mark_lesson(lesson, clarity)
        trace: dict[str, Any] = {
            "schema": "lesson_trace_v1",
            "intentional": True,
            "action": individual.last_action,
            "affordance": affordance,
            "valence": round(individual.last_valence, 6),
            "energy_delta": round(individual.last_energy_delta, 6),
            "inscription_quality": round(clarity, 6),
            "writing_quality": round(writing_quality, 6),
            "lesson_value": round(lesson_value, 6),
            "coherence": round(coherence, 6),
            "lesson": encoded,
        }
        trace["skill"] = round(float(lesson.get("skill", individual.tool_skill.get(affordance, 0.0))) * (0.55 + clarity * 0.45), 6)
        trace["method_quality"] = round(float(lesson.get("method_quality", 0.0) or 0.0) * (0.50 + clarity * 0.50), 6)
        trace["tool_feedback"] = round(max(float(lesson.get("gain", 0.0) or 0.0), float(lesson.get("score", 0.0) or 0.0)) * clarity, 6)
        return trace

    def _encoded_mark_lesson(self, lesson: dict[str, Any], clarity: float) -> dict[str, Any]:
        affordance = str(lesson.get("affordance", ""))
        encoded: dict[str, Any] = {
            "kind": lesson.get("kind", "tool"),
            "affordance": affordance,
            "success": bool(lesson.get("success", False)),
        }
        if clarity >= 0.24:
            problem = lesson.get("problem", {}) if isinstance(lesson.get("problem"), dict) else {}
            encoded_problem: dict[str, Any] = {
                "kind": problem.get("kind", "unknown"),
                "required_affordance": problem.get("required_affordance", affordance),
            }
            for key in ("obstacle", "resource", "payoff_energy", "required_capability"):
                if key in problem and clarity >= 0.34:
                    encoded_problem[key] = problem[key]
            for key in ("severity", "value", "remaining", "difficulty"):
                if key in problem and clarity >= 0.42:
                    encoded_problem[key] = round(float(problem[key]), 6)
            if "sequence" in problem and clarity >= 0.62:
                encoded_problem["sequence"] = list(problem["sequence"])[:4]
            encoded["problem"] = encoded_problem
        if clarity >= 0.36:
            encoded["score"] = round(float(lesson.get("score", 0.0) or 0.0), 6)
            encoded["gain"] = round(float(lesson.get("gain", 0.0) or 0.0), 6)
        if clarity >= 0.48 and isinstance(lesson.get("components"), dict):
            components = lesson["components"]
            limit = 1 + int(clarity * 4.0)
            encoded["components"] = dict(sorted(components.items(), key=lambda item: item[1], reverse=True)[:limit])
        if clarity >= 0.58 and lesson.get("sequence"):
            encoded["sequence"] = list(lesson["sequence"])[:5]
        if clarity >= 0.68:
            encoded["method_quality"] = round(float(lesson.get("method_quality", 0.0) or 0.0), 6)
        return encoded

    def _clone_mutate(self, individual: Individual, feedback: dict[str, float]) -> None:
        self.reproduction_attempts["clone_mutate"] += 1
        if not individual.adult() or self.living_total >= self.config.max_population:
            self.reproduction_failures["clone_mutate_not_adult_or_cap"] += 1
            individual.energy -= 0.02
            return
        place = self.world.places[individual.location]
        if len(self._living_ids_at(individual.location)) >= place.capacity:
            self.reproduction_failures["clone_mutate_local_capacity"] += 1
            individual.energy -= 0.015
            return
        decision = self.optimization.plan_clone_mutate(individual)
        if decision.failure or decision.plan is None:
            self.reproduction_failures[decision.failure or "clone_mutate_no_plan"] += 1
            individual.energy -= decision.energy_penalty
            return
        child = self._instantiate_offspring(decision.plan)
        if child:
            self._apply_parent_costs_and_counts(decision.plan)
            self.births_by_mode[decision.plan.operator] += 1
            feedback["reproduction"] += 1.0
            self.observer.observe(
                self.tick,
                "birth",
                {
                    "mode": decision.plan.operator,
                    "child_id": child.id,
                    "parent_ids": list(decision.plan.parent_ids),
                    "kind": child.kind,
                    "place": child.location,
                    "generation": child.cycle,
                    "lineage_root_id": child.lineage_root_id,
                    "parent_lineage_ids": list(child.parent_lineage_ids),
                    "inherited_brain_template": child.inherited_brain_template,
                    "complexity": child.params.complexity(),
                },
                subjects=self._subjects(
                    child,
                    extra=[
                        f"organism:{individual.id}",
                        f"lineage:{individual.lineage_root_id or individual.id}",
                        "mode:clone_mutate",
                    ],
                ),
                score=0.35 + child.cycle * 0.08 + child.params.complexity() * 0.08,
                rarity_key=f"birth:{decision.plan.operator}:{child.kind}",
            )
            if self.config.event_detail:
                self.logger.event(
                    self.tick,
                    "birth",
                    {
                        "mode": decision.plan.operator,
                        "child_id": child.id,
                        "parent_ids": list(decision.plan.parent_ids),
                        "kind": child.kind,
                        "lineage_root_id": child.lineage_root_id,
                        "parent_lineage_ids": list(child.parent_lineage_ids),
                        "inherited_brain_template": child.inherited_brain_template,
                    },
                )
        else:
            self.reproduction_failures["clone_mutate_add_failed"] += 1

    def _resolve_recombine(self, place_ids: set[int], feedback: dict[int, dict[str, float]]) -> None:
        for place_id in place_ids:
            candidates = [
                individual
                for individual in self.organisms.values()
                if individual.alive
                and individual.location == place_id
                and individual.adult()
                and individual.recombine_intent_until >= self.tick
            ]
            if len(candidates) < 2:
                if candidates:
                    self.reproduction_failures["recombine_no_partner"] += len(candidates)
                continue
            self.rng.shuffle(candidates)
            paired: set[int] = set()
            choices: dict[int, int] = {}
            for individual in candidates:
                self.reproduction_attempts["recombine_pairing"] += 1
                if individual.energy < self._recombine_reserve_threshold(individual):
                    self.reproduction_failures["recombine_low_energy"] += 1
                    continue
                viable = [
                    other
                    for other in candidates
                    if other.id != individual.id
                    and other.id not in paired
                    and individual.energy >= self.optimization.recombine_reserve_threshold(individual)
                    and other.energy >= self.optimization.recombine_reserve_threshold(other)
                    and self.optimization.compatible_for_recombine(individual, other)
                ]
                if not viable:
                    self.reproduction_failures["recombine_no_compatible_partner"] += 1
                    continue
                choices[individual.id] = max(viable, key=lambda other: self._partner_score(individual, other)).id
            for individual in candidates:
                if individual.id in paired or individual.id not in choices:
                    continue
                partner_id = choices[individual.id]
                partner = self.organisms.get(partner_id)
                if partner is None or not partner.alive or partner.id in paired:
                    continue
                if choices.get(partner.id) != individual.id and self.rng.random() > 0.35:
                    self.reproduction_failures["recombine_unreciprocated_choice"] += 1
                    continue
                child = self._recombine(individual, partner)
                if child:
                    paired.add(individual.id)
                    paired.add(partner.id)
                    individual.recombine_intent_until = -1
                    partner.recombine_intent_until = -1
                    feedback[individual.id]["reproduction"] += 1.0
                    feedback[partner.id]["reproduction"] += 1.0
                    if self.config.event_detail:
                        self.logger.event(
                            self.tick,
                            "birth",
                            {
                                "mode": "recombine",
                                "child_id": child.id,
                                "parent_ids": [individual.id, partner.id],
                                "kind": child.kind,
                                "lineage_root_id": child.lineage_root_id,
                                "parent_lineage_ids": list(child.parent_lineage_ids),
                                "inherited_brain_template": child.inherited_brain_template,
                            },
                        )

    def _partner_score(self, chooser: Individual, candidate: Individual) -> float:
        visible_fitness = (
            candidate.health * 0.35
            + min(1.0, candidate.energy / max(1.0, candidate.storage_limit())) * 0.25
            + candidate.params.mobility * 0.10
            + candidate.params.manipulator * 0.10
            + self._skill_breadth(candidate) * 0.10
            + min(1.0, candidate.offspring_count / 5.0) * 0.10
        )
        selectivity = chooser.params.pairing_selectivity
        return visible_fitness * (0.3 + selectivity) - chooser.params.distance(candidate.params) * 0.25 + self.rng.random() * 0.05

    def _recombine_reserve_threshold(self, individual: Individual) -> float:
        return self.optimization.recombine_reserve_threshold(individual)

    def _recombine(self, a: Individual, b: Individual) -> Individual | None:
        if self.living_total >= self.config.max_population:
            self.reproduction_failures["recombine_population_cap"] += 1
            return None
        decision = self.optimization.plan_recombine(a, b)
        if decision.failure or decision.plan is None:
            self.reproduction_failures[decision.failure or "recombine_no_plan"] += 1
            return None
        child = self._instantiate_offspring(decision.plan)
        if child:
            self._apply_parent_costs_and_counts(decision.plan)
            self.births_by_mode[decision.plan.operator] += 1
            self.observer.observe(
                self.tick,
                "birth",
                {
                    "mode": decision.plan.operator,
                    "child_id": child.id,
                    "parent_ids": [a.id, b.id],
                    "kind": child.kind,
                    "place": child.location,
                    "generation": child.cycle,
                    "lineage_root_id": child.lineage_root_id,
                    "parent_lineage_ids": list(child.parent_lineage_ids),
                    "inherited_brain_template": child.inherited_brain_template,
                    "complexity": child.params.complexity(),
                },
                subjects=self._subjects(
                    child,
                    extra=[
                        f"organism:{a.id}",
                        f"organism:{b.id}",
                        f"lineage:{a.lineage_root_id or a.id}",
                        f"lineage:{b.lineage_root_id or b.id}",
                        "mode:combine",
                    ],
                ),
                score=0.55 + child.cycle * 0.09 + child.params.complexity() * 0.10,
                rarity_key=f"birth:{decision.plan.operator}:{child.kind}",
            )
        else:
            self.reproduction_failures["recombine_add_failed"] += 1
        return child

    def _instantiate_offspring(self, plan: OffspringPlan) -> Individual | None:
        return self.add_individual(
            plan.child_kind,
            plan.child_genome,
            plan.location,
            plan.child_energy,
            plan.cycle,
            plan.parent_ids,
            plan.controller_template,
        )

    def _apply_parent_costs_and_counts(self, plan: OffspringPlan) -> None:
        for parent_id, cost in plan.parent_costs.items():
            parent = self.organisms.get(parent_id)
            if parent is None:
                continue
            parent.energy -= cost
            parent.offspring_count += 1
            parent.record_success("reproduction", 1.0)

    def _readable_mark_candidates(self, individual: Individual, place: Place) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for mark in place.marks[-16:]:
            if mark.trace and mark.trace.get("intentional") and mark.trace.get("schema") == "lesson_trace_v1":
                candidates.append(
                    {
                        "kind": "place",
                        "mark": mark,
                        "source_id": mark.source_id,
                        "holder_id": None,
                        "token": mark.token,
                        "intensity": mark.intensity,
                        "durability": mark.durability,
                        "trace": mark.trace,
                        "reads": mark.reads,
                        "value_transmitted": mark.value_transmitted,
                    }
                )
        for holder_id in self._living_ids_at(place.id):
            holder = self.organisms.get(holder_id)
            if holder is None:
                continue
            for artifact in holder.artifacts:
                if not artifact.inscriptions:
                    continue
                artifact_factor = max(0.0, min(1.0, artifact.durability / 100.0))
                for inscription in artifact.inscriptions[-4:]:
                    trace = inscription.get("trace", {}) if isinstance(inscription.get("trace"), dict) else {}
                    if trace and trace.get("intentional") and trace.get("schema") == "lesson_trace_v1":
                        candidates.append(
                            {
                                "kind": "portable",
                                "artifact": artifact,
                                "inscription": inscription,
                                "source_id": int(inscription.get("source_id", holder.id)),
                                "holder_id": holder.id,
                                "token": int(inscription.get("token", 0)) % 8,
                                "intensity": float(inscription.get("intensity", 0.0)) * (0.80 + artifact_factor * 0.20),
                                "durability": min(float(inscription.get("durability", 0.0)), artifact.durability),
                                "trace": trace,
                                "reads": int(inscription.get("reads", 0)),
                                "value_transmitted": float(inscription.get("value_transmitted", 0.0)),
                            }
                        )
        return candidates

    def _read_mark_trace(self, individual: Individual, feedback: dict[str, float]) -> bool:
        place = self.world.places[individual.location]
        readable = self._readable_mark_candidates(individual, place)
        if not readable:
            return False
        planning = self._interaction_control(individual)
        candidate = max(
            readable,
            key=lambda item: (
                float(item["intensity"]) * min(1.0, float(item["durability"]) / 140.0)
                + float(item["trace"].get("writing_quality", item["trace"].get("inscription_quality", 0.0))) * 0.18
                + min(0.25, int(item["reads"]) * 0.025 + float(item["value_transmitted"]) * 0.08)
                + (0.05 if item["holder_id"] == individual.id else 0.0)
                + self.rng.random() * 0.03
            ),
        )
        trace = candidate["trace"]
        lesson = trace.get("lesson", {}) if isinstance(trace.get("lesson"), dict) else {}
        affordance = str(trace.get("affordance") or lesson.get("affordance", ""))
        if affordance not in individual.tool_skill:
            return False
        interpretation_skill = individual.tool_skill.get("interpret_mark", 0.0)
        inscription_quality = max(0.0, min(1.0, float(trace.get("inscription_quality", 0.0))))
        writing_quality = max(0.0, min(1.0, float(trace.get("writing_quality", inscription_quality))))
        coherence = max(0.0, min(1.0, float(trace.get("coherence", 0.5))))
        lesson_value = max(0.0, min(2.5, float(trace.get("lesson_value", self._lesson_value(lesson) if lesson else 0.0))))
        method_quality = max(0.0, min(1.0, float(trace.get("method_quality", 0.0))))
        tool_feedback = max(0.0, min(1.5, float(trace.get("tool_feedback", 0.0))))
        token = int(candidate["token"])
        token_bias = max(0.0, individual.signal_values[token] if 0 <= token < len(individual.signal_values) else 0.0)
        fidelity = max(
            0.0,
            min(
                1.0,
                float(candidate["intensity"])
                * min(1.0, float(candidate["durability"]) / 140.0)
                * (inscription_quality * 0.42 + writing_quality * 0.58)
                * (0.44 + interpretation_skill * 0.34 + planning * 0.14 + coherence * 0.08),
            ),
        )
        attention = max(
            0.0,
            min(
                1.0,
                individual.params.sensor_range * 0.34
                + individual.params.memory_budget / 22.0
                + planning * 0.24
                + interpretation_skill * 0.26
                + token_bias * 0.10,
            ),
        )
        gain = fidelity * attention * (
            0.0025
            + tool_feedback * 0.012
            + method_quality * 0.016
            + interpretation_skill * 0.006
            + writing_quality * 0.006
            + min(1.0, lesson_value / 1.5) * 0.007
        )
        interpretation_gain = 0.0010 + fidelity * 0.002 + min(0.010, gain * 0.22)
        self._increase_skill(individual, "interpret_mark", interpretation_gain)
        if candidate["kind"] == "place":
            mark = candidate["mark"]
            mark.reads += 1
            mark.last_read_tick = self.tick
            reads = mark.reads
        else:
            inscription = candidate["inscription"]
            inscription["reads"] = int(inscription.get("reads", 0)) + 1
            inscription["last_read_tick"] = self.tick
            reads = int(inscription["reads"])
        if gain <= 0.0005:
            individual.energy -= 0.006
            return True
        if candidate["kind"] == "place":
            candidate["mark"].value_transmitted = min(100.0, candidate["mark"].value_transmitted + gain)
        else:
            inscription = candidate["inscription"]
            inscription["value_transmitted"] = min(100.0, float(inscription.get("value_transmitted", 0.0)) + gain)
            self.portable_mark_reads[affordance] += 1
        self._increase_skill(individual, affordance, gain, transfer=0.08)
        if method_quality > 0.0 or lesson.get("components"):
            self._increase_skill(individual, "craft", gain * 0.55, transfer=0.08)
            individual.last_craft_target = affordance
            individual.last_artifact_method = max(individual.last_artifact_method, method_quality * fidelity)
        individual.last_tool_affordance = affordance
        if lesson:
            copied_lesson = dict(lesson)
            copied_lesson["kind"] = f"read_{copied_lesson.get('kind', 'lesson')}"
            copied_lesson["gain"] = gain
            copied_lesson["score"] = max(float(copied_lesson.get("score", 0.0) or 0.0), fidelity)
            individual.record_lesson(copied_lesson)
        individual.record_success("written_learning", gain * 12.0)
        self.mark_lessons[affordance] += 1
        self.mark_read_value[affordance] += gain
        self._apply_mark_author_feedback(int(candidate["source_id"]), individual.id, place.id, token, affordance, gain, fidelity, writing_quality, reads)
        feedback["social"] += 0.05 + fidelity * 0.04
        feedback["tool"] = feedback.get("tool", 0.0) + min(0.12, gain * 4.0)
        individual.energy -= 0.010 + (1.0 - attention) * 0.010
        portable = candidate["kind"] == "portable"
        self.observer.observe(
            self.tick,
            "portable_mark_read" if portable else "mark_lesson_read",
            {
                "organism_id": individual.id,
                "source_id": int(candidate["source_id"]),
                "holder_id": candidate["holder_id"],
                "place": place.id,
                "token": token,
                "affordance": affordance,
                "gain": gain,
                "fidelity": fidelity,
                "clarity": inscription_quality,
                "writing_quality": writing_quality,
                "coherence": coherence,
                "reads": reads,
                "portable": portable,
                "self_read": int(candidate["source_id"]) == individual.id,
                "artifact": candidate.get("artifact").name if portable else None,
            },
            subjects=self._subjects(individual, place.id, [f"organism:{candidate['source_id']}", f"affordance:{affordance}", f"mark_token:{token}"]),
            score=fidelity + writing_quality * 0.45 + gain * 18.0 + min(0.5, reads * 0.04),
            rarity_key=f"mark_lesson_read:{affordance}",
        )
        return True

    def _apply_mark_author_feedback(self, source_id: int, reader_id: int, place_id: int, token: int, affordance: str, gain: float, fidelity: float, writing_quality: float, reads: int) -> None:
        if source_id == reader_id:
            return
        author = self.organisms.get(source_id)
        if author is None or not author.alive:
            return
        # Feedback is local: the writer only learns when still present where the mark is used.
        if author.location != place_id:
            return
        if fidelity <= 0.0 or gain <= 0.0:
            return
        feedback_gain = min(0.020, 0.0015 + gain * 0.14 + fidelity * 0.002 + writing_quality * 0.004)
        self._increase_skill(author, "inscribe", feedback_gain, transfer=0.08)
        author.learn_signal_value(token, gain * 6.0 + writing_quality)
        author.record_success("knowledge_transmitted", gain * 10.0)
        self.mark_author_feedbacks[affordance] += gain
        self.observer.observe(
            self.tick,
            "mark_author_feedback",
            {
                "organism_id": author.id,
                "place": place_id,
                "token": token,
                "affordance": affordance,
                "gain": gain,
                "fidelity": fidelity,
                "writing_quality": writing_quality,
                "reads": reads,
                "skill_gain": feedback_gain,
            },
            subjects=self._subjects(author, place_id, [f"affordance:{affordance}", f"mark_token:{token}"]),
            score=writing_quality * 0.65 + gain * 12.0 + min(0.35, reads * 0.035),
            rarity_key=f"mark_author_feedback:{affordance}",
        )

    def _observe_others(self, individual: Individual, feedback: dict[str, float]) -> None:
        demos = self.demonstrations.get(individual.location, [])
        read_mark = self._read_mark_trace(individual, feedback)
        if not demos:
            if not read_mark:
                individual.energy -= 0.015
            return
        source_id, affordance, success = self.rng.choice(demos)
        if source_id == individual.id:
            return
        planning = self._interaction_control(individual)
        gain = (0.012 if success else 0.004) * (0.5 + individual.params.sensor_range + planning * 1.10)
        self._increase_skill(individual, affordance, gain, transfer=0.10)
        individual.last_tool_affordance = affordance
        individual.record_success("social_learning", 0.2 if success else 0.05)
        feedback["social"] += 0.2 if success else 0.05
        feedback["tool"] = feedback.get("tool", 0.0) + (0.10 if success else 0.03)
        individual.energy -= 0.018

    def _remember_place(self, individual: Individual) -> None:
        if individual.params.memory_budget <= 0.0:
            return
        place = self.world.places[individual.location]
        value = (
            place.total_accessible_energy() / 420.0
            + place.sealed_essence / 500.0
            + place.physics.get("shelter", 0.0) * 0.08
            + place.physics.get("interiority", 0.0) * place.physics.get("boundary_permeability", 0.0) * 0.04
        )
        old = individual.place_memory.get(place.id, 0.0)
        individual.place_memory[place.id] = old * 0.90 + value * 0.10
        limit = max(2, int(individual.params.memory_budget * 3))
        if len(individual.place_memory) > limit:
            weakest = min(individual.place_memory, key=individual.place_memory.get)
            del individual.place_memory[weakest]

    def _deactivate(self, individual: Individual, cause: str) -> None:
        if not individual.alive:
            return
        individual.alive = False
        self.living_total = max(0, self.living_total - 1)
        self.living_by_kind[individual.kind] = max(0, self.living_by_kind[individual.kind] - 1)
        if individual.neural:
            self.living_neural = max(0, self.living_neural - 1)
        self.deaths_by_cause[cause] += 1
        self.deaths_by_kind_cause[f"{individual.kind}:{cause}"] += 1
        place = self.world.places[individual.location]
        place.resources["organic_store"] = min(180.0, place.resources["organic_store"] + max(0.0, individual.energy) * 0.35 + 2.0)
        if cause in {"predation", "starvation"}:
            place.materials["bone"] = min(99, place.materials.get("bone", 0) + 1)
        checkpoint_score = self._checkpoint_score(individual) if individual.controller is not None else 0.0
        notable = (
            individual.offspring_count >= 3
            or individual.successful_tools >= 2
            or individual.success_profile.get("causal_unlock", 0.0) > 0.0
            or individual.success_profile.get("prediction_fit", 0.0) >= 2.0
            or individual.success_profile.get("written_learning", 0.0) > 0.0
            or individual.success_profile.get("knowledge_transmitted", 0.0) > 0.0
        )
        if individual.controller is not None and (
            notable
        ):
            self.checkpoints.save_brain(
                self.tick,
                individual,
                f"death_{cause}",
                {"place": place.to_summary()},
                bucket="notable_death",
                score=checkpoint_score,
            )
        if notable:
            self.observer.observe(
                self.tick,
                "notable_death",
                {
                    "organism_id": individual.id,
                    "kind": individual.kind,
                    "cause": cause,
                    "place": place.id,
                    "lineage_root_id": individual.lineage_root_id,
                    "offspring_count": individual.offspring_count,
                    "successful_tools": individual.successful_tools,
                    "score": checkpoint_score,
                    "success_profile": dict(individual.success_profile),
                },
                subjects=self._subjects(individual, place.id, [f"cause:{cause}"]),
                score=0.75 + min(4.0, checkpoint_score / 8.0),
                rarity_key=f"death:{cause}",
            )
        if self.config.event_detail:
            self.logger.event(self.tick, "death", {"organism_id": individual.id, "cause": cause, "kind": individual.kind})
        individual.controller = None
        individual.controller_template = None
        individual.inventory.clear()
        individual.artifacts.clear()

    def _apply_interventions(self) -> None:
        if self.config.run_mode != "garden":
            return
        interventions = self.interventions.get(self.tick, [])
        for intervention in interventions:
            self._apply_intervention(intervention)

    def _apply_intervention(self, intervention: Intervention) -> None:
        payload = intervention.payload
        if intervention.kind == "add_resource":
            place = self.world.places[int(payload.get("place", 0)) % len(self.world.places)]
            energy = str(payload.get("energy", "essence"))
            amount = float(payload.get("amount", 10.0))
            if energy in place.resources:
                place.resources[energy] = min(180.0, place.resources[energy] + amount)
        elif intervention.kind == "disaster":
            places = self.world.places if payload.get("place", "all") == "all" else [self.world.places[int(payload.get("place", 0)) % len(self.world.places)]]
            resource_loss = float(payload.get("resource_loss", 0.25))
            damage = float(payload.get("damage", 0.10))
            affected_ids = {place.id for place in places}
            for place in places:
                for key in place.resources:
                    place.resources[key] *= max(0.0, 1.0 - resource_loss)
            for individual in self.organisms.values():
                if individual.alive and individual.location in affected_ids:
                    individual.health -= damage
                    if individual.health <= 0.0:
                        self._deactivate(individual, "intervention_disaster")
        elif intervention.kind == "climate_shift":
            self.world.climate_drift = max(-0.5, min(0.5, self.world.climate_drift + float(payload.get("amount", 0.0))))
        elif intervention.kind == "add_organisms":
            kind = str(payload.get("kind", "plant"))
            count = int(payload.get("count", 1))
            place = int(payload.get("place", self.rng.randrange(len(self.world.places))))
            for _ in range(count):
                params = ParamVector.neural(self.rng) if kind == "agent" else ParamVector.fungus(self.rng) if kind == "fungus" else ParamVector.plant(self.rng)
                self.add_individual(kind, params, place, float(payload.get("energy", 25.0)))
        record = {"tick": self.tick, "kind": intervention.kind, "payload": payload, "reason": intervention.reason}
        self.interventions_applied.append(record)
        self.logger.event(self.tick, "intervention", record)

    def _checkpoint_score(self, individual: Individual) -> float:
        energy_ratio = individual.energy / max(1.0, individual.storage_limit())
        profile = individual.success_profile
        profile_score = (
            math.log1p(profile.get("energy_gain", 0.0)) * 0.9
            + math.log1p(profile.get("prediction_fit", 0.0)) * 1.3
            + math.log1p(profile.get("tool_make", 0.0)) * 1.2
            + math.log1p(profile.get("tool_use", 0.0)) * 1.4
            + math.log1p(profile.get("structure", 0.0)) * 1.5
            + math.log1p(profile.get("causal_step", 0.0)) * 1.2
            + math.log1p(profile.get("causal_unlock", 0.0)) * 2.4
            + math.log1p(profile.get("social_learning", 0.0)) * 1.0
            + math.log1p(profile.get("written_learning", 0.0)) * 1.1
            + math.log1p(profile.get("knowledge_transmitted", 0.0)) * 1.0
            + math.log1p(profile.get("reproduction", 0.0)) * 1.6
        )
        return (
            individual.offspring_count * 6.0
            + individual.successful_tools * 2.0
            + individual.cycle * 0.75
            + individual.age / 450.0
            + energy_ratio * 2.0
            + individual.params.complexity() * 0.5
            + profile_score
        )

    def _checkpoint_context(self, label: str, criterion: str) -> dict[str, Any]:
        return {
            "label": label,
            "criterion": criterion,
            "population": population_counts(self.organisms),
            "world_energy": world_energy_summary(self.world),
            "world_physics": world_physics_summary(self.world),
            "lineages": self._lineage_summary(limit=5),
        }

    def _save_checkpoint_candidate(self, individual: Individual, label: str, criterion: str, bucket: str) -> None:
        score = self._checkpoint_score(individual)
        saved = self.checkpoints.save_brain(
            self.tick,
            individual,
            f"{label}_{criterion}",
            self._checkpoint_context(label, criterion),
            bucket=bucket,
            score=score,
        )
        if saved:
            self.observer.observe(
                self.tick,
                "checkpoint_saved",
                {
                    "organism_id": individual.id,
                    "place": individual.location,
                    "criterion": criterion,
                    "bucket": bucket,
                    "score": score,
                },
                subjects=self._subjects(individual, extra=[f"checkpoint:{bucket}"]),
                score=1.1 + min(3.0, score / 10.0),
                rarity_key=f"checkpoint:{bucket}:{criterion}",
            )

    def _best_checkpoint_candidate(self, candidates: list[Individual], excluded_ids: set[int], key: Any) -> Individual | None:
        available = [individual for individual in candidates if individual.id not in excluded_ids]
        if not available:
            return None
        return max(available, key=key)

    def _checkpoint_champions(self, label: str) -> None:
        candidates = [individual for individual in self.organisms.values() if individual.alive and individual.controller is not None]
        if not candidates:
            return
        saved_ids: set[int] = set()

        overall = max(candidates, key=self._checkpoint_score)
        self._save_checkpoint_candidate(overall, label, "overall_champion", "interval_champion")
        saved_ids.add(overall.id)

        reproductive = self._best_checkpoint_candidate(candidates, saved_ids, lambda individual: (individual.offspring_count, individual.cycle, individual.energy, individual.age))
        if reproductive is not None and reproductive.offspring_count > 0:
            self._save_checkpoint_candidate(reproductive, label, "reproductive_champion", "reproductive_champion")
            saved_ids.add(reproductive.id)

        tool_user = self._best_checkpoint_candidate(candidates, saved_ids, lambda individual: (individual.successful_tools, individual.offspring_count, individual.energy, individual.age))
        if tool_user is not None and tool_user.successful_tools > 0:
            self._save_checkpoint_candidate(tool_user, label, "tool_champion", "tool_champion")
            saved_ids.add(tool_user.id)

        causal = self._best_checkpoint_candidate(
            candidates,
            saved_ids,
            lambda individual: (
                individual.success_profile.get("causal_unlock", 0.0),
                individual.success_profile.get("causal_step", 0.0),
                individual.successful_tools,
                individual.energy,
            ),
        )
        if causal is not None and causal.success_profile.get("causal_step", 0.0) > 0.0:
            self._save_checkpoint_candidate(causal, label, "causal_champion", "causal_champion")
            saved_ids.add(causal.id)

        learner = self._best_checkpoint_candidate(
            candidates,
            saved_ids,
            lambda individual: (
                individual.success_profile.get("prediction_fit", 0.0),
                -sum(abs(value) for value in individual.prediction_error_profile),
                individual.age,
                individual.energy,
            ),
        )
        if learner is not None and learner.success_profile.get("prediction_fit", 0.0) > 0.0:
            self._save_checkpoint_candidate(learner, label, "learner_champion", "learner_champion")
            saved_ids.add(learner.id)

        lineage = self._best_checkpoint_candidate(candidates, saved_ids, lambda individual: (individual.cycle, individual.offspring_count, individual.energy, individual.age))
        if lineage is not None and (lineage.cycle > 0 or lineage.offspring_count > 0):
            self._save_checkpoint_candidate(lineage, label, "lineage_founder", "lineage_founder")

    def _lineage_summary(self, limit: int = 8) -> dict[str, Any]:
        profile_keys = (
            "energy_gain",
            "prediction_fit",
            "tool_make",
            "tool_use",
            "structure",
            "causal_step",
            "causal_unlock",
            "collaboration",
            "social_learning",
            "written_learning",
            "knowledge_transmitted",
            "reproduction",
        )
        rows: dict[int, dict[str, Any]] = {}
        members: dict[int, list[Individual]] = defaultdict(list)
        for individual in self.organisms.values():
            if individual.kind != "agent":
                continue
            root = individual.lineage_root_id or individual.id
            if root not in rows:
                rows[root] = {
                    "lineage_root_id": root,
                    "born": 0,
                    "living": 0,
                    "living_neural": 0,
                    "dead": 0,
                    "max_generation": 0,
                    "offspring_total": 0,
                    "successful_tools_total": 0,
                    "tool_users": 0,
                    "inherited_template_count": 0,
                    "profile": Counter(),
                    "tool_use_counts": Counter(),
                    "energy_total": 0.0,
                    "health_total": 0.0,
                }
            row = rows[root]
            row["born"] += 1
            row["max_generation"] = max(row["max_generation"], individual.cycle)
            row["offspring_total"] += individual.offspring_count
            row["successful_tools_total"] += individual.successful_tools
            row["tool_users"] += int(individual.successful_tools > 0 or any(count > 0 for count in individual.tool_use_counts.values()))
            row["inherited_template_count"] += int(individual.inherited_brain_template)
            for key in profile_keys:
                row["profile"][key] += individual.success_profile.get(key, 0.0)
            for key, value in individual.tool_use_counts.items():
                row["tool_use_counts"][key] += value
            if individual.alive:
                row["living"] += 1
                row["living_neural"] += int(individual.neural)
                row["energy_total"] += individual.energy
                row["health_total"] += individual.health
            else:
                row["dead"] += 1
            members[root].append(individual)

        def member_score(individual: Individual) -> float:
            return (
                individual.offspring_count * 4.0
                + individual.successful_tools * 1.8
                + individual.success_profile.get("prediction_fit", 0.0) * 1.1
                + individual.success_profile.get("collaboration", 0.0) * 0.8
                + individual.success_profile.get("tool_use", 0.0) * 1.0
                + individual.success_profile.get("structure", 0.0) * 1.0
                + individual.success_profile.get("reproduction", 0.0) * 1.2
                + individual.energy / max(1.0, individual.storage_limit())
            )

        summaries: list[dict[str, Any]] = []
        for root, row in rows.items():
            living_members = [individual for individual in members[root] if individual.alive]
            top_members = sorted(living_members, key=member_score, reverse=True)[:5]
            living = int(row["living"])
            profile = row["profile"]
            lineage_score = (
                living * 6.0
                + int(row["max_generation"]) * 1.8
                + int(row["offspring_total"]) * 1.2
                + int(row["successful_tools_total"]) * 1.6
                + float(profile.get("prediction_fit", 0.0)) * 0.9
                + float(profile.get("collaboration", 0.0)) * 0.7
                + float(profile.get("tool_use", 0.0)) * 1.0
                + float(profile.get("structure", 0.0)) * 1.0
            )
            summaries.append(
                {
                    "lineage_root_id": root,
                    "born": int(row["born"]),
                    "living": living,
                    "living_neural": int(row["living_neural"]),
                    "dead": int(row["dead"]),
                    "max_generation": int(row["max_generation"]),
                    "offspring_total": int(row["offspring_total"]),
                    "successful_tools_total": int(row["successful_tools_total"]),
                    "tool_users": int(row["tool_users"]),
                    "inherited_template_count": int(row["inherited_template_count"]),
                    "avg_living_energy": round(float(row["energy_total"]) / max(1, living), 5),
                    "avg_living_health": round(float(row["health_total"]) / max(1, living), 5),
                    "score": round(lineage_score, 5),
                    "top_living_ids": [individual.id for individual in top_members],
                    "profile": {key: round(value, 5) for key, value in sorted(profile.items()) if value > 0.0},
                    "tool_use_counts": dict(row["tool_use_counts"].most_common(8)),
                }
            )
        top_living = sorted(
            (row for row in summaries if row["living"] > 0),
            key=lambda row: (row["living"], row["max_generation"], row["offspring_total"], row["successful_tools_total"], row["score"]),
            reverse=True,
        )[:limit]
        top_all_time = sorted(summaries, key=lambda row: row["score"], reverse=True)[:limit]
        return {
            "agent_lineages_total": len(summaries),
            "living_agent_lineages": sum(1 for row in summaries if row["living"] > 0),
            "top_living": top_living,
            "top_all_time": top_all_time,
        }

    def _log_aggregate(self) -> None:
        living = [individual for individual in self.organisms.values() if individual.alive]
        neural = [individual for individual in living if individual.neural]
        avg_energy = sum(individual.energy for individual in living) / max(1, len(living))
        avg_complexity = sum(individual.params.complexity() for individual in living) / max(1, len(living))
        # Controller capacity & attention stats — these only make sense for neural agents
        # with controllers. With controller growth active and neuroplastic attention active,
        # tracking how these distributions optimize over the run is what tells you
        # whether the substrate is selecting for richer cognition.
        brain_sizes = [individual.controller.hidden_size for individual in neural if individual.controller is not None]
        if brain_sizes:
            brain_capacity = {
                "count": len(brain_sizes),
                "mean": round(sum(brain_sizes) / len(brain_sizes), 3),
                "max": max(brain_sizes),
                "min": min(brain_sizes),
                "p90": sorted(brain_sizes)[int(len(brain_sizes) * 0.9)] if len(brain_sizes) >= 10 else max(brain_sizes),
            }
        else:
            brain_capacity = {"count": 0, "mean": 0.0, "max": 0, "min": 0, "p90": 0}
        # Architecture census: the signal a structural-evolution run is FOR.
        # Tracks how many lineages run modular controllers, how many blocks
        # they carry, and how much capacity sits behind nonzero gates.
        modular = [
            individual.controller
            for individual in neural
            if individual.controller is not None and hasattr(individual.controller, "blocks")
        ]
        if modular:
            block_counts = [len(c.blocks) for c in modular]
            active_blocks = [sum(1 for b in c.blocks if b.out_gate != 0.0) for c in modular]
            gates = [c.neuromodulation() for c in modular]
            plasticity = [
                sum(b.plasticity_scale for b in c.blocks) / len(c.blocks) for c in modular
            ]
            architecture_stats = {
                "modular": len(modular),
                "legacy": len(brain_sizes) - len(modular),
                "blocks_mean": round(sum(block_counts) / len(block_counts), 3),
                "blocks_max": max(block_counts),
                "active_blocks_mean": round(sum(active_blocks) / len(active_blocks), 3),
                "capacity_mean": round(sum(c.capacity for c in modular) / len(modular), 3),
                "capacity_max": max(c.capacity for c in modular),
                # Learning-gate evolution: drift away from the neutral 1.0
                # means context-dependent learning is being selected.
                "neuromod_mean": round(sum(gates) / len(gates), 4),
                "neuromod_min": round(min(gates), 4),
                "neuromod_max": round(max(gates), 4),
                "plasticity_scale_mean": round(sum(plasticity) / len(plasticity), 4),
            }
        else:
            architecture_stats = {"modular": 0, "legacy": len(brain_sizes)}
        attention_stats: dict[str, float | int] = {"count": 0}
        attended_brains = [
            individual.controller
            for individual in neural
            if individual.controller is not None and individual.controller._has_attention() and individual.controller.last_attention.size
        ]
        if attended_brains:
            # Concentration measure: max(fidelity) - mean(fidelity). High when controller
            # is focusing on a few features, low when spread uniformly. Good signal
            # for "how trained" the attention head is on this individual.
            concentrations: list[float] = []
            mean_max_fidelity = 0.0
            for controller in attended_brains:
                fidelity = controller.last_attention
                concentrations.append(float(fidelity.max() - fidelity.mean()))
                mean_max_fidelity += float(fidelity.max())
            attention_stats = {
                "count": len(attended_brains),
                "mean_max_fidelity": round(mean_max_fidelity / len(attended_brains), 4),
                "mean_concentration": round(sum(concentrations) / len(concentrations), 4),
                "max_concentration": round(max(concentrations), 4),
            }
        aggregate = {
            "tick": self.tick,
            "population": population_counts(self.organisms),
            "avg_energy": round(avg_energy, 5),
            "avg_complexity": round(avg_complexity, 5),
            "neural_avg_energy": round(sum(o.energy for o in neural) / max(1, len(neural)), 5),
            "brain_capacity": brain_capacity,
            "architecture": architecture_stats,
            "attention_stats": attention_stats,
            "births": dict(self.births_by_mode),
            "deaths": dict(self.deaths_by_cause),
            "deaths_by_kind_cause": dict(self.deaths_by_kind_cause),
            "tool_successes": dict(self.tool_successes),
            "causal_steps": dict(self.causal_steps),
            "causal_unlocks": dict(self.causal_unlocks),
            "collaboration_events": dict(self.collaboration_events),
            "patch_recovery_triggers": self.patch_recovery_triggers,
            "movement": self._movement_summary(),
            "success_profile": success_profile_summary(self.organisms),
            "lineages": self._lineage_summary(),
            "marks_created": dict(self.marks_created),
            "mark_lessons": dict(self.mark_lessons),
            "mark_lesson_packets": dict(self.mark_lesson_packets),
            "mark_read_value": {key: round(value, 5) for key, value in self.mark_read_value.items()},
            "mark_author_feedbacks": {key: round(value, 5) for key, value in self.mark_author_feedbacks.items()},
            "portable_marks_created": dict(self.portable_marks_created),
            "portable_mark_reads": dict(self.portable_mark_reads),
            "artifacts_created": dict(self.artifacts_created),
            "artifacts_broken": dict(self.artifacts_broken),
            "structures_built": dict(self.structures_built),
            "structures_extended": dict(self.structures_extended),
            "physics_events": dict(self.physics_events),
            "reproduction_attempts": dict(self.reproduction_attempts),
            "reproduction_failures": dict(self.reproduction_failures),
            "action_counts": dict(self.action_counts),
            "action_energy_delta": {key: round(value, 5) for key, value in self.action_energy_delta.items()},
            "action_avg_energy_delta": {
                key: round(self.action_energy_delta[key] / max(1, self.action_counts[key]), 5)
                for key in self.action_counts
            },
            "world_energy": world_energy_summary(self.world),
            "world_physics": world_physics_summary(self.world),
            "observer": self.observer.to_summary(),
        }
        self.aggregate_history.append(aggregate)
        if len(self.aggregate_history) > 500:
            self.aggregate_history = self.aggregate_history[-500:]
        self.logger.event(self.tick, "aggregate", aggregate)
        self.logger.flush()
