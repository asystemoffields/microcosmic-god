from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from random import Random
from typing import Any

from .brain import TinyBrain
from .checkpoints import CheckpointManager
from .config import RunConfig
from .debrief import build_debrief, population_counts, success_profile_summary, world_energy_summary, world_physics_summary
from .energy import (
    AFFORDANCES,
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
from .evolution import EvolutionEngine, OffspringPlan
from .genome import MEMORY_BUDGET_MAX, NEURAL_BUDGET_MAX, Genome
from .interventions import Intervention, load_interventions
from .observer import EventObserver
from .organisms import ACTIONS, ACTION_INDEX, OBSERVATION_SIZE, Organism, organism_from_genome
from .runlog import RunLogger
from .world import Place, World


class Simulation:
    def __init__(self, config: RunConfig):
        self.config = config
        self.rng = Random(config.seed)
        self.world = World.generate(self.rng, config)
        self.logger = RunLogger(config)
        self.observer = EventObserver(self.logger)
        self.checkpoints = CheckpointManager(self.logger.checkpoint_dir, config.neural_checkpoint_limit)
        self.evolution = EvolutionEngine(self.rng, config)
        self.interventions = load_interventions(config.interventions_path) if config.run_mode == "garden" else {}
        self.organisms: dict[int, Organism] = {}
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
        self._seed_initial_life()

    def _seed_initial_life(self) -> None:
        for _ in range(self.config.initial_plants):
            self.add_organism("plant", Genome.plant(self.rng), self.rng.randrange(len(self.world.places)), self.rng.uniform(10.0, 35.0))
        for _ in range(self.config.initial_fungi):
            self.add_organism("fungus", Genome.fungus(self.rng), self.rng.randrange(len(self.world.places)), self.rng.uniform(8.0, 28.0))
        for _ in range(self.config.initial_agents):
            self.add_organism("agent", Genome.neural(self.rng), self.rng.randrange(len(self.world.places)), self.rng.uniform(22.0, 55.0))
        self.logger.event(0, "seeded", {"population": population_counts(self.organisms)})

    def add_organism(
        self,
        kind: str,
        genome: Genome,
        location: int,
        energy: float,
        generation: int = 0,
        parent_ids: tuple[int, ...] = (),
        brain_template: TinyBrain | None = None,
    ) -> Organism | None:
        if self.living_total >= self.config.max_population:
            return None
        if kind != "agent":
            genome.neural_budget = 0.0
            genome.memory_budget = 0.0
            brain_template = None
        organism = organism_from_genome(
            self.rng,
            id_=self.next_id,
            kind=kind,
            genome=genome,
            location=location % len(self.world.places),
            energy=max(0.1, energy),
            generation=generation,
            parent_ids=parent_ids,
            brain_template=brain_template,
        )
        self.organisms[organism.id] = organism
        self.next_id += 1
        self.living_total += 1
        self.living_by_kind[organism.kind] += 1
        if organism.neural:
            self.living_neural += 1
        return organism

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
        physics_events = self.world.update_environment(self.rng)
        self.physics_events.update(physics_events)
        self._apply_physics_transport()
        self._apply_interventions()
        self.demonstrations.clear()

        # Perception is a tick-start snapshot; action effects below resolve sequentially
        # against current organism locations, births, and deaths.
        rosters = self._rosters()
        contexts: dict[int, tuple[list[float], int, float, float, list[int]]] = {}
        intents: dict[int, str] = {}
        feedback: dict[int, dict[str, float]] = defaultdict(lambda: {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        for organism in list(self.organisms.values()):
            if not organism.alive:
                continue
            self._metabolize(organism)
            self._habitat_stress(organism)
            if not organism.alive:
                continue
            observed_tokens = self._observed_tokens(organism)
            observation = self._observe(organism, rosters)
            action = self._choose_action(organism, observation)
            intents[organism.id] = action
            contexts[organism.id] = (observation, ACTION_INDEX[action], organism.energy, organism.health, observed_tokens)

        active_recombine_places: set[int] = set()
        for organism_id, action in list(intents.items()):
            organism = self.organisms.get(organism_id)
            if organism is None or not organism.alive:
                continue
            if action == "coordinate":
                self._coordinate_recombine(organism, feedback[organism_id])
                active_recombine_places.add(organism.location)
                continue
            self._resolve_action(organism, action, feedback[organism_id])

        for organism in self.organisms.values():
            if organism.alive and organism.recombine_intent_until >= self.tick:
                active_recombine_places.add(organism.location)
        self._resolve_recombine(active_recombine_places, feedback)

        for organism_id, context in contexts.items():
            organism = self.organisms.get(organism_id)
            if organism is None:
                continue
            observation, action_index, before_energy, before_health, observed_tokens = context
            action_name = ACTIONS[action_index]
            energy_delta = organism.energy - before_energy
            self.action_counts[action_name] += 1
            self.action_energy_delta[action_name] += energy_delta
            if not organism.alive or organism.brain is None:
                continue
            health_delta = organism.health - before_health
            damage = max(0.0, -health_delta)
            extra = feedback[organism_id]
            movement_hazard = damage * 4.0 if action_name == "move" else 0.0
            valence = (
                organism.genome.valence_energy * (energy_delta / 10.0)
                + organism.genome.valence_health * (health_delta * 4.0)
                - organism.genome.valence_damage * (damage * 4.0)
                + organism.genome.valence_reproduction * extra.get("reproduction", 0.0)
                + organism.genome.valence_social * extra.get("social", 0.0)
            )
            prediction_error = organism.brain.learn(
                action_index=action_index,
                valence=valence,
                energy_delta=energy_delta / 10.0,
                learning_rate=organism.genome.learning_rate,
                plasticity=organism.genome.plasticity_rate,
                prediction_weight=organism.genome.prediction_weight,
                outcome_targets={
                    "damage": damage * 4.0,
                    "reproduction": extra.get("reproduction", 0.0),
                    "social": extra.get("social", 0.0),
                    "tool": extra.get("tool", 0.0),
                    "hazard": movement_hazard,
                },
            )
            organism.last_valence = valence
            organism.record_action_result(
                action_index=action_index,
                energy_delta=energy_delta,
                health_delta=health_delta,
                damage=damage,
                prediction_error=prediction_error,
                reproduction_feedback=extra.get("reproduction", 0.0),
                social_feedback=extra.get("social", 0.0),
                tool_feedback=extra.get("tool", 0.0),
                prediction_errors=organism.brain.last_prediction_errors,
            )
            for token in observed_tokens:
                organism.learn_signal_value(token, valence + prediction_error * 0.05)
            self._remember_place(organism)

        for organism in list(self.organisms.values()):
            if organism.alive:
                organism.repair_or_decay()
                self._age_risk(organism)

        if self.tick % self.config.log_every == 0:
            self._log_aggregate()
        if self.tick % self.config.checkpoint_every == 0:
            self._checkpoint_champions("interval")

    def _rosters(self) -> dict[int, list[int]]:
        rosters: dict[int, list[int]] = {place.id: [] for place in self.world.places}
        for organism in self.organisms.values():
            if organism.alive:
                rosters[organism.location].append(organism.id)
        return rosters

    def _subjects(self, organism: Organism | None = None, place_id: int | None = None, extra: list[str] | None = None) -> list[str]:
        subjects: list[str] = []
        if organism is not None:
            subjects.append(f"organism:{organism.id}")
            subjects.append(f"place:{organism.location}")
        if place_id is not None:
            subject = f"place:{place_id}"
            if subject not in subjects:
                subjects.append(subject)
        if extra:
            subjects.extend(extra)
        return subjects

    def _living_ids_at(self, place_id: int) -> list[int]:
        return [organism.id for organism in self.organisms.values() if organism.alive and organism.location == place_id]

    def _fast_counts(self) -> dict[str, int]:
        counts = {kind: count for kind, count in self.living_by_kind.items() if count > 0}
        counts["neural"] = self.living_neural
        counts["total"] = self.living_total
        return counts

    def _interaction_control(self, organism: Organism) -> float:
        if organism.brain is None:
            return 0.0
        has_history = (
            abs(organism.last_energy_delta)
            + abs(organism.recent_health_delta)
            + sum(abs(value) for value in organism.event_memory)
        ) > 0.0001
        average_error = sum(abs(value) for value in organism.prediction_error_profile) / max(1, len(organism.prediction_error_profile))
        prediction_fit = max(0.0, 1.0 - average_error / 1.5) if has_history else 0.0
        memory_signal = min(1.0, sum(abs(value) for value in organism.event_memory) / max(1.0, len(organism.event_memory) * 0.35))
        learned_skill = max(organism.tool_skill.values(), default=0.0)
        place_knowledge = min(1.0, len(organism.place_memory) / max(1.0, organism.genome.memory_budget * 2.0))
        return max(0.0, min(1.0, prediction_fit * 0.38 + memory_signal * 0.24 + learned_skill * 0.24 + place_knowledge * 0.14))

    def _salient_problem(self, organism: Organism, place: Place, target_affordance: str = "") -> dict[str, Any]:
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
            }

        obstacle_options = [
            (place.obstacles.get("height", 0.0), "height", "lever", "traverse"),
            (place.obstacles.get("water", 0.0) + place.physics.get("current_exposure", 0.0) * 0.35, "water", "contain", "float"),
            (place.obstacles.get("thorn", 0.0), "thorn", "cut", "cut"),
            (place.obstacles.get("heat", 0.0), "heat", "concentrate_heat", "insulate"),
            (place.physics.get("salinity", 0.0), "salinity", "filter", "filter"),
            (place.physics.get("pressure", 0.0), "pressure", "contain", "anchor"),
        ]
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
            (place.locked_chemical / 180.0, "locked_chemical", "crack", "chemical"),
            (place.resources.get("mechanical", 0.0) / 180.0 + place.physics.get("current_exposure", 0.0) * 0.18, "mechanical_gradient", "contain", "mechanical"),
            (place.resources.get("electrical", 0.0) / 140.0 + place.resources.get("high_density", 0.0) / 80.0, "electrical_source", "conduct", "electrical"),
            (place.resources.get("radiant", 0.0) / 220.0 + place.resources.get("thermal", 0.0) / 240.0, "heat_source", "concentrate_heat", "thermal"),
            (place.resources.get("chemical", 0.0) / 220.0 + place.resources.get("biological_storage", 0.0) / 220.0, "surface_food", "filter", "chemical"),
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
        organism: Organism,
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
        if affordance not in organism.tool_skill and affordance not in {"build", "craft"}:
            return
        lesson: dict[str, Any] = {
            "kind": kind,
            "place": place.id,
            "affordance": affordance,
            "success": bool(success),
            "gain": gain,
            "score": score,
            "method_quality": method_quality,
            "skill": organism.tool_skill.get(affordance, 0.0),
            "problem": self._salient_problem(organism, place, affordance),
        }
        if components:
            lesson["components"] = {name: int(qty) for name, qty in components.items() if qty > 0}
        if sequence:
            lesson["sequence"] = list(sequence)
        organism.record_lesson(lesson)

    def _apply_physics_transport(self) -> None:
        for organism in list(self.organisms.values()):
            if not organism.alive:
                continue
            place = self.world.places[organism.location]
            physics = place.physics
            fluid = physics.get("fluid_level", 0.0)
            current = physics.get("current_exposure", 0.0)
            downstream = self.world.downstream_neighbor(place.id)
            if downstream and fluid > 0.35 and current > 0.08:
                float_cap = artifact_capability(organism.artifacts, "float")
                anchor = artifact_capability(organism.artifacts, "anchor")
                traverse = artifact_capability(organism.artifacts, "traverse")
                structure_anchor = structure_capability(place.structures, "anchor")
                structure_support = structure_capability(place.structures, "support")
                structure_shelter = structure_capability(place.structures, "shelter")
                resistance = max(
                    organism.genome.aquatic_affinity * 0.65 + organism.genome.buoyancy * 0.35,
                    organism.genome.mobility * 0.30,
                    float_cap * 0.75,
                    anchor * 0.80,
                    traverse * 0.55,
                    structure_anchor * 0.45,
                    structure_support * 0.30,
                    structure_shelter * 0.55,
                )
                drift_chance = max(0.0, downstream[1] * fluid * (1.0 - resistance)) * 0.020
                if self.rng.random() < drift_chance:
                    organism.location = downstream[0]
                    organism.energy -= 0.010 + current * 0.012
                    if organism.genome.aquatic_affinity < fluid * 0.45 and float_cap < 0.20:
                        organism.health -= fluid * 0.006
                    self.physics_events["current_transport"] += 1
                    if organism.health <= 0.0:
                        self._kill(organism, "current_washout")
                        continue

            steep_edges = [
                edge
                for edge in self.world.edges_from(place.id)
                if edge.slope_from(place.id) < -0.35 and edge.danger > 0.10
            ]
            if steep_edges and organism.genome.mobility < 0.65:
                edge = min(steep_edges, key=lambda item: item.slope_from(place.id))
                traverse = artifact_capability(organism.artifacts, "traverse")
                anchor = artifact_capability(organism.artifacts, "anchor")
                structure_support = structure_capability(place.structures, "support")
                structure_anchor = structure_capability(place.structures, "anchor")
                footing = max(organism.genome.mobility, traverse * 0.65, anchor * 0.75, structure_support * 0.35, structure_anchor * 0.40)
                fall_chance = max(0.0, abs(edge.slope_from(place.id)) * edge.danger * (1.0 - footing)) * 0.004
                if self.rng.random() < fall_chance:
                    organism.location = edge.other(place.id)
                    damage = max(0.0, abs(edge.slope_from(place.id)) - footing) * 0.035
                    organism.health -= damage
                    organism.energy -= 0.025
                    self.physics_events["gravity_fall"] += 1
                    if organism.health <= 0.0:
                        self._kill(organism, "fall")

    def _metabolize(self, organism: Organism) -> None:
        organism.age += 1
        organism.energy -= organism.metabolic_cost()
        if organism.energy < 0.0:
            organism.health += organism.energy * 0.030
            organism.energy = 0.0
        if organism.health <= 0.0:
            self._kill(organism, "starvation")

    def _habitat_stress(self, organism: Organism) -> None:
        place = self.world.places[organism.location]
        aquatic = place.habitat.get("aquatic", 0.0)
        depth = place.habitat.get("depth", 0.0)
        physics = place.physics
        salinity = physics.get("salinity", place.habitat.get("salinity", 0.0))
        humidity = physics.get("humidity", place.habitat.get("humidity", 0.5))
        temperature = physics.get("temperature", 0.5)
        pressure = physics.get("pressure", depth)
        current = physics.get("current_exposure", 0.0)
        shelter = max(physics.get("shelter", 0.0), structure_capability(place.structures, "shelter"))
        interiority = max(physics.get("interiority", 0.0), structure_capability(place.structures, "enclose"))
        permeability = max(physics.get("boundary_permeability", 0.0), structure_capability(place.structures, "permeable"))
        insulation = max(artifact_capability(organism.artifacts, "insulate"), shelter * 0.55)
        float_cap = artifact_capability(organism.artifacts, "float")
        anchor = max(artifact_capability(organism.artifacts, "anchor"), structure_capability(place.structures, "anchor") * 0.35)
        traverse = max(artifact_capability(organism.artifacts, "traverse"), structure_capability(place.structures, "support") * 0.25)
        exposure_damping = 1.0 - shelter * 0.35
        drowning = max(0.0, aquatic * depth - organism.genome.aquatic_affinity * 0.85 - organism.genome.mobility * 0.15 - float_cap * 0.20 - shelter * 0.08)
        desiccation = max(0.0, organism.genome.aquatic_affinity * (1.0 - humidity) - organism.genome.desiccation_tolerance * 0.55 - shelter * 0.14)
        salinity_stress = max(0.0, abs(salinity - organism.genome.salinity_tolerance) - 0.55)
        heat_stress = max(0.0, temperature - (0.58 + organism.genome.thermal_tolerance * 0.42 + insulation * 0.25))
        cold_stress = max(0.0, 0.18 - temperature - organism.genome.thermal_tolerance * 0.12 - insulation * 0.18)
        pressure_stress = max(0.0, pressure - (organism.genome.pressure_tolerance * 1.05 + organism.genome.aquatic_affinity * 0.20 + organism.genome.armor * 0.12))
        current_stress = max(0.0, current * aquatic - max(organism.genome.buoyancy, float_cap, anchor * 0.80, traverse * 0.55, organism.genome.mobility * 0.25))
        stagnant_interior = max(0.0, interiority - shelter) * max(0.0, 1.0 - permeability) * max(0.0, pressure + temperature - 0.80)
        stress = (
            drowning * 0.020 * exposure_damping
            + desiccation * 0.015 * exposure_damping
            + salinity_stress * 0.010
            + heat_stress * 0.018
            + cold_stress * 0.012
            + pressure_stress * 0.012
            + current_stress * 0.009 * exposure_damping
            + stagnant_interior * 0.006
        )
        if stress <= 0.0:
            return
        organism.energy -= stress * 2.0
        organism.health -= stress
        if organism.health <= 0.0:
            if pressure_stress > max(drowning, desiccation, salinity_stress, heat_stress, cold_stress):
                self._kill(organism, "pressure_stress")
            elif heat_stress > max(drowning, desiccation, salinity_stress, pressure_stress, cold_stress):
                self._kill(organism, "thermal_stress")
            elif current_stress > max(drowning, desiccation, salinity_stress, pressure_stress, heat_stress):
                self._kill(organism, "current_exposure")
            else:
                self._kill(organism, "habitat_mismatch")

    def _age_risk(self, organism: Organism) -> None:
        life_expectancy = 850 + organism.genome.storage_capacity * 800 + organism.genome.armor * 300
        if organism.age > life_expectancy:
            risk = min(0.030, (organism.age - life_expectancy) / max(1.0, life_expectancy) * 0.020)
            if self.rng.random() < risk:
                self._kill(organism, "senescence")

    def _observe(self, organism: Organism, rosters: dict[int, list[int]]) -> list[float]:
        place = self.world.places[organism.location]
        resources = [place.resources[kind] / 120.0 for kind in ("radiant", "chemical", "biological_storage", "thermal", "mechanical", "electrical", "high_density")]
        local_ids = rosters.get(place.id, [])
        local_neural = sum(1 for oid in local_ids if self.organisms[oid].neural)
        best_skill = max(organism.tool_skill.values()) if organism.tool_skill else 0.0
        season = math.sin(2.0 * math.pi * self.world.tick / max(2, self.world.season_length))
        features = [
            organism.energy / max(1.0, organism.storage_limit()),
            organism.health,
            min(1.0, organism.age / 2_000.0),
            *resources,
            place.locked_chemical / 160.0,
            len(local_ids) / max(1.0, place.capacity),
            local_neural / max(1.0, len(local_ids)),
            organism.inventory_count() / max(1.0, organism.inventory_limit()),
            organism.genome.mobility,
            organism.genome.manipulator,
            organism.genome.armor,
            organism.genome.sensor_range,
            organism.genome.neural_budget / NEURAL_BUDGET_MAX,
            organism.genome.memory_budget / MEMORY_BUDGET_MAX,
            organism.genome.prediction_weight,
            organism.genome.plasticity_rate,
            max(-1.0, min(1.0, organism.last_valence)),
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
            place.physics.get("biological_activity", 0.0),
            place.physics.get("abrasion", 0.0),
            place.physics.get("wet_dry_cycle", 0.0),
            place.physics.get("elevation", 0.5),
            place.habitat.get("aquatic", 0.0),
            place.habitat.get("depth", 0.0),
            place.habitat.get("salinity", 0.0),
            place.habitat.get("humidity", 0.5),
            *organism.recent_trace(),
            *organism.prediction_error_profile,
            *organism.event_memory,
            *organism.signal_values,
        ]
        if len(features) != OBSERVATION_SIZE:
            raise AssertionError(f"observation size drifted to {len(features)}")
        return [max(-1.0, min(1.5, float(value))) for value in features]

    def _observed_tokens(self, organism: Organism) -> list[int]:
        place = self.world.places[organism.location]
        tokens: list[int] = []
        for signal in place.signals:
            if signal.source_id != organism.id:
                tokens.append(signal.token)
        for mark in place.marks[-8:]:
            if mark.source_id != organism.id:
                tokens.append(mark.token)
        return tokens[:4]

    def _choose_action(self, organism: Organism, observation: list[float]) -> str:
        if organism.brain is None:
            non_neural_birth_rate = 0.025 if organism.kind == "plant" else 0.035
            if organism.energy > self.evolution.clone_mutate_reserve_threshold(organism) and self.rng.random() < non_neural_birth_rate:
                return "clone_mutate"
            if organism.kind == "plant":
                return "absorb_radiant"
            if organism.kind == "fungus":
                return "eat" if self.rng.random() < 0.72 else "forage"
            return "rest"

        outputs = organism.brain.forward(observation)
        exploration = 0.025 + organism.genome.plasticity_rate * 0.055 + organism.genome.mutation_rate * 0.25
        if self.rng.random() < exploration:
            return self.rng.choice(ACTIONS)
        energy_ratio = organism.energy / max(1.0, organism.storage_limit())
        if organism.adult() and energy_ratio > 0.62:
            reproductive_drive = organism.genome.valence_reproduction * (energy_ratio - 0.62)
            outputs[ACTION_INDEX["coordinate"]] += reproductive_drive * (0.9 + organism.genome.mate_selectivity)
            outputs[ACTION_INDEX["clone_mutate"]] += reproductive_drive * (0.7 + (1.0 - organism.genome.mate_selectivity) * 0.4)
        ranked = sorted(range(len(outputs)), key=lambda i: outputs[i], reverse=True)
        for index in ranked:
            action = ACTIONS[index]
            if self._action_feasible(organism, action):
                return action
        return "rest"

    def _action_feasible(self, organism: Organism, action: str) -> bool:
        if action in {"coordinate", "clone_mutate"} and not organism.adult():
            return False
        if action == "pickup" and organism.inventory_count() >= organism.inventory_limit():
            return False
        if action == "craft" and (organism.inventory_count() < 2 or len(organism.artifacts) >= organism.artifact_limit()):
            return False
        if action == "build" and (organism.inventory_count() < 3 or organism.genome.manipulator < 0.18):
            return False
        if action == "move" and organism.genome.mobility < 0.05:
            return False
        if action == "use_tool" and organism.inventory_count() == 0 and not organism.artifacts:
            return False
        if action == "mark" and organism.genome.manipulator < 0.12:
            return False
        return True

    def _resolve_action(
        self,
        organism: Organism,
        action: str,
        feedback: dict[str, float],
    ) -> None:
        organism.last_action = action
        if action == "rest":
            organism.energy -= 0.004
            return
        if action == "move":
            self._move(organism)
        elif action == "eat":
            self._eat(organism)
        elif action == "absorb_radiant":
            self._absorb_radiant(organism)
        elif action == "forage":
            self._forage(organism)
        elif action == "pickup":
            self._pickup(organism)
        elif action == "craft":
            self._craft(organism, feedback)
        elif action == "build":
            self._build_structure(organism, feedback)
        elif action == "use_tool":
            self._use_tool(organism, feedback)
        elif action == "attack":
            self._attack(organism)
        elif action == "signal":
            self._signal(organism, feedback)
        elif action == "mark":
            self._mark(organism, feedback)
        elif action == "clone_mutate":
            self._clone_mutate(organism, feedback)
        elif action == "observe":
            self._observe_others(organism, feedback)

    def _move(self, organism: Organism) -> None:
        place = self.world.places[organism.location]
        if not place.neighbors:
            return
        planning = self._interaction_control(organism)
        efficiency = 1.0 - planning * 0.14
        organism.energy -= (0.05 + organism.genome.mobility * 0.08) * efficiency
        if organism.brain and organism.place_memory:
            scored = []
            for neighbor in place.neighbors:
                memory = organism.place_memory.get(neighbor, 0.0)
                neighbor_place = self.world.places[neighbor]
                crowd = len(self._living_ids_at(neighbor)) / max(1.0, neighbor_place.capacity)
                scored.append((memory - crowd * (0.20 + planning * 0.80) + self.rng.random() * 0.08, neighbor))
            destination_id = max(scored)[1]
        else:
            destination_id = self.rng.choice(place.neighbors)
        destination = self.world.places[destination_id]
        edge = self.world.edge_between(place.id, destination_id)
        structure_traverse = max(
            structure_capability(place.structures, "support") * 0.25,
            structure_capability(destination.structures, "support") * 0.18,
            structure_capability(place.structures, "channel") * 0.20,
            structure_capability(destination.structures, "channel") * 0.12,
        )
        traverse = max(artifact_capability(organism.artifacts, "traverse"), structure_traverse)
        insulation = max(artifact_capability(organism.artifacts, "insulate"), structure_capability(destination.structures, "shelter") * 0.35)
        float_cap = artifact_capability(organism.artifacts, "float")
        anchor = max(artifact_capability(organism.artifacts, "anchor"), structure_capability(place.structures, "anchor") * 0.25)
        cut = max(organism.tool_skill.get("cut", 0.0), artifact_capability(organism.artifacts, "cut"))
        aquatic_fit = organism.genome.aquatic_affinity
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
            + destination.obstacles.get("height", 0.0) * (1.0 - max(traverse, organism.genome.mobility))
            + destination.obstacles.get("thorn", 0.0) * (1.0 - max(cut, organism.genome.armor))
            + destination.obstacles.get("heat", 0.0) * (1.0 - max(insulation, organism.genome.thermal_tolerance))
            + edge_required * (1.0 - max(traverse, float_cap, anchor * 0.65, organism.genome.mobility))
            + uphill * (1.0 - max(traverse, organism.genome.mobility))
            + against_current * (1.0 - max(traverse, float_cap, aquatic_fit, anchor * 0.55))
            + downhill * (1.0 - max(traverse, anchor, organism.genome.mobility)) * 0.35
            + boundary * (1.0 - max(traverse, organism.genome.manipulator * 0.35, cut * 0.25))
        ) / 8.35
        success = (
            organism.genome.mobility
            + traverse * 0.65
            + organism.genome.sensor_range * 0.10
            + planning * 0.10
            + with_current * max(float_cap, aquatic_fit) * 0.10
            + anchor * 0.04
            + self.rng.gauss(0.0, 0.04)
        )
        if success >= barrier:
            organism.location = destination_id
            organism.energy -= (barrier * 0.06 + distance * 0.012 + uphill * 0.020) * efficiency
            if with_current > 0.1 and destination.obstacles.get("water", 0.0) > 0.3:
                self.physics_events["current_assisted_move"] += 1
        else:
            organism.energy -= (barrier * 0.18 + distance * 0.016) * efficiency
            organism.health -= max(0.0, barrier - success) * (0.035 + downhill * 0.015) * (1.0 - planning * 0.18)
            if organism.health <= 0.0:
                self._kill(organism, "movement_hazard")

    def _eat(self, organism: Organism) -> None:
        place = self.world.places[organism.location]
        appetite = 2.0 + organism.genome.digestion * 7.0 + organism.genome.chemical_metabolism * 3.0
        chemical = min(place.resources["chemical"], appetite * 0.55)
        place.resources["chemical"] -= chemical
        biological = min(place.resources["biological_storage"], appetite - chemical)
        place.resources["biological_storage"] -= biological
        gain = chemical * organism.genome.chemical_metabolism + biological * (0.45 + organism.genome.digestion * 0.80)
        organism.energy += gain

    def _absorb_radiant(self, organism: Organism) -> None:
        place = self.world.places[organism.location]
        gain = place.resources["radiant"] * 0.018 * organism.genome.radiant_metabolism * (0.2 + organism.genome.photosynthesis_surface)
        thermal_stress = max(0.0, place.resources["thermal"] / 120.0 - organism.genome.thermal_tolerance)
        organism.energy += gain
        organism.health -= thermal_stress * 0.003
        place.resources["biological_storage"] += gain * 0.18
        if organism.health <= 0.0:
            self._kill(organism, "thermal_stress")

    def _forage(self, organism: Organism) -> None:
        place = self.world.places[organism.location]
        planning = self._interaction_control(organism)
        organism.energy -= (0.025 + organism.genome.sensor_range * 0.020) * (1.0 - planning * 0.10)
        if self.rng.random() < 0.18 + organism.genome.sensor_range * 0.45 + planning * 0.09:
            found = self.rng.choice(("chemical", "biological_storage", "mechanical"))
            amount = self.rng.uniform(0.2, 1.6) * (0.5 + organism.genome.sensor_range) * (1.0 + planning * 0.25)
            place.resources[found] = min(180.0, place.resources[found] + amount)
        if self.rng.random() < 0.08 + organism.genome.sensor_range * 0.12 + planning * 0.04:
            material = self.rng.choice(tuple(MATERIALS.keys()))
            place.materials[material] = min(99, place.materials.get(material, 0) + 1)

    def _pickup(self, organism: Organism) -> None:
        if organism.genome.manipulator < 0.08 or organism.inventory_count() >= organism.inventory_limit():
            organism.energy -= 0.025
            return
        place = self.world.places[organism.location]
        available = [name for name, qty in place.materials.items() if qty > 0]
        if not available:
            organism.energy -= 0.015
            return
        material = self.rng.choice(available)
        place.materials[material] -= 1
        organism.inventory[material] = organism.inventory.get(material, 0) + 1
        organism.energy -= 0.035

    def _craft_target_affordance(self, organism: Organism, place: Place, planning: float) -> str:
        challenge = place.causal_challenge
        if challenge is not None:
            expected = challenge.expected_affordance()
            if expected in AFFORDANCES and self.rng.random() < 0.45 + planning * 0.45:
                return expected
        scores = {
            "crack": place.locked_chemical / 180.0 + place.mineral_richness * 0.25,
            "cut": place.obstacles.get("thorn", 0.0) * 0.62 + place.resources["biological_storage"] / 220.0,
            "bind": organism.tool_skill.get("craft", 0.0) * 0.35 + organism.inventory_count() / max(1.0, organism.inventory_limit()) * 0.16,
            "contain": place.obstacles.get("water", 0.0) * 0.30 + place.physics.get("current_exposure", 0.0) * 0.34 + place.resources["mechanical"] / 240.0,
            "concentrate_heat": place.resources["radiant"] / 220.0 + place.resources["thermal"] / 260.0 + place.obstacles.get("heat", 0.0) * 0.12,
            "conduct": place.resources["electrical"] / 160.0 + place.resources["high_density"] / 90.0 + place.mineral_richness * place.geothermal * 0.55,
            "lever": place.locked_chemical / 230.0 + place.obstacles.get("height", 0.0) * 0.40,
            "filter": place.physics.get("current_exposure", 0.0) * 0.36 + place.physics.get("salinity", 0.0) * 0.18 + place.resources["chemical"] / 260.0,
        }
        if organism.last_craft_target in scores:
            scores[organism.last_craft_target] += planning * 0.16
        if organism.last_tool_affordance in scores:
            scores[organism.last_tool_affordance] += planning * 0.10
        return max(scores, key=lambda name: scores[name] + self.rng.random() * (0.18 - planning * 0.12))

    def _select_craft_components(self, organism: Organism, target: str, draws: int, planning: float) -> dict[str, int]:
        components: dict[str, int] = {}
        for _ in range(draws):
            choices = [name for name, qty in organism.inventory.items() if qty > components.get(name, 0)]
            if not choices:
                break
            if self.rng.random() > 0.24 + planning * 0.68 + organism.tool_skill.get("craft", 0.0) * 0.18:
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
        bind_fit = capabilities.get("bind", 0.0)
        durability_fit = properties.get("hard", 0.0) * 0.20 + properties.get("bindable", 0.0) * 0.12 + properties.get("flexible", 0.0) * 0.05
        diversity = len(trial) / max(1.0, sum(trial.values()))
        return target_fit * 0.60 + bind_fit * 0.16 + durability_fit + diversity * planning * 0.08 + self.rng.random() * (0.10 - planning * 0.06)

    def _craft(self, organism: Organism, feedback: dict[str, float]) -> None:
        if organism.genome.manipulator < 0.12 or organism.inventory_count() < 2 or len(organism.artifacts) >= organism.artifact_limit():
            organism.energy -= 0.025
            return
        available = [name for name, qty in organism.inventory.items() if qty > 0]
        if len(available) < 2:
            organism.energy -= 0.020
            return
        planning = self._interaction_control(organism)
        target = self._craft_target_affordance(organism, self.world.places[organism.location], planning)
        draws = min(3 + int((planning + organism.tool_skill.get("craft", 0.0)) > 1.15), organism.inventory_count())
        components = self._select_craft_components(organism, target, draws, planning)
        if sum(components.values()) < 2:
            organism.energy -= 0.020
            return
        component_affordances = derive_affordances(components)
        bind_help = component_affordances.get("bind", 0.0)
        target_fit = derive_artifact_capabilities(component_properties(components)).get(target, 0.0)
        best_skill = max(organism.tool_skill.values(), default=0.0)
        craft_skill = organism.tool_skill.get("craft", 0.0)
        target_skill = organism.tool_skill.get(target, 0.0)
        material_gate = max(0.0, min(1.0, target_fit * 0.78 + bind_help * 0.16 + min(1.0, len(components) / max(1, sum(components.values()))) * 0.06))
        method_quality = max(
            0.0,
            min(
                1.0,
                material_gate
                * (
                    planning * 0.30
                    + craft_skill * 0.24
                    + target_skill * 0.18
                    + bind_help * 0.10
                    + target_fit * 0.18
                ),
            ),
        )
        chance = min(
            0.96,
            organism.genome.manipulator * 0.26
            + bind_help * 0.22
            + best_skill * 0.08
            + planning * 0.12
            + method_quality * 0.20
            + target_fit * 0.24,
        )
        organism.energy -= (0.12 + 0.04 * sum(components.values())) * (1.0 - method_quality * 0.18)
        if self.rng.random() > chance:
            lost = self._lose_failed_craft_components(organism, components, bind_help)
            skill_gain = 0.0015 + lost * 0.0035
            organism.tool_skill["bind"] = min(1.0, organism.tool_skill.get("bind", 0.0) + skill_gain)
            organism.tool_skill["craft"] = min(1.0, craft_skill + skill_gain * 0.80)
            feedback["tool"] = feedback.get("tool", 0.0) + skill_gain
            self._record_tool_lesson(
                organism,
                self.world.places[organism.location],
                kind="craft",
                affordance=target,
                success=False,
                score=target_fit,
                method_quality=method_quality,
                components=components,
            )
            return
        for name, qty in components.items():
            organism.inventory[name] -= qty
            if organism.inventory[name] <= 0:
                del organism.inventory[name]
        artifact = build_artifact(components, method_quality=method_quality, target_affordance=target)
        organism.artifacts.append(artifact)
        self.artifacts_created[artifact.name] += 1
        organism.successful_tools += 1
        organism.record_success("tool_make", 1.0 + method_quality)
        self.tool_successes["craft"] += 1
        feedback["social"] += 0.06
        feedback["tool"] = feedback.get("tool", 0.0) + 0.30 + method_quality * 0.30
        organism.tool_skill["craft"] = min(1.0, craft_skill + 0.020 * (1.0 - craft_skill) + method_quality * 0.010)
        organism.tool_skill[target] = min(1.0, organism.tool_skill.get(target, 0.0) + artifact.capabilities.get(target, 0.0) * 0.012 + method_quality * 0.010)
        organism.last_craft_target = target
        organism.last_tool_affordance = target
        organism.last_artifact_method = method_quality
        self._record_tool_lesson(
            organism,
            self.world.places[organism.location],
            kind="craft",
            affordance=target,
            success=True,
            gain=feedback.get("tool", 0.0),
            score=artifact.capabilities.get(target, 0.0),
            method_quality=method_quality,
            components=components,
        )
        for capability, value in artifact.capabilities.items():
            if capability in organism.tool_skill:
                organism.tool_skill[capability] = min(1.0, organism.tool_skill.get(capability, 0.0) + value * 0.010)
        self.observer.observe(
            self.tick,
            "crafted_tool",
            {
                "organism_id": organism.id,
                "place": organism.location,
                "artifact": artifact.name,
                "target": target,
                "method_quality": method_quality,
                "target_fit": artifact.capabilities.get(target, 0.0),
                "components": dict(components),
            },
            subjects=self._subjects(organism, extra=[f"affordance:{target}"]),
            score=method_quality * 1.1 + artifact.capabilities.get(target, 0.0) * 0.8,
            rarity_key=f"crafted_tool:{target}",
        )
        self.checkpoints.save_first_tool(self.tick, organism, "craft", {"place": self.world.places[organism.location].to_summary(), "artifact": artifact.to_dict()})

    def _build_structure(self, organism: Organism, feedback: dict[str, float]) -> None:
        if organism.genome.manipulator < 0.18 or organism.inventory_count() < 3:
            organism.energy -= 0.035
            return
        available = [name for name, qty in organism.inventory.items() if qty > 0]
        if len(available) < 2:
            organism.energy -= 0.025
            return

        components: dict[str, int] = {}
        draws = min(8, organism.inventory_count())
        for _ in range(draws):
            choices = [name for name, qty in organism.inventory.items() if qty > components.get(name, 0)]
            if not choices:
                break
            chosen = self.rng.choice(choices)
            components[chosen] = components.get(chosen, 0) + 1
        material_count = sum(components.values())
        if material_count < 3:
            organism.energy -= 0.025
            return

        affordances = derive_affordances(components)
        bind_help = affordances.get("bind", 0.0)
        build_skill = organism.tool_skill.get("build", 0.0)
        general_skill = max(organism.tool_skill.values(), default=0.0)
        mass_bonus = min(1.0, material_count / 8.0) * 0.12
        planning = self._interaction_control(organism)
        chance = min(
            0.90,
            organism.genome.manipulator * 0.30
            + bind_help * 0.26
            + build_skill * 0.22
            + general_skill * 0.08
            + planning * 0.16
            + mass_bonus,
        )
        organism.energy -= (0.16 + 0.035 * material_count) * (1.0 - planning * 0.10)
        if self.rng.random() > chance:
            lost = self._lose_failed_craft_components(organism, components, bind_help)
            organism.tool_skill["build"] = min(1.0, build_skill + 0.003 + lost * 0.003)
            organism.tool_skill["bind"] = min(1.0, organism.tool_skill.get("bind", 0.0) + 0.0015 + lost * 0.002)
            feedback["tool"] = feedback.get("tool", 0.0) + 0.04 + lost * 0.01
            self.demonstrations[organism.location].append((organism.id, "build", False))
            self._record_tool_lesson(
                organism,
                self.world.places[organism.location],
                kind="build",
                affordance="build",
                success=False,
                score=bind_help,
                components=components,
            )
            return

        for name, qty in components.items():
            organism.inventory[name] -= qty
            if organism.inventory[name] <= 0:
                del organism.inventory[name]

        place = self.world.places[organism.location]
        if place.structures and (len(place.structures) >= 16 or self.rng.random() < 0.62):
            target = max(place.structures, key=lambda structure: (structure.durability, structure.scale))
            extend_structure(target, components)
            built_name = target.name
            self.structures_extended[built_name] += 1
            structure_summary = target.to_dict()
            extended_action = True
        else:
            structure = build_structure(components, builder_id=organism.id)
            place.structures.append(structure)
            built_name = structure.name
            self.structures_built[built_name] += 1
            structure_summary = structure.to_dict()
            extended_action = False

        organism.successful_tools += 1
        organism.record_success("structure", 1.0 + min(1.5, structure_summary["scale"] / 8.0))
        organism.tool_skill["build"] = min(1.0, build_skill + 0.035 * (1.0 - build_skill) + 0.004)
        for capability, value in structure_summary["capabilities"].items():
            if capability in organism.tool_skill:
                organism.tool_skill[capability] = min(1.0, organism.tool_skill.get(capability, 0.0) + float(value) * 0.006)
        self.tool_successes["build"] += 1
        feedback["social"] += 0.10
        feedback["tool"] = feedback.get("tool", 0.0) + 0.75
        self.demonstrations[place.id].append((organism.id, "build", True))
        self._record_tool_lesson(
            organism,
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
                "organism_id": organism.id,
                "place": place.id,
                "structure": built_name,
                "scale": structure_summary["scale"],
                "best_capability": best_structure_capability,
                "extended": extended_action,
            },
            subjects=self._subjects(organism),
            score=min(2.5, structure_summary["scale"] / 6.0 + best_structure_capability),
            rarity_key=f"structure:{built_name}",
        )
        if self.config.event_detail:
            self.logger.event(
                self.tick,
                "structure_built",
                {"organism_id": organism.id, "structure": built_name, "place": place.id, "scale": structure_summary["scale"]},
            )
        self.checkpoints.save_first_tool(self.tick, organism, "build", {"place": place.to_summary(), "structure": structure_summary})

    def _use_tool(self, organism: Organism, feedback: dict[str, float]) -> None:
        place = self.world.places[organism.location]
        affordance, score = best_affordance(organism.inventory, organism.tool_skill, organism.artifacts)
        if score < 0.08:
            organism.energy -= 0.05
            return
        skill = organism.tool_skill.get(affordance, 0.0)
        resistance = self._affordance_resistance(place, affordance)
        planning = self._interaction_control(organism)
        overmatch = score + skill * 0.25 + organism.genome.manipulator * 0.10 + planning * 0.22 - resistance
        chance = min(0.96, max(0.02, score * 0.34 + skill * 0.30 + organism.genome.manipulator * 0.16 + planning * 0.26 + overmatch * 0.24))
        organism.energy -= (0.12 + score * 0.10) * (1.0 - planning * 0.20)
        success = self.rng.random() < chance
        if success:
            gain = self._tool_effect(organism, place, affordance, score, skill)
            competence = 0.45 + score * 0.35 + skill * 0.20
            gain += self._advance_causal_challenge(organism, place, affordance, competence, feedback)
            organism.energy += gain
            self._wear_artifacts(organism, affordance, amount=0.18 + score * 0.10)
            organism.tool_skill[affordance] = min(1.0, skill + 0.055 * (1.0 - skill) + 0.005)
            organism.successful_tools += 1
            organism.last_tool_affordance = affordance
            organism.record_success("tool_use", 1.0 + min(2.0, max(0.0, gain) / 8.0))
            self.tool_successes[affordance] += 1
            feedback["social"] += 0.2
            feedback["tool"] = feedback.get("tool", 0.0) + 1.0
            self.demonstrations[place.id].append((organism.id, affordance, True))
            self._record_tool_lesson(
                organism,
                place,
                kind="tool_use",
                affordance=affordance,
                success=True,
                gain=gain,
                score=score,
                method_quality=organism.last_artifact_method,
            )
            self.observer.observe(
                self.tick,
                "tool_success",
                {
                    "organism_id": organism.id,
                    "place": place.id,
                    "affordance": affordance,
                    "gain": gain,
                    "score": score,
                    "skill": organism.tool_skill[affordance],
                },
                subjects=self._subjects(organism, extra=[f"affordance:{affordance}"]),
                score=min(3.0, max(0.0, gain) / 6.0 + score * 0.45),
                rarity_key=f"tool_success:{affordance}",
            )
            if self.config.event_detail:
                self.logger.event(
                    self.tick,
                    "tool_success",
                    {"organism_id": organism.id, "affordance": affordance, "gain": round(gain, 5), "place": place.id},
                )
            self.checkpoints.save_first_tool(self.tick, organism, affordance, {"place": place.to_summary(), "gain": gain})
        else:
            organism.tool_skill[affordance] = min(1.0, skill + 0.010 * (1.0 - skill))
            organism.last_tool_affordance = affordance
            feedback["tool"] = feedback.get("tool", 0.0) + 0.05
            if self.rng.random() < 0.10:
                organism.health -= 0.015 + score * 0.030
            overmatch_penalty = max(0.0, resistance - score)
            self._wear_artifacts(organism, affordance, amount=0.35 + score * 0.18 + overmatch_penalty * 2.2)
            self.demonstrations[place.id].append((organism.id, affordance, False))
            self._record_tool_lesson(
                organism,
                place,
                kind="tool_use",
                affordance=affordance,
                success=False,
                score=score,
                method_quality=organism.last_artifact_method,
            )
            if organism.health <= 0.0:
                self._kill(organism, "tool_accident")

    def _affordance_resistance(self, place: Place, affordance: str) -> float:
        physics = place.physics
        if affordance in {"crack", "lever"}:
            return min(1.0, 0.22 + place.mineral_richness * 0.45 + place.locked_chemical / 420.0 + physics.get("pressure", 0.0) * 0.05)
        if affordance == "cut":
            return min(1.0, 0.10 + place.obstacles.get("thorn", 0.0) * 0.60)
        if affordance == "contain":
            return min(1.0, 0.10 + place.obstacles.get("water", 0.0) * 0.22 + physics.get("pressure", 0.0) * 0.10 + physics.get("current_exposure", 0.0) * 0.12)
        if affordance == "concentrate_heat":
            return min(1.0, 0.15 + place.obstacles.get("heat", 0.0) * 0.25 + place.volatility * 0.20 + physics.get("humidity", 0.5) * 0.06)
        if affordance == "conduct":
            return min(1.0, 0.30 + place.mineral_richness * 0.25 + physics.get("fluid_level", 0.0) * 0.08)
        if affordance == "filter":
            return min(1.0, 0.10 + physics.get("current_exposure", 0.0) * 0.18 + physics.get("salinity", 0.0) * 0.10)
        if affordance == "bind":
            return 0.18
        return 0.25

    def _tool_effect(self, organism: Organism, place: Place, affordance: str, score: float, skill: float) -> float:
        competence = 0.45 + score * 0.35 + skill * 0.20
        if affordance == "crack":
            amount = min(place.locked_chemical, 2.0 + competence * 9.0)
            place.locked_chemical -= amount
            return amount * (0.25 + organism.genome.chemical_metabolism * 0.65 + organism.genome.digestion * 0.30)
        if affordance == "cut":
            amount = min(place.resources["biological_storage"], 1.5 + competence * 8.0)
            place.resources["biological_storage"] -= amount
            return amount * (0.35 + organism.genome.digestion * 0.85)
        if affordance == "bind":
            for name in organism.tool_skill:
                organism.tool_skill[name] = min(1.0, organism.tool_skill[name] + 0.004)
            return 0.5 + competence * 1.4
        if affordance == "contain":
            amount = min(place.resources["mechanical"], 0.8 + competence * 4.0 + place.physics.get("current_exposure", 0.0) * 2.5)
            place.resources["mechanical"] -= amount * 0.15
            place.physics["fluid_level"] = max(0.0, place.physics.get("fluid_level", 0.0) - amount * 0.0008)
            return amount * (0.15 + organism.genome.storage_capacity * 0.25)
        if affordance == "concentrate_heat":
            radiant = min(place.resources["radiant"], 2.0 + competence * 10.0)
            place.resources["thermal"] = min(180.0, place.resources["thermal"] + radiant * 0.15)
            place.physics["temperature"] = min(1.45, place.physics.get("temperature", 0.5) + radiant * 0.0008)
            return radiant * (0.04 + organism.genome.thermal_tolerance * 0.09 + organism.genome.radiant_metabolism * 0.06)
        if affordance == "conduct":
            amount = min(place.resources["electrical"], 0.5 + competence * 5.0 + place.mineral_richness * 1.5)
            place.resources["electrical"] -= amount
            structure_conduct = structure_capability(place.structures, "conduct")
            storage = structure_capability(place.structures, "energy_storage")
            gradient = max(structure_capability(place.structures, "gradient_harvest"), place.geothermal * place.mineral_richness)
            tap_pressure = max(0.0, competence + structure_conduct * 0.45 + storage * 0.25 + gradient * 0.35 - 0.92)
            high_density = min(place.resources["high_density"], tap_pressure * (1.0 + competence * 3.0))
            place.resources["high_density"] -= high_density
            place.resources["electrical"] = min(180.0, place.resources["electrical"] + high_density * (0.25 + storage * 0.25))
            return amount * (0.1 + organism.genome.electrical_use * 1.3) + high_density * (0.35 + organism.genome.electrical_use * 1.8)
        if affordance == "lever":
            amount = min(place.locked_chemical, 1.0 + competence * 5.5)
            place.locked_chemical -= amount
            return amount * (0.15 + organism.genome.mechanical_use * 0.55 + organism.genome.chemical_metabolism * 0.25)
        if affordance == "filter":
            flow_bonus = place.physics.get("current_exposure", 0.0) * 3.0 + place.physics.get("fluid_level", 0.0) * 1.5
            chemical = min(place.resources["chemical"], 0.5 + competence * 3.0 + flow_bonus)
            biological = min(place.resources["biological_storage"], 0.3 + competence * 1.8 + flow_bonus * 0.40)
            place.resources["chemical"] -= chemical * 0.55
            place.resources["biological_storage"] -= biological * 0.45
            return chemical * (0.12 + organism.genome.chemical_metabolism * 0.45) + biological * (0.15 + organism.genome.digestion * 0.38)
        return 0.0

    def _advance_causal_challenge(
        self,
        organism: Organism,
        place: Place,
        affordance: str,
        competence: float,
        feedback: dict[str, float],
    ) -> float:
        challenge = place.causal_challenge
        if challenge is None or challenge.payoff_remaining <= 0.0:
            return 0.0
        expected = challenge.expected_affordance()
        if expected is None:
            return 0.0
        challenge.attempts += 1
        planning = self._interaction_control(organism)
        threshold = challenge.difficulty * (0.76 + challenge.progress * 0.14)
        margin = competence + organism.tool_skill.get(affordance, 0.0) * 0.12 + planning * 0.30 + self.rng.gauss(0.0, 0.025)
        if affordance != expected or margin < threshold:
            if affordance in challenge.sequence and self.rng.random() < 0.35:
                challenge.progress = 0
            return 0.0

        challenge.progress += 1
        signature = challenge.signature()
        self.causal_steps[signature] += 1
        organism.record_success("causal_step", 1.0)
        feedback["tool"] = feedback.get("tool", 0.0) + 0.10
        if challenge.progress < len(challenge.sequence):
            self._record_tool_lesson(
                organism,
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
        release = min(challenge.payoff_remaining, 3.0 + competence * 13.0 + planning * 6.0 + sequence_bonus)
        challenge.payoff_remaining -= release
        place.resources[challenge.payoff_energy] = min(180.0, place.resources[challenge.payoff_energy] + release)
        self.causal_unlocks[signature] += 1
        organism.record_success("causal_unlock", 1.0 + min(2.0, release / 8.0))
        feedback["tool"] = feedback.get("tool", 0.0) + 0.70
        self._record_tool_lesson(
            organism,
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
                "organism_id": organism.id,
                "place": place.id,
                "sequence": list(challenge.sequence),
                "energy": challenge.payoff_energy,
                "released": release,
                "remaining": challenge.payoff_remaining,
            },
            subjects=self._subjects(organism, place.id, [f"challenge:{signature}"]),
            score=1.0 + min(4.0, release / 6.0),
            rarity_key=f"causal_unlock:{signature}",
        )
        if self.config.event_detail:
            self.logger.event(
                self.tick,
                "causal_unlock",
                {
                    "organism_id": organism.id,
                    "place": place.id,
                    "sequence": list(challenge.sequence),
                    "energy": challenge.payoff_energy,
                    "released": round(release, 5),
                },
            )
        return release * (0.30 + organism.genome.sensor_range * 0.10 + organism.genome.prediction_weight * 0.12 + planning * 0.10)

    def _wear_artifacts(self, organism: Organism, affordance: str, amount: float) -> None:
        kept = []
        for artifact in organism.artifacts:
            if artifact.capabilities.get(affordance, 0.0) > 0.05:
                artifact.durability -= amount
                artifact.age += 1
            if artifact.durability > 0.0:
                kept.append(artifact)
            else:
                self.artifacts_broken[artifact.name] += 1
                self.world.places[organism.location].resources["chemical"] += 0.1
        organism.artifacts = kept

    def _lose_failed_craft_components(self, organism: Organism, components: dict[str, int], bind_help: float) -> int:
        place = self.world.places[organism.location]
        lost = 0
        break_chance = max(0.18, min(0.72, 0.48 - bind_help * 0.20 + (1.0 - organism.genome.manipulator) * 0.12))
        for name, qty in components.items():
            for _ in range(qty):
                if organism.inventory.get(name, 0) <= 0 or self.rng.random() >= break_chance:
                    continue
                organism.inventory[name] -= 1
                if organism.inventory[name] <= 0:
                    del organism.inventory[name]
                lost += 1
                if self.rng.random() < 0.35:
                    place.materials[name] = min(99, place.materials.get(name, 0) + 1)
        return lost

    def _attack(self, organism: Organism) -> None:
        local = [self.organisms[oid] for oid in self._living_ids_at(organism.location) if oid != organism.id]
        if not local:
            organism.energy -= 0.04
            return
        target = min(local, key=lambda candidate: (candidate.health + candidate.genome.armor * 0.7, -candidate.energy))
        attack_power = organism.genome.mobility * 0.40 + organism.genome.manipulator * 0.35 + organism.genome.mechanical_use * 0.25
        defense = target.genome.armor * 0.45 + target.genome.mobility * 0.25 + target.health * 0.20
        damage = max(0.005, attack_power - defense + self.rng.gauss(0.0, 0.05))
        organism.energy -= 0.10 + attack_power * 0.08
        if damage > 0.0:
            target.health -= damage
        if target.health <= 0.0:
            gained = target.energy * (0.30 + organism.genome.digestion * 0.45)
            organism.energy += max(0.0, gained)
            self._kill(target, "predation")

    def _signal(self, organism: Organism, feedback: dict[str, float]) -> None:
        intensity = organism.genome.signal_strength * (0.5 + organism.energy / max(1.0, organism.storage_limit()))
        if intensity <= 0.01:
            organism.energy -= 0.01
            return
        token = organism.choose_signal_token()
        organism.energy -= 0.025 + intensity * 0.045
        self.world.emit_signal(organism.location, organism.id, token, intensity)
        feedback["social"] += intensity * 0.1

    def _coordinate_recombine(self, organism: Organism, feedback: dict[str, float]) -> None:
        self.reproduction_attempts["coordinate"] += 1
        if not organism.adult():
            self.reproduction_failures["coordinate_not_adult"] += 1
            organism.energy -= 0.015
            return
        if organism.energy < self._recombine_reserve_threshold(organism) * 0.82:
            self.reproduction_failures["coordinate_low_energy"] += 1
            organism.energy -= 0.020
            return
        window = 6 + int(organism.genome.signal_strength * 8.0 + organism.genome.mate_selectivity * 5.0)
        token = organism.choose_signal_token()
        intensity = 0.10 + organism.genome.signal_strength * 0.45 + organism.genome.mate_selectivity * 0.10
        organism.recombine_intent_until = max(organism.recombine_intent_until, self.tick + window)
        organism.coordination_token = token
        organism.energy -= 0.035 + intensity * 0.040
        self.world.emit_signal(organism.location, organism.id, token, intensity)
        feedback["social"] += intensity * 0.12

    def _mark(self, organism: Organism, feedback: dict[str, float]) -> None:
        if organism.genome.manipulator < 0.12:
            organism.energy -= 0.02
            return
        token = organism.choose_signal_token()
        affordances = derive_affordances(organism.inventory)
        inscription_help = max(affordances.get("cut", 0.0), affordances.get("bind", 0.0), affordances.get("concentrate_heat", 0.0))
        intensity = 0.20 + organism.genome.signal_strength * 0.35 + organism.genome.memory_budget / 40.0
        durability = 45.0 + organism.genome.manipulator * 90.0 + organism.genome.memory_budget * 12.0 + inscription_help * 180.0
        trace_intent = self._trace_inscription_intent(organism, inscription_help)
        trace = self._mark_trace(organism, inscription_help) if trace_intent else {}
        clarity = float(trace.get("inscription_quality", 0.0)) if trace else 0.0
        organism.energy -= 0.08 + intensity * 0.12 + durability * 0.0008 + clarity * 0.055
        self.world.create_mark(organism.location, organism.id, token, intensity, durability, trace=trace)
        self.marks_created[str(token)] += 1
        if trace:
            affordance = str(trace.get("affordance", "unknown"))
            organism.tool_skill["inscribe"] = min(1.0, organism.tool_skill.get("inscribe", 0.0) + 0.004 + clarity * 0.010)
            self.mark_lesson_packets[affordance] += 1
            lesson = trace.get("lesson", {})
            problem = lesson.get("problem", {}) if isinstance(lesson, dict) else {}
            self.observer.observe(
                self.tick,
                "mark_lesson_written",
                {
                    "organism_id": organism.id,
                    "place": organism.location,
                    "token": token,
                    "affordance": affordance,
                    "clarity": clarity,
                    "problem_kind": problem.get("kind") if isinstance(problem, dict) else None,
                    "lesson_kind": lesson.get("kind") if isinstance(lesson, dict) else None,
                },
                subjects=self._subjects(organism, extra=[f"affordance:{affordance}", f"mark_token:{token}"]),
                score=clarity * 1.2,
                rarity_key=f"mark_lesson_written:{affordance}",
            )
        feedback["social"] += 0.04 + intensity * 0.08 + clarity * 0.04

    def _trace_inscription_intent(self, organism: Organism, inscription_help: float) -> bool:
        if not organism.lesson_memory:
            return False
        planning = self._interaction_control(organism)
        skill = organism.tool_skill.get("inscribe", 0.0)
        memory = min(1.0, organism.genome.memory_budget / 14.0)
        lesson_value = self._lesson_value(self._select_mark_lesson(organism))
        capacity = (
            organism.genome.manipulator * 0.20
            + organism.genome.signal_strength * 0.10
            + organism.genome.sensor_range * 0.10
            + memory * 0.18
            + min(1.0, inscription_help) * 0.14
        )
        chance = capacity + planning * 0.16 + skill * 0.42 + lesson_value * 0.08 - 0.24
        if skill < 0.04:
            chance *= 0.45
        return self.rng.random() < max(0.0, min(0.82, chance))

    def _select_mark_lesson(self, organism: Organism) -> dict[str, Any]:
        return max(organism.lesson_memory[-5:], key=self._lesson_value)

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

    def _mark_trace(self, organism: Organism, inscription_help: float) -> dict[str, Any]:
        if not organism.lesson_memory:
            return {}
        lesson = self._select_mark_lesson(organism)
        affordance = str(lesson.get("affordance") or organism.last_tool_affordance or organism.last_craft_target)
        if affordance not in organism.tool_skill:
            return {}
        planning = self._interaction_control(organism)
        inscribe_skill = organism.tool_skill.get("inscribe", 0.0)
        clarity = max(
            0.0,
            min(
                1.0,
                organism.genome.memory_budget / 18.0 * 0.22
                + organism.genome.sensor_range * 0.14
                + organism.genome.manipulator * 0.15
                + organism.genome.signal_strength * 0.08
                + min(1.0, inscription_help) * 0.16
                + planning * 0.12
                + inscribe_skill * 0.33
                + self.rng.gauss(0.0, 0.025),
            ),
        )
        if clarity < 0.12:
            return {}
        encoded = self._encoded_mark_lesson(lesson, clarity)
        trace: dict[str, Any] = {
            "schema": "lesson_trace_v1",
            "intentional": True,
            "action": organism.last_action,
            "affordance": affordance,
            "valence": round(organism.last_valence, 6),
            "energy_delta": round(organism.last_energy_delta, 6),
            "inscription_quality": round(clarity, 6),
            "lesson": encoded,
        }
        trace["skill"] = round(float(lesson.get("skill", organism.tool_skill.get(affordance, 0.0))) * (0.55 + clarity * 0.45), 6)
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

    def _clone_mutate(self, organism: Organism, feedback: dict[str, float]) -> None:
        self.reproduction_attempts["clone_mutate"] += 1
        if not organism.adult() or self.living_total >= self.config.max_population:
            self.reproduction_failures["clone_mutate_not_adult_or_cap"] += 1
            organism.energy -= 0.02
            return
        place = self.world.places[organism.location]
        if len(self._living_ids_at(organism.location)) >= place.capacity:
            self.reproduction_failures["clone_mutate_local_capacity"] += 1
            organism.energy -= 0.015
            return
        decision = self.evolution.plan_clone_mutate(organism)
        if decision.failure or decision.plan is None:
            self.reproduction_failures[decision.failure or "clone_mutate_no_plan"] += 1
            organism.energy -= decision.energy_penalty
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
                    "generation": child.generation,
                    "complexity": child.genome.complexity(),
                },
                subjects=self._subjects(child, extra=[f"organism:{organism.id}", f"lineage:{organism.id}", "mode:clone_mutate"]),
                score=0.35 + child.generation * 0.08 + child.genome.complexity() * 0.08,
                rarity_key=f"birth:{decision.plan.operator}:{child.kind}",
            )
            if self.config.event_detail:
                self.logger.event(
                    self.tick,
                    "birth",
                    {"mode": decision.plan.operator, "child_id": child.id, "parent_ids": list(decision.plan.parent_ids), "kind": child.kind},
                )
        else:
            self.reproduction_failures["clone_mutate_add_failed"] += 1

    def _resolve_recombine(self, place_ids: set[int], feedback: dict[int, dict[str, float]]) -> None:
        for place_id in place_ids:
            candidates = [
                organism
                for organism in self.organisms.values()
                if organism.alive
                and organism.location == place_id
                and organism.adult()
                and organism.recombine_intent_until >= self.tick
            ]
            if len(candidates) < 2:
                if candidates:
                    self.reproduction_failures["recombine_no_partner"] += len(candidates)
                continue
            self.rng.shuffle(candidates)
            paired: set[int] = set()
            choices: dict[int, int] = {}
            for organism in candidates:
                self.reproduction_attempts["recombine_pairing"] += 1
                if organism.energy < self._recombine_reserve_threshold(organism):
                    self.reproduction_failures["recombine_low_energy"] += 1
                    continue
                viable = [
                    other
                    for other in candidates
                    if other.id != organism.id
                    and other.id not in paired
                    and organism.energy >= self.evolution.recombine_reserve_threshold(organism)
                    and other.energy >= self.evolution.recombine_reserve_threshold(other)
                    and self.evolution.compatible_for_recombine(organism, other)
                ]
                if not viable:
                    self.reproduction_failures["recombine_no_compatible_partner"] += 1
                    continue
                choices[organism.id] = max(viable, key=lambda other: self._partner_score(organism, other)).id
            for organism in candidates:
                if organism.id in paired or organism.id not in choices:
                    continue
                partner_id = choices[organism.id]
                partner = self.organisms.get(partner_id)
                if partner is None or not partner.alive or partner.id in paired:
                    continue
                if choices.get(partner.id) != organism.id and self.rng.random() > 0.35:
                    self.reproduction_failures["recombine_unreciprocated_choice"] += 1
                    continue
                child = self._recombine(organism, partner)
                if child:
                    paired.add(organism.id)
                    paired.add(partner.id)
                    organism.recombine_intent_until = -1
                    partner.recombine_intent_until = -1
                    feedback[organism.id]["reproduction"] += 1.0
                    feedback[partner.id]["reproduction"] += 1.0
                    if self.config.event_detail:
                        self.logger.event(
                            self.tick,
                            "birth",
                            {"mode": "recombine", "child_id": child.id, "parent_ids": [organism.id, partner.id], "kind": child.kind},
                        )

    def _partner_score(self, chooser: Organism, candidate: Organism) -> float:
        visible_fitness = (
            candidate.health * 0.35
            + min(1.0, candidate.energy / max(1.0, candidate.storage_limit())) * 0.25
            + candidate.genome.mobility * 0.10
            + candidate.genome.manipulator * 0.10
            + max(candidate.tool_skill.values(), default=0.0) * 0.10
            + min(1.0, candidate.offspring_count / 5.0) * 0.10
        )
        selectivity = chooser.genome.mate_selectivity
        return visible_fitness * (0.3 + selectivity) - chooser.genome.distance(candidate.genome) * 0.25 + self.rng.random() * 0.05

    def _recombine_reserve_threshold(self, organism: Organism) -> float:
        return self.evolution.recombine_reserve_threshold(organism)

    def _recombine(self, a: Organism, b: Organism) -> Organism | None:
        if self.living_total >= self.config.max_population:
            self.reproduction_failures["recombine_population_cap"] += 1
            return None
        decision = self.evolution.plan_recombine(a, b)
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
                    "generation": child.generation,
                    "complexity": child.genome.complexity(),
                },
                subjects=self._subjects(child, extra=[f"organism:{a.id}", f"organism:{b.id}", f"lineage:{a.id}", f"lineage:{b.id}", "mode:recombine"]),
                score=0.55 + child.generation * 0.09 + child.genome.complexity() * 0.10,
                rarity_key=f"birth:{decision.plan.operator}:{child.kind}",
            )
        else:
            self.reproduction_failures["recombine_add_failed"] += 1
        return child

    def _instantiate_offspring(self, plan: OffspringPlan) -> Organism | None:
        return self.add_organism(
            plan.child_kind,
            plan.child_genome,
            plan.location,
            plan.child_energy,
            plan.generation,
            plan.parent_ids,
            plan.brain_template,
        )

    def _apply_parent_costs_and_counts(self, plan: OffspringPlan) -> None:
        for parent_id, cost in plan.parent_costs.items():
            parent = self.organisms.get(parent_id)
            if parent is None:
                continue
            parent.energy -= cost
            parent.offspring_count += 1
            parent.record_success("reproduction", 1.0)

    def _read_mark_trace(self, organism: Organism, feedback: dict[str, float]) -> bool:
        place = self.world.places[organism.location]
        readable = [
            mark
            for mark in place.marks
            if mark.source_id != organism.id
            and mark.trace
            and mark.trace.get("intentional")
            and mark.trace.get("schema") == "lesson_trace_v1"
        ]
        if not readable:
            return False
        planning = self._interaction_control(organism)
        mark = max(
            readable[-12:],
            key=lambda item: item.intensity * min(1.0, item.durability / 140.0) + self.rng.random() * 0.03,
        )
        trace = mark.trace
        lesson = trace.get("lesson", {}) if isinstance(trace.get("lesson"), dict) else {}
        affordance = str(trace.get("affordance") or lesson.get("affordance", ""))
        if affordance not in organism.tool_skill:
            return False
        interpretation_skill = organism.tool_skill.get("interpret_mark", 0.0)
        inscription_quality = max(0.0, min(1.0, float(trace.get("inscription_quality", 0.0))))
        method_quality = max(0.0, min(1.0, float(trace.get("method_quality", 0.0))))
        tool_feedback = max(0.0, min(1.5, float(trace.get("tool_feedback", 0.0))))
        token_bias = max(0.0, organism.signal_values[mark.token] if 0 <= mark.token < len(organism.signal_values) else 0.0)
        fidelity = max(
            0.0,
            min(
                1.0,
                mark.intensity
                * min(1.0, mark.durability / 140.0)
                * inscription_quality
                * (0.48 + interpretation_skill * 0.36 + planning * 0.16),
            ),
        )
        attention = max(
            0.0,
            min(
                1.0,
                organism.genome.sensor_range * 0.34
                + organism.genome.memory_budget / 22.0
                + planning * 0.24
                + interpretation_skill * 0.26
                + token_bias * 0.10,
            ),
        )
        gain = fidelity * attention * (0.0025 + tool_feedback * 0.014 + method_quality * 0.018 + interpretation_skill * 0.006)
        organism.tool_skill["interpret_mark"] = min(1.0, interpretation_skill + 0.0015 + fidelity * 0.003)
        if gain <= 0.0005:
            organism.energy -= 0.006
            return True
        organism.tool_skill[affordance] = min(1.0, organism.tool_skill.get(affordance, 0.0) + gain)
        if method_quality > 0.0 or lesson.get("components"):
            organism.tool_skill["craft"] = min(1.0, organism.tool_skill.get("craft", 0.0) + gain * 0.55)
            organism.last_craft_target = affordance
            organism.last_artifact_method = max(organism.last_artifact_method, method_quality * fidelity)
        organism.last_tool_affordance = affordance
        if lesson:
            copied_lesson = dict(lesson)
            copied_lesson["kind"] = f"read_{copied_lesson.get('kind', 'lesson')}"
            copied_lesson["gain"] = gain
            copied_lesson["score"] = max(float(copied_lesson.get("score", 0.0) or 0.0), fidelity)
            organism.record_lesson(copied_lesson)
        organism.record_success("written_learning", gain * 12.0)
        self.mark_lessons[affordance] += 1
        feedback["social"] += 0.05 + fidelity * 0.04
        feedback["tool"] = feedback.get("tool", 0.0) + min(0.12, gain * 4.0)
        organism.energy -= 0.010 + (1.0 - attention) * 0.010
        self.observer.observe(
            self.tick,
            "mark_lesson_read",
            {
                "organism_id": organism.id,
                "source_id": mark.source_id,
                "place": place.id,
                "token": mark.token,
                "affordance": affordance,
                "gain": gain,
                "fidelity": fidelity,
                "clarity": inscription_quality,
            },
            subjects=self._subjects(organism, place.id, [f"organism:{mark.source_id}", f"affordance:{affordance}", f"mark_token:{mark.token}"]),
            score=fidelity + gain * 18.0,
            rarity_key=f"mark_lesson_read:{affordance}",
        )
        return True

    def _observe_others(self, organism: Organism, feedback: dict[str, float]) -> None:
        demos = self.demonstrations.get(organism.location, [])
        read_mark = self._read_mark_trace(organism, feedback)
        if not demos:
            if not read_mark:
                organism.energy -= 0.015
            return
        source_id, affordance, success = self.rng.choice(demos)
        if source_id == organism.id:
            return
        planning = self._interaction_control(organism)
        gain = (0.012 if success else 0.004) * (0.5 + organism.genome.sensor_range + planning * 1.10)
        organism.tool_skill[affordance] = min(1.0, organism.tool_skill.get(affordance, 0.0) + gain)
        organism.last_tool_affordance = affordance
        organism.record_success("social_learning", 0.2 if success else 0.05)
        feedback["social"] += 0.2 if success else 0.05
        feedback["tool"] = feedback.get("tool", 0.0) + (0.10 if success else 0.03)
        organism.energy -= 0.018

    def _remember_place(self, organism: Organism) -> None:
        if organism.genome.memory_budget <= 0.0:
            return
        place = self.world.places[organism.location]
        value = (
            place.total_accessible_energy() / 420.0
            + place.locked_chemical / 500.0
            + place.physics.get("shelter", 0.0) * 0.08
            + place.physics.get("interiority", 0.0) * place.physics.get("boundary_permeability", 0.0) * 0.04
        )
        old = organism.place_memory.get(place.id, 0.0)
        organism.place_memory[place.id] = old * 0.90 + value * 0.10
        limit = max(2, int(organism.genome.memory_budget * 3))
        if len(organism.place_memory) > limit:
            weakest = min(organism.place_memory, key=organism.place_memory.get)
            del organism.place_memory[weakest]

    def _kill(self, organism: Organism, cause: str) -> None:
        if not organism.alive:
            return
        organism.alive = False
        self.living_total = max(0, self.living_total - 1)
        self.living_by_kind[organism.kind] = max(0, self.living_by_kind[organism.kind] - 1)
        if organism.neural:
            self.living_neural = max(0, self.living_neural - 1)
        self.deaths_by_cause[cause] += 1
        self.deaths_by_kind_cause[f"{organism.kind}:{cause}"] += 1
        place = self.world.places[organism.location]
        place.resources["biological_storage"] = min(180.0, place.resources["biological_storage"] + max(0.0, organism.energy) * 0.35 + 2.0)
        if cause in {"predation", "senescence", "starvation"}:
            place.materials["bone"] = min(99, place.materials.get("bone", 0) + 1)
        checkpoint_score = self._checkpoint_score(organism) if organism.brain is not None else 0.0
        notable = (
            organism.offspring_count >= 3
            or organism.successful_tools >= 2
            or organism.success_profile.get("causal_unlock", 0.0) > 0.0
            or organism.success_profile.get("prediction_fit", 0.0) >= 2.0
            or organism.success_profile.get("written_learning", 0.0) > 0.0
        )
        if organism.brain is not None and (
            notable
        ):
            self.checkpoints.save_brain(
                self.tick,
                organism,
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
                    "organism_id": organism.id,
                    "kind": organism.kind,
                    "cause": cause,
                    "place": place.id,
                    "offspring_count": organism.offspring_count,
                    "successful_tools": organism.successful_tools,
                    "score": checkpoint_score,
                    "success_profile": dict(organism.success_profile),
                },
                subjects=self._subjects(organism, place.id, [f"cause:{cause}"]),
                score=0.75 + min(4.0, checkpoint_score / 8.0),
                rarity_key=f"death:{cause}",
            )
        if self.config.event_detail:
            self.logger.event(self.tick, "death", {"organism_id": organism.id, "cause": cause, "kind": organism.kind})
        organism.brain = None
        organism.brain_template = None
        organism.inventory.clear()
        organism.artifacts.clear()

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
            energy = str(payload.get("energy", "chemical"))
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
            for organism in self.organisms.values():
                if organism.alive and organism.location in affected_ids:
                    organism.health -= damage
                    if organism.health <= 0.0:
                        self._kill(organism, "intervention_disaster")
        elif intervention.kind == "climate_shift":
            self.world.climate_drift = max(-0.5, min(0.5, self.world.climate_drift + float(payload.get("amount", 0.0))))
        elif intervention.kind == "add_organisms":
            kind = str(payload.get("kind", "plant"))
            count = int(payload.get("count", 1))
            place = int(payload.get("place", self.rng.randrange(len(self.world.places))))
            for _ in range(count):
                genome = Genome.neural(self.rng) if kind == "agent" else Genome.fungus(self.rng) if kind == "fungus" else Genome.plant(self.rng)
                self.add_organism(kind, genome, place, float(payload.get("energy", 25.0)))
        record = {"tick": self.tick, "kind": intervention.kind, "payload": payload, "reason": intervention.reason}
        self.interventions_applied.append(record)
        self.logger.event(self.tick, "intervention", record)

    def _checkpoint_score(self, organism: Organism) -> float:
        energy_ratio = organism.energy / max(1.0, organism.storage_limit())
        profile = organism.success_profile
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
            + math.log1p(profile.get("reproduction", 0.0)) * 1.6
        )
        return (
            organism.offspring_count * 6.0
            + organism.successful_tools * 2.0
            + organism.generation * 0.75
            + organism.age / 450.0
            + energy_ratio * 2.0
            + organism.genome.complexity() * 0.5
            + profile_score
        )

    def _checkpoint_context(self, label: str, criterion: str) -> dict[str, Any]:
        return {
            "label": label,
            "criterion": criterion,
            "population": population_counts(self.organisms),
            "world_energy": world_energy_summary(self.world),
            "world_physics": world_physics_summary(self.world),
        }

    def _save_checkpoint_candidate(self, organism: Organism, label: str, criterion: str, bucket: str) -> None:
        score = self._checkpoint_score(organism)
        saved = self.checkpoints.save_brain(
            self.tick,
            organism,
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
                    "organism_id": organism.id,
                    "place": organism.location,
                    "criterion": criterion,
                    "bucket": bucket,
                    "score": score,
                },
                subjects=self._subjects(organism, extra=[f"checkpoint:{bucket}"]),
                score=1.1 + min(3.0, score / 10.0),
                rarity_key=f"checkpoint:{bucket}:{criterion}",
            )

    def _best_checkpoint_candidate(self, candidates: list[Organism], excluded_ids: set[int], key: Any) -> Organism | None:
        available = [organism for organism in candidates if organism.id not in excluded_ids]
        if not available:
            return None
        return max(available, key=key)

    def _checkpoint_champions(self, label: str) -> None:
        candidates = [organism for organism in self.organisms.values() if organism.alive and organism.brain is not None]
        if not candidates:
            return
        saved_ids: set[int] = set()

        overall = max(candidates, key=self._checkpoint_score)
        self._save_checkpoint_candidate(overall, label, "overall_champion", "interval_champion")
        saved_ids.add(overall.id)

        reproductive = self._best_checkpoint_candidate(candidates, saved_ids, lambda organism: (organism.offspring_count, organism.generation, organism.energy, organism.age))
        if reproductive is not None and reproductive.offspring_count > 0:
            self._save_checkpoint_candidate(reproductive, label, "reproductive_champion", "reproductive_champion")
            saved_ids.add(reproductive.id)

        tool_user = self._best_checkpoint_candidate(candidates, saved_ids, lambda organism: (organism.successful_tools, organism.offspring_count, organism.energy, organism.age))
        if tool_user is not None and tool_user.successful_tools > 0:
            self._save_checkpoint_candidate(tool_user, label, "tool_champion", "tool_champion")
            saved_ids.add(tool_user.id)

        causal = self._best_checkpoint_candidate(
            candidates,
            saved_ids,
            lambda organism: (
                organism.success_profile.get("causal_unlock", 0.0),
                organism.success_profile.get("causal_step", 0.0),
                organism.successful_tools,
                organism.energy,
            ),
        )
        if causal is not None and causal.success_profile.get("causal_step", 0.0) > 0.0:
            self._save_checkpoint_candidate(causal, label, "causal_champion", "causal_champion")
            saved_ids.add(causal.id)

        learner = self._best_checkpoint_candidate(
            candidates,
            saved_ids,
            lambda organism: (
                organism.success_profile.get("prediction_fit", 0.0),
                -sum(abs(value) for value in organism.prediction_error_profile),
                organism.age,
                organism.energy,
            ),
        )
        if learner is not None and learner.success_profile.get("prediction_fit", 0.0) > 0.0:
            self._save_checkpoint_candidate(learner, label, "learner_champion", "learner_champion")
            saved_ids.add(learner.id)

        lineage = self._best_checkpoint_candidate(candidates, saved_ids, lambda organism: (organism.generation, organism.offspring_count, organism.energy, organism.age))
        if lineage is not None and (lineage.generation > 0 or lineage.offspring_count > 0):
            self._save_checkpoint_candidate(lineage, label, "lineage_founder", "lineage_founder")

    def _log_aggregate(self) -> None:
        living = [organism for organism in self.organisms.values() if organism.alive]
        neural = [organism for organism in living if organism.neural]
        avg_energy = sum(organism.energy for organism in living) / max(1, len(living))
        avg_complexity = sum(organism.genome.complexity() for organism in living) / max(1, len(living))
        aggregate = {
            "tick": self.tick,
            "population": population_counts(self.organisms),
            "avg_energy": round(avg_energy, 5),
            "avg_complexity": round(avg_complexity, 5),
            "neural_avg_energy": round(sum(o.energy for o in neural) / max(1, len(neural)), 5),
            "births": dict(self.births_by_mode),
            "deaths": dict(self.deaths_by_cause),
            "deaths_by_kind_cause": dict(self.deaths_by_kind_cause),
            "tool_successes": dict(self.tool_successes),
            "causal_steps": dict(self.causal_steps),
            "causal_unlocks": dict(self.causal_unlocks),
            "success_profile": success_profile_summary(self.organisms),
            "marks_created": dict(self.marks_created),
            "mark_lessons": dict(self.mark_lessons),
            "mark_lesson_packets": dict(self.mark_lesson_packets),
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
