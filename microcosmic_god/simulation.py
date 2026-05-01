from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from random import Random
from typing import Any

from .brain import TinyBrain
from .checkpoints import CheckpointManager
from .config import RunConfig
from .debrief import build_debrief, population_counts, world_energy_summary
from .energy import MATERIALS, best_affordance, derive_affordances
from .genome import Genome
from .interventions import Intervention, load_interventions
from .organisms import ACTIONS, ACTION_INDEX, OBSERVATION_SIZE, Organism, organism_from_genome
from .runlog import RunLogger
from .world import Place, World


class Simulation:
    def __init__(self, config: RunConfig):
        self.config = config
        self.rng = Random(config.seed)
        self.world = World.generate(self.rng, config)
        self.logger = RunLogger(config)
        self.checkpoints = CheckpointManager(self.logger.checkpoint_dir, config.neural_checkpoint_limit)
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
        self.marks_created: Counter[str] = Counter()
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
            debrief = build_debrief(self, reason, elapsed)
            self.logger.write_json("summary.json", debrief)
            self.logger.write_json("world_final.json", self.world.to_summary())
            self.logger.close()
        return debrief

    def step(self) -> None:
        self.tick += 1
        self.world.update_environment(self.rng)
        self._apply_interventions()
        self.demonstrations.clear()

        rosters = self._rosters()
        contexts: dict[int, tuple[list[float], int, float, float, list[int]]] = {}
        intents: dict[int, str] = {}
        feedback: dict[int, dict[str, float]] = defaultdict(lambda: {"reproduction": 0.0, "social": 0.0})

        for organism in list(self.organisms.values()):
            if not organism.alive:
                continue
            self._metabolize(organism)
            if not organism.alive:
                continue
            observed_tokens = self._observed_tokens(organism)
            observation = self._observe(organism, rosters)
            action = self._choose_action(organism, observation)
            intents[organism.id] = action
            contexts[organism.id] = (observation, ACTION_INDEX[action], organism.energy, organism.health, observed_tokens)

        active_mating_places: set[int] = set()
        for organism_id, action in list(intents.items()):
            organism = self.organisms.get(organism_id)
            if organism is None or not organism.alive:
                continue
            if action == "mate":
                self._courtship(organism, feedback[organism_id])
                active_mating_places.add(organism.location)
                continue
            self._resolve_action(organism, action, rosters, feedback[organism_id])

        for organism in self.organisms.values():
            if organism.alive and organism.mate_intent_until >= self.tick:
                active_mating_places.add(organism.location)
        self._resolve_mating(active_mating_places, feedback)

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
            )
            organism.last_valence = valence
            organism.last_energy_delta = energy_delta
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
            self._checkpoint_champion()

    def _rosters(self) -> dict[int, list[int]]:
        rosters: dict[int, list[int]] = {place.id: [] for place in self.world.places}
        for organism in self.organisms.values():
            if organism.alive:
                rosters[organism.location].append(organism.id)
        return rosters

    def _fast_counts(self) -> dict[str, int]:
        counts = {kind: count for kind, count in self.living_by_kind.items() if count > 0}
        counts["neural"] = self.living_neural
        counts["total"] = self.living_total
        return counts

    def _metabolize(self, organism: Organism) -> None:
        organism.age += 1
        organism.energy -= organism.metabolic_cost()
        if organism.energy < 0.0:
            organism.health += organism.energy * 0.030
            organism.energy = 0.0
        if organism.health <= 0.0:
            self._kill(organism, "starvation")

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
            organism.genome.neural_budget / 32.0,
            organism.genome.memory_budget / 16.0,
            organism.genome.prediction_weight,
            organism.genome.plasticity_rate,
            max(-1.0, min(1.0, organism.last_valence)),
            best_skill,
            season,
            self.world.climate_drift,
            *organism.signal_values[:6],
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
            if organism.energy > organism.asexual_energy_threshold() and self.rng.random() < non_neural_birth_rate:
                return "asexual_reproduce"
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
            outputs[ACTION_INDEX["mate"]] += reproductive_drive * (0.9 + organism.genome.mate_selectivity)
            outputs[ACTION_INDEX["asexual_reproduce"]] += reproductive_drive * (0.7 + (1.0 - organism.genome.mate_selectivity) * 0.4)
        ranked = sorted(range(len(outputs)), key=lambda i: outputs[i], reverse=True)
        for index in ranked:
            action = ACTIONS[index]
            if self._action_feasible(organism, action):
                return action
        return "rest"

    def _action_feasible(self, organism: Organism, action: str) -> bool:
        if action in {"mate", "asexual_reproduce"} and not organism.adult():
            return False
        if action == "pickup" and organism.inventory_count() >= organism.inventory_limit():
            return False
        if action == "move" and organism.genome.mobility < 0.05:
            return False
        if action == "use_tool" and organism.inventory_count() == 0:
            return False
        if action == "mark" and organism.genome.manipulator < 0.12:
            return False
        return True

    def _resolve_action(
        self,
        organism: Organism,
        action: str,
        rosters: dict[int, list[int]],
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
        elif action == "use_tool":
            self._use_tool(organism, feedback)
        elif action == "attack":
            self._attack(organism, rosters)
        elif action == "signal":
            self._signal(organism, feedback)
        elif action == "mark":
            self._mark(organism, feedback)
        elif action == "asexual_reproduce":
            self._asexual_reproduce(organism, feedback, rosters)
        elif action == "observe":
            self._observe_others(organism, feedback)

    def _move(self, organism: Organism) -> None:
        place = self.world.places[organism.location]
        if not place.neighbors:
            return
        organism.energy -= 0.05 + organism.genome.mobility * 0.08
        if organism.brain and organism.place_memory:
            scored = []
            for neighbor in place.neighbors:
                memory = organism.place_memory.get(neighbor, 0.0)
                crowd = 0.0
                scored.append((memory - crowd + self.rng.random() * 0.08, neighbor))
            organism.location = max(scored)[1]
        else:
            organism.location = self.rng.choice(place.neighbors)

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
        organism.energy -= 0.025 + organism.genome.sensor_range * 0.020
        if self.rng.random() < 0.18 + organism.genome.sensor_range * 0.45:
            found = self.rng.choice(("chemical", "biological_storage", "mechanical"))
            amount = self.rng.uniform(0.2, 1.6) * (0.5 + organism.genome.sensor_range)
            place.resources[found] = min(180.0, place.resources[found] + amount)
        if self.rng.random() < 0.08 + organism.genome.sensor_range * 0.12:
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

    def _use_tool(self, organism: Organism, feedback: dict[str, float]) -> None:
        place = self.world.places[organism.location]
        affordance, score = best_affordance(organism.inventory, organism.tool_skill)
        if score < 0.08:
            organism.energy -= 0.05
            return
        skill = organism.tool_skill.get(affordance, 0.0)
        chance = min(0.94, score * 0.42 + skill * 0.34 + organism.genome.manipulator * 0.20 + organism.genome.prediction_weight * 0.06)
        organism.energy -= 0.12 + score * 0.10
        success = self.rng.random() < chance
        if success:
            gain = self._tool_effect(organism, place, affordance, score, skill)
            organism.energy += gain
            organism.tool_skill[affordance] = min(1.0, skill + 0.055 * (1.0 - skill) + 0.005)
            organism.successful_tools += 1
            self.tool_successes[affordance] += 1
            feedback["social"] += 0.2
            self.demonstrations[place.id].append((organism.id, affordance, True))
            if self.config.event_detail:
                self.logger.event(
                    self.tick,
                    "tool_success",
                    {"organism_id": organism.id, "affordance": affordance, "gain": round(gain, 5), "place": place.id},
                )
            self.checkpoints.save_first_tool(self.tick, organism, affordance, {"place": place.to_summary(), "gain": gain})
        else:
            organism.tool_skill[affordance] = min(1.0, skill + 0.010 * (1.0 - skill))
            if self.rng.random() < 0.10:
                organism.health -= 0.015 + score * 0.030
            self.demonstrations[place.id].append((organism.id, affordance, False))
            if organism.health <= 0.0:
                self._kill(organism, "tool_accident")

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
            amount = min(place.resources["mechanical"], 0.8 + competence * 4.0)
            place.resources["mechanical"] -= amount * 0.15
            return amount * (0.15 + organism.genome.storage_capacity * 0.25)
        if affordance == "concentrate_heat":
            radiant = min(place.resources["radiant"], 2.0 + competence * 10.0)
            place.resources["thermal"] = min(180.0, place.resources["thermal"] + radiant * 0.15)
            return radiant * (0.04 + organism.genome.thermal_tolerance * 0.09 + organism.genome.radiant_metabolism * 0.06)
        if affordance == "conduct":
            amount = min(place.resources["electrical"], 0.5 + competence * 5.0)
            place.resources["electrical"] -= amount
            return amount * (0.1 + organism.genome.electrical_use * 1.3)
        if affordance == "lever":
            amount = min(place.locked_chemical, 1.0 + competence * 5.5)
            place.locked_chemical -= amount
            return amount * (0.15 + organism.genome.mechanical_use * 0.55 + organism.genome.chemical_metabolism * 0.25)
        return 0.0

    def _attack(self, organism: Organism, rosters: dict[int, list[int]]) -> None:
        local = [self.organisms[oid] for oid in rosters.get(organism.location, []) if oid != organism.id and self.organisms[oid].alive]
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

    def _courtship(self, organism: Organism, feedback: dict[str, float]) -> None:
        self.reproduction_attempts["courtship"] += 1
        if not organism.adult():
            self.reproduction_failures["courtship_not_adult"] += 1
            organism.energy -= 0.015
            return
        if organism.energy < organism.sexual_energy_threshold() * 0.72:
            self.reproduction_failures["courtship_low_energy"] += 1
            organism.energy -= 0.020
            return
        window = 6 + int(organism.genome.signal_strength * 8.0 + organism.genome.mate_selectivity * 5.0)
        token = organism.choose_signal_token()
        intensity = 0.10 + organism.genome.signal_strength * 0.45 + organism.genome.mate_selectivity * 0.10
        organism.mate_intent_until = max(organism.mate_intent_until, self.tick + window)
        organism.courtship_token = token
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
        organism.energy -= 0.08 + intensity * 0.12 + durability * 0.0008
        self.world.create_mark(organism.location, organism.id, token, intensity, durability)
        self.marks_created[str(token)] += 1
        feedback["social"] += 0.08

    def _asexual_reproduce(self, organism: Organism, feedback: dict[str, float], rosters: dict[int, list[int]]) -> None:
        self.reproduction_attempts["asexual"] += 1
        if not organism.adult() or self.living_total >= self.config.max_population:
            self.reproduction_failures["asexual_not_adult_or_cap"] += 1
            organism.energy -= 0.02
            return
        place = self.world.places[organism.location]
        if len(rosters.get(organism.location, [])) >= place.capacity:
            self.reproduction_failures["asexual_local_capacity"] += 1
            organism.energy -= 0.015
            return
        threshold = organism.asexual_energy_threshold()
        if organism.energy < threshold:
            self.reproduction_failures["asexual_low_energy"] += 1
            organism.energy -= 0.03
            return
        if organism.genome.complexity() > self.config.asexual_complexity_ceiling and organism.neural:
            self.reproduction_failures["asexual_complexity_ceiling"] += 1
            organism.energy -= 0.08 + organism.genome.complexity() * 0.04
            return
        cost = threshold * (0.32 + organism.genome.offspring_investment * 0.28)
        child_energy = cost * 0.42
        child_genome = organism.genome.mutate(self.rng, strength=0.055)
        child_kind = organism.kind
        template = self._inherit_template_asexual(organism, child_genome)
        child = self.add_organism(
            child_kind,
            child_genome,
            organism.location,
            child_energy,
            organism.generation + 1,
            (organism.id,),
            template,
        )
        if child:
            organism.energy -= cost
            organism.offspring_count += 1
            self.births_by_mode["asexual"] += 1
            feedback["reproduction"] += 1.0
            if self.config.event_detail:
                self.logger.event(self.tick, "birth", {"mode": "asexual", "child_id": child.id, "parent_ids": [organism.id], "kind": child.kind})
        else:
            self.reproduction_failures["asexual_add_failed"] += 1

    def _inherit_template_asexual(self, parent: Organism, child_genome: Genome) -> TinyBrain | None:
        if parent.brain_template is None or child_genome.neural_budget < 2.0:
            return None
        if int(round(child_genome.neural_budget)) != parent.brain_template.hidden_size:
            return None
        return parent.brain_template.clone_for_offspring(self.rng, mutation_scale=0.025 + child_genome.mutation_rate * 0.25)

    def _resolve_mating(self, place_ids: set[int], feedback: dict[int, dict[str, float]]) -> None:
        for place_id in place_ids:
            candidates = [
                organism
                for organism in self.organisms.values()
                if organism.alive
                and organism.location == place_id
                and organism.adult()
                and organism.mate_intent_until >= self.tick
            ]
            if len(candidates) < 2:
                if candidates:
                    self.reproduction_failures["sexual_no_partner"] += len(candidates)
                continue
            self.rng.shuffle(candidates)
            paired: set[int] = set()
            choices: dict[int, int] = {}
            for organism in candidates:
                self.reproduction_attempts["sexual_pairing"] += 1
                if organism.energy < organism.sexual_energy_threshold():
                    self.reproduction_failures["sexual_low_energy"] += 1
                    continue
                viable = [
                    other
                    for other in candidates
                    if other.id != organism.id
                    and other.id not in paired
                    and organism.energy >= organism.sexual_energy_threshold()
                    and other.energy >= other.sexual_energy_threshold()
                    and organism.genome.distance(other.genome) < 0.50
                ]
                if not viable:
                    self.reproduction_failures["sexual_no_compatible_partner"] += 1
                    continue
                choices[organism.id] = max(viable, key=lambda other: self._mate_score(organism, other)).id
            for organism in candidates:
                if organism.id in paired or organism.id not in choices:
                    continue
                partner_id = choices[organism.id]
                partner = self.organisms.get(partner_id)
                if partner is None or not partner.alive or partner.id in paired:
                    continue
                if choices.get(partner.id) != organism.id and self.rng.random() > 0.35:
                    self.reproduction_failures["sexual_unreciprocated_choice"] += 1
                    continue
                child = self._sexual_reproduce(organism, partner)
                if child:
                    paired.add(organism.id)
                    paired.add(partner.id)
                    organism.mate_intent_until = -1
                    partner.mate_intent_until = -1
                    feedback[organism.id]["reproduction"] += 1.0
                    feedback[partner.id]["reproduction"] += 1.0
                    if self.config.event_detail:
                        self.logger.event(
                            self.tick,
                            "birth",
                            {"mode": "sexual", "child_id": child.id, "parent_ids": [organism.id, partner.id], "kind": child.kind},
                        )

    def _mate_score(self, chooser: Organism, candidate: Organism) -> float:
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

    def _sexual_reproduce(self, a: Organism, b: Organism) -> Organism | None:
        if self.living_total >= self.config.max_population:
            self.reproduction_failures["sexual_population_cap"] += 1
            return None
        cost_a = a.sexual_energy_threshold() * (0.25 + a.genome.offspring_investment * 0.22)
        cost_b = b.sexual_energy_threshold() * (0.25 + b.genome.offspring_investment * 0.22)
        if a.energy < cost_a or b.energy < cost_b:
            self.reproduction_failures["sexual_cost_energy"] += 1
            return None
        child_genome = Genome.recombine(self.rng, a.genome, b.genome)
        child_genome.developmental_complexity = min(1.0, child_genome.developmental_complexity + self.rng.uniform(0.00, 0.04))
        child_energy = (cost_a + cost_b) * 0.34
        template = self._inherit_template_sexual(a, b, child_genome)
        child = self.add_organism("agent", child_genome, a.location, child_energy, max(a.generation, b.generation) + 1, (a.id, b.id), template)
        if child:
            a.energy -= cost_a
            b.energy -= cost_b
            a.offspring_count += 1
            b.offspring_count += 1
            self.births_by_mode["sexual"] += 1
        else:
            self.reproduction_failures["sexual_add_failed"] += 1
        return child

    def _inherit_template_sexual(self, a: Organism, b: Organism, child_genome: Genome) -> TinyBrain | None:
        hidden = int(round(child_genome.neural_budget))
        templates = [parent.brain_template for parent in (a, b) if parent.brain_template and parent.brain_template.hidden_size == hidden]
        if templates:
            return self.rng.choice(templates).clone_for_offspring(self.rng, mutation_scale=0.035 + child_genome.mutation_rate * 0.20)
        return None

    def _observe_others(self, organism: Organism, feedback: dict[str, float]) -> None:
        demos = self.demonstrations.get(organism.location, [])
        if not demos:
            organism.energy -= 0.015
            return
        source_id, affordance, success = self.rng.choice(demos)
        if source_id == organism.id:
            return
        gain = (0.012 if success else 0.004) * (0.5 + organism.genome.sensor_range + organism.genome.memory_budget / 16.0)
        organism.tool_skill[affordance] = min(1.0, organism.tool_skill.get(affordance, 0.0) + gain)
        feedback["social"] += 0.2 if success else 0.05
        organism.energy -= 0.018

    def _remember_place(self, organism: Organism) -> None:
        if organism.genome.memory_budget <= 0.0:
            return
        place = self.world.places[organism.location]
        value = place.total_accessible_energy() / 420.0 + place.locked_chemical / 500.0
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
        if organism.brain is not None and (organism.offspring_count >= 3 or organism.successful_tools >= 2):
            self.checkpoints.save_brain(self.tick, organism, f"death_{cause}", {"place": place.to_summary()})
        if self.config.event_detail:
            self.logger.event(self.tick, "death", {"organism_id": organism.id, "cause": cause, "kind": organism.kind})
        organism.brain = None
        organism.brain_template = None
        organism.inventory.clear()

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

    def _checkpoint_champion(self) -> None:
        candidates = [organism for organism in self.organisms.values() if organism.alive and organism.brain is not None]
        if not candidates:
            return
        champion = max(candidates, key=lambda organism: (organism.offspring_count, organism.successful_tools, organism.energy, organism.age))
        self.checkpoints.save_brain(self.tick, champion, "interval_champion", {"population": population_counts(self.organisms)})

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
            "marks_created": dict(self.marks_created),
            "reproduction_attempts": dict(self.reproduction_attempts),
            "reproduction_failures": dict(self.reproduction_failures),
            "action_counts": dict(self.action_counts),
            "action_energy_delta": {key: round(value, 5) for key, value in self.action_energy_delta.items()},
            "action_avg_energy_delta": {
                key: round(self.action_energy_delta[key] / max(1, self.action_counts[key]), 5)
                for key in self.action_counts
            },
            "world_energy": world_energy_summary(self.world),
        }
        self.aggregate_history.append(aggregate)
        if len(self.aggregate_history) > 500:
            self.aggregate_history = self.aggregate_history[-500:]
        self.logger.event(self.tick, "aggregate", aggregate)
        self.logger.flush()
