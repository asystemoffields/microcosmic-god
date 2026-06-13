from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from random import Random
from typing import Any

from .backends import ControllerLearningCase, make_controller_runtime
from .controller import TinyController
from .checkpoints import CheckpointManager
from .config import RunConfig
from .debrief import build_debrief, pool_counts, success_profile_summary, world_energy_summary, world_physics_summary
from .optimization import Optimizer, ChildPlan
from .params import MEMORY_BUDGET_MAX, NEURAL_BUDGET_MAX, ParamVector
from .interventions import Intervention, load_interventions
from .observer import EventObserver
from .individuals import (
    ACTIONS,
    ACTION_INDEX,
    OBSERVATION_SIZE,
    Individual,
    individual_from_params,
    make_modular_controller_for_params,
)
from .runlog import RunLogger
from .world import Place, World


# Era 2: candidate cue channels for the tap contract. Each is an existing
# observable place-physics dim. With tap_cue_drift=1 the active channel is
# re-drawn from this set at every world refresh, so the contract must be
# re-discovered in-lifetime through tap outcomes (docs/ENV_AXIS_REVIEW.md).
TAP_CUE_CHANNELS: tuple[str, ...] = ("residue_activity", "current_exposure", "wet_dry_cycle")


class Simulation:
    def __init__(self, config: RunConfig):
        self.config = config
        self.rng = Random(config.seed)
        self.world = World.generate(self.rng, config)
        self.logger = RunLogger(config)
        self.observer = EventObserver(self.logger)
        self.checkpoints = CheckpointManager(self.logger.checkpoint_dir, config.neural_checkpoint_limit)
        self.controller_runtime = make_controller_runtime(config.compute_backend, config.device)
        self.optimization = Optimizer(self.rng, config)
        self.interventions = load_interventions(config.interventions_path) if config.run_mode == "garden" else {}
        self.individuals: dict[int, Individual] = {}
        self.next_id = 1
        self.tick = 0
        self.active_total = 0
        self.active_by_kind: Counter[str] = Counter()
        self.active_neural = 0
        self.births_by_mode: Counter[str] = Counter()
        self.deaths_by_cause: Counter[str] = Counter()
        self.deaths_by_kind_cause: Counter[str] = Counter()
        self.tap_outcomes: Counter[str] = Counter()
        self.patch_recovery_triggers: int = 0
        self.structural_steps: Counter[str] = Counter()
        self.physics_events: Counter[str] = Counter()
        self.spawn_attempts: Counter[str] = Counter()
        self.spawn_failures: Counter[str] = Counter()
        self.action_counts: Counter[str] = Counter()
        self.action_energy_delta: Counter[str] = Counter()
        self.infeasible_commits: Counter[str] = Counter()
        self.aggregate_history: list[dict[str, Any]] = []
        self.interventions_applied: list[dict[str, Any]] = []
        # Era 2: the active tap cue channel (re-drawn per refresh when
        # tap_cue_drift=1; fixed at the first entry otherwise). The gate
        # level sits at a percentile of the channel's current distribution
        # so every contract keeps a comparable fraction of places tappable
        # (an absolute level would leave some channels with zero tappable
        # places - a dead action, not a demand).
        self.tap_cue_channel: str = TAP_CUE_CHANNELS[0]
        self.tap_cue_level: float = self._compute_tap_cue_level()
        # Era 2.1: the staple cue (gates eat/absorb_solar). Inits to a
        # different channel than tap so the two demands aren't trivially the
        # same read; both drift independently when their drift knob is on.
        self.staple_cue_channel: str = TAP_CUE_CHANNELS[1 % len(TAP_CUE_CHANNELS)]
        self.staple_cue_level: float = self._compute_staple_cue_level()
        self._seed_initial_pool()

    def _cue_level(self, channel: str, fraction: float) -> float:
        fraction = min(0.99, max(0.0, fraction))
        values = sorted(place.physics.get(channel, 0.0) for place in self.world.places)
        if not values:
            return 0.05
        index = min(len(values) - 1, int(fraction * len(values)))
        return max(0.05, float(values[index]))

    def _compute_tap_cue_level(self) -> float:
        return self._cue_level(self.tap_cue_channel, float(getattr(self.config, "tap_cue_threshold", 0.70)))

    def _compute_staple_cue_level(self) -> float:
        return self._cue_level(self.staple_cue_channel, float(getattr(self.config, "staple_cue_threshold", 0.70)))

    def _staple_cue_factor(self, individual: Individual, place: Place) -> float:
        """Era 2.1 payoff multiplier on eat/absorb_solar. 1.0 above the cue
        gate, staple_cue_floor below it. floor>=1.0 disables gating (legacy).

        Applies ONLY to neural agents — the perception demand falls on the
        population we want to evolve perception. The scripted collector/
        converter food base has no controller and cannot read the cue, so
        gating it would only starve the substrate with no selection upside.
        """
        floor = float(getattr(self.config, "staple_cue_floor", 1.0))
        if floor >= 1.0 or not individual.neural:
            return 1.0
        cue = float(place.physics.get(self.staple_cue_channel, 0.0))
        return 1.0 if cue >= self.staple_cue_level else max(0.0, floor)

    def _environment_harshness(self) -> float:
        return max(0.2, float(getattr(self.config, "environment_harshness", 1.0)))

    def _refresh_world(self) -> None:
        """Replace physics and obstacles with fresh ones, but keep
        accumulated resource state.

        The point of multi-world is to invalidate controllers' memorized
        solutions (place 7 is safe to tap, etc.) so that only controllers
        that read current state persist. We do NOT want to also reset resource
        depletion, because that's a free buff to the pool — it lets
        energy-depleted lines get re-supplied every refresh cycle, which masks the
        cognitive ranking pressure with a basic persistence-pressure release.

        So the refresh swaps:
          - place.physics (temperature, fluid_level, abrasion, pressure, ...)
          - place.obstacles (water/thorn/heat/etc. barriers)
          - place.archetype, sun_exposure, water_flow, geothermal, mineral_richness, volatility
          - signals (cleared: they reference the old context)
          - the tap cue channel, when tap_cue_drift=1 (the era-2 re-mapping pressure)
        and KEEPS:
          - place.resources (current essence / residue_store / thermal / etc.)
          - place.sealed_essence (the tappable reserve)
          - place.id, name, neighbors (graph topology preserved so movement stays valid)
        """
        from random import Random as _Random
        new_rng = _Random((self.config.seed * 1_000_003) ^ (self.tick * 1_009))
        new_world = self.world.__class__.generate(new_rng, self.config)
        new_world.tick = self.world.tick

        # Carry over the resource state place-by-place. The new world has the
        # same place count and matching ids (both are 0..N-1 by construction),
        # but freshly generated physics/obstacles. Signals are NOT carried:
        # they reference the old context.
        n_places = min(len(self.world.places), len(new_world.places))
        for i in range(n_places):
            old = self.world.places[i]
            new = new_world.places[i]
            new.resources = dict(old.resources)
            new.regen_recovery_until = old.regen_recovery_until
            new.sealed_essence = old.sealed_essence

        # Edges/topology stay attached to the new world (which was just
        # generated, so its edges already match its place ids). We swap the
        # whole world object.
        for individual in self.individuals.values():
            if not individual.alive:
                continue
            if individual.location >= n_places:
                individual.location = self.rng.randrange(n_places)
        self.world = new_world
        if int(getattr(self.config, "tap_cue_drift", 0)) > 0:
            self.tap_cue_channel = new_rng.choice(TAP_CUE_CHANNELS)
        if int(getattr(self.config, "staple_cue_drift", 0)) > 0:
            self.staple_cue_channel = new_rng.choice(TAP_CUE_CHANNELS)
        self.tap_cue_level = self._compute_tap_cue_level()
        self.staple_cue_level = self._compute_staple_cue_level()
        self.logger.event(
            self.tick,
            "world_refreshed",
            {
                "new_world_seed_basis": self.tick,
                "resources_preserved": True,
                "tap_cue_channel": self.tap_cue_channel,
                "tap_cue_level": round(self.tap_cue_level, 5),
                "staple_cue_channel": self.staple_cue_channel,
                "staple_cue_level": round(self.staple_cue_level, 5),
            },
        )

    def _seed_initial_pool(self) -> None:
        for _ in range(self.config.initial_collectors):
            self.add_individual("collector", ParamVector.collector(self.rng), self.rng.randrange(len(self.world.places)), self.rng.uniform(10.0, 35.0))
        for _ in range(self.config.initial_converters):
            self.add_individual("converter", ParamVector.converter(self.rng), self.rng.randrange(len(self.world.places)), self.rng.uniform(8.0, 28.0))
        for _ in range(self.config.initial_agents):
            params = ParamVector.neural(self.rng)
            template = None
            if self.rng.random() < self.config.initial_modular_fraction:
                n_blocks = self.rng.randint(1, max(1, self.config.initial_modular_max_blocks))
                _, template = make_modular_controller_for_params(self.rng, params, n_blocks=n_blocks)
            self.add_individual(
                "agent", params, self.rng.randrange(len(self.world.places)), self.rng.uniform(22.0, 55.0),
                controller_template=template,
            )
        self.logger.event(0, "seeded", {"pool": pool_counts(self.individuals)})

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
        if self.active_total >= self.config.max_pool:
            return None
        if kind != "agent":
            params.neural_budget = 0.0
            params.memory_budget = 0.0
            controller_template = None
        individual = individual_from_params(
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
        parent_lines = tuple(
            self.individuals[parent_id].line_root_id or parent_id
            for parent_id in parent_ids
            if parent_id in self.individuals
        )
        if parent_ids:
            individual.parent_line_ids = parent_lines or parent_ids
            individual.line_root_id = individual.parent_line_ids[0]
        else:
            individual.line_root_id = individual.id
        individual.neural_upkeep_grace_ticks = self.config.neural_upkeep_grace_ticks
        individual.neural_upkeep_grace_floor = self.config.neural_upkeep_grace_floor
        self.individuals[individual.id] = individual
        self.next_id += 1
        self.active_total += 1
        self.active_by_kind[individual.kind] += 1
        if individual.neural:
            self.active_neural += 1
        return individual

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        self._run_started = started
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
                if self.config.stop_on_full_washout and counts.get("total", 0) == 0:
                    reason = "full_washout"
                    break
                if self.config.stop_on_neural_washout and self.tick > 10 and counts.get("neural", 0) == 0:
                    reason = "neural_washout"
                    break
                self.step()
        except KeyboardInterrupt:
            # Graceful park (SIGINT): label honestly and still write the debrief.
            reason = "interrupted"
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
        # underlying causal rules persist. Selects FOR generalization.
        refresh_every = int(getattr(self.config, "world_refresh_every", 0) or 0)
        if refresh_every > 0 and self.tick > 1 and self.tick % refresh_every == 0:
            self._refresh_world()
        physics_events = self.world.update_environment(self.rng)
        self.physics_events.update(physics_events)
        self._apply_physics_transport()
        self._apply_interventions()

        # Perception is a tick-start snapshot; action effects below resolve sequentially
        # against current individual locations, spawns, and removals.
        rosters = self._rosters()
        context_bases: dict[int, tuple[list[float], float, float, list[int]]] = {}
        contexts: dict[int, tuple[list[float], int, float, float, list[int]]] = {}
        intents: dict[int, str] = {}
        intent_slots: list[tuple[int, str | None]] = []
        neural_choice_rows: list[tuple[Individual, list[float]]] = []
        neural_actions: dict[int, str] = {}
        feedback: dict[int, dict[str, float]] = defaultdict(lambda: {"spawning": 0.0, "social": 0.0, "tap": 0.0})

        for individual in list(self.individuals.values()):
            if not individual.alive:
                continue
            self._apply_upkeep(individual)
            self._terrain_stress(individual)
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
            neural_controllers: list[TinyController] = []
            neural_observations: list[list[float]] = []
            for individual, observation in neural_choice_rows:
                assert individual.controller is not None
                neural_controllers.append(individual.controller)
                neural_observations.append(observation)
            outputs = self.controller_runtime.forward_many(neural_controllers, neural_observations)
            for (individual, _observation), action_outputs in zip(neural_choice_rows, outputs):
                neural_actions[individual.id] = self._choose_action_from_outputs(individual, action_outputs)

        for individual_id, action in intent_slots:
            resolved_action = action if action is not None else neural_actions.get(individual_id, "rest")
            intents[individual_id] = resolved_action
            observation, before_energy, before_health, observed_tokens = context_bases[individual_id]
            contexts[individual_id] = (observation, ACTION_INDEX[resolved_action], before_energy, before_health, observed_tokens)

        active_combine_places: set[int] = set()
        for individual_id, action in list(intents.items()):
            individual = self.individuals.get(individual_id)
            if individual is None or not individual.alive:
                continue
            if action == "coordinate":
                self._coordinate_combine(individual, feedback[individual_id])
                active_combine_places.add(individual.location)
                continue
            self._resolve_action(individual, action, feedback[individual_id])

        for individual in self.individuals.values():
            if individual.alive and individual.combine_intent_until >= self.tick:
                active_combine_places.add(individual.location)
        self._resolve_combine(active_combine_places, feedback)

        learning_rows: list[tuple[Individual, int, float, float, float, float, dict[str, float], list[int]]] = []
        learning_cases: list[ControllerLearningCase] = []
        for individual_id, context in contexts.items():
            individual = self.individuals.get(individual_id)
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
            extra = feedback[individual_id]
            movement_hazard = damage * 4.0 if action_name == "move" else 0.0
            valence = (
                individual.params.valence_energy * (energy_delta / 10.0)
                + individual.params.valence_health * (health_delta * 4.0)
                - individual.params.valence_damage * (damage * 4.0)
                + individual.params.valence_spawn * extra.get("spawning", 0.0)
                + individual.params.valence_social * extra.get("social", 0.0)
            )
            outcome_targets = {
                "damage": damage * 4.0,
                "spawning": extra.get("spawning", 0.0),
                "social": extra.get("social", 0.0),
                "tap": extra.get("tap", 0.0),
                "hazard": movement_hazard,
            }
            learning_rows.append((individual, action_index, energy_delta, health_delta, damage, valence, extra, observed_tokens))
            learning_cases.append(
                ControllerLearningCase(
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

        prediction_errors = self.controller_runtime.learn_many(learning_cases)
        for row, prediction_error in zip(learning_rows, prediction_errors):
            individual, action_index, energy_delta, health_delta, damage, valence, extra, observed_tokens = row
            individual.last_valence = valence
            individual.record_action_result(
                action_index=action_index,
                energy_delta=energy_delta,
                health_delta=health_delta,
                damage=damage,
                prediction_error=prediction_error,
                spawn_feedback=extra.get("spawning", 0.0),
                social_feedback=extra.get("social", 0.0),
                tap_feedback=extra.get("tap", 0.0),
                prediction_errors=individual.controller.last_prediction_errors,
            )
            for token in observed_tokens:
                individual.learn_signal_value(token, valence + prediction_error * 0.05)

        for individual in list(self.individuals.values()):
            if individual.alive:
                individual.repair_or_decay()

        if self.tick % self.config.log_every == 0:
            self._log_aggregate()
        if self.tick % self.config.checkpoint_every == 0:
            self._checkpoint_champions("interval")

    def _rosters(self) -> dict[int, list[int]]:
        rosters: dict[int, list[int]] = {place.id: [] for place in self.world.places}
        for individual in self.individuals.values():
            if individual.alive:
                rosters[individual.location].append(individual.id)
        return rosters

    def _subjects(self, individual: Individual | None = None, place_id: int | None = None, extra: list[str] | None = None) -> list[str]:
        subjects: list[str] = []
        if individual is not None:
            subjects.append(f"individual:{individual.id}")
            subjects.append(f"place:{individual.location}")
            line_root_id = getattr(individual, "line_root_id", 0)
            if line_root_id:
                subjects.append(f"line:{line_root_id}")
        if place_id is not None:
            subject = f"place:{place_id}"
            if subject not in subjects:
                subjects.append(subject)
        if extra:
            subjects.extend(extra)
        return subjects

    def _active_ids_at(self, place_id: int) -> list[int]:
        return [individual.id for individual in self.individuals.values() if individual.alive and individual.location == place_id]

    def _fast_counts(self) -> dict[str, int]:
        counts = {kind: count for kind, count in self.active_by_kind.items() if count > 0}
        counts["neural"] = self.active_neural
        counts["total"] = self.active_total
        return counts

    def _signal_intensity_from(self, place: Place, source_id: int) -> float:
        return max((signal.intensity for signal in place.signals if signal.source_id == source_id), default=0.0)

    def _place_exposure_pressure(self, place: Place) -> dict[str, Any]:
        physics = place.physics
        temperature = physics.get("temperature", 0.5)
        humidity = physics.get("humidity", place.terrain.get("humidity", 0.5))
        fluid = physics.get("fluid_level", place.terrain.get("aquatic", 0.0))
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
        score = max(0.0, min(1.5, cold * 0.72 + heat * 0.54 + wet * 0.42 + abrasion_pressure * 0.36))
        return {
            "kind": "exposure",
            "place": place.id,
            "severity": round(score, 6),
            "primary": primary,
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

    def _relocation_shock(self, individual: Individual, origin: Place, destination: Place) -> float:
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
        # Body-only mitigation (resilience is an observable own-param dim);
        # the artifact/memory/skill mitigation sources were cut with their
        # subsystems (docs/ENV_AXIS_REVIEW.md).
        mitigation = min(0.72, individual.params.resilience * 0.25)
        return max(0.0, raw * (1.0 - mitigation))

    def _apply_physics_transport(self) -> None:
        for individual in list(self.individuals.values()):
            if not individual.alive:
                continue
            place = self.world.places[individual.location]
            physics = place.physics
            fluid = physics.get("fluid_level", 0.0)
            current = physics.get("current_exposure", 0.0)
            downstream = self.world.downstream_neighbor(place.id)
            if downstream and fluid > 0.35 and current > 0.08:
                resistance = max(
                    individual.params.aquatic_affinity * 0.65 + individual.params.buoyancy * 0.35,
                    individual.params.mobility * 0.30,
                )
                drift_chance = max(0.0, downstream[1] * fluid * (1.0 - resistance)) * 0.020
                if self.rng.random() < drift_chance:
                    individual.location = downstream[0]
                    individual.energy -= 0.010 + current * 0.012
                    if individual.params.aquatic_affinity < fluid * 0.45:
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
                footing = individual.params.mobility
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
        hardship = max(0.0, self._environment_harshness() - 1.0)
        hardship_multiplier = 1.0 + hardship * (0.16 if individual.kind == "agent" else 0.08)
        individual.energy -= individual.upkeep_cost() * hardship_multiplier
        if individual.energy < 0.0:
            individual.health += individual.energy * 0.030
            individual.energy = 0.0
        if individual.health <= 0.0:
            self._deactivate(individual, "exhaustion")

    def _terrain_stress(self, individual: Individual) -> None:
        place = self.world.places[individual.location]
        aquatic = place.terrain.get("aquatic", 0.0)
        depth = place.terrain.get("depth", 0.0)
        physics = place.physics
        salinity = physics.get("salinity", place.terrain.get("salinity", 0.0))
        humidity = physics.get("humidity", place.terrain.get("humidity", 0.5))
        temperature = physics.get("temperature", 0.5)
        pressure = physics.get("pressure", depth)
        current = physics.get("current_exposure", 0.0)
        shelter = physics.get("shelter", 0.0)
        interiority = physics.get("interiority", 0.0)
        permeability = physics.get("boundary_permeability", 0.0)
        insulation = shelter * 0.55
        hardship = max(0.0, self._environment_harshness() - 1.0)
        exposure = self._place_exposure_pressure(place)
        exposure_severity = float(exposure["severity"]) * (1.0 + hardship * 0.50)
        exposure_damping = max(0.45, 1.0 - shelter * 0.35)
        drowning = max(0.0, aquatic * depth - individual.params.aquatic_affinity * 0.85 - individual.params.mobility * 0.15 - shelter * 0.08)
        desiccation = max(0.0, individual.params.aquatic_affinity * (1.0 - humidity) - individual.params.desiccation_tolerance * 0.55 - shelter * 0.14)
        salinity_stress = max(0.0, abs(salinity - individual.params.salinity_tolerance) - 0.55)
        heat_stress = max(0.0, temperature - (0.58 + individual.params.thermal_tolerance * 0.42 + insulation * 0.25))
        cold_stress = max(0.0, 0.24 - temperature - individual.params.thermal_tolerance * 0.14 - insulation * 0.20)
        pressure_stress = max(0.0, pressure - (individual.params.pressure_tolerance * 1.05 + individual.params.aquatic_affinity * 0.20 + individual.params.resilience * 0.12))
        current_stress = max(0.0, current * aquatic - max(individual.params.buoyancy, individual.params.mobility * 0.25))
        stagnant_interior = max(0.0, interiority - shelter) * max(0.0, 1.0 - permeability) * max(0.0, pressure + temperature - 0.80)
        exposure_buffer = (
            insulation * 0.24
            + shelter * 0.16
            + individual.params.thermal_tolerance * 0.10
            + individual.params.resilience * 0.05
        )
        exposure_stress = max(0.0, exposure_severity - exposure_buffer)
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
                self._deactivate(individual, "terrain_mismatch")

    def _observe(self, individual: Individual, rosters: dict[int, list[int]]) -> list[float]:
        place = self.world.places[individual.location]
        resources = [place.resources[kind] / 120.0 for kind in ("solar", "essence", "residue_store", "thermal", "mechanical", "electrical", "dense_node")]
        local_ids = rosters.get(place.id, [])
        local_neural = sum(1 for oid in local_ids if self.individuals[oid].neural)
        season = math.sin(2.0 * math.pi * self.world.tick / max(2, self.world.season_length))
        features = [
            individual.energy / max(1.0, individual.storage_limit()),
            individual.health,
            min(1.0, individual.age / 2_000.0),
            *resources,
            place.sealed_essence / 160.0,
            len(local_ids) / max(1.0, place.capacity),
            local_neural / max(1.0, len(local_ids)),
            individual.params.mobility,
            individual.params.manipulator,
            individual.params.resilience,
            individual.params.sensor_range,
            individual.params.neural_budget / NEURAL_BUDGET_MAX,
            individual.params.memory_budget / MEMORY_BUDGET_MAX,
            individual.params.prediction_weight,
            individual.params.plasticity_rate,
            max(-1.0, min(1.0, individual.last_valence)),
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
            place.physics.get("residue_activity", 0.0),
            place.physics.get("abrasion", 0.0),
            place.physics.get("wet_dry_cycle", 0.0),
            place.physics.get("elevation", 0.5),
            place.terrain.get("aquatic", 0.0),
            place.terrain.get("depth", 0.0),
            place.terrain.get("salinity", 0.0),
            place.terrain.get("humidity", 0.5),
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
        return tokens[:4]

    def _choose_action(self, individual: Individual, observation: list[float]) -> str:
        if individual.controller is None:
            non_neural_birth_rate = 0.025 if individual.kind == "collector" else 0.035
            if individual.energy > self.optimization.clone_perturb_reserve_threshold(individual) and self.rng.random() < non_neural_birth_rate:
                return "clone_perturb"
            if individual.kind == "collector":
                return "absorb_solar"
            if individual.kind == "converter":
                return "eat" if self.rng.random() < 0.72 else "forage"
            return "rest"

        outputs = self.controller_runtime.forward_many([individual.controller], [observation])[0]
        return self._choose_action_from_outputs(individual, outputs)

    def _choose_action_from_outputs(self, individual: Individual, outputs: list[float]) -> str:
        floor = float(getattr(self.config, "exploration_floor", 0.025))
        exploration = floor + individual.params.plasticity_rate * 0.055 + individual.params.perturbation_rate * 0.25
        if self.rng.random() < exploration:
            return self.rng.choice(ACTIONS)
        energy_ratio = individual.energy / max(1.0, individual.storage_limit())
        drive_scale = float(getattr(self.config, "drive_injection_scale", 1.0))
        if drive_scale > 0.0 and individual.adult() and energy_ratio > 0.62:
            spawn_drive = individual.params.valence_spawn * (energy_ratio - 0.62) * drive_scale
            outputs[ACTION_INDEX["coordinate"]] += spawn_drive * (0.9 + individual.params.pairing_selectivity)
            outputs[ACTION_INDEX["clone_perturb"]] += spawn_drive * (0.7 + (1.0 - individual.params.pairing_selectivity) * 0.4)
        ranked = sorted(range(len(outputs)), key=lambda i: outputs[i], reverse=True)
        depth = int(getattr(self.config, "action_search_depth", 0))
        search = ranked if depth <= 0 else ranked[:depth]
        for index in search:
            action = ACTIONS[index]
            if self._action_feasible(individual, action):
                return action
        if depth <= 0:
            return "rest"
        # Bounded search exhausted: commit to the top-ranked action anyway and
        # let the action handlers adjudicate it — every handler already no-ops
        # an infeasible attempt with a small energy cost (the exploration branch
        # relies on this), so a wrong ranking wastes the tick instead of falling
        # through to a free feasibility oracle.
        committed = ACTIONS[ranked[0]]
        self.infeasible_commits[committed] += 1
        return committed

    def _action_feasible(self, individual: Individual, action: str) -> bool:
        if action in {"coordinate", "clone_perturb"} and not individual.adult():
            return False
        if action == "move" and individual.params.mobility < 0.05:
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
        elif action == "tap":
            self._tap(individual, feedback)
        elif action == "drain":
            self._drain(individual)
        elif action == "signal":
            self._signal(individual, feedback)
        elif action == "clone_perturb":
            self._clone_perturb(individual, feedback)

    def _move(self, individual: Individual, feedback: dict[str, float] | None = None) -> None:
        place = self.world.places[individual.location]
        if not place.neighbors:
            return
        individual.energy -= 0.055 + individual.params.mobility * 0.055
        # Destination is uniform-random among neighbors: the controller owns
        # the leave/stay decision and the harness adds no destination
        # cognition (the era-1 place-memory scorer answered that question on
        # the controller's behalf - docs/ENV_AXIS_REVIEW.md).
        destination_id = self.rng.choice(place.neighbors)
        destination = self.world.places[destination_id]
        edge = self.world.edge_between(place.id, destination_id)
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
            destination.obstacles.get("water", 0.0) * (1.0 - aquatic_fit)
            + destination.obstacles.get("height", 0.0) * (1.0 - individual.params.mobility)
            + destination.obstacles.get("thorn", 0.0) * (1.0 - individual.params.resilience)
            + destination.obstacles.get("heat", 0.0) * (1.0 - individual.params.thermal_tolerance)
            + edge_required * (1.0 - individual.params.mobility)
            + uphill * (1.0 - individual.params.mobility)
            + against_current * (1.0 - max(aquatic_fit, individual.params.buoyancy))
            + downhill * (1.0 - individual.params.mobility) * 0.35
            + boundary * (1.0 - individual.params.manipulator * 0.35)
        ) / 6.10
        solo_success = (
            individual.params.mobility
            + individual.params.sensor_range * 0.10
            + with_current * max(individual.params.buoyancy, aquatic_fit) * 0.10
            + self.rng.gauss(0.0, 0.04)
        )
        edge_danger = edge.danger if edge else 0.0
        relocation_shock = self._relocation_shock(individual, place, destination)
        effective_barrier = max(0.0, barrier + relocation_shock * 0.22 + edge_danger * 0.12)
        if solo_success >= effective_barrier:
            individual.location = destination_id
            individual.energy -= (
                effective_barrier * 0.125
                + distance * 0.026
                + uphill * 0.040
                + destination.physics.get("pressure", 0.0) * 0.016
                + edge_danger * 0.035
                + relocation_shock * 0.58
            )
            individual.health -= relocation_shock * (0.030 + edge_danger * 0.012)
            if with_current > 0.1 and destination.obstacles.get("water", 0.0) > 0.3:
                self.physics_events["current_assisted_move"] += 1
            if individual.health <= 0.0:
                self._deactivate(individual, "relocation_shock")
        else:
            individual.energy -= effective_barrier * 0.340 + distance * 0.032 + edge_danger * 0.060 + relocation_shock * 0.32
            individual.health -= (
                max(0.0, effective_barrier - solo_success) * (0.052 + downhill * 0.018 + edge_danger * 0.020)
                + relocation_shock * 0.012
            )
            if individual.health <= 0.0:
                self._deactivate(individual, "movement_hazard")

    def _eat(self, individual: Individual) -> None:
        place = self.world.places[individual.location]
        # Era 2.1 staple cue gate: below-cue eating is less efficient
        # (appetite scaled), so a blind "eat anywhere" policy is dominated by
        # a cue-reader. factor 1.0 when gating off. Scaling appetite (not just
        # gain) keeps consumption proportional so off-cue eating does not
        # strip the commons and worsen the overshoot.
        appetite = (2.0 + individual.params.essence_conversion * 7.0 + individual.params.essence_energy_gain * 3.0) * self._staple_cue_factor(individual, place)
        essence = min(place.resources["essence"], appetite * 0.55)
        place.resources["essence"] -= essence
        residue = min(place.resources["residue_store"], appetite - essence)
        place.resources["residue_store"] -= residue
        gain = essence * individual.params.essence_energy_gain + residue * (0.45 + individual.params.essence_conversion * 0.80)
        individual.energy += gain
        if essence + residue > 0.5 and self.world.note_patch_depletion(place.id, self.rng):
            self.patch_recovery_triggers += 1

    def _absorb_solar(self, individual: Individual) -> None:
        place = self.world.places[individual.location]
        gain = place.resources["solar"] * 0.018 * individual.params.solar_energy_gain * (0.2 + individual.params.solar_capture_area) * self._staple_cue_factor(individual, place)
        thermal_stress = max(0.0, place.resources["thermal"] / 120.0 - individual.params.thermal_tolerance)
        individual.energy += gain
        individual.health -= thermal_stress * 0.003
        place.resources["residue_store"] += gain * 0.18
        if individual.health <= 0.0:
            self._deactivate(individual, "thermal_stress")

    def _forage(self, individual: Individual) -> None:
        place = self.world.places[individual.location]
        individual.energy -= 0.025 + individual.params.sensor_range * 0.020
        if self.rng.random() < 0.18 + individual.params.sensor_range * 0.45:
            found = self.rng.choice(("essence", "residue_store", "mechanical"))
            amount = self.rng.uniform(0.2, 1.6) * (0.5 + individual.params.sensor_range)
            place.resources[found] = min(180.0, place.resources[found] + amount)

    def _tap(self, individual: Individual, feedback: dict[str, float]) -> None:
        # Era 2's installed perception demand (docs/ENV_AXIS_REVIEW.md):
        # release the place's sealed reserve, keyed to the active cue
        # channel. The reserve level and the cue are both observable dims, so
        # a discriminating tap is pure perception; a constant-tap policy
        # bleeds on misfires. With tap_cue_drift=1 the cue channel identity
        # re-draws each refresh, so the contract must be re-discovered
        # in-lifetime - the re-mapping pressure.
        place = self.world.places[individual.location]
        cue = float(place.physics.get(self.tap_cue_channel, 0.0))
        if place.sealed_essence <= 0.05 or cue < self.tap_cue_level:
            individual.energy -= 0.05
            individual.health -= 0.004
            individual.record_tap(success=False)
            self.tap_outcomes["mistap"] += 1
            if individual.health <= 0.0:
                self._deactivate(individual, "overload")
            return
        headroom = min(1.0, (cue - self.tap_cue_level) / max(0.05, 1.0 - self.tap_cue_level) + 0.25)
        release = min(place.sealed_essence, 2.0 + 8.0 * headroom)
        place.sealed_essence -= release
        share = release * (0.45 + individual.params.essence_conversion * 0.25)
        individual.energy += share
        place.resources["essence"] = min(180.0, place.resources["essence"] + max(0.0, release - share) * 0.8)
        individual.record_tap(success=True)
        individual.record_success("tap", min(1.0, release / 10.0))
        self.tap_outcomes["tap"] += 1
        feedback["tap"] += min(1.0, release / 10.0)

    def _drain(self, individual: Individual) -> None:
        # Energy-transfer contest between co-located individuals. The actor
        # commits without target information: the recipient is drawn at
        # random from the co-located roster (the era-1 weakest-target pick
        # was harness-side cognition - docs/ENV_AXIS_REVIEW.md). The
        # perception demand is the precondition: act only where the place is
        # crowded (occupancy is an observable dim).
        local = [self.individuals[oid] for oid in self._active_ids_at(individual.location) if oid != individual.id]
        if not local:
            individual.energy -= 0.04
            return
        target = self.rng.choice(local)
        draw_load = individual.params.mobility * 0.40 + individual.params.manipulator * 0.35 + individual.params.mechanical_use * 0.25
        resistance = target.params.resilience * 0.45 + target.params.mobility * 0.25 + target.health * 0.20
        strain = max(0.0, draw_load - resistance + self.rng.gauss(0.0, 0.05))
        individual.energy -= 0.10 + draw_load * 0.08
        if strain > 0.0:
            target.health -= strain
        if target.kind == "agent" and target.health > 0.0:
            feedback_base = (
                target.params.resilience * 0.16
                + target.params.mobility * 0.14
                + target.params.manipulator * 0.10
            )
            feedback_window = feedback_base - draw_load * 0.38 + self.rng.gauss(0.0, 0.035)
            if feedback_window > 0.12:
                feedback_load = min(0.32, (feedback_window - 0.12) * 0.42)
                individual.health -= feedback_load
                target.energy -= 0.025 + feedback_load * 0.05
                if individual.health <= 0.0:
                    self._deactivate(individual, "overload")
        if target.health <= 0.0 and individual.alive:
            gained = target.energy * (0.30 + individual.params.essence_conversion * 0.45)
            individual.energy += max(0.0, gained)
            self._deactivate(target, "depletion")

    def _signal(self, individual: Individual, feedback: dict[str, float]) -> None:
        intensity = individual.params.signal_strength * (0.5 + individual.energy / max(1.0, individual.storage_limit()))
        if intensity <= 0.01:
            individual.energy -= 0.01
            return
        token = individual.choose_signal_token()
        individual.energy -= 0.025 + intensity * 0.045
        self.world.emit_signal(individual.location, individual.id, token, intensity)
        feedback["social"] += intensity * 0.1

    def _coordinate_combine(self, individual: Individual, feedback: dict[str, float]) -> None:
        self.spawn_attempts["coordinate"] += 1
        if not individual.adult():
            self.spawn_failures["coordinate_not_adult"] += 1
            individual.energy -= 0.015
            return
        if individual.energy < self._combine_reserve_threshold(individual) * 0.82:
            self.spawn_failures["coordinate_low_energy"] += 1
            individual.energy -= 0.020
            return
        window_scale = max(0.0, float(getattr(self.config, "combine_intent_window_scale", 1.0)))
        window = int(round((6 + individual.params.signal_strength * 8.0 + individual.params.pairing_selectivity * 5.0) * window_scale))
        token = individual.choose_signal_token()
        intensity = 0.10 + individual.params.signal_strength * 0.45 + individual.params.pairing_selectivity * 0.10
        individual.combine_intent_until = max(individual.combine_intent_until, self.tick + window)
        individual.coordination_token = token
        individual.energy -= 0.035 + intensity * 0.040
        self.world.emit_signal(individual.location, individual.id, token, intensity)
        feedback["social"] += intensity * 0.12

    def _clone_perturb(self, individual: Individual, feedback: dict[str, float]) -> None:
        self.spawn_attempts["clone_perturb"] += 1
        if not individual.adult() or self.active_total >= self.config.max_pool:
            self.spawn_failures["clone_perturb_not_adult_or_cap"] += 1
            individual.energy -= 0.02
            return
        place = self.world.places[individual.location]
        if len(self._active_ids_at(individual.location)) >= place.capacity:
            self.spawn_failures["clone_perturb_local_capacity"] += 1
            individual.energy -= 0.015
            return
        decision = self.optimization.plan_clone_perturb(individual)
        if decision.failure or decision.plan is None:
            self.spawn_failures[decision.failure or "clone_perturb_no_plan"] += 1
            individual.energy -= decision.energy_penalty
            return
        child = self._instantiate_child(decision.plan)
        if child:
            self._apply_parent_costs_and_counts(decision.plan)
            self.births_by_mode[decision.plan.operator] += 1
            feedback["spawning"] += 1.0
            self.observer.observe(
                self.tick,
                "birth",
                {
                    "mode": decision.plan.operator,
                    "child_id": child.id,
                    "parent_ids": list(decision.plan.parent_ids),
                    "kind": child.kind,
                    "place": child.location,
                    "cycle": child.cycle,
                    "line_root_id": child.line_root_id,
                    "parent_line_ids": list(child.parent_line_ids),
                    "inherited_controller_template": child.inherited_controller_template,
                    "complexity": child.params.complexity(),
                },
                subjects=self._subjects(
                    child,
                    extra=[
                        f"individual:{individual.id}",
                        f"line:{individual.line_root_id or individual.id}",
                        "mode:clone_perturb",
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
                        "line_root_id": child.line_root_id,
                        "parent_line_ids": list(child.parent_line_ids),
                        "inherited_controller_template": child.inherited_controller_template,
                    },
                )
        else:
            self.spawn_failures["clone_perturb_add_failed"] += 1

    def _resolve_combine(self, place_ids: set[int], feedback: dict[int, dict[str, float]]) -> None:
        for place_id in place_ids:
            candidates = [
                individual
                for individual in self.individuals.values()
                if individual.alive
                and individual.location == place_id
                and individual.adult()
                and individual.combine_intent_until >= self.tick
            ]
            if len(candidates) < 2:
                if candidates:
                    self.spawn_failures["combine_no_partner"] += len(candidates)
                continue
            self.rng.shuffle(candidates)
            paired: set[int] = set()
            choices: dict[int, int] = {}
            for individual in candidates:
                self.spawn_attempts["combine_pairing"] += 1
                if individual.energy < self._combine_reserve_threshold(individual):
                    self.spawn_failures["combine_low_energy"] += 1
                    continue
                viable = [
                    other
                    for other in candidates
                    if other.id != individual.id
                    and other.id not in paired
                    and individual.energy >= self.optimization.combine_reserve_threshold(individual)
                    and other.energy >= self.optimization.combine_reserve_threshold(other)
                    and self.optimization.compatible_for_combine(individual, other)
                ]
                if not viable:
                    self.spawn_failures["combine_no_compatible_partner"] += 1
                    continue
                choices[individual.id] = max(viable, key=lambda other: self._partner_score(individual, other)).id
            for individual in candidates:
                if individual.id in paired or individual.id not in choices:
                    continue
                partner_id = choices[individual.id]
                partner = self.individuals.get(partner_id)
                if partner is None or not partner.alive or partner.id in paired:
                    continue
                if choices.get(partner.id) != individual.id and self.rng.random() > 0.35:
                    self.spawn_failures["combine_unreciprocated_choice"] += 1
                    continue
                child = self._combine(individual, partner)
                if child:
                    paired.add(individual.id)
                    paired.add(partner.id)
                    individual.combine_intent_until = -1
                    partner.combine_intent_until = -1
                    feedback[individual.id]["spawning"] += 1.0
                    feedback[partner.id]["spawning"] += 1.0
                    if self.config.event_detail:
                        self.logger.event(
                            self.tick,
                            "birth",
                            {
                                "mode": "combine",
                                "child_id": child.id,
                                "parent_ids": [individual.id, partner.id],
                                "kind": child.kind,
                                "line_root_id": child.line_root_id,
                                "parent_line_ids": list(child.parent_line_ids),
                                "inherited_controller_template": child.inherited_controller_template,
                            },
                        )

    def _partner_score(self, chooser: Individual, candidate: Individual) -> float:
        # Era 2: condition-only quality. The skill-breadth and child-count
        # terms were accumulation leaks (docs/ENV_AXIS_REVIEW.md).
        visible_quality = (
            candidate.health * 0.40
            + min(1.0, candidate.energy / max(1.0, candidate.storage_limit())) * 0.35
            + candidate.params.mobility * 0.125
            + candidate.params.manipulator * 0.125
        )
        selectivity = chooser.params.pairing_selectivity
        return visible_quality * (0.3 + selectivity) - chooser.params.distance(candidate.params) * 0.25 + self.rng.random() * 0.05

    def _combine_reserve_threshold(self, individual: Individual) -> float:
        return self.optimization.combine_reserve_threshold(individual)

    def _combine(self, a: Individual, b: Individual) -> Individual | None:
        if self.active_total >= self.config.max_pool:
            self.spawn_failures["combine_pool_cap"] += 1
            return None
        decision = self.optimization.plan_combine(a, b)
        if decision.failure or decision.plan is None:
            self.spawn_failures[decision.failure or "combine_no_plan"] += 1
            return None
        child = self._instantiate_child(decision.plan)
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
                    "cycle": child.cycle,
                    "line_root_id": child.line_root_id,
                    "parent_line_ids": list(child.parent_line_ids),
                    "inherited_controller_template": child.inherited_controller_template,
                    "complexity": child.params.complexity(),
                },
                subjects=self._subjects(
                    child,
                    extra=[
                        f"individual:{a.id}",
                        f"individual:{b.id}",
                        f"line:{a.line_root_id or a.id}",
                        f"line:{b.line_root_id or b.id}",
                        "mode:combine",
                    ],
                ),
                score=0.55 + child.cycle * 0.09 + child.params.complexity() * 0.10,
                rarity_key=f"birth:{decision.plan.operator}:{child.kind}",
            )
        else:
            self.spawn_failures["combine_add_failed"] += 1
        return child

    def _instantiate_child(self, plan: ChildPlan) -> Individual | None:
        child = self.add_individual(
            plan.child_kind,
            plan.child_params,
            plan.location,
            plan.child_energy,
            plan.cycle,
            plan.parent_ids,
            plan.controller_template,
        )
        if child is not None:
            note = getattr(child.controller_template, "birth_structural_op", None)
            if note:
                child.controller_template.birth_structural_op = None
                self.structural_steps[note["op"]] += 1
                self.logger.structure_event(
                    {
                        "tick": self.tick,
                        "child_id": child.id,
                        "parent_ids": list(plan.parent_ids),
                        "line_root_id": child.line_root_id,
                        "mode": plan.operator,
                        **note,
                    }
                )
        return child

    def _apply_parent_costs_and_counts(self, plan: ChildPlan) -> None:
        for parent_id, cost in plan.parent_costs.items():
            parent = self.individuals.get(parent_id)
            if parent is None:
                continue
            parent.energy -= cost
            parent.child_count += 1
            parent.record_success("spawning", 1.0)

    def _deactivate(self, individual: Individual, cause: str) -> None:
        if not individual.alive:
            return
        individual.alive = False
        self.active_total = max(0, self.active_total - 1)
        self.active_by_kind[individual.kind] = max(0, self.active_by_kind[individual.kind] - 1)
        if individual.neural:
            self.active_neural = max(0, self.active_neural - 1)
        self.deaths_by_cause[cause] += 1
        self.deaths_by_kind_cause[f"{individual.kind}:{cause}"] += 1
        place = self.world.places[individual.location]
        place.resources["residue_store"] = min(180.0, place.resources["residue_store"] + max(0.0, individual.energy) * 0.35 + 2.0)
        checkpoint_score = self._checkpoint_score(individual) if individual.controller is not None else 0.0
        notable = (
            individual.child_count >= 3
            or individual.successful_taps >= 2
            or individual.success_profile.get("prediction_fit", 0.0) >= 2.0
        )
        if individual.controller is not None and (
            notable
        ):
            self.checkpoints.save_controller(
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
                    "individual_id": individual.id,
                    "kind": individual.kind,
                    "cause": cause,
                    "place": place.id,
                    "age": individual.age,
                    "line_root_id": individual.line_root_id,
                    "child_count": individual.child_count,
                    "successful_taps": individual.successful_taps,
                    "score": checkpoint_score,
                    "success_profile": dict(individual.success_profile),
                    "architecture": (
                        {
                            "blocks": len(individual.controller.blocks),
                            "capacity": individual.controller.capacity,
                        }
                        if individual.controller is not None and hasattr(individual.controller, "blocks")
                        else None
                    ),
                },
                subjects=self._subjects(individual, place.id, [f"cause:{cause}"]),
                score=0.75 + min(4.0, checkpoint_score / 8.0),
                rarity_key=f"death:{cause}",
            )
        if self.config.event_detail:
            self.logger.event(self.tick, "death", {"individual_id": individual.id, "cause": cause, "kind": individual.kind})
        individual.controller = None
        individual.controller_template = None

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
            for individual in self.individuals.values():
                if individual.alive and individual.location in affected_ids:
                    individual.health -= damage
                    if individual.health <= 0.0:
                        self._deactivate(individual, "intervention_disaster")
        elif intervention.kind == "climate_shift":
            self.world.climate_drift = max(-0.5, min(0.5, self.world.climate_drift + float(payload.get("amount", 0.0))))
        elif intervention.kind == "add_individuals":
            kind = str(payload.get("kind", "collector"))
            count = int(payload.get("count", 1))
            place = int(payload.get("place", self.rng.randrange(len(self.world.places))))
            for _ in range(count):
                params = ParamVector.neural(self.rng) if kind == "agent" else ParamVector.converter(self.rng) if kind == "converter" else ParamVector.collector(self.rng)
                self.add_individual(kind, params, place, float(payload.get("energy", 25.0)))
        record = {"tick": self.tick, "kind": intervention.kind, "payload": payload, "reason": intervention.reason}
        self.interventions_applied.append(record)
        self.logger.event(self.tick, "intervention", record)

    def _prediction_fit_rate(self, individual: Individual) -> float:
        # Fit per tick alive: the era-1 accumulator rewarded longevity, not
        # learning quality (docs/ENV_AXIS_REVIEW.md).
        return individual.success_profile.get("prediction_fit", 0.0) / max(1.0, float(individual.age)) * 100.0

    def _tap_discrimination(self, individual: Individual) -> float:
        attempts = individual.successful_taps + individual.mistap_count
        if attempts == 0:
            return 0.0
        return individual.successful_taps / attempts * math.log1p(attempts)

    def _checkpoint_score(self, individual: Individual) -> float:
        # Era 2 re-aim: lead with perception-coupled signals (fit rate, tap
        # discrimination); demote accumulation (child_count, age, capacity).
        energy_ratio = individual.energy / max(1.0, individual.storage_limit())
        profile = individual.success_profile
        profile_score = (
            math.log1p(profile.get("energy_gain", 0.0)) * 0.9
            + math.log1p(profile.get("tap", 0.0)) * 1.6
            + math.log1p(profile.get("spawning", 0.0)) * 1.2
        )
        return (
            self._prediction_fit_rate(individual) * 3.0
            + self._tap_discrimination(individual) * 2.0
            + individual.child_count * 1.5
            + individual.cycle * 0.25
            + energy_ratio * 1.0
            + profile_score
        )

    def _checkpoint_context(self, label: str, criterion: str) -> dict[str, Any]:
        return {
            "label": label,
            "criterion": criterion,
            "pool": pool_counts(self.individuals),
            "world_energy": world_energy_summary(self.world),
            "world_physics": world_physics_summary(self.world),
            "lines": self._line_summary(limit=5),
        }

    def _save_checkpoint_candidate(self, individual: Individual, label: str, criterion: str, bucket: str) -> None:
        score = self._checkpoint_score(individual)
        saved = self.checkpoints.save_controller(
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
                    "individual_id": individual.id,
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
        candidates = [individual for individual in self.individuals.values() if individual.alive and individual.controller is not None]
        if not candidates:
            return
        saved_ids: set[int] = set()

        overall = max(candidates, key=self._checkpoint_score)
        self._save_checkpoint_candidate(overall, label, "overall_champion", "interval_champion")
        saved_ids.add(overall.id)

        spawn = self._best_checkpoint_candidate(candidates, saved_ids, lambda individual: (individual.child_count, individual.cycle, individual.energy, individual.age))
        if spawn is not None and spawn.child_count > 0:
            self._save_checkpoint_candidate(spawn, label, "spawn_champion", "spawn_champion")
            saved_ids.add(spawn.id)

        tapper = self._best_checkpoint_candidate(candidates, saved_ids, lambda individual: (self._tap_discrimination(individual), individual.successful_taps, individual.energy))
        if tapper is not None and tapper.successful_taps > 0:
            self._save_checkpoint_candidate(tapper, label, "tap_champion", "tap_champion")
            saved_ids.add(tapper.id)

        learner = self._best_checkpoint_candidate(
            candidates,
            saved_ids,
            lambda individual: (
                self._prediction_fit_rate(individual),
                -sum(abs(value) for value in individual.prediction_error_profile),
                individual.energy,
            ),
        )
        if learner is not None and learner.success_profile.get("prediction_fit", 0.0) > 0.0:
            self._save_checkpoint_candidate(learner, label, "learner_champion", "learner_champion")
            saved_ids.add(learner.id)

        line = self._best_checkpoint_candidate(candidates, saved_ids, lambda individual: (individual.cycle, individual.child_count, individual.energy, individual.age))
        if line is not None and (line.cycle > 0 or line.child_count > 0):
            self._save_checkpoint_candidate(line, label, "line_founder", "line_founder")

    def _line_summary(self, limit: int = 8) -> dict[str, Any]:
        profile_keys = (
            "energy_gain",
            "prediction_fit",
            "tap",
            "spawning",
        )
        rows: dict[int, dict[str, Any]] = {}
        members: dict[int, list[Individual]] = defaultdict(list)
        for individual in self.individuals.values():
            if individual.kind != "agent":
                continue
            root = individual.line_root_id or individual.id
            if root not in rows:
                rows[root] = {
                    "line_root_id": root,
                    "born": 0,
                    "active": 0,
                    "active_neural": 0,
                    "dead": 0,
                    "max_generation": 0,
                    "child_total": 0,
                    "successful_taps_total": 0,
                    "tappers": 0,
                    "inherited_template_count": 0,
                    "profile": Counter(),
                    "energy_total": 0.0,
                    "health_total": 0.0,
                }
            row = rows[root]
            row["born"] += 1
            row["max_generation"] = max(row["max_generation"], individual.cycle)
            row["child_total"] += individual.child_count
            row["successful_taps_total"] += individual.successful_taps
            row["tappers"] += int(individual.successful_taps > 0)
            row["inherited_template_count"] += int(individual.inherited_controller_template)
            for key in profile_keys:
                row["profile"][key] += individual.success_profile.get(key, 0.0)
            if individual.alive:
                row["active"] += 1
                row["active_neural"] += int(individual.neural)
                row["energy_total"] += individual.energy
                row["health_total"] += individual.health
            else:
                row["dead"] += 1
            members[root].append(individual)

        def member_score(individual: Individual) -> float:
            return (
                individual.child_count * 4.0
                + individual.successful_taps * 1.8
                + individual.success_profile.get("prediction_fit", 0.0) * 1.1
                + individual.success_profile.get("tap", 0.0) * 1.0
                + individual.success_profile.get("spawning", 0.0) * 1.2
                + individual.energy / max(1.0, individual.storage_limit())
            )

        summaries: list[dict[str, Any]] = []
        for root, row in rows.items():
            active_members = [individual for individual in members[root] if individual.alive]
            top_members = sorted(active_members, key=member_score, reverse=True)[:5]
            active = int(row["active"])
            profile = row["profile"]
            line_score = (
                active * 6.0
                + int(row["max_generation"]) * 1.8
                + int(row["child_total"]) * 1.2
                + int(row["successful_taps_total"]) * 1.6
                + float(profile.get("prediction_fit", 0.0)) * 0.9
                + float(profile.get("tap", 0.0)) * 1.0
            )
            summaries.append(
                {
                    "line_root_id": root,
                    "born": int(row["born"]),
                    "active": active,
                    "active_neural": int(row["active_neural"]),
                    "dead": int(row["dead"]),
                    "max_generation": int(row["max_generation"]),
                    "child_total": int(row["child_total"]),
                    "successful_taps_total": int(row["successful_taps_total"]),
                    "tappers": int(row["tappers"]),
                    "inherited_template_count": int(row["inherited_template_count"]),
                    "avg_active_energy": round(float(row["energy_total"]) / max(1, active), 5),
                    "avg_active_health": round(float(row["health_total"]) / max(1, active), 5),
                    "score": round(line_score, 5),
                    "top_active_ids": [individual.id for individual in top_members],
                    "profile": {key: round(value, 5) for key, value in sorted(profile.items()) if value > 0.0},
                }
            )
        top_active = sorted(
            (row for row in summaries if row["active"] > 0),
            key=lambda row: (row["active"], row["max_generation"], row["child_total"], row["successful_taps_total"], row["score"]),
            reverse=True,
        )[:limit]
        top_all_time = sorted(summaries, key=lambda row: row["score"], reverse=True)[:limit]
        return {
            "agent_lines_total": len(summaries),
            "active_agent_lines": sum(1 for row in summaries if row["active"] > 0),
            "top_active": top_active,
            "top_all_time": top_all_time,
        }

    def _log_aggregate(self) -> None:
        active = [individual for individual in self.individuals.values() if individual.alive]
        neural = [individual for individual in active if individual.neural]
        avg_energy = sum(individual.energy for individual in active) / max(1, len(active))
        avg_complexity = sum(individual.params.complexity() for individual in active) / max(1, len(active))
        # Controller capacity & attention stats — these only make sense for neural agents
        # with controllers. With controller growth active and neuroplastic attention active,
        # tracking how these distributions optimize over the run is what tells you
        # whether the substrate is selecting for richer cognition.
        controller_sizes = [individual.controller.hidden_size for individual in neural if individual.controller is not None]
        if controller_sizes:
            controller_capacity = {
                "count": len(controller_sizes),
                "mean": round(sum(controller_sizes) / len(controller_sizes), 3),
                "max": max(controller_sizes),
                "min": min(controller_sizes),
                "p90": sorted(controller_sizes)[int(len(controller_sizes) * 0.9)] if len(controller_sizes) >= 10 else max(controller_sizes),
            }
        else:
            controller_capacity = {"count": 0, "mean": 0.0, "max": 0, "min": 0, "p90": 0}
        # Architecture census: the signal a structural-evolution run is FOR.
        # Tracks how many lines run modular controllers, how many blocks
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
                "legacy": len(controller_sizes) - len(modular),
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
            architecture_stats = {"modular": 0, "legacy": len(controller_sizes)}
        attention_stats: dict[str, float | int] = {"count": 0}
        attended_controllers = [
            individual.controller
            for individual in neural
            if individual.controller is not None and individual.controller._has_attention() and individual.controller.last_attention.size
        ]
        if attended_controllers:
            # Concentration measure: max(fidelity) - mean(fidelity). High when controller
            # is focusing on a few features, low when spread uniformly. Good signal
            # for "how trained" the attention head is on this individual.
            concentrations: list[float] = []
            mean_max_fidelity = 0.0
            for controller in attended_controllers:
                fidelity = controller.last_attention
                concentrations.append(float(fidelity.max() - fidelity.mean()))
                mean_max_fidelity += float(fidelity.max())
            attention_stats = {
                "count": len(attended_controllers),
                "mean_max_fidelity": round(mean_max_fidelity / len(attended_controllers), 4),
                "mean_concentration": round(sum(concentrations) / len(concentrations), 4),
                "max_concentration": round(max(concentrations), 4),
            }
        aggregate = {
            "tick": self.tick,
            "pool": pool_counts(self.individuals),
            "avg_energy": round(avg_energy, 5),
            "avg_complexity": round(avg_complexity, 5),
            "neural_avg_energy": round(sum(o.energy for o in neural) / max(1, len(neural)), 5),
            "controller_capacity": controller_capacity,
            "architecture": architecture_stats,
            "attention_stats": attention_stats,
            "births": dict(self.births_by_mode),
            "deaths": dict(self.deaths_by_cause),
            "deaths_by_kind_cause": dict(self.deaths_by_kind_cause),
            "tap_outcomes": dict(self.tap_outcomes),
            "tap_cue_channel": self.tap_cue_channel,
            "tap_cue_level": round(self.tap_cue_level, 5),
            "staple_cue_channel": self.staple_cue_channel,
            "staple_cue_level": round(self.staple_cue_level, 5),
            "patch_recovery_triggers": self.patch_recovery_triggers,
            "structural_steps": dict(self.structural_steps),
            "success_profile": success_profile_summary(self.individuals),
            "lines": self._line_summary(),
            "physics_events": dict(self.physics_events),
            "spawn_attempts": dict(self.spawn_attempts),
            "spawn_failures": dict(self.spawn_failures),
            "action_counts": dict(self.action_counts),
            "infeasible_commits": dict(self.infeasible_commits),
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
        # Live heartbeat: a small atomically-replaced snapshot so a running
        # sim can be checked (`cat status.json`) without parsing events.jsonl.
        elapsed = time.monotonic() - getattr(self, "_run_started", time.monotonic())
        self.logger.write_json_atomic(
            "status.json",
            {
                "tick": self.tick,
                "elapsed_seconds": round(elapsed, 1),
                "ticks_per_second": round(self.tick / elapsed, 2) if elapsed > 0 else 0.0,
                "pool": aggregate["pool"],
                "neural_avg_energy": aggregate["neural_avg_energy"],
                "architecture": aggregate["architecture"],
                "controller_capacity": aggregate["controller_capacity"],
                "births": aggregate["births"],
                "deaths": aggregate["deaths"],
                "structural_steps": dict(self.structural_steps),
                "patch_recovery_triggers": self.patch_recovery_triggers,
                "stories_promoted": self.observer.promoted,
            },
        )
        self.logger.flush()
