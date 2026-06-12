from __future__ import annotations

import json
import tempfile
import unittest
from random import Random

from microcosmic_god.backends import ControllerLearningCase
from microcosmic_god.controller import PREDICTION_HEADS, TinyController
from microcosmic_god.config import RunConfig
from microcosmic_god.params import ParamVector
from microcosmic_god.individuals import ACTIONS, OBSERVATION_SIZE
from microcosmic_god.simulation import Simulation


def make_sim(seed: int = 101, places: int = 3, environment_harshness: float = 1.0) -> Simulation:
    tmp = tempfile.TemporaryDirectory()
    config = RunConfig(
        seed=seed,
        profile="test",
        max_ticks=10,
        max_wall_seconds=0,
        places=places,
        initial_collectors=0,
        initial_converters=0,
        initial_agents=0,
        max_pool=50,
        output_dir=tmp.name,
        event_detail=False,
        environment_harshness=environment_harshness,
    )
    sim = Simulation(config)
    sim._tmpdir = tmp  # type: ignore[attr-defined]
    return sim


def close_sim(sim: Simulation) -> None:
    sim.logger.close()
    sim._tmpdir.cleanup()  # type: ignore[attr-defined]


class CoreContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        sim = getattr(self, "sim", None)
        if sim is not None:
            sim.logger.close()
            sim._tmpdir.cleanup()  # type: ignore[attr-defined]

    def test_minute_profile_defaults_to_harsher_environment(self) -> None:
        self.assertEqual(RunConfig.from_profile("smoke").environment_harshness, 1.0)
        self.assertGreater(RunConfig.from_profile("minute").environment_harshness, 1.0)
        self.assertGreaterEqual(RunConfig.from_profile("modal").environment_harshness, RunConfig.from_profile("minute").environment_harshness)

    def test_environment_harshness_reduces_easy_persistence_budget(self) -> None:
        mild = make_sim(seed=909, places=12, environment_harshness=1.0)
        harsh = make_sim(seed=909, places=12, environment_harshness=1.6)
        try:
            mild_capacity = sum(place.capacity for place in mild.world.places)
            harsh_capacity = sum(place.capacity for place in harsh.world.places)
            mild_easy_energy = sum(place.resources["essence"] + place.resources["residue_store"] for place in mild.world.places)
            harsh_easy_energy = sum(place.resources["essence"] + place.resources["residue_store"] for place in harsh.world.places)
            mild_sealed_energy = sum(place.sealed_essence + place.resources["mechanical"] for place in mild.world.places)
            harsh_sealed_energy = sum(place.sealed_essence + place.resources["mechanical"] for place in harsh.world.places)
            mild_exposure = sum(float(mild._place_exposure_pressure(place)["severity"]) for place in mild.world.places)
            harsh_exposure = sum(float(harsh._place_exposure_pressure(place)["severity"]) for place in harsh.world.places)

            self.assertLess(harsh_capacity, mild_capacity)
            self.assertLess(harsh_easy_energy, mild_easy_energy)
            self.assertGreater(harsh_sealed_energy, mild_sealed_energy)
            self.assertGreater(harsh_exposure, mild_exposure)
            self.assertEqual(harsh.world.to_summary()["environment_harshness"], 1.6)
        finally:
            close_sim(mild)
            close_sim(harsh)

    def test_harshness_amplifies_unbuffered_exposure_damage(self) -> None:
        mild = make_sim(seed=919, places=1, environment_harshness=1.0)
        harsh = make_sim(seed=919, places=1, environment_harshness=1.6)
        try:
            losses: list[float] = []
            for sim in (mild, harsh):
                place = sim.world.places[0]
                place.volatility = 0.32
                place.geothermal = 0.0
                place.resources["thermal"] = 0.0
                place.physics.update(
                    {
                        "temperature": 0.08,
                        "humidity": 0.95,
                        "fluid_level": 0.12,
                        "pressure": 0.0,
                        "current_exposure": 0.58,
                        "elevation": 0.86,
                        "abrasion": 0.65,
                        "wet_dry_cycle": 0.76,
                        "shelter": 0.0,
                        "salinity": 0.0,
                    }
                )
                place.terrain.update({"aquatic": 0.0, "depth": 0.0, "humidity": 0.95, "salinity": 0.0})
                params = ParamVector.neural(sim.rng)
                params.thermal_tolerance = 0.0
                params.resilience = 0.0
                agent = sim.add_individual("agent", params, 0, 80.0)
                assert agent is not None

                sim._terrain_stress(agent)

                losses.append(1.0 - agent.health)

            self.assertGreater(losses[0], 0.0)
            self.assertGreater(losses[1], losses[0] * 1.25)
            self.assertGreater(harsh.physics_events["harsh_exposure_pressure"], 0)
        finally:
            close_sim(mild)
            close_sim(harsh)

    def test_all_signal_tokens_are_observed(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 50.0)
        assert agent is not None
        agent.event_memory = [index / 20.0 for index in range(8)]
        agent.signal_values = [index / 10.0 for index in range(8)]

        observation = self.sim._observe(agent, self.sim._rosters())

        self.assertEqual(len(observation), OBSERVATION_SIZE)
        self.assertEqual(observation[-16:-8], agent.event_memory)
        self.assertEqual(observation[-8:], agent.signal_values)

    def test_action_results_feed_short_event_memory(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 50.0)
        assert agent is not None

        agent.record_action_result(
            action_index=ACTIONS.index("tap"),
            energy_delta=4.0,
            health_delta=-0.05,
            damage=0.05,
            prediction_error=0.7,
            spawn_feedback=0.0,
            social_feedback=0.2,
            tap_feedback=1.0,
        )

        self.assertEqual(agent.last_action, "tap")
        self.assertGreater(agent.event_memory[0], 0.0)
        self.assertGreater(agent.event_memory[3], 0.0)
        self.assertGreater(agent.event_memory[6], 0.0)
        self.assertGreater(agent.event_memory[7], 0.0)

    def test_checkpoints_capture_cognitive_context(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 50.0)
        assert agent is not None
        agent.record_action_result(ACTIONS.index("tap"), 3.0, 0.0, 0.0, 0.25, 0.0, 0.1, 1.0)
        saved = self.sim.checkpoints.save_controller(42, agent, "test_cognition", {}, bucket="general")
        self.assertTrue(saved)

        checkpoint = next(self.sim.logger.checkpoint_dir.glob("controller_t00000042_*.json"))
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertIn("cognition", payload)
        self.assertEqual(payload["cognition"]["last_action"], "tap")
        self.assertIn("prediction_errors", payload["cognition"])
        self.assertGreater(payload["cognition"]["event_memory"]["tap"], 0.0)

    def test_line_metadata_tracks_inherited_agent_templates(self) -> None:
        self.sim = make_sim(places=1)
        parent = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 80.0)
        assert parent is not None and parent.controller_template is not None

        child = self.sim.add_individual(
            "agent",
            ParamVector.neural(self.sim.rng),
            0,
            40.0,
            cycle=parent.cycle + 1,
            parent_ids=(parent.id,),
            controller_template=parent.controller_template,
        )
        assert child is not None

        self.assertEqual(parent.line_root_id, parent.id)
        self.assertEqual(child.line_root_id, parent.id)
        self.assertEqual(child.parent_line_ids, (parent.id,))
        self.assertTrue(child.inherited_controller_template)
        self.assertEqual(child.to_summary()["line_root_id"], parent.id)
        self.assertEqual(child.cognitive_snapshot()["line"]["root_id"], parent.id)

        line_summary = self.sim._line_summary()
        self.assertEqual(line_summary["agent_lines_total"], 1)
        self.assertEqual(line_summary["top_active"][0]["active"], 2)
        self.assertEqual(line_summary["top_active"][0]["inherited_template_count"], 1)

    def test_drain_uses_current_location_not_tick_start_roster(self) -> None:
        self.sim = make_sim(places=2)
        drainer = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 80.0)
        target = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 80.0)
        assert drainer is not None and target is not None
        _stale_roster = self.sim._rosters()
        target.location = 1
        before_health = target.health

        self.sim._drain(drainer)

        self.assertEqual(target.health, before_health)
        self.assertEqual(target.location, 1)

    def test_drain_resistance_and_return_come_from_params_only(self) -> None:
        # The era-2 drain has no helpers, skills, or shelter terms: the target
        # is drawn at random from the co-located roster and the contest reads
        # body params + health only. A weak drainer against a resilient agent
        # transfers no strain and takes the feedback load itself.
        self.sim = make_sim(places=1)
        drainer_params = ParamVector.neural(self.sim.rng)
        drainer_params.mobility = 0.20
        drainer_params.manipulator = 0.20
        drainer_params.mechanical_use = 0.20
        target_params = ParamVector.neural(self.sim.rng)
        target_params.resilience = 1.00
        target_params.mobility = 1.00
        target_params.manipulator = 1.00
        drainer = self.sim.add_individual("agent", drainer_params, 0, 80.0)
        target = self.sim.add_individual("agent", target_params, 0, 80.0)
        assert drainer is not None and target is not None
        drainer.health = 0.50
        target.health = 1.00

        class ZeroRng:
            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

        self.sim.rng = ZeroRng()  # type: ignore[assignment]
        before_target_health = target.health
        before_target_energy = target.energy
        before_drainer_health = drainer.health
        before_drainer_energy = drainer.energy

        self.sim._drain(drainer)

        # strain = draw_load 0.20 - resistance 0.90 <= 0: no health transfer.
        self.assertAlmostEqual(target.health, before_target_health)
        # feedback window 0.40 - 0.20*0.38 = 0.324 > 0.12: the load returns.
        self.assertLess(drainer.health, before_drainer_health)
        self.assertLess(target.energy, before_target_energy)
        self.assertLess(drainer.energy, before_drainer_energy)
        self.assertTrue(drainer.alive)
        self.assertTrue(target.alive)

    def test_clone_perturb_capacity_uses_current_local_pool(self) -> None:
        self.sim = make_sim(places=2)
        parent_params = ParamVector.collector(self.sim.rng)
        parent = self.sim.add_individual("collector", parent_params, 0, 200.0)
        neighbor = self.sim.add_individual("collector", ParamVector.collector(self.sim.rng), 0, 20.0)
        assert parent is not None and neighbor is not None
        self.sim.world.places[0].capacity = 2
        parent.age = 100
        neighbor.location = 1

        self.sim._clone_perturb(parent, {"spawning": 0.0, "social": 0.0})

        self.assertEqual(self.sim.births_by_mode["clone_perturb"], 1)
        self.assertEqual(len(self.sim._active_ids_at(0)), 2)

    def test_spawning_actions_are_operator_labels(self) -> None:
        self.assertIn("clone_perturb", ACTIONS)
        self.assertIn("coordinate", ACTIONS)
        self.assertNotIn("solo_spawn", ACTIONS)
        self.assertNotIn("mate", ACTIONS)

    def test_complex_neural_agents_can_clone_with_soft_strain(self) -> None:
        self.sim = make_sim()
        self.sim.config.clone_complexity_soft_limit = 0.0
        parent_params = ParamVector.neural(self.sim.rng)
        parent_params.neural_budget = 32.0
        parent_params.memory_budget = 16.0
        parent = self.sim.add_individual("agent", parent_params, 0, 1_000.0)
        assert parent is not None
        parent.age = 100

        self.sim._clone_perturb(parent, {"spawning": 0.0, "social": 0.0})

        self.assertEqual(self.sim.births_by_mode["clone_perturb"], 1)
        self.assertNotIn("clone_perturb_complexity_ceiling", self.sim.spawn_failures)

    def test_exposure_pressure_is_buffered_by_shelter_physics(self) -> None:
        # Era 2: exposure buffering is place physics (shelter), not artifacts.
        self.sim = make_sim(places=2)
        for place in self.sim.world.places[:2]:
            place.volatility = 0.30
            place.geothermal = 0.0
            place.resources["thermal"] = 0.0
            place.physics.update(
                {
                    "temperature": 0.07,
                    "humidity": 0.95,
                    "fluid_level": 0.12,
                    "pressure": 0.0,
                    "current_exposure": 0.55,
                    "elevation": 0.86,
                    "abrasion": 0.62,
                    "wet_dry_cycle": 0.74,
                    "shelter": 0.0,
                    "salinity": 0.0,
                }
            )
            place.terrain.update({"aquatic": 0.0, "depth": 0.0, "humidity": 0.95, "salinity": 0.0})
        self.sim.world.places[1].physics["shelter"] = 0.8
        fragile = ParamVector.neural(self.sim.rng)
        fragile.thermal_tolerance = 0.0
        fragile.resilience = 0.0
        exposed = self.sim.add_individual("agent", fragile, 0, 80.0)
        sheltered = self.sim.add_individual("agent", fragile, 1, 80.0)
        assert exposed is not None and sheltered is not None

        self.sim._terrain_stress(exposed)
        self.sim._terrain_stress(sheltered)

        self.assertLess(exposed.health, 1.0)
        self.assertGreater(sheltered.health, exposed.health)
        self.assertGreater(self.sim.physics_events["exposure_pressure"], 0)

    def test_inside_boundary_is_distinct_from_shelter(self) -> None:
        # interiority behind a low-permeability boundary is a hazard
        # (stagnant interior), not protection; shelter is the protective dim.
        self.sim = make_sim(places=3)
        open_place, sealed_place, sheltered_place = self.sim.world.places[:3]
        base_physics = {
            "temperature": 0.55,
            "humidity": 0.50,
            "fluid_level": 0.0,
            "pressure": 0.45,
            "current_exposure": 0.0,
            "elevation": 0.5,
            "abrasion": 0.0,
            "wet_dry_cycle": 0.0,
            "salinity": 0.0,
            "interiority": 0.0,
            "boundary_permeability": 0.0,
            "shelter": 0.0,
        }
        for place in (open_place, sealed_place, sheltered_place):
            place.volatility = 0.05
            place.geothermal = 0.0
            place.resources["thermal"] = 0.0
            place.physics.update(dict(base_physics))
            place.terrain.update({"aquatic": 0.0, "depth": 0.0, "humidity": 0.50, "salinity": 0.0})
        sealed_place.physics.update({"interiority": 0.9, "boundary_permeability": 0.05})
        sheltered_place.physics.update({"interiority": 0.9, "boundary_permeability": 0.05, "shelter": 0.9})

        self.assertGreater(
            self.sim._place_hazard_pressure(sealed_place),
            self.sim._place_hazard_pressure(open_place),
        )

        params = ParamVector.neural(self.sim.rng)
        params.thermal_tolerance = 0.5
        params.pressure_tolerance = 0.8
        params.resilience = 0.1
        params.aquatic_affinity = 0.0
        params.salinity_tolerance = 0.0
        in_open = self.sim.add_individual("agent", params, 0, 80.0)
        in_sealed = self.sim.add_individual("agent", params, 1, 80.0)
        in_sheltered = self.sim.add_individual("agent", params, 2, 80.0)
        assert in_open is not None and in_sealed is not None and in_sheltered is not None

        self.sim._terrain_stress(in_open)
        self.sim._terrain_stress(in_sealed)
        self.sim._terrain_stress(in_sheltered)

        self.assertLess(in_sealed.health, in_open.health)
        self.assertGreater(in_sheltered.health, in_sealed.health)

    def test_successful_movement_spends_energy_even_when_easy(self) -> None:
        self.sim = make_sim(places=2)
        params = ParamVector.neural(self.sim.rng)
        params.mobility = 1.0
        params.sensor_range = 0.0
        agent = self.sim.add_individual("agent", params, 0, 100.0)
        assert agent is not None
        origin = self.sim.world.places[0]
        destination = self.sim.world.places[1]
        origin.physics.update({"temperature": 0.50, "fluid_level": 0.0, "pressure": 0.0, "humidity": 0.45, "salinity": 0.0, "elevation": 0.20, "oxygen": 0.40})
        destination.physics.update({"temperature": 0.52, "fluid_level": 0.0, "pressure": 0.0, "humidity": 0.44, "salinity": 0.0, "elevation": 0.22, "oxygen": 0.41})
        destination.obstacles.update({"water": 0.0, "height": 0.0, "thorn": 0.0, "heat": 0.0})
        edge = self.sim.world.edge_between(0, 1)
        assert edge is not None
        edge.traversal_required = 0.0
        edge.distance = 1.0
        edge.danger = 0.0
        edge.slope = 0.0
        edge.current = 0.0

        class EasyMoveRng:
            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

        self.sim.rng = EasyMoveRng()  # type: ignore[assignment]
        before = agent.energy

        self.sim._move(agent)

        self.assertEqual(agent.location, 1)
        self.assertLess(agent.energy, before)

    def test_failed_movement_spends_energy(self) -> None:
        self.sim = make_sim(places=2)
        params = ParamVector.neural(self.sim.rng)
        params.mobility = 0.02
        params.sensor_range = 0.0
        params.aquatic_affinity = 0.0
        params.thermal_tolerance = 0.0
        params.pressure_tolerance = 0.0
        agent = self.sim.add_individual("agent", params, 0, 100.0)
        assert agent is not None
        origin = self.sim.world.places[0]
        destination = self.sim.world.places[1]
        origin.physics.update({"temperature": 0.92, "fluid_level": 0.0, "pressure": 0.0, "humidity": 0.08, "salinity": 0.0, "elevation": 0.90, "oxygen": 0.45})
        destination.physics.update({"temperature": 0.12, "fluid_level": 1.0, "pressure": 1.20, "humidity": 0.98, "salinity": 0.95, "elevation": 0.02, "oxygen": 0.12})
        destination.obstacles.update({"water": 1.0, "height": 1.0, "thorn": 1.0, "heat": 1.0})
        edge = self.sim.world.edge_between(0, 1)
        assert edge is not None
        edge.traversal_required = 1.0
        edge.distance = 1.8
        edge.danger = 1.0
        edge.slope = 1.0
        edge.current = -1.0

        class HardMoveRng:
            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

            def random(self) -> float:
                return 0.99

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

        self.sim.rng = HardMoveRng()  # type: ignore[assignment]
        before_energy = agent.energy
        before_health = agent.health

        self.sim._move(agent)

        self.assertEqual(agent.location, 0)
        self.assertLess(agent.energy, before_energy)
        self.assertLess(agent.health, before_health)

    def test_environment_generation_has_hostile_treasure_biomes(self) -> None:
        self.sim = make_sim(seed=55, places=10)
        places = {place.archetype: place for place in self.sim.world.places}

        self.assertEqual(self.sim.world.places[0].archetype, "pelagic")
        self.assertGreaterEqual(places["pelagic"].obstacles["water"], 0.72)
        self.assertGreater(places["trench"].physics["pressure"], 0.75)
        self.assertGreater(places["hydrothermal_vent"].resources["thermal"], 40.0)
        self.assertGreaterEqual(places["high_ridge"].physics["elevation"], 0.72)
        self.assertGreater(places["mineral_scree"].sealed_essence, 20.0)

    def test_controller_plasticity_can_update_representations(self) -> None:
        # Tests the core (non-attention) plasticity path. Attention plasticity
        # is exercised separately in AttentionTests below.
        import numpy as np

        controller = TinyController.random(Random(7), input_size=5, hidden_size=4, output_size=3, with_attention=False)
        inputs = [0.8, -0.2, 0.5, 0.0, 0.3]
        controller.forward(inputs)
        before_in = controller.weights_in.copy()
        before_out = controller.weights_out.copy()

        controller.learn(
            action_index=1,
            valence=1.2,
            energy_delta=0.6,
            learning_rate=0.20,
            plasticity=0.80,
            prediction_weight=0.70,
        )

        self.assertFalse(np.array_equal(before_out, controller.weights_out))
        self.assertFalse(np.array_equal(before_in, controller.weights_in))
        self.assertEqual(controller.input_trace.size, 5)
        self.assertEqual(controller.hidden_trace.size, 4)

    def test_controller_learns_multiple_prediction_heads(self) -> None:
        import numpy as np

        controller = TinyController.random(Random(9), input_size=5, hidden_size=4, output_size=3)
        controller.forward([0.5, -0.1, 0.4, 0.7, 0.2])
        before_damage = controller.auxiliary_prediction_weights["damage"].copy()
        before_tap = controller.auxiliary_prediction_weights["tap"].copy()

        controller.learn(
            action_index=2,
            valence=0.8,
            energy_delta=0.4,
            learning_rate=0.18,
            plasticity=0.90,
            prediction_weight=0.85,
            outcome_targets={"damage": 0.3, "spawning": 0.0, "social": 0.2, "tap": 1.0, "hazard": 0.1},
        )

        self.assertEqual(set(controller.last_prediction_errors), set(PREDICTION_HEADS))
        self.assertFalse(np.array_equal(before_damage, controller.auxiliary_prediction_weights["damage"]))
        self.assertFalse(np.array_equal(before_tap, controller.auxiliary_prediction_weights["tap"]))

    def test_zero_plasticity_keeps_controller_weights_stable(self) -> None:
        controller = TinyController.random(Random(8), input_size=5, hidden_size=4, output_size=3)
        controller.forward([0.3, 0.1, -0.4, 0.7, 0.0])
        before = controller.to_dict(include_state=False)

        controller.learn(
            action_index=0,
            valence=1.0,
            energy_delta=0.5,
            learning_rate=0.20,
            plasticity=0.0,
            prediction_weight=0.80,
        )

        self.assertEqual(before, controller.to_dict(include_state=False))

    def test_death_checkpoints_do_not_crowd_out_active_champions(self) -> None:
        self.sim = make_sim()
        death_candidate = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 80.0)
        tap_champion = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 80.0)
        spawn_champion = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 80.0)
        spawn_runner = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 80.0)
        line_founder = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, 80.0)
        assert (
            death_candidate is not None
            and tap_champion is not None
            and spawn_champion is not None
            and spawn_runner is not None
            and line_founder is not None
        )
        tap_champion.successful_taps = 80
        spawn_champion.child_count = 18
        spawn_champion.cycle = 3
        spawn_runner.child_count = 6
        spawn_runner.cycle = 1
        line_founder.cycle = 9
        line_founder.child_count = 4

        for index in range(30):
            self.sim.checkpoints.save_controller(index, death_candidate, f"death_predation_{index}", {}, bucket="notable_death")

        death_bucket = self.sim.checkpoints.to_summary()["buckets"]["notable_death"]
        self.assertEqual(death_bucket, self.sim.checkpoints.bucket_limits["notable_death"])

        self.sim.tick = 123
        self.sim._checkpoint_champions("final")
        summary = self.sim.checkpoints.to_summary()

        self.assertIn("final_overall_champion", summary["reasons"])
        self.assertIn("final_spawn_champion", summary["reasons"])
        self.assertIn("final_tap_champion", summary["reasons"])
        self.assertIn("final_line_founder", summary["reasons"])
        self.assertGreater(summary["buckets"].get("spawn_champion", 0), 0)
        self.assertGreater(summary["buckets"].get("tap_champion", 0), 0)
        self.assertGreater(summary["buckets"].get("line_founder", 0), 0)
        self.assertEqual(summary["buckets"]["notable_death"], death_bucket)

    def _torch_runtime_or_skip(self):  # type: ignore[no-untyped-def]
        try:
            from microcosmic_god.backends.torch_gpu import TorchBrainRuntime

            return TorchBrainRuntime(device="cpu")
        except Exception as exc:
            raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

    def test_torch_controller_batch_forward_matches_cpu_reference(self) -> None:
        # The torch backend does not yet implement the attention head; for now,
        # parity is verified on attention-disabled controllers. Attention behavior
        # has dedicated CPU tests in AttentionTests.
        runtime = self._torch_runtime_or_skip()
        rng = Random(123)
        cpu_controllers = [
            TinyController.random(rng, input_size=5, hidden_size=3, output_size=4, with_attention=False),
            TinyController.random(rng, input_size=5, hidden_size=4, output_size=4, with_attention=False),
            TinyController.random(rng, input_size=5, hidden_size=3, output_size=4, with_attention=False),
        ]
        torch_controllers = [TinyController.from_dict(controller.to_dict(include_state=True)) for controller in cpu_controllers]
        observations = [
            [0.2, -0.1, 0.7, 0.0, 0.5],
            [-0.3, 0.4, 0.1, 0.9, -0.2],
            [0.8, 0.0, -0.5, 0.3, 0.2],
        ]

        expected = [controller.forward(observation) for controller, observation in zip(cpu_controllers, observations)]
        actual = runtime.forward_many(torch_controllers, observations)

        for expected_row, actual_row in zip(expected, actual):
            for expected_value, actual_value in zip(expected_row, actual_row):
                self.assertAlmostEqual(expected_value, actual_value, places=5)
        for cpu_controller, torch_controller in zip(cpu_controllers, torch_controllers):
            for expected_value, actual_value in zip(cpu_controller.hidden, torch_controller.hidden):
                self.assertAlmostEqual(expected_value, actual_value, places=5)
            for expected_value, actual_value in zip(cpu_controller.input_trace, torch_controller.input_trace):
                self.assertAlmostEqual(expected_value, actual_value, places=6)

    def test_torch_controller_batch_learning_matches_cpu_reference(self) -> None:
        runtime = self._torch_runtime_or_skip()
        rng = Random(321)
        cpu_controllers = [
            TinyController.random(rng, input_size=5, hidden_size=4, output_size=3, with_attention=False),
            TinyController.random(rng, input_size=5, hidden_size=4, output_size=3, with_attention=False),
        ]
        torch_controllers = [TinyController.from_dict(controller.to_dict(include_state=True)) for controller in cpu_controllers]
        observations = [[0.3, -0.2, 0.8, 0.1, 0.0], [-0.4, 0.9, 0.2, 0.0, 0.5]]
        for controller, observation in zip(cpu_controllers, observations):
            controller.forward(observation)
        runtime.forward_many(torch_controllers, observations)

        params = [
            {
                "action_index": 1,
                "valence": 0.8,
                "energy_delta": 0.4,
                "learning_rate": 0.16,
                "plasticity": 0.75,
                "prediction_weight": 0.60,
                "outcome_targets": {"damage": 0.1, "spawning": 0.0, "social": 0.2, "tap": 1.0, "hazard": 0.1},
            },
            {
                "action_index": 2,
                "valence": -0.5,
                "energy_delta": -0.3,
                "learning_rate": 0.12,
                "plasticity": 0.90,
                "prediction_weight": 0.80,
                "outcome_targets": {"damage": 0.4, "spawning": 0.0, "social": -0.1, "tap": 0.0, "hazard": 0.5},
            },
        ]
        expected_errors = [controller.learn(**param) for controller, param in zip(cpu_controllers, params)]
        actual_errors = runtime.learn_many(
            [
                ControllerLearningCase(controller=controller, **param)
                for controller, param in zip(torch_controllers, params)
            ]
        )

        for expected_value, actual_value in zip(expected_errors, actual_errors):
            self.assertAlmostEqual(expected_value, actual_value, places=5)
        for cpu_controller, torch_controller in zip(cpu_controllers, torch_controllers):
            # weights_out is now a (output_size, hidden_size) matrix; flatten to compare element-wise.
            for expected_value, actual_value in zip(cpu_controller.weights_out.flatten(), torch_controller.weights_out.flatten()):
                self.assertAlmostEqual(float(expected_value), float(actual_value), places=5)
            for expected_value, actual_value in zip(cpu_controller.prediction_weights, torch_controller.prediction_weights):
                self.assertAlmostEqual(float(expected_value), float(actual_value), places=5)
            self.assertEqual(set(torch_controller.last_prediction_errors), set(PREDICTION_HEADS))

    def test_torch_backend_can_run_small_simulation(self) -> None:
        self._torch_runtime_or_skip()
        tmp = tempfile.TemporaryDirectory()
        config = RunConfig(
            seed=909,
            profile="test",
            max_ticks=3,
            max_wall_seconds=0,
            places=4,
            initial_collectors=2,
            initial_converters=1,
            initial_agents=2,
            max_pool=20,
            output_dir=tmp.name,
            event_detail=False,
            compute_backend="torch",
            device="cpu",
        )
        self.sim = Simulation(config)
        self.sim._tmpdir = tmp  # type: ignore[attr-defined]

        debrief = self.sim.run()

        self.assertEqual(debrief["reason"], "max_ticks")
        self.assertEqual(debrief["tick"], 3)
        self.assertEqual(self.sim.config.compute_backend, "torch")


class AttentionTests(unittest.TestCase):
    """Information-as-attention: controllers learn during their active span what to attend to.
    Total fidelity is bounded; what isn't attended to gets noise. The mechanism
    must be neuroplastic (lifetime learning), inheritable (clone with perturbation),
    backward-compatible (legacy checkpoints work), and not require signals
    (transfer-clean to environments without durable symbol encoding)."""

    def test_default_controller_has_attention_head(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=6, hidden_size=4, output_size=3)
        self.assertEqual(controller.attention_weights.shape, (4, 6))
        self.assertEqual(controller.attention_bias.shape, (6,))
        self.assertTrue(controller._has_attention())

    def test_controller_can_be_constructed_without_attention(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=6, hidden_size=4, output_size=3, with_attention=False)
        self.assertEqual(controller.attention_weights.size, 0)
        self.assertEqual(controller.attention_bias.size, 0)
        self.assertFalse(controller._has_attention())

    def test_attention_total_fidelity_bounded_by_budget(self) -> None:
        from microcosmic_god.controller import TinyController, ATTENTION_BUDGET_FRACTION

        import numpy as np

        controller = TinyController.random(Random(11), input_size=8, hidden_size=4, output_size=3)
        # Saturate attention bias to push raw attention well above budget.
        controller.attention_bias = np.full(controller.input_size, 10.0, dtype=np.float64)
        inputs = [0.5] * controller.input_size
        controller.forward(inputs)
        budget = controller.input_size * ATTENTION_BUDGET_FRACTION
        # Floating-point slack is acceptable; the bound should hold to ~1e-6.
        self.assertLessEqual(sum(controller.last_attention), budget + 1e-6)

    def test_attention_passthrough_when_controller_has_no_attention(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=5, hidden_size=4, output_size=3, with_attention=False)
        inputs = [0.8, -0.2, 0.5, 0.0, 0.3]
        controller.forward(inputs)
        # Without attention, last_inputs should equal inputs exactly (no noise injected).
        for expected, actual in zip(inputs, controller.last_inputs):
            self.assertAlmostEqual(expected, actual, places=10)
        # last_attention reports uniform 1.0 (full fidelity) for the no-attention path.
        self.assertEqual(controller.last_attention.tolist(), [1.0] * 5)

    def test_attention_weights_change_with_surprise_and_valence(self) -> None:
        from microcosmic_god.controller import TinyController

        import numpy as np

        controller = TinyController.random(Random(11), input_size=6, hidden_size=5, output_size=3)
        before = controller.attention_weights.copy()
        # Drive several iterations of forward + learn with strong surprise/valence.
        for _ in range(10):
            controller.forward([0.7, -0.4, 0.6, 0.1, -0.5, 0.3])
            controller.learn(
                action_index=2,
                valence=1.5,
                energy_delta=0.8,
                learning_rate=0.20,
                plasticity=0.95,
                prediction_weight=0.85,
            )
        self.assertFalse(np.array_equal(before, controller.attention_weights))
        # And the attention bias should also have shifted on at least one feature.
        self.assertTrue(np.any(np.abs(controller.attention_bias) > 1e-6))

    def test_attention_serializes_round_trip(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=6, hidden_size=4, output_size=3)
        # Modify a couple of attention weights to ensure they're persisted.
        # attention_weights is now (hidden_size, input_size) — index as [h, i].
        controller.attention_weights[0, 0] = 0.5
        controller.attention_weights[1, 5] = -0.3
        controller.attention_bias[2] = 1.2
        data = controller.to_dict(include_state=True)
        restored = TinyController.from_dict(data)
        self.assertEqual(controller.attention_weights.shape, restored.attention_weights.shape)
        for original, recovered in zip(controller.attention_weights.flatten(), restored.attention_weights.flatten()):
            self.assertAlmostEqual(float(original), float(recovered), places=6)
        for original, recovered in zip(controller.attention_bias, restored.attention_bias):
            self.assertAlmostEqual(float(original), float(recovered), places=6)

    def test_legacy_checkpoint_without_attention_loads_cleanly(self) -> None:
        from microcosmic_god.controller import TinyController

        # Construct a checkpoint dict that predates attention (no fields).
        legacy_controller = TinyController.random(Random(11), input_size=5, hidden_size=3, output_size=2, with_attention=False)
        data = legacy_controller.to_dict(include_state=True)
        data.pop("attention_weights", None)
        data.pop("attention_bias", None)
        data.pop("last_attention", None)
        restored = TinyController.from_dict(data)
        self.assertFalse(restored._has_attention())
        # And the controller still produces forward outputs.
        outputs = restored.forward([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(len(outputs), 2)

    def test_clone_propagates_attention_with_perturbation(self) -> None:
        from microcosmic_god.controller import TinyController
        import numpy as np

        rng = Random(11)
        parent = TinyController.random(rng, input_size=5, hidden_size=3, output_size=2)
        child = parent.clone_for_child(Random(12), perturbation_scale=0.05)
        self.assertEqual(child.attention_weights.shape, parent.attention_weights.shape)
        self.assertEqual(child.attention_bias.shape, parent.attention_bias.shape)
        # Perturbation should have nudged at least some values.
        self.assertFalse(np.array_equal(parent.attention_weights, child.attention_weights))


class ControllerGrowthTests(unittest.TestCase):
    """Controllers can grow or shrink across spawning without losing the parent's
    learned function. This gives optimization real freedom to find appropriate
    capacity for each operating regime, rather than capping all controllers at one
    fixed size."""

    def test_controller_grow_preserves_function_for_pre_existing_inputs(self) -> None:
        from microcosmic_god.controller import TinyController

        rng = Random(31)
        controller = TinyController.random(rng, input_size=4, hidden_size=5, output_size=3, with_attention=False)
        # Drive a forward pass and snapshot the original output so we can verify
        # function preservation after growth.
        inputs = [0.4, -0.2, 0.6, 0.1]
        before = controller.forward(inputs)
        # Reset hidden state so the comparison is clean.
        controller.hidden = [0.0 for _ in range(controller.hidden_size)]
        controller.resize_hidden(rng, 9)
        self.assertEqual(controller.hidden_size, 9)
        # Newly-added hidden units have small random in-weights so the output is
        # close to but not identical to the original. The shape is what we care
        # about - growth should not catastrophically distort behavior.
        after = controller.forward(inputs)
        self.assertEqual(len(after), len(before))

    def test_controller_shrink_keeps_top_magnitude_units(self) -> None:
        from microcosmic_god.controller import TinyController

        rng = Random(31)
        controller = TinyController.random(rng, input_size=4, hidden_size=8, output_size=3, with_attention=False)
        # Make unit 3 dominant via incoming weights, unit 5 dominant via outgoing.
        # weights_in is (hidden_size, input_size); weights_out is (output_size, hidden_size).
        for h in range(controller.hidden_size):
            for i in range(controller.input_size):
                controller.weights_in[h, i] = 2.5 if h == 3 else 0.01
        for o in range(controller.output_size):
            for h in range(controller.hidden_size):
                controller.weights_out[o, h] = 2.5 if h == 5 else 0.01
        controller.resize_hidden(rng, 2)
        self.assertEqual(controller.hidden_size, 2)
        # After shrink, the two persisting units (kept in original index order)
        # are the dominant ones: unit 3 had big incoming weights (2.5 each) and
        # unit 5 had big outgoing weights (2.5 each). The first persisting slot
        # should reflect unit 3's incoming weights; the second slot should
        # reflect unit 5's outgoing weights.
        self.assertEqual(controller.weights_in[0, 0], 2.5)  # unit 3 in slot 0
        self.assertEqual(controller.weights_out[0, 1], 2.5)  # unit 5's out, in slot 1

    def test_clone_for_child_resizes_when_target_differs(self) -> None:
        from microcosmic_god.controller import TinyController

        rng = Random(31)
        parent = TinyController.random(rng, input_size=4, hidden_size=5, output_size=3, with_attention=False)
        child = parent.clone_for_child(Random(32), perturbation_scale=0.02, target_hidden_size=8)
        self.assertEqual(child.hidden_size, 8)
        self.assertEqual(parent.hidden_size, 5)  # parent unchanged
        # Child should still produce sensible outputs.
        outputs = child.forward([0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(outputs), 3)

    def test_resize_preserves_attention_when_present(self) -> None:
        from microcosmic_god.controller import TinyController

        rng = Random(31)
        controller = TinyController.random(rng, input_size=4, hidden_size=5, output_size=3, with_attention=True)
        self.assertEqual(controller.attention_weights.shape, (5, 4))
        controller.resize_hidden(rng, 9)
        self.assertEqual(controller.attention_weights.shape, (9, 4))
        controller.resize_hidden(rng, 3)
        self.assertEqual(controller.attention_weights.shape, (3, 4))
        self.assertTrue(controller._has_attention())

    def test_controller_max_cap_raised_above_legacy_128(self) -> None:
        from microcosmic_god.controller import TinyController, CONTROLLER_HIDDEN_MAX

        # The legacy cap was 128; growth requires headroom beyond that.
        self.assertGreater(CONTROLLER_HIDDEN_MAX, 128)
        rng = Random(31)
        big_controller = TinyController.random(rng, input_size=4, hidden_size=200, output_size=3, with_attention=False)
        self.assertEqual(big_controller.hidden_size, 200)


class EpisodicMemoryTests(unittest.TestCase):
    """Episodic memory: optional v2 controller feature. When capacity > 0, the
    controller has a content-addressable bank of past hidden-state snapshots.
    Storage is gated by surprise + valence; retrieval is similarity-weighted
    cross-attention; replay during rest averages two episodes and pushes
    the result through the recurrent core."""

    def test_default_controller_has_no_episodic_memory(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=6, hidden_size=4, output_size=3)
        self.assertFalse(controller._has_episodic())
        self.assertEqual(controller.episodic_slots.size, 0)

    def test_controller_with_capacity_has_episodic_memory(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=6, hidden_size=4, output_size=3, episodic_capacity=8)
        self.assertTrue(controller._has_episodic())
        self.assertEqual(controller.episodic_slots.shape, (8, 4))
        self.assertEqual(controller.episodic_age.shape, (8,))
        # All slots start empty (age=-1).
        self.assertTrue(all(controller.episodic_age == -1.0))

    def test_episodic_storage_writes_on_surprise(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=5, hidden_size=4, output_size=3, episodic_capacity=4)
        controller.forward([0.5, 0.3, -0.2, 0.4, 0.1])
        # High surprise + high valence -> should write a slot.
        wrote = controller._store_episode(surprise=0.8, valence=1.0)
        self.assertTrue(wrote)
        self.assertEqual((controller.episodic_age >= 0.0).sum(), 1)

    def test_episodic_storage_skips_when_uneventful(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=5, hidden_size=4, output_size=3, episodic_capacity=4)
        controller.forward([0.5, 0.3, -0.2, 0.4, 0.1])
        # Low surprise + low valence -> should NOT write.
        wrote = controller._store_episode(surprise=0.05, valence=0.10)
        self.assertFalse(wrote)
        self.assertEqual((controller.episodic_age >= 0.0).sum(), 0)

    def test_episodic_retrieval_returns_zero_when_empty(self) -> None:
        from microcosmic_god.controller import TinyController
        import numpy as np

        controller = TinyController.random(Random(11), input_size=5, hidden_size=4, output_size=3, episodic_capacity=4)
        controller.forward([0.5, 0.3, -0.2, 0.4, 0.1])
        retrieved = controller._retrieve_episodes()
        self.assertEqual(retrieved.shape, (4,))
        self.assertTrue(np.allclose(retrieved, 0.0))

    def test_replay_requires_two_or_more_stored_episodes(self) -> None:
        from microcosmic_god.controller import TinyController

        controller = TinyController.random(Random(11), input_size=5, hidden_size=4, output_size=3, episodic_capacity=4)
        controller.forward([0.5, 0.3, -0.2, 0.4, 0.1])
        # No stored episodes yet -> replay is a no-op.
        self.assertFalse(controller.replay_episode(Random(0)))
        # Store two episodes.
        controller._store_episode(surprise=0.8, valence=1.0)
        controller.forward([0.1, -0.4, 0.6, 0.2, -0.1])
        controller._store_episode(surprise=0.8, valence=1.0)
        # Now replay should succeed.
        self.assertTrue(controller.replay_episode(Random(0)))

    def test_episodic_serializes_round_trip(self) -> None:
        from microcosmic_god.controller import TinyController
        import numpy as np

        controller = TinyController.random(Random(11), input_size=5, hidden_size=4, output_size=3, episodic_capacity=4)
        controller.forward([0.5, 0.3, -0.2, 0.4, 0.1])
        controller._store_episode(surprise=0.8, valence=1.0)
        data = controller.to_dict(include_state=True)
        restored = TinyController.from_dict(data)
        self.assertTrue(restored._has_episodic())
        self.assertEqual(restored.episodic_slots.shape, controller.episodic_slots.shape)
        for original, recovered in zip(controller.episodic_slots.flatten(), restored.episodic_slots.flatten()):
            self.assertAlmostEqual(float(original), float(recovered), places=6)

    def test_clone_inherits_capacity_but_clears_memories(self) -> None:
        from microcosmic_god.controller import TinyController

        parent = TinyController.random(Random(11), input_size=5, hidden_size=4, output_size=3, episodic_capacity=4)
        parent.forward([0.5, 0.3, -0.2, 0.4, 0.1])
        parent._store_episode(surprise=0.8, valence=1.0)
        # Parent has 1 stored episode.
        self.assertEqual((parent.episodic_age >= 0.0).sum(), 1)
        # Child should inherit capacity but start with empty slots (no acquired-state inheritance).
        child = parent.clone_for_child(Random(12), perturbation_scale=0.02)
        self.assertEqual(child.episodic_slots.shape, parent.episodic_slots.shape)
        self.assertEqual((child.episodic_age >= 0.0).sum(), 0, "child should have no inherited episodes")


class MultiWorldSelectionTests(unittest.TestCase):
    """When `world_refresh_every` is set, the simulation should swap the world
    every N ticks. Physics and obstacles are re-drawn; resource state (including
    the sealed reserve) carries over; signals are cleared because they reference
    the old context."""

    def _sim(self, refresh_every: int):
        tmp = tempfile.TemporaryDirectory()
        config = RunConfig(
            seed=42,
            profile="test",
            max_ticks=10,
            max_wall_seconds=0,
            places=8,
            initial_collectors=4,
            initial_converters=2,
            initial_agents=4,
            max_pool=30,
            output_dir=tmp.name,
            event_detail=False,
            world_refresh_every=refresh_every,
        )
        sim = Simulation(config)
        sim._tmpdir = tmp  # type: ignore[attr-defined]
        return sim, tmp

    def test_world_refresh_swaps_physics_but_carries_resources(self) -> None:
        sim, tmp = self._sim(refresh_every=3)

        original_world = sim.world

        # Run up to (but not past) the first refresh point.
        for _ in range(2):
            sim.step()
        self.assertIs(sim.world, original_world, "world must not refresh before tick 3")

        place = sim.world.places[0]
        place.resources["essence"] = 123.4
        place.sealed_essence = 77.0
        sim.world.emit_signal(0, source_id=999, token=3, intensity=1.0)

        sim._refresh_world()
        self.assertIsNot(sim.world, original_world)
        refreshed = sim.world.places[0]
        # Resource state (including the sealed reserve) carries over exactly...
        self.assertAlmostEqual(refreshed.resources["essence"], 123.4, places=6)
        self.assertAlmostEqual(refreshed.sealed_essence, 77.0, places=6)
        # ...while signals are cleared (they reference the old context).
        self.assertEqual(refreshed.signals, [])

        # The tick boundary fires the swap too (tick 3 with refresh_every=3).
        swapped_world = sim.world
        sim.step()
        self.assertIsNot(sim.world, swapped_world, "world must refresh at tick 3")

        sim.logger.close()
        tmp.cleanup()

    def test_world_refresh_zero_means_legacy_single_world(self) -> None:
        sim, tmp = self._sim(refresh_every=0)

        original_world = sim.world
        for _ in range(5):
            sim.step()
        self.assertIs(sim.world, original_world)

        sim.logger.close()
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
