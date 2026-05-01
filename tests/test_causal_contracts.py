from __future__ import annotations

import tempfile
import unittest
from random import Random

from microcosmic_god.brain import TinyBrain
from microcosmic_god.config import RunConfig
from microcosmic_god.energy import build_structure, structure_decay_channels
from microcosmic_god.genome import Genome
from microcosmic_god.organisms import OBSERVATION_SIZE
from microcosmic_god.simulation import Simulation


def make_sim(seed: int = 101, places: int = 3) -> Simulation:
    tmp = tempfile.TemporaryDirectory()
    config = RunConfig(
        seed=seed,
        profile="test",
        max_ticks=10,
        max_wall_seconds=0,
        places=places,
        initial_plants=0,
        initial_fungi=0,
        initial_agents=0,
        max_population=50,
        output_dir=tmp.name,
        event_detail=False,
    )
    sim = Simulation(config)
    sim._tmpdir = tmp  # type: ignore[attr-defined]
    return sim


class CausalContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        sim = getattr(self, "sim", None)
        if sim is not None:
            sim.logger.close()
            sim._tmpdir.cleanup()  # type: ignore[attr-defined]

    def test_all_signal_tokens_are_observed(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 50.0)
        assert agent is not None
        agent.signal_values = [index / 10.0 for index in range(8)]

        observation = self.sim._observe(agent, self.sim._rosters())

        self.assertEqual(len(observation), OBSERVATION_SIZE)
        self.assertEqual(observation[-8:], agent.signal_values)

    def test_attack_uses_current_location_not_tick_start_roster(self) -> None:
        self.sim = make_sim(places=2)
        attacker = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        target = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert attacker is not None and target is not None
        _stale_roster = self.sim._rosters()
        target.location = 1
        before_health = target.health

        self.sim._attack(attacker)

        self.assertEqual(target.health, before_health)
        self.assertEqual(target.location, 1)

    def test_asexual_capacity_uses_current_local_population(self) -> None:
        self.sim = make_sim(places=2)
        parent_genome = Genome.plant(self.sim.rng)
        parent = self.sim.add_organism("plant", parent_genome, 0, 200.0)
        neighbor = self.sim.add_organism("plant", Genome.plant(self.sim.rng), 0, 20.0)
        assert parent is not None and neighbor is not None
        self.sim.world.places[0].capacity = 2
        parent.age = 100
        neighbor.location = 1

        self.sim._asexual_reproduce(parent, {"reproduction": 0.0, "social": 0.0})

        self.assertEqual(self.sim.births_by_mode["asexual"], 1)
        self.assertEqual(len(self.sim._living_ids_at(0)), 2)

    def test_failed_craft_risks_material_loss(self) -> None:
        self.sim = make_sim()
        agent_genome = Genome.neural(self.sim.rng)
        agent_genome.manipulator = 0.12
        agent = self.sim.add_organism("agent", agent_genome, 0, 100.0)
        assert agent is not None
        agent.inventory = {"stone": 1, "crystal": 1}

        class FailingCraftRng:
            def __init__(self) -> None:
                self.calls = 0

            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

            def random(self) -> float:
                self.calls += 1
                return 0.99 if self.calls == 1 else 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

        self.sim.rng = FailingCraftRng()  # type: ignore[assignment]

        self.sim._craft(agent, {"reproduction": 0.0, "social": 0.0})

        self.assertLess(agent.inventory_count(), 2)
        self.assertGreater(agent.tool_skill["bind"], 0.0)

    def test_inside_boundary_is_distinct_from_shelter(self) -> None:
        structure = build_structure({"shell": 3, "resin": 2, "fiber": 1})

        self.assertIn("enclose", structure.capabilities)
        self.assertIn("shelter", structure.capabilities)
        self.assertIn("permeable", structure.capabilities)
        self.assertGreater(structure.capabilities["enclose"], 0.0)
        self.assertNotEqual(round(structure.capabilities["enclose"], 5), round(structure.capabilities["shelter"], 5))

    def test_build_action_creates_persistent_structure(self) -> None:
        self.sim = make_sim()
        agent_genome = Genome.neural(self.sim.rng)
        agent_genome.manipulator = 1.0
        agent = self.sim.add_organism("agent", agent_genome, 0, 100.0)
        assert agent is not None
        agent.inventory = {"stone": 3, "branch": 3, "fiber": 3, "resin": 2}

        class SuccessfulBuildRng:
            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

        self.sim.rng = SuccessfulBuildRng()  # type: ignore[assignment]

        self.sim._build_structure(agent, {"reproduction": 0.0, "social": 0.0})

        place = self.sim.world.places[agent.location]
        self.assertEqual(len(place.structures), 1)
        self.assertGreater(place.structures[0].scale, 0)
        self.assertEqual(self.sim.tool_successes["build"], 1)
        self.assertGreater(agent.tool_skill["build"], 0.0)

    def test_structure_decay_couples_materials_to_environment(self) -> None:
        conductive = build_structure({"crystal": 6})
        dry = {
            "temperature": 0.45,
            "fluid_level": 0.0,
            "humidity": 0.12,
            "salinity": 0.0,
            "oxygen": 0.25,
            "acidity": 0.04,
            "biological_activity": 0.02,
            "abrasion": 0.02,
            "wet_dry_cycle": 0.03,
            "current_exposure": 0.0,
            "pressure": 0.0,
            "light": 0.10,
            "flow_gradient": 0.0,
        }
        salty_wet = {
            **dry,
            "fluid_level": 0.85,
            "humidity": 0.95,
            "salinity": 0.90,
            "oxygen": 0.80,
            "acidity": 0.28,
            "wet_dry_cycle": 0.42,
            "current_exposure": 0.30,
            "flow_gradient": 0.45,
        }

        dry_decay = structure_decay_channels(conductive, dry)
        wet_decay = structure_decay_channels(conductive, salty_wet)

        self.assertGreater(wet_decay["chemical"], dry_decay["chemical"] * 3.0)
        self.assertGreater(sum(wet_decay.values()), sum(dry_decay.values()))

    def test_durable_materials_decay_less_than_organic_materials_in_wet_biology(self) -> None:
        stone = build_structure({"stone": 6})
        organic = build_structure({"branch": 3, "fiber": 3})
        wet_biology = {
            "temperature": 0.52,
            "fluid_level": 0.70,
            "humidity": 0.95,
            "salinity": 0.20,
            "oxygen": 0.50,
            "acidity": 0.18,
            "biological_activity": 0.92,
            "abrasion": 0.10,
            "wet_dry_cycle": 0.30,
            "current_exposure": 0.08,
            "pressure": 0.25,
            "light": 0.32,
            "flow_gradient": 0.08,
        }

        stone_decay = sum(structure_decay_channels(stone, wet_biology).values())
        organic_decay = sum(structure_decay_channels(organic, wet_biology).values())

        self.assertGreater(organic_decay, stone_decay * 2.0)

    def test_brain_plasticity_can_update_representations(self) -> None:
        brain = TinyBrain.random(Random(7), input_size=5, hidden_size=4, output_size=3)
        inputs = [0.8, -0.2, 0.5, 0.0, 0.3]
        brain.forward(inputs)
        before_in = list(brain.weights_in)
        before_out = list(brain.weights_out)

        brain.learn(
            action_index=1,
            valence=1.2,
            energy_delta=0.6,
            learning_rate=0.20,
            plasticity=0.80,
            prediction_weight=0.70,
        )

        self.assertNotEqual(before_out, brain.weights_out)
        self.assertNotEqual(before_in, brain.weights_in)
        self.assertEqual(len(brain.input_trace), 5)
        self.assertEqual(len(brain.hidden_trace), 4)

    def test_zero_plasticity_keeps_brain_weights_stable(self) -> None:
        brain = TinyBrain.random(Random(8), input_size=5, hidden_size=4, output_size=3)
        brain.forward([0.3, 0.1, -0.4, 0.7, 0.0])
        before = brain.to_dict(include_state=False)

        brain.learn(
            action_index=0,
            valence=1.0,
            energy_delta=0.5,
            learning_rate=0.20,
            plasticity=0.0,
            prediction_weight=0.80,
        )

        self.assertEqual(before, brain.to_dict(include_state=False))


if __name__ == "__main__":
    unittest.main()
