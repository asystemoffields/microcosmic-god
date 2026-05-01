from __future__ import annotations

import tempfile
import unittest

from microcosmic_god.config import RunConfig
from microcosmic_god.energy import build_structure
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


if __name__ == "__main__":
    unittest.main()
