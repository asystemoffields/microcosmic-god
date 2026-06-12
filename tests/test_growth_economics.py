import unittest
from random import Random

from microcosmic_god.config import RunConfig
from microcosmic_god.individuals import individual_from_params
from microcosmic_god.params import ParamVector
from microcosmic_god.simulation import Simulation


def _agent(grace_ticks: int = 0, grace_floor: float = 0.35, budget: float = 24.0):
    rng = Random(7)
    params = ParamVector.neural(rng)
    params.neural_budget = budget
    individual = individual_from_params(rng, 1, "agent", params, 0, 50.0)
    individual.neural_upkeep_grace_ticks = grace_ticks
    individual.neural_upkeep_grace_floor = grace_floor
    return individual


class GrowthEconomicsTest(unittest.TestCase):
    def test_grace_disabled_matches_legacy_price(self):
        legacy = _agent(grace_ticks=0)
        subsidized_expired = _agent(grace_ticks=100)
        subsidized_expired.age = 100
        self.assertAlmostEqual(legacy.upkeep_cost(), subsidized_expired.upkeep_cost(), places=9)

    def test_grace_discounts_young_individuals(self):
        young = _agent(grace_ticks=100)
        adult = _agent(grace_ticks=100)
        adult.age = 100
        self.assertLess(young.upkeep_cost(), adult.upkeep_cost())

    def test_grace_ramp_is_monotonic_in_age(self):
        costs = []
        for age in (0, 25, 50, 75, 100):
            individual = _agent(grace_ticks=100)
            individual.age = age
            costs.append(individual.upkeep_cost())
        self.assertEqual(costs, sorted(costs))

    def test_floor_scales_only_the_neural_term(self):
        # With the subsidy at floor f, the discount equals (1-f) x neural term.
        full = _agent(grace_ticks=0)
        floored = _agent(grace_ticks=100, grace_floor=0.5)
        zero_budget_full = _agent(grace_ticks=0, budget=0.0)
        zero_budget_floored = _agent(grace_ticks=100, grace_floor=0.5, budget=0.0)
        neural_with = full.upkeep_cost() - floored.upkeep_cost()
        neural_without = zero_budget_full.upkeep_cost() - zero_budget_floored.upkeep_cost()
        # The discount must grow with neural_budget; the non-neural terms are untouched.
        self.assertGreater(neural_with, neural_without)
        base_and_body_full = zero_budget_full.upkeep_cost()
        base_and_body_floored = zero_budget_floored.upkeep_cost()
        # budget=0 still leaves prediction/plasticity/memory in the neural term,
        # so the floored cost is lower, but never below floor x neural share.
        self.assertLessEqual(base_and_body_floored, base_and_body_full)

    def test_simulation_wires_grace_from_config(self):
        config = RunConfig.from_profile(
            "smoke",
            seed=11,
            max_ticks=5,
            max_wall_seconds=0.0,
            neural_upkeep_grace_ticks=120,
            neural_upkeep_grace_floor=0.4,
            log_every=10**9,
            checkpoint_every=10**9,
            neural_checkpoint_limit=0,
            output_dir="/tmp/mcg_grace_wire_test",
        )
        sim = Simulation(config)
        agents = [o for o in sim.individuals.values() if o.kind == "agent"]
        self.assertTrue(agents)
        self.assertTrue(all(a.neural_upkeep_grace_ticks == 120 for a in agents))
        self.assertTrue(all(abs(a.neural_upkeep_grace_floor - 0.4) < 1e-9 for a in agents))


if __name__ == "__main__":
    unittest.main()
