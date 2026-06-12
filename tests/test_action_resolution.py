import unittest
from random import Random

from microcosmic_god.config import RunConfig
from microcosmic_god.organisms import ACTION_INDEX, ACTIONS, individual_from_genome
from microcosmic_god.params import ParamVector
from microcosmic_god.simulation import Simulation


class _NoExploration:
    def random(self) -> float:
        return 0.99


def _sim(**overrides) -> Simulation:
    config = RunConfig.from_profile("smoke", **overrides)
    sim = Simulation(config)
    sim.rng = _NoExploration()
    return sim


def _juvenile():
    rng = Random(7)
    params = ParamVector.neural(rng)
    individual = individual_from_genome(rng, 1, "agent", params, 0, 50.0)
    individual.age = 0  # not adult: coordinate / clone_mutate are infeasible
    return individual


def _outputs(*ranked_actions: str) -> list[float]:
    outputs = [0.0] * len(ACTIONS)
    for rank, action in enumerate(ranked_actions):
        outputs[ACTION_INDEX[action]] = 10.0 - rank
    return outputs


class ActionSearchDepthTest(unittest.TestCase):
    def test_legacy_walks_past_infeasible_to_next_feasible(self):
        sim = _sim()
        self.assertEqual(sim.config.action_search_depth, 0)
        action = sim._choose_action_from_outputs(_juvenile(), _outputs("clone_mutate", "rest"))
        self.assertEqual(action, "rest")

    def test_depth_one_commits_to_infeasible_top_choice(self):
        sim = _sim(action_search_depth=1)
        action = sim._choose_action_from_outputs(_juvenile(), _outputs("clone_mutate", "rest"))
        self.assertEqual(action, "clone_mutate")

    def test_depth_one_returns_feasible_top_choice(self):
        sim = _sim(action_search_depth=1)
        action = sim._choose_action_from_outputs(_juvenile(), _outputs("rest", "clone_mutate"))
        self.assertEqual(action, "rest")

    def test_depth_two_still_searches_within_bound(self):
        sim = _sim(action_search_depth=2)
        action = sim._choose_action_from_outputs(_juvenile(), _outputs("clone_mutate", "rest"))
        self.assertEqual(action, "rest")

    def test_depth_two_commits_when_bound_exhausted(self):
        sim = _sim(action_search_depth=2)
        action = sim._choose_action_from_outputs(
            _juvenile(), _outputs("clone_mutate", "coordinate", "rest")
        )
        self.assertEqual(action, "clone_mutate")


if __name__ == "__main__":
    unittest.main()
