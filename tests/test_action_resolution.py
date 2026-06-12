import unittest
from random import Random

from microcosmic_god.config import RunConfig
from microcosmic_god.individuals import ACTION_INDEX, ACTIONS, individual_from_params
from microcosmic_god.params import ParamVector
from microcosmic_god.simulation import Simulation


class _NoExploration:
    def random(self) -> float:
        return 0.99


def _sim(case: unittest.TestCase, **overrides) -> Simulation:
    config = RunConfig.from_profile("smoke", output_dir="/tmp/mcg_test_runs", **overrides)
    sim = Simulation(config)
    case.addCleanup(sim.logger.close)
    sim.rng = _NoExploration()
    return sim


def _juvenile():
    rng = Random(7)
    params = ParamVector.neural(rng)
    individual = individual_from_params(rng, 1, "agent", params, 0, 50.0)
    individual.age = 0  # not adult: coordinate / clone_perturb are infeasible
    return individual


def _immobile_adult():
    individual = _juvenile()
    individual.age = 100  # adult: spawning actions are feasible
    individual.params.mobility = 0.0  # below the 0.05 gate: move is infeasible
    return individual


def _outputs(*ranked_actions: str) -> list[float]:
    outputs = [0.0] * len(ACTIONS)
    for rank, action in enumerate(ranked_actions):
        outputs[ACTION_INDEX[action]] = 10.0 - rank
    return outputs


class ActionSearchDepthTest(unittest.TestCase):
    def test_era2_action_list_is_ten_actions(self):
        # Era 2 (docs/ENV_AXIS_REVIEW.md): the gate-blind verbs are gone; the
        # only remaining feasibility gates are adult (coordinate/clone_perturb)
        # and mobility (move).
        self.assertEqual(len(ACTIONS), 10)
        self.assertIn("tap", ACTIONS)
        for removed in ("pickup", "craft", "build", "use_tool", "mark", "observe"):
            self.assertNotIn(removed, ACTIONS)

    def test_legacy_walks_past_infeasible_to_next_feasible(self):
        sim = _sim(self)
        self.assertEqual(sim.config.action_search_depth, 0)
        action = sim._choose_action_from_outputs(_juvenile(), _outputs("clone_perturb", "rest"))
        self.assertEqual(action, "rest")

    def test_depth_one_commits_to_infeasible_top_choice(self):
        sim = _sim(self, action_search_depth=1)
        action = sim._choose_action_from_outputs(_juvenile(), _outputs("clone_perturb", "rest"))
        self.assertEqual(action, "clone_perturb")

    def test_depth_one_returns_feasible_top_choice(self):
        sim = _sim(self, action_search_depth=1)
        action = sim._choose_action_from_outputs(_juvenile(), _outputs("rest", "clone_perturb"))
        self.assertEqual(action, "rest")

    def test_depth_two_still_searches_within_bound(self):
        sim = _sim(self, action_search_depth=2)
        action = sim._choose_action_from_outputs(_juvenile(), _outputs("clone_perturb", "rest"))
        self.assertEqual(action, "rest")

    def test_depth_two_commits_when_bound_exhausted(self):
        sim = _sim(self, action_search_depth=2)
        action = sim._choose_action_from_outputs(
            _juvenile(), _outputs("clone_perturb", "coordinate", "rest")
        )
        self.assertEqual(action, "clone_perturb")

    def test_commit_increments_infeasible_counter(self):
        sim = _sim(self, action_search_depth=1)
        self.assertEqual(sum(sim.infeasible_commits.values()), 0)
        sim._choose_action_from_outputs(_juvenile(), _outputs("clone_perturb", "rest"))
        self.assertEqual(sim.infeasible_commits["clone_perturb"], 1)
        sim._choose_action_from_outputs(_juvenile(), _outputs("rest", "clone_perturb"))
        self.assertEqual(sum(sim.infeasible_commits.values()), 1)

    def test_mobility_gate_commits_immobile_move_at_depth_one(self):
        sim = _sim(self, action_search_depth=1)
        action = sim._choose_action_from_outputs(_immobile_adult(), _outputs("move", "rest"))
        self.assertEqual(action, "move")
        self.assertEqual(sim.infeasible_commits["move"], 1)

    def test_mobility_gate_walks_to_feasible_at_legacy_depth(self):
        sim = _sim(self)
        action = sim._choose_action_from_outputs(_immobile_adult(), _outputs("move", "rest"))
        self.assertEqual(action, "rest")


def _rich_adult():
    individual = _juvenile()
    individual.age = 100  # adult
    individual.energy = individual.storage_limit()  # energy ratio 1.0 > 0.62
    individual.params.valence_spawn = 1.0
    return individual


def _flat_outputs_with_small_rest_lead() -> list[float]:
    # The injection is additive and small (~0.3-0.7 for a saturated adult), so
    # the controller's own preference must be of comparable size for the test
    # to discriminate.
    outputs = [0.0] * len(ACTIONS)
    outputs[ACTION_INDEX["rest"]] = 0.05
    return outputs


class DriveInjectionScaleTest(unittest.TestCase):
    def test_legacy_default_injects_drive(self):
        sim = _sim(self)
        self.assertEqual(sim.config.drive_injection_scale, 1.0)
        action = sim._choose_action_from_outputs(_rich_adult(), _flat_outputs_with_small_rest_lead())
        self.assertIn(action, ("coordinate", "clone_perturb"))

    def test_zero_scale_leaves_controller_outputs_alone(self):
        sim = _sim(self, drive_injection_scale=0.0)
        action = sim._choose_action_from_outputs(_rich_adult(), _flat_outputs_with_small_rest_lead())
        self.assertEqual(action, "rest")


if __name__ == "__main__":
    unittest.main()
