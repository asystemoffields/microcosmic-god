import unittest

from microcosmic_god.config import RunConfig
from microcosmic_god.modular import ModularController
from microcosmic_god.simulation import Simulation


def _smoke_config(**overrides):
    return RunConfig.from_profile(
        "smoke",
        seed=29,
        max_ticks=overrides.pop("max_ticks", 120),
        max_wall_seconds=0.0,
        log_every=10**9,
        checkpoint_every=10**9,
        neural_checkpoint_limit=0,
        event_detail=False,
        stop_on_neural_extinction=False,
        stop_on_full_extinction=False,
        output_dir="/tmp/mcg_modular_wirein",
        initial_modular_fraction=overrides.pop("initial_modular_fraction", 1.0),
        **overrides,
    )


class ModularWireinTest(unittest.TestCase):
    def test_all_modular_run_steps_and_agents_act(self):
        sim = Simulation(_smoke_config())
        agents = [o for o in sim.organisms.values() if o.kind == "agent"]
        self.assertTrue(agents)
        self.assertTrue(all(isinstance(a.controller, ModularController) for a in agents))
        for _ in range(120):
            sim.step()
        survivors = [o for o in sim.organisms.values() if o.kind == "agent" and o.alive]
        acted = [a for a in survivors if a.last_action != "rest"]
        self.assertTrue(survivors, "modular cohort went extinct in 120 smoke ticks")
        self.assertTrue(acted, "no modular agent ever chose an action")

    def test_modular_children_inherit_modular_and_budget_tracks_capacity(self):
        sim = Simulation(_smoke_config(max_ticks=400))
        for _ in range(400):
            sim.step()
        children = [
            o
            for o in sim.organisms.values()
            if o.kind == "agent" and o.parent_ids and o.controller is not None
        ]
        if not children:
            self.skipTest("no reproduction in 400 smoke ticks (world too harsh this seed)")
        for child in children:
            self.assertIsInstance(child.controller, ModularController)
            self.assertEqual(int(round(child.params.neural_budget)), child.controller.capacity)

    def test_mixed_population_runs(self):
        sim = Simulation(_smoke_config(initial_modular_fraction=0.5))
        kinds = {type(o.controller).__name__ for o in sim.organisms.values() if o.kind == "agent"}
        self.assertEqual(kinds, {"TinyController", "ModularController"})
        for _ in range(60):
            sim.step()


if __name__ == "__main__":
    unittest.main()
