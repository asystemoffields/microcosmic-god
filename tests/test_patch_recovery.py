import unittest
from random import Random

from microcosmic_god.config import RunConfig
from microcosmic_god.simulation import Simulation
from microcosmic_god.world import World


def _world(seed: int = 5, **overrides) -> World:
    config = RunConfig.from_profile("smoke", **overrides)
    return World.generate(Random(seed), config)


class PatchRecoveryTest(unittest.TestCase):
    def test_disabled_by_default(self):
        world = _world()
        self.assertEqual(world.patch_recovery_ticks, 0)
        self.assertFalse(world.note_patch_depletion(0, Random(1)))
        events = world.update_environment(Random(2))
        self.assertNotIn("patch_recovering", events)
        self.assertEqual(world.places[0].regen_recovery_until, 0)

    def test_window_suppresses_regen_only_at_depleted_place(self):
        # Paired worlds, identical seeds and rng streams; one notes a depletion.
        # The recovery branch draws no rng, so all divergence is the suppression.
        control = _world(patch_recovery_ticks=50)
        treated = _world(patch_recovery_ticks=50)
        self.assertTrue(treated.note_patch_depletion(0, Random(1)))
        self.assertEqual(treated.places[0].regen_recovery_until, treated.tick + 50)
        events = None
        for step in range(3):
            control.update_environment(Random(100 + step))
            events = treated.update_environment(Random(100 + step))
        self.assertEqual(events.get("patch_recovering"), 1)
        self.assertLess(treated.places[0].resources["essence"], control.places[0].resources["essence"])
        for i in range(1, len(control.places)):
            self.assertAlmostEqual(
                treated.places[i].resources["essence"],
                control.places[i].resources["essence"],
                places=9,
            )

    def test_window_expires(self):
        world = _world(patch_recovery_ticks=2)
        world.note_patch_depletion(0, Random(1))
        self.assertIn("patch_recovering", world.update_environment(Random(3)))
        self.assertIn("patch_recovering", world.update_environment(Random(4)))
        self.assertNotIn("patch_recovering", world.update_environment(Random(5)))

    def test_repeat_depletion_extends_never_shrinks(self):
        world = _world(patch_recovery_ticks=50)
        world.note_patch_depletion(0, Random(1))
        first = world.places[0].regen_recovery_until
        world.tick += 10
        world.note_patch_depletion(0, Random(2))
        self.assertGreater(world.places[0].regen_recovery_until, first)

    def test_floor_keeps_partial_regen(self):
        floored = _world(patch_recovery_ticks=50, patch_recovery_floor=0.5)
        hard = _world(patch_recovery_ticks=50, patch_recovery_floor=0.0)
        floored.note_patch_depletion(0, Random(1))
        hard.note_patch_depletion(0, Random(1))
        floored.update_environment(Random(7))
        hard.update_environment(Random(7))
        self.assertGreater(floored.places[0].resources["essence"], hard.places[0].resources["essence"])

    def test_jitter_randomizes_window_with_same_mean(self):
        # jitter 0: window is exactly the configured length. jitter 1: windows
        # vary per event but keep the configured mean (scrambled control).
        fixed = _world(patch_recovery_ticks=100)
        fixed.note_patch_depletion(0, Random(1))
        self.assertEqual(fixed.places[0].regen_recovery_until, 100 + fixed.tick)

        scrambled = _world(patch_recovery_ticks=100, patch_recovery_jitter=1.0)
        rng = Random(9)
        durations = []
        for place_id in range(len(scrambled.places)):
            scrambled.note_patch_depletion(place_id, rng)
            durations.append(scrambled.places[place_id].regen_recovery_until - scrambled.tick)
        self.assertGreater(len(set(durations)), 1)
        mean = sum(durations) / len(durations)
        self.assertGreater(mean, 30.0)
        self.assertLess(mean, 300.0)

    def test_substantial_feed_starts_recovery(self):
        config = RunConfig.from_profile(
            "smoke", patch_recovery_ticks=40, seed=11, output_dir="/tmp/mcg_patch_recovery_test"
        )
        sim = Simulation(config)
        self.addCleanup(sim.logger.close)
        agent = next(ind for ind in sim.individuals.values() if ind.kind == "agent" and ind.alive)
        agent.location = 0
        sim.world.places[0].resources["essence"] = 60.0
        sim._eat(agent)
        self.assertGreaterEqual(sim.patch_recovery_triggers, 1)
        self.assertGreater(sim.world.places[0].regen_recovery_until, 0)

    def test_refresh_carries_recovery_state(self):
        config = RunConfig.from_profile(
            "smoke",
            patch_recovery_ticks=500,
            world_refresh_every=50,
            seed=11,
            output_dir="/tmp/mcg_patch_recovery_test",
        )
        sim = Simulation(config)
        self.addCleanup(sim.logger.close)
        sim.world.note_patch_depletion(0, Random(1))
        until = sim.world.places[0].regen_recovery_until
        sim.tick = 50
        sim._refresh_world()
        self.assertEqual(sim.world.places[0].regen_recovery_until, until)
        self.assertEqual(sim.world.patch_recovery_ticks, 500)


if __name__ == "__main__":
    unittest.main()
