from __future__ import annotations

import tempfile
import unittest

from microcosmic_god.config import RunConfig
from microcosmic_god.individuals import ACTIONS, ACTION_INDEX, OBSERVATION_SIZE
from microcosmic_god.params import ParamVector
from microcosmic_god.simulation import Simulation, TAP_CUE_CHANNELS


def make_sim(seed: int = 101, places: int = 3, **overrides) -> Simulation:
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
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    sim = Simulation(config)
    sim._tmpdir = tmp  # type: ignore[attr-defined]
    return sim


class TapContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        sim = getattr(self, "sim", None)
        if sim is not None:
            sim.logger.close()
            sim._tmpdir.cleanup()  # type: ignore[attr-defined]

    def _agent(self, energy: float = 50.0):
        agent = self.sim.add_individual("agent", ParamVector.neural(self.sim.rng), 0, energy)
        assert agent is not None
        return agent

    def test_tap_cue_level_sits_at_configured_percentile(self) -> None:
        self.sim = make_sim()
        self.assertEqual(self.sim.config.tap_cue_threshold, 0.70)
        for place, value in zip(self.sim.world.places, (0.1, 0.8, 0.5)):
            place.physics[self.sim.tap_cue_channel] = value

        # percentile 0.70 of 3 places -> index 2 of the sorted values.
        self.assertAlmostEqual(self.sim._compute_tap_cue_level(), 0.8, places=6)

        # percentile 0.0 -> the lowest value.
        self.sim.config.tap_cue_threshold = 0.0
        self.assertAlmostEqual(self.sim._compute_tap_cue_level(), 0.1, places=6)

        # The gate never sits below the 0.05 floor.
        for place in self.sim.world.places:
            place.physics[self.sim.tap_cue_channel] = 0.0
        self.assertAlmostEqual(self.sim._compute_tap_cue_level(), 0.05, places=6)

    def test_tap_above_cue_level_releases_sealed_reserve(self) -> None:
        self.sim = make_sim()
        agent = self._agent()
        agent.params.essence_conversion = 0.2
        place = self.sim.world.places[0]
        place.sealed_essence = 50.0
        place.resources["essence"] = 10.0
        place.physics[self.sim.tap_cue_channel] = 0.9
        self.sim.tap_cue_level = 0.45
        feedback = {"spawning": 0.0, "social": 0.0, "tap": 0.0}
        before_energy = agent.energy

        self.sim._tap(agent, feedback)

        # headroom = min(1, (0.9-0.45)/0.55 + 0.25) = 1.0;
        # release = min(50, 2 + 8) = 10; share = 10 * (0.45 + 0.2*0.25) = 5;
        # spill = (10 - 5) * 0.8 = 4 into place essence.
        self.assertAlmostEqual(place.sealed_essence, 40.0, places=6)
        self.assertAlmostEqual(agent.energy, before_energy + 5.0, places=6)
        self.assertAlmostEqual(place.resources["essence"], 14.0, places=6)
        self.assertEqual(self.sim.tap_outcomes["tap"], 1)
        self.assertEqual(self.sim.tap_outcomes["mistap"], 0)
        self.assertEqual(agent.successful_taps, 1)
        self.assertEqual(agent.mistap_count, 0)
        self.assertGreater(feedback["tap"], 0.0)
        self.assertGreater(agent.success_profile["tap"], 0.0)

    def test_tap_below_cue_level_misfires(self) -> None:
        self.sim = make_sim()
        agent = self._agent()
        place = self.sim.world.places[0]
        place.sealed_essence = 50.0
        place.physics[self.sim.tap_cue_channel] = 0.10
        self.sim.tap_cue_level = 0.45
        feedback = {"spawning": 0.0, "social": 0.0, "tap": 0.0}
        before_energy = agent.energy
        before_health = agent.health

        self.sim._tap(agent, feedback)

        self.assertAlmostEqual(agent.energy, before_energy - 0.05, places=6)
        self.assertAlmostEqual(agent.health, before_health - 0.004, places=6)
        self.assertAlmostEqual(place.sealed_essence, 50.0, places=6)
        self.assertEqual(self.sim.tap_outcomes["mistap"], 1)
        self.assertEqual(self.sim.tap_outcomes["tap"], 0)
        self.assertEqual(agent.mistap_count, 1)
        self.assertEqual(agent.successful_taps, 0)
        self.assertEqual(feedback["tap"], 0.0)

    def test_tap_with_no_sealed_reserve_misfires_even_on_high_cue(self) -> None:
        self.sim = make_sim()
        agent = self._agent()
        place = self.sim.world.places[0]
        place.sealed_essence = 0.0
        place.physics[self.sim.tap_cue_channel] = 0.95
        self.sim.tap_cue_level = 0.45
        feedback = {"spawning": 0.0, "social": 0.0, "tap": 0.0}
        before_energy = agent.energy

        self.sim._tap(agent, feedback)

        self.assertLess(agent.energy, before_energy)
        self.assertEqual(self.sim.tap_outcomes["mistap"], 1)
        self.assertEqual(self.sim.tap_outcomes["tap"], 0)
        self.assertEqual(agent.mistap_count, 1)
        self.assertAlmostEqual(place.sealed_essence, 0.0, places=6)

    def _drift_channels_for_seed(self, seed: int, drift: int, steps: int = 24) -> list[str]:
        sim = make_sim(seed=seed, places=6, world_refresh_every=2, tap_cue_drift=drift)
        try:
            channels = [sim.tap_cue_channel]
            for _ in range(steps):
                sim.step()
                channels.append(sim.tap_cue_channel)
                self.assertGreaterEqual(sim.tap_cue_level, 0.05)
            return channels
        finally:
            sim.logger.close()
            sim._tmpdir.cleanup()  # type: ignore[attr-defined]

    def test_tap_cue_drift_redraws_channel_across_refreshes(self) -> None:
        # Pick a seed deterministically: the first of a small fixed set where
        # the channel changes at least once across 12 refreshes.
        drifting_seed = None
        drifting_channels: list[str] = []
        for seed in (1, 2, 3, 4, 5, 6):
            channels = self._drift_channels_for_seed(seed, drift=1)
            self.assertTrue(all(channel in TAP_CUE_CHANNELS for channel in channels))
            if len(set(channels)) > 1:
                drifting_seed = seed
                drifting_channels = channels
                break
        self.assertIsNotNone(drifting_seed, "no seed in 1..6 re-drew the cue channel across 12 refreshes")
        self.assertGreater(len(set(drifting_channels)), 1)

        # drift=0 keeps the fixed era-2.0 channel through the same refreshes.
        fixed_channels = self._drift_channels_for_seed(drifting_seed, drift=0)
        self.assertEqual(set(fixed_channels), {TAP_CUE_CHANNELS[0]})

    def test_zero_combine_intent_window_scale_does_not_persist_past_tick(self) -> None:
        self.sim = make_sim(combine_intent_window_scale=0.0)
        agent = self._agent(energy=500.0)
        agent.age = 100  # adult
        feedback = {"spawning": 0.0, "social": 0.0, "tap": 0.0}

        self.sim._coordinate_combine(agent, feedback)

        # window = 0: the intent expires with the current tick.
        self.assertEqual(agent.combine_intent_until, self.sim.tick)
        self.assertLess(agent.combine_intent_until, self.sim.tick + 1)

    def test_legacy_combine_intent_window_persists_across_ticks(self) -> None:
        self.sim = make_sim(combine_intent_window_scale=1.0)
        agent = self._agent(energy=500.0)
        agent.age = 100
        feedback = {"spawning": 0.0, "social": 0.0, "tap": 0.0}

        self.sim._coordinate_combine(agent, feedback)

        self.assertGreaterEqual(agent.combine_intent_until, self.sim.tick + 6)

    def test_zero_exploration_floor_makes_chooser_deterministic(self) -> None:
        self.sim = make_sim(exploration_floor=0.0, drive_injection_scale=0.0)
        agent = self._agent()
        agent.params.plasticity_rate = 0.0
        agent.params.perturbation_rate = 0.0
        outputs_template = [0.0] * len(ACTIONS)
        outputs_template[ACTION_INDEX["eat"]] = 1.0

        chosen = {self.sim._choose_action_from_outputs(agent, list(outputs_template)) for _ in range(100)}

        self.assertEqual(chosen, {"eat"})

    def test_drain_on_empty_place_costs_energy_and_nothing_else(self) -> None:
        self.sim = make_sim()
        agent = self._agent()
        before_energy = agent.energy
        before_health = agent.health
        before_location = agent.location

        self.sim._drain(agent)

        self.assertAlmostEqual(agent.energy, before_energy - 0.04, places=6)
        self.assertEqual(agent.health, before_health)
        self.assertEqual(agent.location, before_location)
        self.assertTrue(agent.alive)
        self.assertEqual(dict(self.sim.deaths_by_cause), {})

    def test_observation_size_matches_schema_and_controller_width(self) -> None:
        self.assertEqual(OBSERVATION_SIZE, 70)
        self.assertEqual(len(ACTIONS), 10)
        self.assertIn("tap", ACTIONS)
        self.sim = make_sim()
        params = ParamVector.neural(self.sim.rng)
        params.neural_budget = 8.0
        agent = self.sim.add_individual("agent", params, 0, 50.0)
        assert agent is not None and agent.controller is not None

        observation = self.sim._observe(agent, self.sim._rosters())

        self.assertEqual(len(observation), OBSERVATION_SIZE)
        self.assertEqual(agent.controller.input_size, OBSERVATION_SIZE)
        self.assertEqual(agent.controller.output_size, len(ACTIONS))


if __name__ == "__main__":
    unittest.main()
