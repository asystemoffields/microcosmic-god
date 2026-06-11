import unittest
from random import Random

import numpy as np

from microcosmic_god.brain import TinyController
from microcosmic_god.modular import ModularController, from_tiny, observation_groups
from microcosmic_god.organisms import ACTIONS, OBSERVATION_SIZE


def _obs(rng: Random) -> list[float]:
    return [rng.uniform(-1.0, 1.5) for _ in range(OBSERVATION_SIZE)]


def _rollout(controller: ModularController, seed: int, ticks: int = 12) -> list[list[float]]:
    rng = Random(seed)
    return [controller.forward(_obs(rng)) for _ in range(ticks)]


class SchemaTest(unittest.TestCase):
    def test_groups_tile_observation_exactly(self):
        groups = observation_groups()
        cursor = 0
        for name, start, length in groups:
            self.assertEqual(start, cursor, f"group {name} misaligned")
            self.assertGreater(length, 0)
            cursor += length
        self.assertEqual(cursor, OBSERVATION_SIZE)


class ForwardTest(unittest.TestCase):
    def test_forward_shapes_and_determinism(self):
        c1 = ModularController.random(Random(3), OBSERVATION_SIZE, len(ACTIONS), n_blocks=3)
        c2 = ModularController.random(Random(3), OBSERVATION_SIZE, len(ACTIONS), n_blocks=3)
        out1, out2 = _rollout(c1, 11), _rollout(c2, 11)
        self.assertEqual(len(out1[0]), len(ACTIONS))
        np.testing.assert_allclose(out1, out2)

    def test_wiring_carries_information_between_blocks(self):
        base = ModularController.random(Random(5), OBSERVATION_SIZE, len(ACTIONS), n_blocks=2)
        wired = ModularController.from_dict(base.to_dict())
        wired.wiring[0, 1] = 1.5
        self.assertFalse(np.allclose(_rollout(base, 7), _rollout(wired, 7)))


class StructuralOperatorTest(unittest.TestCase):
    def test_add_block_is_exactly_function_preserving(self):
        # Build both sides from the same serialized state so the comparison
        # isolates the operator from to_dict's 7-decimal rounding.
        seed_state = ModularController.random(Random(9), OBSERVATION_SIZE, len(ACTIONS), n_blocks=2).to_dict()
        base = ModularController.from_dict(seed_state)
        grown = ModularController.from_dict(seed_state)
        grown.add_block(Random(42), hidden_size=10)
        np.testing.assert_allclose(_rollout(base, 13), _rollout(grown, 13), atol=1e-12)
        self.assertEqual(len(grown.blocks), 3)
        self.assertEqual(grown.capacity, base.capacity + 10)

    def test_duplicate_block_is_exactly_function_preserving(self):
        proto = ModularController.random(Random(17), OBSERVATION_SIZE, len(ACTIONS), n_blocks=2)
        proto.wiring[0, 1] = 0.7
        proto.wiring[1, 0] = -0.4
        seed_state = proto.to_dict()
        base = ModularController.from_dict(seed_state)
        doubled = ModularController.from_dict(seed_state)
        doubled.duplicate_block(0)
        np.testing.assert_allclose(_rollout(base, 19), _rollout(doubled, 19), atol=1e-10)

    def test_duplicated_blocks_diverge_under_perturbation(self):
        c = ModularController.random(Random(23), OBSERVATION_SIZE, len(ACTIONS), n_blocks=1)
        new = c.duplicate_block(0)
        c.blocks[new].weights_in += 0.05
        reference = ModularController.random(Random(23), OBSERVATION_SIZE, len(ACTIONS), n_blocks=1)
        self.assertFalse(np.allclose(_rollout(reference, 29), _rollout(c, 29)))

    def test_prune_block(self):
        c = ModularController.random(Random(31), OBSERVATION_SIZE, len(ACTIONS), n_blocks=3)
        c.prune_block(1)
        self.assertEqual(len(c.blocks), 2)
        self.assertEqual(c.wiring.shape, (2, 2))
        _rollout(c, 37)  # still runs
        with self.assertRaises(ValueError):
            c.prune_block(0), c.prune_block(0)


class SerializationTest(unittest.TestCase):
    def test_round_trip(self):
        c = ModularController.random(Random(41), OBSERVATION_SIZE, len(ACTIONS), n_blocks=2)
        c.add_block(Random(43))
        c.wiring[0, 1] = 0.9
        clone = ModularController.from_dict(c.to_dict())
        np.testing.assert_allclose(_rollout(c, 47), _rollout(clone, 47), atol=1e-6)


class LegacyImportTest(unittest.TestCase):
    def test_tiny_imports_exactly_without_attention(self):
        rng = Random(53)
        tiny = TinyController.random(rng, OBSERVATION_SIZE, 10, len(ACTIONS), with_attention=False)
        modular = from_tiny(tiny.to_dict(include_state=False))
        obs_rng = Random(59)
        for _ in range(15):
            obs = _obs(obs_rng)
            np.testing.assert_allclose(tiny.forward(obs), modular.forward(obs), atol=1e-6)
        self.assertAlmostEqual(tiny.predict_next_energy(), modular.predict_next_energy(), places=6)

    def test_imported_tiny_can_then_grow(self):
        rng = Random(61)
        tiny = TinyController.random(rng, OBSERVATION_SIZE, 8, len(ACTIONS), with_attention=False)
        modular = from_tiny(tiny.to_dict(include_state=False))
        modular.add_block(Random(67), hidden_size=16)
        obs_rng = Random(71)
        for _ in range(10):
            obs = _obs(obs_rng)
            np.testing.assert_allclose(tiny.forward(obs), modular.forward(obs), atol=1e-6)


class PredictionLearningTest(unittest.TestCase):
    def test_energy_prediction_error_shrinks(self):
        c = ModularController.random(Random(73), OBSERVATION_SIZE, len(ACTIONS), n_blocks=2)
        obs = _obs(Random(79))
        errors = []
        for _ in range(200):
            c.forward(obs)
            errors.append(abs(c.learn_energy_prediction(0.8, learning_rate=0.2, plasticity=1.0, prediction_weight=1.0)))
        self.assertLess(errors[-1], errors[0])


if __name__ == "__main__":
    unittest.main()
