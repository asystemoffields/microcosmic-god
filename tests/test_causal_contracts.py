from __future__ import annotations

import json
import tempfile
import unittest
from random import Random

from microcosmic_god.backends import BrainLearningCase
from microcosmic_god.brain import PREDICTION_HEADS, TinyBrain
from microcosmic_god.config import RunConfig
from microcosmic_god.energy import build_artifact, build_structure, structure_decay_channels
from microcosmic_god.genome import Genome
from microcosmic_god.organisms import ACTIONS, OBSERVATION_SIZE
from microcosmic_god.simulation import Simulation
from microcosmic_god.world import CausalChallenge


def make_sim(seed: int = 101, places: int = 3, environment_harshness: float = 1.0) -> Simulation:
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
        environment_harshness=environment_harshness,
    )
    sim = Simulation(config)
    sim._tmpdir = tmp  # type: ignore[attr-defined]
    return sim


def close_sim(sim: Simulation) -> None:
    sim.logger.close()
    sim._tmpdir.cleanup()  # type: ignore[attr-defined]


class CausalContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        sim = getattr(self, "sim", None)
        if sim is not None:
            sim.logger.close()
            sim._tmpdir.cleanup()  # type: ignore[attr-defined]

    def test_minute_profile_defaults_to_harsher_environment(self) -> None:
        self.assertEqual(RunConfig.from_profile("smoke").environment_harshness, 1.0)
        self.assertGreater(RunConfig.from_profile("minute").environment_harshness, 1.0)
        self.assertGreaterEqual(RunConfig.from_profile("modal").environment_harshness, RunConfig.from_profile("minute").environment_harshness)

    def test_environment_harshness_reduces_easy_survival_budget(self) -> None:
        mild = make_sim(seed=909, places=12, environment_harshness=1.0)
        harsh = make_sim(seed=909, places=12, environment_harshness=1.6)
        try:
            mild_capacity = sum(place.capacity for place in mild.world.places)
            harsh_capacity = sum(place.capacity for place in harsh.world.places)
            mild_easy_energy = sum(place.resources["chemical"] + place.resources["biological_storage"] for place in mild.world.places)
            harsh_easy_energy = sum(place.resources["chemical"] + place.resources["biological_storage"] for place in harsh.world.places)
            mild_toolable_energy = sum(place.locked_chemical + place.resources["mechanical"] for place in mild.world.places)
            harsh_toolable_energy = sum(place.locked_chemical + place.resources["mechanical"] for place in harsh.world.places)
            mild_exposure = sum(float(mild._place_exposure_pressure(place)["severity"]) for place in mild.world.places)
            harsh_exposure = sum(float(harsh._place_exposure_pressure(place)["severity"]) for place in harsh.world.places)

            self.assertLess(harsh_capacity, mild_capacity)
            self.assertLess(harsh_easy_energy, mild_easy_energy)
            self.assertGreater(harsh_toolable_energy, mild_toolable_energy)
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
                place.habitat.update({"aquatic": 0.0, "depth": 0.0, "humidity": 0.95, "salinity": 0.0})
                genome = Genome.neural(sim.rng)
                genome.thermal_tolerance = 0.0
                genome.armor = 0.0
                agent = sim.add_organism("agent", genome, 0, 80.0)
                assert agent is not None

                sim._habitat_stress(agent)

                losses.append(1.0 - agent.health)

            self.assertGreater(losses[0], 0.0)
            self.assertGreater(losses[1], losses[0] * 1.25)
            self.assertGreater(harsh.physics_events["harsh_exposure_pressure"], 0)
        finally:
            close_sim(mild)
            close_sim(harsh)

    def test_all_signal_tokens_are_observed(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 50.0)
        assert agent is not None
        agent.event_memory = [index / 20.0 for index in range(8)]
        agent.signal_values = [index / 10.0 for index in range(8)]

        observation = self.sim._observe(agent, self.sim._rosters())

        self.assertEqual(len(observation), OBSERVATION_SIZE)
        self.assertEqual(observation[-16:-8], agent.event_memory)
        self.assertEqual(observation[-8:], agent.signal_values)

    def test_action_results_feed_short_event_memory(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 50.0)
        assert agent is not None

        agent.record_action_result(
            action_index=6,
            energy_delta=4.0,
            health_delta=-0.05,
            damage=0.05,
            prediction_error=0.7,
            reproduction_feedback=0.0,
            social_feedback=0.2,
            tool_feedback=1.0,
        )

        self.assertEqual(agent.last_action, "craft")
        self.assertGreater(agent.event_memory[0], 0.0)
        self.assertGreater(agent.event_memory[3], 0.0)
        self.assertGreater(agent.event_memory[6], 0.0)
        self.assertGreater(agent.event_memory[7], 0.0)

    def test_checkpoints_capture_cognitive_context(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 50.0)
        assert agent is not None
        agent.record_action_result(8, 3.0, 0.0, 0.0, 0.25, 0.0, 0.1, 1.0)
        saved = self.sim.checkpoints.save_brain(42, agent, "test_cognition", {}, bucket="general")
        self.assertTrue(saved)

        checkpoint = next(self.sim.logger.checkpoint_dir.glob("brain_t00000042_*.json"))
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertIn("cognition", payload)
        self.assertEqual(payload["cognition"]["last_action"], "use_tool")
        self.assertIn("prediction_errors", payload["cognition"])
        self.assertGreater(payload["cognition"]["event_memory"]["tool"], 0.0)

    def test_lineage_metadata_tracks_inherited_agent_templates(self) -> None:
        self.sim = make_sim(places=1)
        parent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert parent is not None and parent.brain_template is not None

        child = self.sim.add_organism(
            "agent",
            Genome.neural(self.sim.rng),
            0,
            40.0,
            generation=parent.generation + 1,
            parent_ids=(parent.id,),
            brain_template=parent.brain_template,
        )
        assert child is not None

        self.assertEqual(parent.lineage_root_id, parent.id)
        self.assertEqual(child.lineage_root_id, parent.id)
        self.assertEqual(child.parent_lineage_ids, (parent.id,))
        self.assertTrue(child.inherited_brain_template)
        self.assertEqual(child.to_summary()["lineage_root_id"], parent.id)
        self.assertEqual(child.cognitive_snapshot()["lineage"]["root_id"], parent.id)

        lineage_summary = self.sim._lineage_summary()
        self.assertEqual(lineage_summary["agent_lineages_total"], 1)
        self.assertEqual(lineage_summary["top_living"][0]["living"], 2)
        self.assertEqual(lineage_summary["top_living"][0]["inherited_template_count"], 1)

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

    def test_agent_defense_can_block_and_counter_predation(self) -> None:
        self.sim = make_sim(places=1)
        attacker_genome = Genome.neural(self.sim.rng)
        attacker_genome.mobility = 0.20
        attacker_genome.manipulator = 0.20
        attacker_genome.mechanical_use = 0.20
        target_genome = Genome.neural(self.sim.rng)
        target_genome.armor = 0.30
        target_genome.mobility = 0.40
        target_genome.manipulator = 1.00
        helper_genome = Genome.neural(self.sim.rng)
        helper_genome.armor = 1.00
        helper_genome.mobility = 1.00
        helper_genome.manipulator = 1.00
        helper_genome.sensor_range = 1.00
        helper_genome.signal_strength = 1.00
        attacker = self.sim.add_organism("agent", attacker_genome, 0, 80.0)
        target = self.sim.add_organism("agent", target_genome, 0, 80.0)
        helper = self.sim.add_organism("agent", helper_genome, 0, 80.0)
        assert attacker is not None and target is not None and helper is not None
        attacker.health = 0.03
        target.health = 0.45
        target.last_action = "observe"
        helper.health = 1.0
        helper.last_action = "signal"
        helper.tool_skill["protect"] = 1.0
        helper.tool_skill["support"] = 1.0
        self.sim.world.places[0].physics["shelter"] = 0.8

        class ZeroRng:
            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

        self.sim.rng = ZeroRng()  # type: ignore[assignment]
        before_target_health = target.health
        before_attacker_health = attacker.health
        before_protect = target.tool_skill["protect"]

        self.sim._attack(attacker)

        self.assertAlmostEqual(target.health, before_target_health)
        self.assertLess(attacker.health, before_attacker_health)
        self.assertGreater(target.tool_skill["protect"], before_protect)
        self.assertGreater(self.sim.collaboration_events["defense"], 0)

    def test_tool_choice_uses_recognized_situation_without_magic(self) -> None:
        self.sim = make_sim(places=1)
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert agent is not None
        agent.inventory = {"stone": 1, "shell": 1}
        place = self.sim.world.places[0]
        place.causal_challenge = None
        place.obstacles["water"] = 1.0
        place.physics["current_exposure"] = 0.5

        class ZeroRng:
            def random(self) -> float:
                return 0.0

        self.sim.rng = ZeroRng()  # type: ignore[assignment]
        base_affordance = "crack"
        base_score = self.sim._affordance_score(agent, base_affordance)

        affordance, _score, directed, context = self.sim._situation_affordance_choice(agent, place, base_affordance, base_score)
        self.assertEqual(affordance, "crack")
        self.assertFalse(directed)
        self.assertEqual(context["problem"]["required_affordance"], "contain")

        agent.record_lesson(
            {
                "kind": "tool_use",
                "affordance": "contain",
                "success": True,
                "gain": 5.0,
                "score": 0.4,
                "problem": {
                    "kind": "obstacle",
                    "obstacle": "water",
                    "severity": 0.8,
                    "required_affordance": "contain",
                },
            }
        )

        affordance, score, directed, context = self.sim._situation_affordance_choice(agent, place, base_affordance, base_score)
        self.assertEqual(affordance, "contain")
        self.assertGreaterEqual(score, 0.08)
        self.assertTrue(directed)
        self.assertGreater(context["recognition"], 0.5)
        self.assertGreater(context["memory_bias"], 0.0)

        place.causal_challenge = CausalChallenge(sequence=("conduct",), payoff_energy="electrical", payoff_remaining=10.0, difficulty=0.4)
        affordance, _score, directed, context = self.sim._situation_affordance_choice(agent, place, base_affordance, base_score)
        self.assertEqual(affordance, "crack")
        self.assertFalse(directed)
        self.assertEqual(context["problem"]["required_affordance"], "conduct")

    def test_tool_lessons_store_situation_separately_from_attempt(self) -> None:
        self.sim = make_sim(places=1)
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert agent is not None
        place = self.sim.world.places[0]
        place.causal_challenge = None
        place.obstacles["water"] = 1.0
        place.physics["current_exposure"] = 0.5

        self.sim._record_tool_lesson(
            agent,
            place,
            kind="tool_use",
            affordance="crack",
            success=False,
            score=0.4,
        )

        lesson = agent.lesson_memory[-1]
        self.assertEqual(lesson["attempted_affordance"], "crack")
        self.assertEqual(lesson["problem"]["required_affordance"], "contain")

    def test_clone_mutate_capacity_uses_current_local_population(self) -> None:
        self.sim = make_sim(places=2)
        parent_genome = Genome.plant(self.sim.rng)
        parent = self.sim.add_organism("plant", parent_genome, 0, 200.0)
        neighbor = self.sim.add_organism("plant", Genome.plant(self.sim.rng), 0, 20.0)
        assert parent is not None and neighbor is not None
        self.sim.world.places[0].capacity = 2
        parent.age = 100
        neighbor.location = 1

        self.sim._clone_mutate(parent, {"reproduction": 0.0, "social": 0.0})

        self.assertEqual(self.sim.births_by_mode["clone_mutate"], 1)
        self.assertEqual(len(self.sim._living_ids_at(0)), 2)

    def test_reproduction_actions_are_operator_labels(self) -> None:
        self.assertIn("clone_mutate", ACTIONS)
        self.assertIn("coordinate", ACTIONS)
        self.assertNotIn("asexual_reproduce", ACTIONS)
        self.assertNotIn("mate", ACTIONS)

    def test_complex_neural_agents_can_clone_with_soft_strain(self) -> None:
        self.sim = make_sim()
        self.sim.config.clone_complexity_soft_limit = 0.0
        parent_genome = Genome.neural(self.sim.rng)
        parent_genome.neural_budget = 32.0
        parent_genome.memory_budget = 16.0
        parent = self.sim.add_organism("agent", parent_genome, 0, 1_000.0)
        assert parent is not None
        parent.age = 100

        self.sim._clone_mutate(parent, {"reproduction": 0.0, "social": 0.0})

        self.assertEqual(self.sim.births_by_mode["clone_mutate"], 1)
        self.assertNotIn("clone_mutate_complexity_ceiling", self.sim.reproduction_failures)

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

    def test_successful_craft_counts_as_tool_making(self) -> None:
        self.sim = make_sim()
        agent_genome = Genome.neural(self.sim.rng)
        agent_genome.manipulator = 1.0
        agent = self.sim.add_organism("agent", agent_genome, 0, 100.0)
        assert agent is not None
        agent.inventory = {"stone": 1, "fiber": 1}

        class SuccessfulCraftRng:
            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

        self.sim.rng = SuccessfulCraftRng()  # type: ignore[assignment]

        self.sim._craft(agent, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertEqual(agent.successful_tools, 1)
        self.assertEqual(self.sim.tool_successes["craft"], 1)
        self.assertGreater(agent.success_profile["tool_make"], 0.0)

    def test_method_quality_amplifies_real_material_fit_without_magic(self) -> None:
        rough_conductor = build_artifact({"crystal": 1, "stone": 1}, target_affordance="conduct")
        worked_conductor = build_artifact({"crystal": 1, "stone": 1}, method_quality=0.85, target_affordance="conduct")
        bad_conductor = build_artifact({"fiber": 2}, method_quality=1.0, target_affordance="conduct")

        self.assertGreater(worked_conductor.capabilities["conduct"], rough_conductor.capabilities["conduct"])
        self.assertGreater(worked_conductor.durability, rough_conductor.durability)
        self.assertLess(bad_conductor.capabilities["conduct"], 0.08)

    def test_only_intentional_marks_transmit_lesson_traces_when_observed(self) -> None:
        self.sim = make_sim()
        agent_genome = Genome.neural(self.sim.rng)
        agent_genome.sensor_range = 1.0
        agent_genome.memory_budget = 12.0
        reader = self.sim.add_organism("agent", agent_genome, 0, 80.0)
        assert reader is not None
        self.sim.world.create_mark(
            0,
            source_id=999,
            token=3,
            intensity=1.0,
            durability=160.0,
            trace={
                "schema": "lesson_trace_v1",
                "intentional": True,
                "affordance": "filter",
                "inscription_quality": 1.0,
                "method_quality": 0.8,
                "tool_feedback": 1.0,
                "lesson": {
                    "kind": "tool_use",
                    "affordance": "filter",
                    "success": True,
                    "gain": 1.0,
                    "score": 0.7,
                    "problem": {"kind": "resource", "required_affordance": "filter"},
                },
            },
        )
        self.sim.world.create_mark(
            0,
            source_id=998,
            token=4,
            intensity=1.0,
            durability=160.0,
            trace={
                "affordance": "conduct",
                "inscription_quality": 1.0,
                "method_quality": 1.0,
                "tool_feedback": 1.0,
            },
        )
        before = reader.tool_skill["filter"]
        before_conduct = reader.tool_skill["conduct"]

        self.sim._observe_others(reader, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertGreater(reader.tool_skill["filter"], before)
        self.assertEqual(reader.tool_skill["conduct"], before_conduct)
        self.assertEqual(self.sim.mark_lessons["filter"], 1)
        self.assertGreater(reader.success_profile["written_learning"], 0.0)
        self.assertGreater(reader.tool_skill["interpret_mark"], 0.0)
        self.assertEqual(reader.lesson_memory[-1]["affordance"], "filter")
        self.assertEqual(self.sim.world.places[0].marks[0].reads, 1)
        self.assertGreater(self.sim.world.places[0].marks[0].value_transmitted, 0.0)

    def test_better_writing_transmits_more_literacy_value(self) -> None:
        low = self._mark_read_gain_for_quality(quality=0.20)
        high = self._mark_read_gain_for_quality(quality=0.95)

        self.assertGreater(high["skill_gain"], low["skill_gain"])
        self.assertGreater(high["interpret_gain"], low["interpret_gain"])
        self.assertGreater(high["mark_value"], low["mark_value"])

    def _mark_read_gain_for_quality(self, quality: float) -> dict[str, float]:
        self.sim = make_sim(seed=int(100 + quality * 100))
        agent_genome = Genome.neural(self.sim.rng)
        agent_genome.sensor_range = 1.0
        agent_genome.memory_budget = 12.0
        reader = self.sim.add_organism("agent", agent_genome, 0, 80.0)
        assert reader is not None
        self.sim.world.create_mark(
            0,
            source_id=999,
            token=3,
            intensity=1.0,
            durability=160.0,
            trace={
                "schema": "lesson_trace_v1",
                "intentional": True,
                "affordance": "filter",
                "inscription_quality": 1.0,
                "writing_quality": quality,
                "coherence": quality,
                "lesson_value": 1.2,
                "method_quality": 0.8,
                "tool_feedback": 1.0,
                "lesson": {
                    "kind": "tool_use",
                    "affordance": "filter",
                    "success": True,
                    "gain": 1.0,
                    "score": 0.7,
                    "problem": {"kind": "resource", "required_affordance": "filter"},
                },
            },
        )

        class ZeroRng:
            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

        self.sim.rng = ZeroRng()  # type: ignore[assignment]
        before_skill = reader.tool_skill["filter"]
        before_interpret = reader.tool_skill["interpret_mark"]

        self.sim._observe_others(reader, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        mark = self.sim.world.places[0].marks[0]
        self.sim.logger.close()
        self.sim._tmpdir.cleanup()  # type: ignore[attr-defined]
        self.sim = None  # type: ignore[assignment]
        return {
            "skill_gain": reader.tool_skill["filter"] - before_skill,
            "interpret_gain": reader.tool_skill["interpret_mark"] - before_interpret,
            "mark_value": mark.value_transmitted,
        }

    def test_useful_reads_feed_back_to_present_authors(self) -> None:
        self.sim = make_sim()
        writer_genome = Genome.neural(self.sim.rng)
        reader_genome = Genome.neural(self.sim.rng)
        reader_genome.sensor_range = 1.0
        reader_genome.memory_budget = 12.0
        writer = self.sim.add_organism("agent", writer_genome, 0, 80.0)
        reader = self.sim.add_organism("agent", reader_genome, 0, 80.0)
        assert writer is not None and reader is not None
        self.sim.world.create_mark(
            0,
            source_id=writer.id,
            token=3,
            intensity=1.0,
            durability=160.0,
            trace={
                "schema": "lesson_trace_v1",
                "intentional": True,
                "affordance": "filter",
                "inscription_quality": 1.0,
                "writing_quality": 0.95,
                "coherence": 0.95,
                "lesson_value": 1.2,
                "method_quality": 0.8,
                "tool_feedback": 1.0,
                "lesson": {
                    "kind": "tool_use",
                    "affordance": "filter",
                    "success": True,
                    "gain": 1.0,
                    "score": 0.7,
                    "problem": {"kind": "resource", "required_affordance": "filter"},
                },
            },
        )

        class ZeroRng:
            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

        self.sim.rng = ZeroRng()  # type: ignore[assignment]
        before = writer.tool_skill["inscribe"]

        self.sim._observe_others(reader, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertGreater(writer.tool_skill["inscribe"], before)
        self.assertGreater(writer.success_profile["knowledge_transmitted"], 0.0)
        self.assertGreater(self.sim.mark_author_feedbacks["filter"], 0.0)

    def test_self_reading_counts_as_memory_not_knowledge_transmission(self) -> None:
        self.sim = make_sim()
        genome = Genome.neural(self.sim.rng)
        genome.sensor_range = 1.0
        genome.memory_budget = 12.0
        agent = self.sim.add_organism("agent", genome, 0, 80.0)
        assert agent is not None
        self.sim.world.create_mark(
            0,
            source_id=agent.id,
            token=3,
            intensity=1.0,
            durability=160.0,
            trace={
                "schema": "lesson_trace_v1",
                "intentional": True,
                "affordance": "filter",
                "inscription_quality": 1.0,
                "writing_quality": 0.95,
                "coherence": 0.95,
                "lesson_value": 1.2,
                "method_quality": 0.8,
                "tool_feedback": 1.0,
                "lesson": {
                    "kind": "tool_use",
                    "affordance": "filter",
                    "success": True,
                    "gain": 1.0,
                    "score": 0.7,
                    "problem": {"kind": "resource", "required_affordance": "filter"},
                },
            },
        )

        class ZeroRng:
            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

        self.sim.rng = ZeroRng()  # type: ignore[assignment]
        before_inscribe = agent.tool_skill["inscribe"]
        before_filter = agent.tool_skill["filter"]

        self.sim._observe_others(agent, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertEqual(agent.tool_skill["inscribe"], before_inscribe)
        self.assertGreater(agent.tool_skill["filter"], before_filter)
        self.assertGreater(agent.success_profile["written_learning"], 0.0)
        self.assertEqual(agent.success_profile["knowledge_transmitted"], 0.0)
        self.assertEqual(self.sim.mark_author_feedbacks["filter"], 0.0)

    def test_record_artifact_can_carry_lesson_trace_across_places(self) -> None:
        self.sim = make_sim(places=4)
        genome = Genome.neural(self.sim.rng)
        genome.sensor_range = 1.0
        genome.memory_budget = 12.0
        agent = self.sim.add_organism("agent", genome, 0, 80.0)
        assert agent is not None
        artifact = build_artifact({"fiber": 1, "resin": 1}, method_quality=1.0, target_affordance="record")
        agent.artifacts.append(artifact)
        trace = {
            "schema": "lesson_trace_v1",
            "intentional": True,
            "affordance": "filter",
            "inscription_quality": 1.0,
            "writing_quality": 0.95,
            "coherence": 0.95,
            "lesson_value": 1.2,
            "method_quality": 0.8,
            "tool_feedback": 1.0,
            "lesson": {
                "kind": "tool_use",
                "affordance": "filter",
                "success": True,
                "gain": 1.0,
                "score": 0.7,
                "problem": {"kind": "resource", "required_affordance": "filter"},
            },
        }
        self.sim._inscribe_portable_mark(agent, artifact, token=3, intensity=1.0, durability=160.0, trace=trace)
        agent.location = 2

        class ZeroRng:
            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

        self.sim.rng = ZeroRng()  # type: ignore[assignment]
        before = agent.tool_skill["filter"]

        self.sim._observe_others(agent, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertGreater(agent.tool_skill["filter"], before)
        self.assertEqual(artifact.inscriptions[0]["reads"], 1)
        self.assertGreater(artifact.inscriptions[0]["value_transmitted"], 0.0)
        self.assertGreater(self.sim.portable_mark_reads["filter"], 0)

    def test_carry_artifact_expands_material_capacity(self) -> None:
        self.sim = make_sim()
        genome = Genome.neural(self.sim.rng)
        genome.manipulator = 0.2
        genome.developmental_complexity = 0.0
        agent = self.sim.add_organism("agent", genome, 0, 80.0)
        assert agent is not None
        base_limit = agent.inventory_limit()
        artifact = build_artifact({"fiber": 1, "shell": 1, "resin": 1}, method_quality=1.0, target_affordance="carry")

        agent.artifacts.append(artifact)

        self.assertGreater(artifact.capabilities["carry"], 0.0)
        self.assertGreater(agent.inventory_limit(), base_limit)

    def test_protective_artifact_reduces_environment_stress(self) -> None:
        self.sim = make_sim()
        place = self.sim.world.places[0]
        place.physics["pressure"] = 1.0
        place.physics["abrasion"] = 0.7
        genome = Genome.neural(self.sim.rng)
        genome.pressure_tolerance = 0.05
        genome.armor = 0.0
        unprotected = self.sim.add_organism("agent", genome, 0, 80.0)
        protected = self.sim.add_organism("agent", genome, 0, 80.0)
        assert unprotected is not None and protected is not None
        protected.artifacts.append(build_artifact({"shell": 2, "fiber": 1}, method_quality=1.0, target_affordance="protect"))

        self.sim._habitat_stress(unprotected)
        self.sim._habitat_stress(protected)

        self.assertLess(protected.health, 1.0)
        self.assertGreater(protected.health, unprotected.health)

    def test_exposure_pressure_favors_insulation_and_heat_control(self) -> None:
        self.sim = make_sim(places=1)
        place = self.sim.world.places[0]
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
        place.habitat.update({"aquatic": 0.0, "depth": 0.0, "humidity": 0.95, "salinity": 0.0})
        fragile = Genome.neural(self.sim.rng)
        fragile.thermal_tolerance = 0.0
        fragile.armor = 0.0
        unprotected = self.sim.add_organism("agent", fragile, 0, 80.0)
        protected = self.sim.add_organism("agent", fragile, 0, 80.0)
        assert unprotected is not None and protected is not None
        protected.artifacts.append(build_artifact({"fiber": 2, "resin": 1}, method_quality=1.0, target_affordance="insulate"))
        protected.artifacts.append(build_artifact({"crystal": 1, "stone": 1}, method_quality=1.0, target_affordance="concentrate_heat"))

        self.sim._habitat_stress(unprotected)
        self.sim._habitat_stress(protected)

        self.assertLess(unprotected.health, 1.0)
        self.assertGreater(protected.health, unprotected.health)
        self.assertGreater(self.sim.physics_events["exposure_pressure"], 0)
        self.assertGreater(self.sim.physics_events["tool_buffered_exposure"], 0)
        self.assertGreater(protected.tool_skill["insulate"], 0.0)

    def test_active_helpers_can_buffer_exposure_without_replacing_tools(self) -> None:
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
            place.habitat.update({"aquatic": 0.0, "depth": 0.0, "humidity": 0.95, "salinity": 0.0})
        fragile = Genome.neural(self.sim.rng)
        fragile.thermal_tolerance = 0.0
        fragile.armor = 0.0
        alone = self.sim.add_organism("agent", fragile, 1, 80.0)
        helped = self.sim.add_organism("agent", fragile, 0, 80.0)
        helper = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert alone is not None and helped is not None and helper is not None
        helper.last_action = "signal"
        helper.tool_skill["protect"] = 1.0
        helper.tool_skill["support"] = 1.0
        helper.genome.signal_strength = 1.0
        helper.genome.sensor_range = 1.0
        helper.genome.manipulator = 1.0

        self.sim._habitat_stress(alone)
        self.sim._habitat_stress(helped)

        self.assertLess(alone.health, 1.0)
        self.assertGreater(helped.health, alone.health)
        self.assertGreater(self.sim.collaboration_events["exposure"], 0)
        self.assertGreater(self.sim.physics_events["collaborative_exposure_buffer"], 0)

    def test_plain_marks_do_not_automatically_encode_recent_lessons(self) -> None:
        self.sim = make_sim()
        agent_genome = Genome.neural(self.sim.rng)
        agent_genome.manipulator = 1.0
        agent_genome.memory_budget = 18.0
        agent_genome.signal_strength = 1.0
        writer = self.sim.add_organism("agent", agent_genome, 0, 100.0)
        assert writer is not None

        class ZeroRng:
            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

        self.sim.rng = ZeroRng()  # type: ignore[assignment]

        self.sim._mark(writer, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertEqual(self.sim.world.places[0].marks[-1].trace, {})
        self.assertEqual(dict(self.sim.mark_lesson_packets), {})

    def test_agents_can_discover_intentional_lesson_inscription_as_a_skill(self) -> None:
        self.sim = make_sim()
        agent_genome = Genome.neural(self.sim.rng)
        agent_genome.manipulator = 1.0
        agent_genome.memory_budget = 18.0
        agent_genome.signal_strength = 1.0
        agent_genome.sensor_range = 1.0
        writer = self.sim.add_organism("agent", agent_genome, 0, 100.0)
        assert writer is not None
        writer.inventory = {"fiber": 2, "resin": 1}
        writer.record_lesson(
            {
                "kind": "tool_use",
                "affordance": "filter",
                "success": True,
                "gain": 6.0,
                "score": 0.8,
                "method_quality": 0.6,
                "components": {"fiber": 2, "resin": 1},
                "problem": {
                    "kind": "obstacle",
                    "obstacle": "water",
                    "severity": 0.8,
                    "required_affordance": "filter",
                    "required_capability": "filter",
                },
            }
        )

        class ZeroRng:
            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

        self.sim.rng = ZeroRng()  # type: ignore[assignment]
        before = writer.tool_skill["inscribe"]

        self.sim._mark(writer, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        trace = self.sim.world.places[0].marks[-1].trace
        self.assertTrue(trace["intentional"])
        self.assertEqual(trace["schema"], "lesson_trace_v1")
        self.assertEqual(trace["lesson"]["affordance"], "filter")
        self.assertIn("problem", trace["lesson"])
        self.assertGreater(writer.tool_skill["inscribe"], before)
        self.assertEqual(self.sim.mark_lesson_packets["filter"], 1)

    def test_causal_challenge_unlocks_after_affordance_sequence(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert agent is not None
        place = self.sim.world.places[0]
        before = place.resources["chemical"]
        place.causal_challenge = CausalChallenge(
            sequence=("contain", "filter"),
            payoff_energy="chemical",
            payoff_remaining=10.0,
            difficulty=0.0,
        )
        feedback = {"reproduction": 0.0, "social": 0.0, "tool": 0.0}

        first_gain = self.sim._advance_causal_challenge(agent, place, "contain", competence=1.0, feedback=feedback)
        second_gain = self.sim._advance_causal_challenge(agent, place, "filter", competence=1.0, feedback=feedback)

        self.assertGreater(first_gain, 0.0)
        self.assertGreater(second_gain, 0.0)
        self.assertGreater(place.resources["chemical"], before)
        self.assertEqual(self.sim.causal_unlocks["contain>filter"], 1)
        self.assertGreater(agent.success_profile["causal_unlock"], 0.0)
        self.sim.logger.flush()
        story_records = [
            json.loads(line)
            for line in self.sim.logger.story_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(any(record["kind"] == "causal_unlock" for record in story_records))

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

    def test_bind_practice_transfers_only_to_related_skills(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert agent is not None

        self.sim._tool_effect(agent, self.sim.world.places[0], "bind", score=1.0, skill=0.0)

        self.assertGreater(agent.tool_skill["bind"], 0.0)
        self.assertGreater(agent.tool_skill["craft"], 0.0)
        self.assertGreater(agent.tool_skill["build"], 0.0)
        self.assertEqual(agent.tool_skill["conduct"], 0.0)
        self.assertEqual(agent.tool_skill["concentrate_heat"], 0.0)

    def test_specialists_keep_cognitive_credit_from_repeated_practice(self) -> None:
        self.sim = make_sim()
        agent = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert agent is not None
        agent.tool_skill["bind"] = 1.0
        agent.tool_use_counts = {"bind": 96}

        control = self.sim._skill_breadth(agent)

        self.assertGreater(control, 0.45)
        self.assertLess(control, 0.70)

    def test_active_helpers_can_supply_build_materials(self) -> None:
        self.sim = make_sim()
        actor_genome = Genome.neural(self.sim.rng)
        actor_genome.manipulator = 1.0
        helper_genome = Genome.neural(self.sim.rng)
        helper_genome.manipulator = 1.0
        helper_genome.mobility = 1.0
        helper_genome.sensor_range = 1.0
        helper_genome.signal_strength = 1.0
        actor = self.sim.add_organism("agent", actor_genome, 0, 100.0)
        helper = self.sim.add_organism("agent", helper_genome, 0, 100.0)
        assert actor is not None and helper is not None
        actor.inventory = {"stone": 1}
        helper.inventory = {"branch": 1, "fiber": 1, "resin": 1}
        helper.last_action = "coordinate"
        helper.recombine_intent_until = self.sim.tick + 4
        helper.tool_skill["build"] = 1.0
        helper.tool_skill["support"] = 1.0

        class SuccessfulBuildRng:
            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

        self.sim.rng = SuccessfulBuildRng()  # type: ignore[assignment]

        self.sim._build_structure(actor, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        place = self.sim.world.places[actor.location]
        self.assertEqual(len(place.structures), 1)
        self.assertLess(helper.inventory_count(), 3)
        self.assertEqual(actor.tool_use_counts["build"], 1)
        self.assertGreater(self.sim.collaboration_events["build"], 0)

    def test_collective_support_and_relocation_shock_are_tracked_for_moves(self) -> None:
        self.sim = make_sim(places=2)
        actor_genome = Genome.neural(self.sim.rng)
        actor_genome.mobility = 0.40
        actor_genome.manipulator = 0.25
        actor_genome.sensor_range = 0.0
        actor_genome.aquatic_affinity = 0.0
        helper_genome = Genome.neural(self.sim.rng)
        helper_genome.mobility = 1.0
        helper_genome.manipulator = 1.0
        helper_genome.sensor_range = 1.0
        helper_genome.signal_strength = 1.0
        actor = self.sim.add_organism("agent", actor_genome, 0, 120.0)
        helper = self.sim.add_organism("agent", helper_genome, 0, 120.0)
        assert actor is not None and helper is not None
        helper.last_action = "coordinate"
        helper.recombine_intent_until = self.sim.tick + 4
        helper.tool_skill["traverse"] = 1.0
        helper.tool_skill["support"] = 1.0
        origin = self.sim.world.places[0]
        destination = self.sim.world.places[1]
        origin.physics.update({"temperature": 0.92, "fluid_level": 0.0, "pressure": 0.0, "humidity": 0.08, "salinity": 0.0, "elevation": 0.85, "oxygen": 0.30})
        origin.obstacles.update({"water": 0.0, "height": 0.2, "thorn": 0.1, "heat": 0.5})
        destination.physics.update({"temperature": 0.22, "fluid_level": 1.0, "pressure": 1.05, "humidity": 0.95, "salinity": 0.85, "elevation": 0.04, "oxygen": 0.18})
        destination.obstacles.update({"water": 1.0, "height": 0.7, "thorn": 0.0, "heat": 0.0})
        edge = self.sim.world.edge_between(0, 1)
        assert edge is not None
        edge.traversal_required = 0.75
        edge.distance = 1.9
        edge.danger = 0.45
        edge.slope = -0.65
        edge.current = 0.0

        class MoveRng:
            def choice(self, values):  # type: ignore[no-untyped-def]
                return tuple(values)[0]

            def random(self) -> float:
                return 0.0

            def gauss(self, _mu: float, _sigma: float) -> float:
                return 0.0

        self.sim.rng = MoveRng()  # type: ignore[assignment]

        self.sim._move(actor, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertEqual(actor.location, 0)
        movement = self.sim._movement_summary()
        self.assertEqual(movement["events"]["attempt"], 1)
        self.assertEqual(movement["events"]["assisted_failure"], 1)
        self.assertGreater(movement["avg_relocation_shock"], 0.0)
        self.assertGreater(movement["avg_energy_cost"], 0.0)
        self.assertGreater(movement["avg_health_cost"], 0.0)
        self.assertLess(actor.energy, 120.0)
        self.assertLess(helper.energy, 120.0)

    def test_successful_movement_spends_energy_even_when_easy(self) -> None:
        self.sim = make_sim(places=2)
        genome = Genome.neural(self.sim.rng)
        genome.mobility = 1.0
        genome.sensor_range = 0.0
        agent = self.sim.add_organism("agent", genome, 0, 100.0)
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

        self.sim._move(agent, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertEqual(agent.location, 1)
        self.assertLess(agent.energy, before)
        self.assertGreater(self.sim._movement_summary()["avg_energy_cost"], 0.0)

    def test_failed_movement_spends_energy(self) -> None:
        self.sim = make_sim(places=2)
        genome = Genome.neural(self.sim.rng)
        genome.mobility = 0.02
        genome.sensor_range = 0.0
        genome.aquatic_affinity = 0.0
        genome.thermal_tolerance = 0.0
        genome.pressure_tolerance = 0.0
        agent = self.sim.add_organism("agent", genome, 0, 100.0)
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
        before = agent.energy

        self.sim._move(agent, {"reproduction": 0.0, "social": 0.0, "tool": 0.0})

        self.assertEqual(agent.location, 0)
        self.assertLess(agent.energy, before)
        movement = self.sim._movement_summary()
        self.assertEqual(movement["events"]["failure"], 1)
        self.assertGreater(movement["avg_energy_cost"], 0.0)

    def test_environment_generation_has_hostile_treasure_biomes(self) -> None:
        self.sim = make_sim(seed=55, places=10)
        places = {place.archetype: place for place in self.sim.world.places}

        self.assertEqual(self.sim.world.places[0].archetype, "pelagic")
        self.assertGreaterEqual(places["pelagic"].obstacles["water"], 0.72)
        self.assertGreater(places["trench"].physics["pressure"], 0.75)
        self.assertGreater(places["hydrothermal_vent"].resources["thermal"], 40.0)
        self.assertGreaterEqual(places["high_ridge"].physics["elevation"], 0.72)
        self.assertGreater(places["mineral_scree"].locked_chemical, 20.0)

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
        # Tests the core (non-attention) plasticity path. Attention plasticity
        # is exercised separately in AttentionTests below.
        brain = TinyBrain.random(Random(7), input_size=5, hidden_size=4, output_size=3, with_attention=False)
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

    def test_brain_learns_multiple_prediction_heads(self) -> None:
        brain = TinyBrain.random(Random(9), input_size=5, hidden_size=4, output_size=3)
        brain.forward([0.5, -0.1, 0.4, 0.7, 0.2])
        before_damage = list(brain.auxiliary_prediction_weights["damage"])
        before_tool = list(brain.auxiliary_prediction_weights["tool"])

        brain.learn(
            action_index=2,
            valence=0.8,
            energy_delta=0.4,
            learning_rate=0.18,
            plasticity=0.90,
            prediction_weight=0.85,
            outcome_targets={"damage": 0.3, "reproduction": 0.0, "social": 0.2, "tool": 1.0, "hazard": 0.1},
        )

        self.assertEqual(set(brain.last_prediction_errors), set(PREDICTION_HEADS))
        self.assertNotEqual(before_damage, brain.auxiliary_prediction_weights["damage"])
        self.assertNotEqual(before_tool, brain.auxiliary_prediction_weights["tool"])

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

    def test_death_checkpoints_do_not_crowd_out_living_champions(self) -> None:
        self.sim = make_sim()
        death_candidate = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        tool_champion = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        reproductive_champion = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        lineage_founder = self.sim.add_organism("agent", Genome.neural(self.sim.rng), 0, 80.0)
        assert death_candidate is not None and tool_champion is not None and reproductive_champion is not None and lineage_founder is not None
        tool_champion.successful_tools = 80
        reproductive_champion.offspring_count = 18
        reproductive_champion.generation = 3
        lineage_founder.generation = 9
        lineage_founder.offspring_count = 4

        for index in range(30):
            self.sim.checkpoints.save_brain(index, death_candidate, f"death_predation_{index}", {}, bucket="notable_death")

        death_bucket = self.sim.checkpoints.to_summary()["buckets"]["notable_death"]
        self.assertEqual(death_bucket, self.sim.checkpoints.bucket_limits["notable_death"])

        self.sim.tick = 123
        self.sim._checkpoint_champions("final")
        summary = self.sim.checkpoints.to_summary()

        self.assertIn("final_overall_champion", summary["reasons"])
        self.assertIn("final_reproductive_champion", summary["reasons"])
        self.assertIn("final_lineage_founder", summary["reasons"])
        self.assertGreater(summary["buckets"].get("reproductive_champion", 0), 0)
        self.assertGreater(summary["buckets"].get("lineage_founder", 0), 0)

    def _torch_runtime_or_skip(self):  # type: ignore[no-untyped-def]
        try:
            from microcosmic_god.backends.torch_gpu import TorchBrainRuntime

            return TorchBrainRuntime(device="cpu")
        except Exception as exc:
            raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

    def test_torch_brain_batch_forward_matches_cpu_reference(self) -> None:
        # The torch backend does not yet implement the attention head; for now,
        # parity is verified on attention-disabled brains. Attention behavior
        # has dedicated CPU tests in AttentionTests.
        runtime = self._torch_runtime_or_skip()
        rng = Random(123)
        cpu_brains = [
            TinyBrain.random(rng, input_size=5, hidden_size=3, output_size=4, with_attention=False),
            TinyBrain.random(rng, input_size=5, hidden_size=4, output_size=4, with_attention=False),
            TinyBrain.random(rng, input_size=5, hidden_size=3, output_size=4, with_attention=False),
        ]
        torch_brains = [TinyBrain.from_dict(brain.to_dict(include_state=True)) for brain in cpu_brains]
        observations = [
            [0.2, -0.1, 0.7, 0.0, 0.5],
            [-0.3, 0.4, 0.1, 0.9, -0.2],
            [0.8, 0.0, -0.5, 0.3, 0.2],
        ]

        expected = [brain.forward(observation) for brain, observation in zip(cpu_brains, observations)]
        actual = runtime.forward_many(torch_brains, observations)

        for expected_row, actual_row in zip(expected, actual):
            for expected_value, actual_value in zip(expected_row, actual_row):
                self.assertAlmostEqual(expected_value, actual_value, places=5)
        for cpu_brain, torch_brain in zip(cpu_brains, torch_brains):
            for expected_value, actual_value in zip(cpu_brain.hidden, torch_brain.hidden):
                self.assertAlmostEqual(expected_value, actual_value, places=5)
            for expected_value, actual_value in zip(cpu_brain.input_trace, torch_brain.input_trace):
                self.assertAlmostEqual(expected_value, actual_value, places=6)

    def test_torch_brain_batch_learning_matches_cpu_reference(self) -> None:
        runtime = self._torch_runtime_or_skip()
        rng = Random(321)
        cpu_brains = [
            TinyBrain.random(rng, input_size=5, hidden_size=4, output_size=3, with_attention=False),
            TinyBrain.random(rng, input_size=5, hidden_size=4, output_size=3, with_attention=False),
        ]
        torch_brains = [TinyBrain.from_dict(brain.to_dict(include_state=True)) for brain in cpu_brains]
        observations = [[0.3, -0.2, 0.8, 0.1, 0.0], [-0.4, 0.9, 0.2, 0.0, 0.5]]
        for brain, observation in zip(cpu_brains, observations):
            brain.forward(observation)
        runtime.forward_many(torch_brains, observations)

        params = [
            {
                "action_index": 1,
                "valence": 0.8,
                "energy_delta": 0.4,
                "learning_rate": 0.16,
                "plasticity": 0.75,
                "prediction_weight": 0.60,
                "outcome_targets": {"damage": 0.1, "reproduction": 0.0, "social": 0.2, "tool": 1.0, "hazard": 0.1},
            },
            {
                "action_index": 2,
                "valence": -0.5,
                "energy_delta": -0.3,
                "learning_rate": 0.12,
                "plasticity": 0.90,
                "prediction_weight": 0.80,
                "outcome_targets": {"damage": 0.4, "reproduction": 0.0, "social": -0.1, "tool": 0.0, "hazard": 0.5},
            },
        ]
        expected_errors = [brain.learn(**param) for brain, param in zip(cpu_brains, params)]
        actual_errors = runtime.learn_many(
            [
                BrainLearningCase(brain=brain, **param)
                for brain, param in zip(torch_brains, params)
            ]
        )

        for expected_value, actual_value in zip(expected_errors, actual_errors):
            self.assertAlmostEqual(expected_value, actual_value, places=5)
        for cpu_brain, torch_brain in zip(cpu_brains, torch_brains):
            for expected_value, actual_value in zip(cpu_brain.weights_out, torch_brain.weights_out):
                self.assertAlmostEqual(expected_value, actual_value, places=5)
            for expected_value, actual_value in zip(cpu_brain.prediction_weights, torch_brain.prediction_weights):
                self.assertAlmostEqual(expected_value, actual_value, places=5)
            self.assertEqual(set(torch_brain.last_prediction_errors), set(PREDICTION_HEADS))

    def test_torch_backend_can_run_small_simulation(self) -> None:
        self._torch_runtime_or_skip()
        tmp = tempfile.TemporaryDirectory()
        config = RunConfig(
            seed=909,
            profile="test",
            max_ticks=3,
            max_wall_seconds=0,
            places=4,
            initial_plants=2,
            initial_fungi=1,
            initial_agents=2,
            max_population=20,
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
    """Information-as-attention: brains learn during life what to attend to.
    Total fidelity is bounded; what isn't attended to gets noise. The mechanism
    must be neuroplastic (lifetime learning), inheritable (clone with mutation),
    backward-compatible (legacy checkpoints work), and not require marks/signals
    (transfer-clean to environments without writing)."""

    def test_default_brain_has_attention_head(self) -> None:
        from microcosmic_god.brain import TinyBrain

        brain = TinyBrain.random(Random(11), input_size=6, hidden_size=4, output_size=3)
        self.assertEqual(len(brain.attention_weights), 6 * 4)
        self.assertEqual(len(brain.attention_bias), 6)
        self.assertTrue(brain._has_attention())

    def test_brain_can_be_constructed_without_attention(self) -> None:
        from microcosmic_god.brain import TinyBrain

        brain = TinyBrain.random(Random(11), input_size=6, hidden_size=4, output_size=3, with_attention=False)
        self.assertEqual(brain.attention_weights, [])
        self.assertEqual(brain.attention_bias, [])
        self.assertFalse(brain._has_attention())

    def test_attention_total_fidelity_bounded_by_budget(self) -> None:
        from microcosmic_god.brain import TinyBrain, ATTENTION_BUDGET_FRACTION

        brain = TinyBrain.random(Random(11), input_size=8, hidden_size=4, output_size=3)
        # Saturate attention bias to push raw attention well above budget.
        brain.attention_bias = [10.0 for _ in range(brain.input_size)]
        inputs = [0.5] * brain.input_size
        brain.forward(inputs)
        budget = brain.input_size * ATTENTION_BUDGET_FRACTION
        # Floating-point slack is acceptable; the bound should hold to ~1e-6.
        self.assertLessEqual(sum(brain.last_attention), budget + 1e-6)

    def test_attention_passthrough_when_brain_has_no_attention(self) -> None:
        from microcosmic_god.brain import TinyBrain

        brain = TinyBrain.random(Random(11), input_size=5, hidden_size=4, output_size=3, with_attention=False)
        inputs = [0.8, -0.2, 0.5, 0.0, 0.3]
        brain.forward(inputs)
        # Without attention, last_inputs should equal inputs exactly (no noise injected).
        for expected, actual in zip(inputs, brain.last_inputs):
            self.assertAlmostEqual(expected, actual, places=10)
        # last_attention reports uniform 1.0 (full fidelity) for the no-attention path.
        self.assertEqual(brain.last_attention, [1.0] * 5)

    def test_attention_weights_change_with_surprise_and_valence(self) -> None:
        from microcosmic_god.brain import TinyBrain

        brain = TinyBrain.random(Random(11), input_size=6, hidden_size=5, output_size=3)
        before = list(brain.attention_weights)
        # Drive several iterations of forward + learn with strong surprise/valence.
        for _ in range(10):
            brain.forward([0.7, -0.4, 0.6, 0.1, -0.5, 0.3])
            brain.learn(
                action_index=2,
                valence=1.5,
                energy_delta=0.8,
                learning_rate=0.20,
                plasticity=0.95,
                prediction_weight=0.85,
            )
        after = brain.attention_weights
        self.assertNotEqual(before, after)
        # And the attention bias should also have shifted on at least one feature.
        self.assertTrue(any(abs(b) > 1e-6 for b in brain.attention_bias))

    def test_attention_serializes_round_trip(self) -> None:
        from microcosmic_god.brain import TinyBrain

        brain = TinyBrain.random(Random(11), input_size=6, hidden_size=4, output_size=3)
        # Modify a couple of attention weights to ensure they're persisted.
        brain.attention_weights[0] = 0.5
        brain.attention_weights[5] = -0.3
        brain.attention_bias[2] = 1.2
        data = brain.to_dict(include_state=True)
        restored = TinyBrain.from_dict(data)
        self.assertEqual(len(brain.attention_weights), len(restored.attention_weights))
        for original, recovered in zip(brain.attention_weights, restored.attention_weights):
            self.assertAlmostEqual(original, recovered, places=6)
        for original, recovered in zip(brain.attention_bias, restored.attention_bias):
            self.assertAlmostEqual(original, recovered, places=6)

    def test_legacy_checkpoint_without_attention_loads_cleanly(self) -> None:
        from microcosmic_god.brain import TinyBrain

        # Construct a checkpoint dict that predates attention (no fields).
        legacy_brain = TinyBrain.random(Random(11), input_size=5, hidden_size=3, output_size=2, with_attention=False)
        data = legacy_brain.to_dict(include_state=True)
        data.pop("attention_weights", None)
        data.pop("attention_bias", None)
        data.pop("last_attention", None)
        restored = TinyBrain.from_dict(data)
        self.assertFalse(restored._has_attention())
        # And the brain still produces forward outputs.
        outputs = restored.forward([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(len(outputs), 2)

    def test_clone_propagates_attention_with_mutation(self) -> None:
        from microcosmic_god.brain import TinyBrain

        rng = Random(11)
        parent = TinyBrain.random(rng, input_size=5, hidden_size=3, output_size=2)
        child = parent.clone_for_offspring(Random(12), mutation_scale=0.05)
        self.assertEqual(len(child.attention_weights), len(parent.attention_weights))
        self.assertEqual(len(child.attention_bias), len(parent.attention_bias))
        # Mutation should have nudged at least some values.
        self.assertNotEqual(parent.attention_weights, child.attention_weights)


class TexturedHarshnessTests(unittest.TestCase):
    """Causal challenges should pick up physics-conditional prep steps so the
    same global rule produces different sequences in different physics regimes,
    selecting against rote memorization."""

    BASELINE_PHYSICS = {
        "temperature": 0.5,
        "fluid_level": 0.1,
        "pressure": 0.1,
        "abrasion": 0.1,
        "current_exposure": 0.1,
        "salinity": 0.1,
    }
    BASELINE_OBSTACLES = {"water": 0.1, "thorn": 0.1, "height": 0.1, "heat": 0.1}
    BASELINE_RESOURCES = {"biological_storage": 30.0}

    def _make(self, **overrides) -> "CausalChallenge | None":
        from microcosmic_god.world import World

        physics = dict(self.BASELINE_PHYSICS)
        obstacles = dict(self.BASELINE_OBSTACLES)
        resources = dict(self.BASELINE_RESOURCES)
        physics.update(overrides.pop("physics", {}))
        obstacles.update(overrides.pop("obstacles", {}))
        resources.update(overrides.pop("resources", {}))
        params = dict(
            locked_chemical=40.0,
            water=0.4,
            sun=0.3,
            geo=0.3,
            mineral=0.7,
            volatility=0.05,
            obstacles=obstacles,
            physics=physics,
        )
        params.update(overrides)
        # Use a fixed RNG seed so the candidate scoring is deterministic.
        return World._make_causal_challenge(Random(7), resources, **params)

    def test_temperate_dry_place_skips_prep(self) -> None:
        challenge = self._make()
        self.assertIsNotNone(challenge)
        # In a temperate, dry, low-pressure place the sequence should not be
        # prefixed with concentrate_heat / contain / bind.
        self.assertNotIn(challenge.sequence[0], {"concentrate_heat", "contain", "bind"})

    def test_cold_place_prepends_concentrate_heat(self) -> None:
        challenge = self._make(physics={"temperature": 0.10})
        self.assertIsNotNone(challenge)
        self.assertEqual(challenge.sequence[0], "concentrate_heat")

    def test_flooded_place_prepends_contain(self) -> None:
        challenge = self._make(
            physics={"fluid_level": 0.55},
            obstacles={"water": 0.50},
        )
        self.assertIsNotNone(challenge)
        # contain may collide with the base sequence (contain, filter); in that
        # case the dedup keeps the sequence unchanged. Either contain is in
        # position 0 or the original sequence already starts with it.
        self.assertEqual(challenge.sequence[0], "contain")

    def test_high_pressure_place_prepends_contain(self) -> None:
        challenge = self._make(physics={"pressure": 0.55})
        self.assertIsNotNone(challenge)
        self.assertEqual(challenge.sequence[0], "contain")

    def test_unstable_place_prepends_bind(self) -> None:
        challenge = self._make(physics={"abrasion": 0.45})
        self.assertIsNotNone(challenge)
        self.assertEqual(challenge.sequence[0], "bind")

    def test_cold_and_flooded_stacks_two_prep_steps(self) -> None:
        challenge = self._make(
            physics={"temperature": 0.10, "fluid_level": 0.55},
            obstacles={"water": 0.50},
        )
        self.assertIsNotNone(challenge)
        self.assertEqual(challenge.sequence[0], "concentrate_heat")
        self.assertEqual(challenge.sequence[1], "contain")

    def test_prep_step_not_duplicated_when_base_sequence_already_contains_it(self) -> None:
        # A water-flow place whose base sequence is (contain, filter) and which
        # is also high-pressure: prep "contain" should be dropped since it's
        # already in the base sequence.
        challenge = self._make(
            water=0.95,
            physics={"pressure": 0.55, "current_exposure": 0.45},
        )
        self.assertIsNotNone(challenge)
        self.assertEqual(challenge.sequence.count("contain"), 1)


if __name__ == "__main__":
    unittest.main()
