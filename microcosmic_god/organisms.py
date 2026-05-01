from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from .brain import TinyBrain
from .energy import AFFORDANCES, STRUCTURE_CAPABILITIES, Artifact
from .genome import Genome

SIGNAL_VALUE_SIZE = 8
RECENT_TRACE_LABELS = (
    "action",
    "energy_delta",
    "health_delta",
    "damage",
    "prediction_error",
    "reproduction",
    "social",
    "tool",
)
EVENT_MEMORY_LABELS = (
    "energy_gain",
    "energy_loss",
    "health_gain",
    "damage",
    "reproduction",
    "social",
    "tool",
    "surprise",
)
RECENT_TRACE_SIZE = len(RECENT_TRACE_LABELS)
EVENT_MEMORY_SIZE = len(EVENT_MEMORY_LABELS)
OBSERVATION_SIZE = 42 + RECENT_TRACE_SIZE + EVENT_MEMORY_SIZE + SIGNAL_VALUE_SIZE

ACTIONS = (
    "rest",
    "move",
    "eat",
    "absorb_radiant",
    "forage",
    "pickup",
    "craft",
    "build",
    "use_tool",
    "attack",
    "signal",
    "mark",
    "mate",
    "asexual_reproduce",
    "observe",
)

ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}


def _clip(value: float, low: float = -1.0, high: float = 1.5) -> float:
    return max(low, min(high, float(value)))


@dataclass(slots=True)
class Organism:
    id: int
    kind: str
    genome: Genome
    location: int
    energy: float
    health: float = 1.0
    age: int = 0
    generation: int = 0
    parent_ids: tuple[int, ...] = ()
    brain: TinyBrain | None = None
    brain_template: TinyBrain | None = None
    inventory: dict[str, int] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    tool_skill: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in (*AFFORDANCES, *STRUCTURE_CAPABILITIES, "build")})
    signal_values: list[float] = field(default_factory=lambda: [0.0 for _ in range(SIGNAL_VALUE_SIZE)])
    place_memory: dict[int, float] = field(default_factory=dict)
    event_memory: list[float] = field(default_factory=lambda: [0.0 for _ in range(EVENT_MEMORY_SIZE)])
    alive: bool = True
    last_action: str = "rest"
    last_valence: float = 0.0
    last_energy_delta: float = 0.0
    recent_action_index: int = 0
    recent_health_delta: float = 0.0
    recent_damage: float = 0.0
    recent_prediction_error: float = 0.0
    recent_reproduction_feedback: float = 0.0
    recent_social_feedback: float = 0.0
    recent_tool_feedback: float = 0.0
    mate_intent_until: int = -1
    courtship_token: int = 0
    successful_tools: int = 0
    offspring_count: int = 0

    @property
    def neural(self) -> bool:
        return self.brain is not None

    def hidden_size(self) -> int:
        return max(0, int(round(self.genome.neural_budget)))

    def storage_limit(self) -> float:
        return 24.0 + self.genome.storage_capacity * 95.0 + self.genome.developmental_complexity * 25.0

    def inventory_limit(self) -> int:
        return max(0, int(1 + self.genome.manipulator * 6.0 + self.genome.developmental_complexity * 3.0))

    def inventory_count(self) -> int:
        return sum(max(0, qty) for qty in self.inventory.values())

    def artifact_limit(self) -> int:
        return max(0, int(1 + self.genome.manipulator * 4.0 + self.genome.developmental_complexity * 2.0))

    def metabolic_cost(self) -> float:
        base = 0.018
        body = (
            self.genome.mobility * 0.030
            + self.genome.manipulator * 0.020
            + self.genome.armor * 0.015
            + self.genome.sensor_range * 0.010
            + self.genome.developmental_complexity * 0.020
        )
        neural = (
            self.genome.neural_budget * 0.0045
            + self.genome.memory_budget * 0.0028
            + self.genome.prediction_weight * 0.018
            + self.genome.plasticity_rate * 0.010
        )
        if self.kind in {"plant", "fungus"}:
            base *= 0.55
            body *= 0.35
        return base + body + neural

    def adult(self) -> bool:
        return self.age >= 25 and self.health > 0.35

    def asexual_energy_threshold(self) -> float:
        return 22.0 + self.genome.asexual_threshold * 45.0 + self.genome.complexity() * 7.0

    def sexual_energy_threshold(self) -> float:
        return 26.0 + self.genome.sexual_threshold * 55.0 + self.genome.complexity() * 8.0

    def choose_signal_token(self) -> int:
        if self.brain and self.brain.last_outputs:
            return int(max(range(min(8, len(self.brain.last_outputs))), key=lambda i: self.brain.last_outputs[i])) % 8
        return (self.id + self.age + int(self.energy)) % 8

    def learn_signal_value(self, token: int, valence: float) -> None:
        if 0 <= token < len(self.signal_values):
            self.signal_values[token] = self.signal_values[token] * 0.96 + valence * 0.04

    def recent_trace(self) -> list[float]:
        return [
            self.recent_action_index / max(1.0, float(len(ACTIONS) - 1)),
            _clip(self.last_energy_delta / 10.0),
            _clip(self.recent_health_delta * 4.0),
            _clip(self.recent_damage * 4.0, 0.0, 1.5),
            _clip(self.recent_prediction_error),
            _clip(self.recent_reproduction_feedback, 0.0, 1.5),
            _clip(self.recent_social_feedback),
            _clip(self.recent_tool_feedback, 0.0, 1.5),
        ]

    def record_action_result(
        self,
        action_index: int,
        energy_delta: float,
        health_delta: float,
        damage: float,
        prediction_error: float,
        reproduction_feedback: float,
        social_feedback: float,
        tool_feedback: float,
    ) -> None:
        self.recent_action_index = max(0, min(len(ACTIONS) - 1, int(action_index)))
        self.last_action = ACTIONS[self.recent_action_index]
        self.last_energy_delta = energy_delta
        self.recent_health_delta = health_delta
        self.recent_damage = damage
        self.recent_prediction_error = prediction_error
        self.recent_reproduction_feedback = reproduction_feedback
        self.recent_social_feedback = social_feedback
        self.recent_tool_feedback = tool_feedback
        self._write_event_memory(
            energy_delta=energy_delta,
            health_delta=health_delta,
            damage=damage,
            prediction_error=prediction_error,
            reproduction_feedback=reproduction_feedback,
            social_feedback=social_feedback,
            tool_feedback=tool_feedback,
        )

    def _write_event_memory(
        self,
        energy_delta: float,
        health_delta: float,
        damage: float,
        prediction_error: float,
        reproduction_feedback: float,
        social_feedback: float,
        tool_feedback: float,
    ) -> None:
        if len(self.event_memory) != EVENT_MEMORY_SIZE:
            self.event_memory = [0.0 for _ in range(EVENT_MEMORY_SIZE)]
        memory_gate = _clip(self.genome.memory_budget / 12.0, 0.0, 1.0)
        decay = 0.86 + memory_gate * 0.10
        write = (0.03 + memory_gate * 0.12) * (0.75 + min(1.0, self.genome.plasticity_rate * 2.5) * 0.25)
        event = [
            _clip(max(0.0, energy_delta) / 10.0, 0.0, 1.5),
            _clip(max(0.0, -energy_delta) / 10.0, 0.0, 1.5),
            _clip(max(0.0, health_delta) * 4.0, 0.0, 1.5),
            _clip(damage * 4.0, 0.0, 1.5),
            _clip(reproduction_feedback, 0.0, 1.5),
            _clip(social_feedback),
            _clip(tool_feedback, 0.0, 1.5),
            _clip(abs(prediction_error), 0.0, 1.5),
        ]
        self.event_memory = [_clip(old * decay + value * write) for old, value in zip(self.event_memory, event)]

    def repair_or_decay(self) -> None:
        if self.energy > self.storage_limit():
            self.energy = self.storage_limit()
        if self.energy > self.metabolic_cost() * 20.0 and self.health < 1.0:
            repair = min(1.0 - self.health, 0.003 + self.genome.storage_capacity * 0.004)
            self.health += repair
            self.energy -= repair * 5.0

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "location": self.location,
            "age": self.age,
            "generation": self.generation,
            "energy": round(self.energy, 4),
            "health": round(self.health, 4),
            "neural": self.neural,
            "offspring_count": self.offspring_count,
            "successful_tools": self.successful_tools,
            "last_action": self.last_action,
            "last_valence": round(self.last_valence, 4),
            "last_energy_delta": round(self.last_energy_delta, 4),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "complexity": round(self.genome.complexity(), 4),
            "parents": list(self.parent_ids),
        }

    def cognitive_snapshot(self) -> dict[str, Any]:
        place_memory = sorted(self.place_memory.items(), key=lambda item: item[1], reverse=True)[:8]
        return {
            "last_action": self.last_action,
            "last_valence": round(self.last_valence, 6),
            "recent_trace": {label: round(value, 6) for label, value in zip(RECENT_TRACE_LABELS, self.recent_trace())},
            "event_memory": {label: round(value, 6) for label, value in zip(EVENT_MEMORY_LABELS, self.event_memory)},
            "signal_values": [round(value, 6) for value in self.signal_values],
            "place_memory": [{"place_id": place_id, "value": round(value, 6)} for place_id, value in place_memory],
        }


def make_brain_for_genome(rng: Random, genome: Genome) -> tuple[TinyBrain | None, TinyBrain | None]:
    hidden = int(round(genome.neural_budget))
    if hidden < 2:
        return None, None
    template = TinyBrain.random(rng, OBSERVATION_SIZE, hidden, len(ACTIONS))
    brain = TinyBrain.from_dict(template.to_dict(include_state=False))
    return brain, template


def organism_from_genome(
    rng: Random,
    id_: int,
    kind: str,
    genome: Genome,
    location: int,
    energy: float,
    generation: int = 0,
    parent_ids: tuple[int, ...] = (),
    brain_template: TinyBrain | None = None,
) -> Organism:
    brain: TinyBrain | None = None
    template: TinyBrain | None = None
    if brain_template is not None:
        template = brain_template
        brain = TinyBrain.from_dict(template.to_dict(include_state=False))
    elif genome.neural_budget >= 2.0 and kind == "agent":
        brain, template = make_brain_for_genome(rng, genome)
    return Organism(
        id=id_,
        kind=kind,
        genome=genome,
        location=location,
        energy=energy,
        generation=generation,
        parent_ids=parent_ids,
        brain=brain,
        brain_template=template,
    )
