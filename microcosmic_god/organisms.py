from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from .brain import TinyBrain
from .energy import AFFORDANCES
from .genome import Genome

OBSERVATION_SIZE = 32

ACTIONS = (
    "rest",
    "move",
    "eat",
    "absorb_radiant",
    "forage",
    "pickup",
    "use_tool",
    "attack",
    "signal",
    "mark",
    "mate",
    "asexual_reproduce",
    "observe",
)

ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}


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
    tool_skill: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in AFFORDANCES})
    signal_values: list[float] = field(default_factory=lambda: [0.0 for _ in range(8)])
    place_memory: dict[int, float] = field(default_factory=dict)
    alive: bool = True
    last_action: str = "rest"
    last_valence: float = 0.0
    last_energy_delta: float = 0.0
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
            "complexity": round(self.genome.complexity(), 4),
            "parents": list(self.parent_ids),
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
