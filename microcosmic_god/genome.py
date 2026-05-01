from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
from typing import Any


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mut_float(rng: Random, value: float, rate: float, scale: float, low: float = 0.0, high: float = 1.0) -> float:
    if rng.random() < rate:
        value += rng.gauss(0.0, scale)
    if rng.random() < rate * 0.06:
        value = rng.uniform(low, high)
    return clamp(value, low, high)


@dataclass(slots=True)
class Genome:
    radiant_metabolism: float
    chemical_metabolism: float
    thermal_tolerance: float
    mechanical_use: float
    electrical_use: float
    storage_capacity: float
    photosynthesis_surface: float
    digestion: float
    mobility: float
    manipulator: float
    armor: float
    sensor_range: float
    neural_budget: float
    memory_budget: float
    prediction_weight: float
    plasticity_rate: float
    learning_rate: float
    signal_strength: float
    mate_selectivity: float
    offspring_investment: float
    asexual_threshold: float
    sexual_threshold: float
    developmental_complexity: float
    mutation_rate: float
    valence_energy: float
    valence_health: float
    valence_damage: float
    valence_reproduction: float
    valence_social: float

    @classmethod
    def plant(cls, rng: Random) -> "Genome":
        return cls(
            radiant_metabolism=rng.uniform(0.60, 0.95),
            chemical_metabolism=rng.uniform(0.02, 0.20),
            thermal_tolerance=rng.uniform(0.35, 0.70),
            mechanical_use=rng.uniform(0.00, 0.05),
            electrical_use=rng.uniform(0.00, 0.02),
            storage_capacity=rng.uniform(0.35, 0.75),
            photosynthesis_surface=rng.uniform(0.55, 1.00),
            digestion=rng.uniform(0.00, 0.12),
            mobility=rng.uniform(0.00, 0.04),
            manipulator=rng.uniform(0.00, 0.03),
            armor=rng.uniform(0.05, 0.35),
            sensor_range=rng.uniform(0.00, 0.10),
            neural_budget=0.0,
            memory_budget=0.0,
            prediction_weight=0.0,
            plasticity_rate=0.0,
            learning_rate=0.0,
            signal_strength=0.0,
            mate_selectivity=0.0,
            offspring_investment=rng.uniform(0.15, 0.45),
            asexual_threshold=rng.uniform(0.25, 0.45),
            sexual_threshold=rng.uniform(0.50, 0.85),
            developmental_complexity=rng.uniform(0.10, 0.35),
            mutation_rate=rng.uniform(0.015, 0.055),
            valence_energy=rng.uniform(0.15, 0.55),
            valence_health=rng.uniform(0.10, 0.35),
            valence_damage=rng.uniform(0.25, 0.70),
            valence_reproduction=rng.uniform(0.00, 0.25),
            valence_social=rng.uniform(0.00, 0.08),
        )

    @classmethod
    def fungus(cls, rng: Random) -> "Genome":
        return cls(
            radiant_metabolism=rng.uniform(0.00, 0.15),
            chemical_metabolism=rng.uniform(0.45, 0.95),
            thermal_tolerance=rng.uniform(0.25, 0.85),
            mechanical_use=rng.uniform(0.00, 0.05),
            electrical_use=rng.uniform(0.00, 0.02),
            storage_capacity=rng.uniform(0.30, 0.70),
            photosynthesis_surface=rng.uniform(0.00, 0.15),
            digestion=rng.uniform(0.45, 0.90),
            mobility=rng.uniform(0.00, 0.06),
            manipulator=rng.uniform(0.00, 0.02),
            armor=rng.uniform(0.00, 0.18),
            sensor_range=rng.uniform(0.00, 0.12),
            neural_budget=0.0,
            memory_budget=0.0,
            prediction_weight=0.0,
            plasticity_rate=0.0,
            learning_rate=0.0,
            signal_strength=0.0,
            mate_selectivity=0.0,
            offspring_investment=rng.uniform(0.12, 0.40),
            asexual_threshold=rng.uniform(0.22, 0.45),
            sexual_threshold=rng.uniform(0.45, 0.80),
            developmental_complexity=rng.uniform(0.10, 0.40),
            mutation_rate=rng.uniform(0.015, 0.065),
            valence_energy=rng.uniform(0.10, 0.45),
            valence_health=rng.uniform(0.05, 0.25),
            valence_damage=rng.uniform(0.15, 0.50),
            valence_reproduction=rng.uniform(0.00, 0.22),
            valence_social=rng.uniform(0.00, 0.08),
        )

    @classmethod
    def neural(cls, rng: Random) -> "Genome":
        return cls(
            radiant_metabolism=rng.uniform(0.00, 0.25),
            chemical_metabolism=rng.uniform(0.35, 0.85),
            thermal_tolerance=rng.uniform(0.30, 0.75),
            mechanical_use=rng.uniform(0.05, 0.45),
            electrical_use=rng.uniform(0.00, 0.10),
            storage_capacity=rng.uniform(0.35, 0.80),
            photosynthesis_surface=rng.uniform(0.00, 0.16),
            digestion=rng.uniform(0.35, 0.90),
            mobility=rng.uniform(0.35, 0.90),
            manipulator=rng.uniform(0.15, 0.75),
            armor=rng.uniform(0.02, 0.45),
            sensor_range=rng.uniform(0.35, 0.90),
            neural_budget=rng.uniform(4.0, 13.0),
            memory_budget=rng.uniform(1.0, 6.0),
            prediction_weight=rng.uniform(0.05, 0.55),
            plasticity_rate=rng.uniform(0.02, 0.35),
            learning_rate=rng.uniform(0.03, 0.22),
            signal_strength=rng.uniform(0.00, 0.55),
            mate_selectivity=rng.uniform(0.10, 0.85),
            offspring_investment=rng.uniform(0.20, 0.70),
            asexual_threshold=rng.uniform(0.35, 0.65),
            sexual_threshold=rng.uniform(0.45, 0.85),
            developmental_complexity=rng.uniform(0.45, 0.95),
            mutation_rate=rng.uniform(0.010, 0.050),
            valence_energy=rng.uniform(0.20, 0.85),
            valence_health=rng.uniform(0.15, 0.65),
            valence_damage=rng.uniform(0.35, 0.95),
            valence_reproduction=rng.uniform(0.00, 0.55),
            valence_social=rng.uniform(0.00, 0.35),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        return cls(**data)

    def copy(self) -> "Genome":
        return Genome.from_dict(self.to_dict())

    def mutate(self, rng: Random, strength: float = 0.08) -> "Genome":
        data = self.to_dict()
        rate = clamp(self.mutation_rate, 0.001, 0.30)
        for key, value in list(data.items()):
            if key in {"neural_budget", "memory_budget"}:
                data[key] = mut_float(rng, value / 32.0, rate, strength, 0.0, 1.0) * 32.0
            else:
                data[key] = mut_float(rng, float(value), rate, strength)
        data["mutation_rate"] = mut_float(rng, data["mutation_rate"], rate, strength * 0.4, 0.001, 0.20)
        return Genome.from_dict(data)

    @staticmethod
    def recombine(rng: Random, a: "Genome", b: "Genome") -> "Genome":
        data_a = a.to_dict()
        data_b = b.to_dict()
        child: dict[str, float] = {}
        for key in data_a:
            va = float(data_a[key])
            vb = float(data_b[key])
            if rng.random() < 0.20:
                value = rng.choice([va, vb])
            else:
                mix = rng.uniform(0.25, 0.75)
                value = va * mix + vb * (1.0 - mix)
            child[key] = value
        genome = Genome.from_dict(child)
        return genome.mutate(rng, strength=0.055)

    def complexity(self) -> float:
        return (
            self.developmental_complexity
            + self.mobility * 0.65
            + self.manipulator * 0.65
            + self.sensor_range * 0.45
            + self.neural_budget / 10.0
            + self.memory_budget / 14.0
            + self.prediction_weight * 0.55
            + self.plasticity_rate * 0.35
            + self.electrical_use * 0.25
        )

    def distance(self, other: "Genome") -> float:
        a = self.to_dict()
        b = other.to_dict()
        total = 0.0
        for key in a:
            scale = 32.0 if key in {"neural_budget", "memory_budget"} else 1.0
            total += abs(float(a[key]) - float(b[key])) / scale
        return total / len(a)

