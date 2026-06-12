from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from random import Random
from typing import Any


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def perturb_float(rng: Random, value: float, rate: float, scale: float, low: float = 0.0, high: float = 1.0) -> float:
    if rng.random() < rate:
        value += rng.gauss(0.0, scale)
    if rng.random() < rate * 0.06:
        value = rng.uniform(low, high)
    return clamp(value, low, high)


NEURAL_BUDGET_MAX = 512.0
# Mutations and combination distances on neural_budget were calibrated for a
# 128-unit operating range. The cap was raised to 512 to allow controllers to grow,
# but mutations should still produce the same absolute step magnitudes, not
# 4x larger ones - otherwise a single perturbation can shove a controller from 8 to 40
# units, which spikes upkeep cost and crashes early lineages.
NEURAL_BUDGET_PERTURBATION_REFERENCE = 128.0
MEMORY_BUDGET_MAX = 48.0
# Episodic memory capacity: number of past hidden-state snapshots a controller can
# store. 0 = no episodic memory (legacy behavior). ParamVector-controlled, mutates
# like other budgets. Upkeep cost is added in organisms.upkeep_cost.
EPISODIC_CAPACITY_MAX = 32.0


# ParamVector field names were neutralized in source, but the on-disk checkpoint keys
# keep their original strings so existing run artifacts and the transfer harness
# round-trip unchanged. Map: source (neutral) name -> on-disk (legacy) key.
_LEGACY_KEYS = {
    "solar_energy_gain": "radiant_metabolism",
    "essence_energy_gain": "chemical_metabolism",
    "solar_capture_area": "photosynthesis_surface",
    "essence_conversion": "digestion",
    "pairing_selectivity": "mate_selectivity",
    "single_parent_threshold": "asexual_threshold",
    "two_parent_threshold": "sexual_threshold",
    "perturbation_rate": "mutation_rate",
    "resilience": "armor",
}
_LEGACY_TO_NEUTRAL = {legacy: neutral for neutral, legacy in _LEGACY_KEYS.items()}


@dataclass(slots=True)
class ParamVector:
    solar_energy_gain: float
    essence_energy_gain: float
    thermal_tolerance: float
    mechanical_use: float
    electrical_use: float
    storage_capacity: float
    aquatic_affinity: float
    salinity_tolerance: float
    desiccation_tolerance: float
    pressure_tolerance: float
    buoyancy: float
    solar_capture_area: float
    essence_conversion: float
    mobility: float
    manipulator: float
    resilience: float
    sensor_range: float
    neural_budget: float
    memory_budget: float
    prediction_weight: float
    plasticity_rate: float
    learning_rate: float
    signal_strength: float
    pairing_selectivity: float
    offspring_investment: float
    single_parent_threshold: float
    two_parent_threshold: float
    developmental_complexity: float
    perturbation_rate: float
    valence_energy: float
    valence_health: float
    valence_damage: float
    valence_reproduction: float
    valence_social: float
    # Episodic memory capacity (number of stored hidden-state snapshots).
    # Optional v2 controller feature; 0 disables. ParamVector-evolvable.
    episodic_capacity: float = 0.0

    @classmethod
    def plant(cls, rng: Random) -> "ParamVector":
        return cls(
            solar_energy_gain=rng.uniform(0.60, 0.95),
            essence_energy_gain=rng.uniform(0.02, 0.20),
            thermal_tolerance=rng.uniform(0.35, 0.70),
            mechanical_use=rng.uniform(0.00, 0.05),
            electrical_use=rng.uniform(0.00, 0.02),
            storage_capacity=rng.uniform(0.35, 0.75),
            aquatic_affinity=rng.uniform(0.05, 0.45),
            salinity_tolerance=rng.uniform(0.05, 0.45),
            desiccation_tolerance=rng.uniform(0.25, 0.90),
            pressure_tolerance=rng.uniform(0.00, 0.35),
            buoyancy=rng.uniform(0.05, 0.45),
            solar_capture_area=rng.uniform(0.55, 1.00),
            essence_conversion=rng.uniform(0.00, 0.12),
            mobility=rng.uniform(0.00, 0.04),
            manipulator=rng.uniform(0.00, 0.03),
            resilience=rng.uniform(0.05, 0.35),
            sensor_range=rng.uniform(0.00, 0.10),
            neural_budget=0.0,
            memory_budget=0.0,
            prediction_weight=0.0,
            plasticity_rate=0.0,
            learning_rate=0.0,
            signal_strength=0.0,
            pairing_selectivity=0.0,
            offspring_investment=rng.uniform(0.15, 0.45),
            single_parent_threshold=rng.uniform(0.25, 0.45),
            two_parent_threshold=rng.uniform(0.50, 0.85),
            developmental_complexity=rng.uniform(0.10, 0.35),
            perturbation_rate=rng.uniform(0.015, 0.055),
            valence_energy=rng.uniform(0.15, 0.55),
            valence_health=rng.uniform(0.10, 0.35),
            valence_damage=rng.uniform(0.25, 0.70),
            valence_reproduction=rng.uniform(0.00, 0.25),
            valence_social=rng.uniform(0.00, 0.08),
        )

    @classmethod
    def fungus(cls, rng: Random) -> "ParamVector":
        return cls(
            solar_energy_gain=rng.uniform(0.00, 0.15),
            essence_energy_gain=rng.uniform(0.45, 0.95),
            thermal_tolerance=rng.uniform(0.25, 0.85),
            mechanical_use=rng.uniform(0.00, 0.05),
            electrical_use=rng.uniform(0.00, 0.02),
            storage_capacity=rng.uniform(0.30, 0.70),
            aquatic_affinity=rng.uniform(0.15, 0.65),
            salinity_tolerance=rng.uniform(0.05, 0.55),
            desiccation_tolerance=rng.uniform(0.05, 0.65),
            pressure_tolerance=rng.uniform(0.05, 0.55),
            buoyancy=rng.uniform(0.10, 0.60),
            solar_capture_area=rng.uniform(0.00, 0.15),
            essence_conversion=rng.uniform(0.45, 0.90),
            mobility=rng.uniform(0.00, 0.06),
            manipulator=rng.uniform(0.00, 0.02),
            resilience=rng.uniform(0.00, 0.18),
            sensor_range=rng.uniform(0.00, 0.12),
            neural_budget=0.0,
            memory_budget=0.0,
            prediction_weight=0.0,
            plasticity_rate=0.0,
            learning_rate=0.0,
            signal_strength=0.0,
            pairing_selectivity=0.0,
            offspring_investment=rng.uniform(0.12, 0.40),
            single_parent_threshold=rng.uniform(0.22, 0.45),
            two_parent_threshold=rng.uniform(0.45, 0.80),
            developmental_complexity=rng.uniform(0.10, 0.40),
            perturbation_rate=rng.uniform(0.015, 0.065),
            valence_energy=rng.uniform(0.10, 0.45),
            valence_health=rng.uniform(0.05, 0.25),
            valence_damage=rng.uniform(0.15, 0.50),
            valence_reproduction=rng.uniform(0.00, 0.22),
            valence_social=rng.uniform(0.00, 0.08),
        )

    @classmethod
    def neural(cls, rng: Random) -> "ParamVector":
        return cls(
            solar_energy_gain=rng.uniform(0.00, 0.25),
            essence_energy_gain=rng.uniform(0.35, 0.85),
            thermal_tolerance=rng.uniform(0.30, 0.75),
            mechanical_use=rng.uniform(0.05, 0.45),
            electrical_use=rng.uniform(0.00, 0.10),
            storage_capacity=rng.uniform(0.35, 0.80),
            aquatic_affinity=rng.uniform(0.00, 0.55),
            salinity_tolerance=rng.uniform(0.00, 0.50),
            desiccation_tolerance=rng.uniform(0.20, 0.90),
            pressure_tolerance=rng.uniform(0.00, 0.55),
            buoyancy=rng.uniform(0.00, 0.65),
            solar_capture_area=rng.uniform(0.00, 0.16),
            essence_conversion=rng.uniform(0.35, 0.90),
            mobility=rng.uniform(0.35, 0.90),
            manipulator=rng.uniform(0.15, 0.75),
            resilience=rng.uniform(0.02, 0.45),
            sensor_range=rng.uniform(0.35, 0.90),
            neural_budget=rng.uniform(4.0, 13.0),
            memory_budget=rng.uniform(1.0, 6.0),
            prediction_weight=rng.uniform(0.05, 0.55),
            plasticity_rate=rng.uniform(0.02, 0.35),
            learning_rate=rng.uniform(0.03, 0.22),
            signal_strength=rng.uniform(0.00, 0.55),
            pairing_selectivity=rng.uniform(0.10, 0.85),
            offspring_investment=rng.uniform(0.20, 0.70),
            single_parent_threshold=rng.uniform(0.35, 0.65),
            two_parent_threshold=rng.uniform(0.45, 0.85),
            developmental_complexity=rng.uniform(0.45, 0.95),
            perturbation_rate=rng.uniform(0.010, 0.050),
            valence_energy=rng.uniform(0.20, 0.85),
            valence_health=rng.uniform(0.15, 0.65),
            valence_damage=rng.uniform(0.35, 0.95),
            valence_reproduction=rng.uniform(0.00, 0.55),
            valence_social=rng.uniform(0.00, 0.35),
            # Initial neural agents start with modest episodic capacity (0-8 slots).
            # Mutation can grow it up to EPISODIC_CAPACITY_MAX or shrink it to 0.
            # Controllers that find episodic memory useful keep it; controllers that don't
            # pay upkeep cost for nothing and lose to lineages that mutated it away.
            episodic_capacity=rng.uniform(0.0, 8.0),
        )

    def to_dict(self) -> dict[str, Any]:
        # Emit legacy on-disk keys so checkpoints/harness stay byte-compatible.
        return {_LEGACY_KEYS.get(k, k): v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParamVector":
        # Accept both legacy on-disk keys and neutral source names.
        data = {_LEGACY_TO_NEUTRAL.get(k, k): v for k, v in data.items()}
        defaults = {
            "aquatic_affinity": 0.25,
            "salinity_tolerance": 0.20,
            "desiccation_tolerance": 0.55,
            "pressure_tolerance": 0.20,
            "buoyancy": 0.25,
            "episodic_capacity": 0.0,  # backward-compat: legacy genomes had no episodic memory
        }
        for field in fields(cls):
            if field.name not in data and field.name in defaults:
                data[field.name] = defaults[field.name]
        return cls(**data)

    def copy(self) -> "ParamVector":
        return ParamVector.from_dict(self.to_dict())

    def perturb(self, rng: Random, strength: float = 0.08) -> "ParamVector":
        data = self.to_dict()
        rate = clamp(self.perturbation_rate, 0.001, 0.30)
        for key, value in list(data.items()):
            if key == "neural_budget":
                # Normalize against the perturbation reference (128) so the scaled
                # gaussian step has legacy magnitude; the upper bound on the
                # normalized value is MAX / REFERENCE so a rare reset can still
                # land anywhere in the legal range.
                upper = NEURAL_BUDGET_MAX / NEURAL_BUDGET_PERTURBATION_REFERENCE
                data[key] = (
                    perturb_float(
                        rng,
                        value / NEURAL_BUDGET_PERTURBATION_REFERENCE,
                        rate,
                        strength,
                        0.0,
                        upper,
                    )
                    * NEURAL_BUDGET_PERTURBATION_REFERENCE
                )
            elif key == "episodic_capacity":
                data[key] = perturb_float(rng, value / EPISODIC_CAPACITY_MAX, rate, strength, 0.0, 1.0) * EPISODIC_CAPACITY_MAX
            elif key == "memory_budget":
                data[key] = perturb_float(rng, value / MEMORY_BUDGET_MAX, rate, strength, 0.0, 1.0) * MEMORY_BUDGET_MAX
            else:
                data[key] = perturb_float(rng, float(value), rate, strength)
        data["mutation_rate"] = perturb_float(rng, data["mutation_rate"], rate, strength * 0.4, 0.001, 0.20)
        return ParamVector.from_dict(data)

    @staticmethod
    def combine(rng: Random, a: "ParamVector", b: "ParamVector") -> "ParamVector":
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
        params = ParamVector.from_dict(child)
        return params.perturb(rng, strength=0.055)

    def complexity(self) -> float:
        return (
            self.developmental_complexity
            + self.mobility * 0.65
            + abs(self.aquatic_affinity - 0.35) * 0.15
            + self.pressure_tolerance * 0.10
            + self.buoyancy * 0.08
            + self.manipulator * 0.65
            + self.sensor_range * 0.45
            + self.neural_budget / 18.0
            + self.memory_budget / 20.0
            + self.prediction_weight * 0.55
            + self.plasticity_rate * 0.35
            + self.electrical_use * 0.25
        )

    def distance(self, other: "ParamVector") -> float:
        a = self.to_dict()
        b = other.to_dict()
        total = 0.0
        for key in a:
            if key == "neural_budget":
                scale = NEURAL_BUDGET_MAX
            elif key == "memory_budget":
                scale = MEMORY_BUDGET_MAX
            else:
                scale = 1.0
            total += abs(float(a[key]) - float(b[key])) / scale
        return total / len(a)
