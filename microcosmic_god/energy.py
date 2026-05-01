from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ENERGY_KINDS = (
    "radiant",
    "chemical",
    "biological_storage",
    "thermal",
    "mechanical",
    "electrical",
    "high_density",
)


def blank_energy(value: float = 0.0) -> dict[str, float]:
    return {kind: float(value) for kind in ENERGY_KINDS}


@dataclass(frozen=True, slots=True)
class Material:
    name: str
    properties: Mapping[str, float]


@dataclass(slots=True)
class Artifact:
    name: str
    components: dict[str, int]
    properties: dict[str, float]
    capabilities: dict[str, float]
    durability: float
    age: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": dict(self.components),
            "properties": {key: round(value, 6) for key, value in self.properties.items()},
            "capabilities": {key: round(value, 6) for key, value in self.capabilities.items()},
            "durability": round(self.durability, 6),
            "age": self.age,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            name=str(data["name"]),
            components={str(k): int(v) for k, v in data["components"].items()},
            properties={str(k): float(v) for k, v in data["properties"].items()},
            capabilities={str(k): float(v) for k, v in data["capabilities"].items()},
            durability=float(data["durability"]),
            age=int(data.get("age", 0)),
        )


MATERIALS: dict[str, Material] = {
    "branch": Material(
        "branch",
        {
            "hard": 0.30,
            "heavy": 0.18,
            "sharp": 0.05,
            "flexible": 0.40,
            "bindable": 0.20,
            "grippable": 0.85,
            "length": 0.90,
            "combustible": 0.65,
            "buoyant": 0.65,
            "porous": 0.25,
            "density": 0.22,
            "thermal_capacity": 0.28,
        },
    ),
    "stone": Material(
        "stone",
        {
            "hard": 0.92,
            "heavy": 0.82,
            "sharp": 0.28,
            "flexible": 0.00,
            "bindable": 0.05,
            "grippable": 0.30,
            "length": 0.12,
            "thermal_mass": 0.70,
            "density": 0.92,
            "conductive": 0.18,
        },
    ),
    "fiber": Material(
        "fiber",
        {
            "hard": 0.03,
            "heavy": 0.03,
            "sharp": 0.00,
            "flexible": 0.95,
            "bindable": 0.96,
            "grippable": 0.35,
            "length": 0.70,
            "combustible": 0.45,
            "buoyant": 0.42,
            "porous": 0.90,
            "absorbent": 0.72,
            "insulating": 0.48,
            "density": 0.06,
        },
    ),
    "shell": Material(
        "shell",
        {
            "hard": 0.54,
            "heavy": 0.22,
            "sharp": 0.18,
            "flexible": 0.05,
            "bindable": 0.15,
            "container": 0.80,
            "grippable": 0.45,
            "buoyant": 0.36,
            "density": 0.32,
            "thermal_capacity": 0.36,
        },
    ),
    "crystal": Material(
        "crystal",
        {
            "hard": 0.68,
            "heavy": 0.34,
            "sharp": 0.42,
            "reflective": 0.82,
            "conductive": 0.55,
            "grippable": 0.25,
            "brittle": 0.75,
            "density": 0.58,
            "thermal_capacity": 0.22,
        },
    ),
    "resin": Material(
        "resin",
        {
            "hard": 0.10,
            "heavy": 0.12,
            "flexible": 0.30,
            "bindable": 0.75,
            "sticky": 0.95,
            "combustible": 0.82,
            "grippable": 0.10,
            "sealant": 0.82,
            "insulating": 0.34,
            "buoyant": 0.28,
            "density": 0.18,
        },
    ),
    "bone": Material(
        "bone",
        {
            "hard": 0.62,
            "heavy": 0.28,
            "sharp": 0.45,
            "length": 0.50,
            "grippable": 0.55,
            "brittle": 0.25,
            "porous": 0.22,
            "density": 0.46,
            "thermal_capacity": 0.18,
        },
    ),
}

AFFORDANCES = ("crack", "cut", "bind", "contain", "concentrate_heat", "conduct", "lever", "filter")
ARTIFACT_CAPABILITIES = (
    "crack",
    "cut",
    "bind",
    "contain",
    "concentrate_heat",
    "conduct",
    "lever",
    "traverse",
    "insulate",
    "energy_storage",
    "filter",
    "float",
    "anchor",
)


def inventory_properties(inventory: Mapping[str, int]) -> dict[str, float]:
    props: dict[str, float] = {}
    count = 0
    for name, qty in inventory.items():
        if qty <= 0 or name not in MATERIALS:
            continue
        count += qty
        for key, value in MATERIALS[name].properties.items():
            props[key] = props.get(key, 0.0) + value * qty
    if count <= 0:
        return props
    return {key: value / count for key, value in props.items()}


def component_properties(components: Mapping[str, int]) -> dict[str, float]:
    return inventory_properties(components)


def derive_affordances(inventory: Mapping[str, int]) -> dict[str, float]:
    props = inventory_properties(inventory)
    return derive_affordances_from_properties(props)


def derive_affordances_from_properties(props: Mapping[str, float]) -> dict[str, float]:
    if not props:
        return {name: 0.0 for name in AFFORDANCES}
    hard = props.get("hard", 0.0)
    heavy = props.get("heavy", 0.0)
    sharp = props.get("sharp", 0.0)
    flexible = props.get("flexible", 0.0)
    bindable = props.get("bindable", 0.0)
    container = props.get("container", 0.0)
    reflective = props.get("reflective", 0.0)
    conductive = props.get("conductive", 0.0)
    porous = props.get("porous", 0.0)
    absorbent = props.get("absorbent", 0.0)
    length = props.get("length", 0.0)
    grippable = props.get("grippable", 0.0)
    return {
        "crack": min(1.0, hard * 0.55 + heavy * 0.35 + grippable * 0.10),
        "cut": min(1.0, sharp * 0.70 + hard * 0.20 + grippable * 0.10),
        "bind": min(1.0, flexible * 0.45 + bindable * 0.45 + length * 0.10),
        "contain": min(1.0, container * 0.85 + hard * 0.05 + bindable * 0.10),
        "concentrate_heat": min(1.0, reflective * 0.70 + hard * 0.15 + grippable * 0.15),
        "conduct": min(1.0, conductive * 0.85 + hard * 0.05 + grippable * 0.10),
        "lever": min(1.0, length * 0.55 + hard * 0.25 + grippable * 0.20),
        "filter": min(1.0, porous * 0.55 + absorbent * 0.20 + flexible * 0.10 + bindable * 0.10 + container * 0.05),
    }


def derive_artifact_capabilities(properties: Mapping[str, float]) -> dict[str, float]:
    affordances = derive_affordances_from_properties(properties)
    hard = properties.get("hard", 0.0)
    heavy = properties.get("heavy", 0.0)
    sharp = properties.get("sharp", 0.0)
    flexible = properties.get("flexible", 0.0)
    bindable = properties.get("bindable", 0.0)
    container = properties.get("container", 0.0)
    reflective = properties.get("reflective", 0.0)
    conductive = properties.get("conductive", 0.0)
    length = properties.get("length", 0.0)
    thermal_mass = properties.get("thermal_mass", 0.0)
    thermal_capacity = max(thermal_mass, properties.get("thermal_capacity", 0.0))
    sticky = properties.get("sticky", 0.0)
    porous = properties.get("porous", 0.0)
    absorbent = properties.get("absorbent", 0.0)
    buoyant = properties.get("buoyant", 0.0)
    density = properties.get("density", heavy)
    insulating = properties.get("insulating", 0.0)
    sealant = properties.get("sealant", 0.0)
    capabilities = dict(affordances)
    capabilities["traverse"] = min(1.0, length * 0.35 + hard * 0.20 + flexible * 0.15 + bindable * 0.15 + sticky * 0.15)
    capabilities["insulate"] = min(1.0, flexible * 0.20 + container * 0.18 + thermal_capacity * 0.20 + hard * 0.08 + sticky * 0.08 + insulating * 0.35 + porous * 0.08)
    capabilities["energy_storage"] = min(1.0, container * 0.35 + conductive * 0.18 + thermal_capacity * 0.28 + hard * 0.10 + sealant * 0.12)
    capabilities["filter"] = min(1.0, capabilities["filter"] + porous * 0.25 + absorbent * 0.12 + container * 0.08)
    capabilities["float"] = max(0.0, min(1.0, buoyant * 0.62 + container * 0.18 + flexible * 0.10 + sealant * 0.10 - density * 0.24))
    capabilities["anchor"] = min(1.0, density * 0.42 + heavy * 0.32 + hard * 0.20 + length * 0.06)
    capabilities["contain"] = min(1.0, capabilities["contain"] + sealant * 0.16 + absorbent * 0.04)
    capabilities["traverse"] = min(1.0, capabilities["traverse"] + capabilities["float"] * 0.20 + capabilities["anchor"] * 0.08)
    capabilities["concentrate_heat"] = min(1.0, capabilities["concentrate_heat"] + reflective * hard * 0.25)
    capabilities["conduct"] = min(1.0, capabilities["conduct"] + conductive * length * 0.25)
    capabilities["cut"] = min(1.0, capabilities["cut"] + sharp * hard * 0.15)
    capabilities["crack"] = min(1.0, capabilities["crack"] + hard * heavy * 0.15)
    return capabilities


def build_artifact(components: Mapping[str, int]) -> Artifact:
    properties = component_properties(components)
    capabilities = derive_artifact_capabilities(properties)
    ranked = sorted(capabilities.items(), key=lambda item: item[1], reverse=True)
    dominant = [name for name, value in ranked[:2] if value > 0.25]
    name = "composite_" + ("_".join(dominant) if dominant else "object")
    durability = 35.0 + properties.get("hard", 0.0) * 70.0 + properties.get("flexible", 0.0) * 25.0 + properties.get("bindable", 0.0) * 35.0
    return Artifact(
        name=name,
        components=dict(components),
        properties=properties,
        capabilities=capabilities,
        durability=durability,
    )


def artifact_affordances(artifacts: list[Artifact]) -> dict[str, float]:
    potentials = {name: 0.0 for name in AFFORDANCES}
    for artifact in artifacts:
        durability_factor = max(0.0, min(1.0, artifact.durability / 100.0))
        for name in AFFORDANCES:
            potentials[name] = max(potentials[name], artifact.capabilities.get(name, 0.0) * durability_factor)
    return potentials


def artifact_capability(artifacts: list[Artifact], capability: str) -> float:
    best = 0.0
    for artifact in artifacts:
        durability_factor = max(0.0, min(1.0, artifact.durability / 100.0))
        best = max(best, artifact.capabilities.get(capability, 0.0) * durability_factor)
    return best


def best_affordance(inventory: Mapping[str, int], skills: Mapping[str, float], artifacts: list[Artifact] | None = None) -> tuple[str, float]:
    potentials = derive_affordances(inventory)
    if artifacts:
        artifact_potentials = artifact_affordances(artifacts)
        for name, value in artifact_potentials.items():
            potentials[name] = max(potentials[name], value)
    best_name = "crack"
    best_score = 0.0
    for name, potential in potentials.items():
        score = potential * (0.65 + 0.35 * skills.get(name, 0.0))
        if score > best_score:
            best_name = name
            best_score = score
    return best_name, best_score
