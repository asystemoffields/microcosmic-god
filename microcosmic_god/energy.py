from __future__ import annotations

import math
from dataclasses import dataclass, field
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

STRUCTURE_DECAY_CHANNELS = (
    "baseline",
    "mechanical",
    "chemical",
    "biological",
    "thermal",
    "solubility",
    "radiation",
    "fatigue",
)


def blank_energy(value: float = 0.0) -> dict[str, float]:
    return {kind: float(value) for kind in ENERGY_KINDS}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
    inscriptions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": dict(self.components),
            "properties": {key: round(value, 6) for key, value in self.properties.items()},
            "capabilities": {key: round(value, 6) for key, value in self.capabilities.items()},
            "durability": round(self.durability, 6),
            "age": self.age,
            "inscriptions": [dict(inscription) for inscription in self.inscriptions[-8:]],
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
            inscriptions=[dict(item) for item in data.get("inscriptions", [])],
        )


@dataclass(slots=True)
class Structure:
    name: str
    components: dict[str, int]
    properties: dict[str, float]
    capabilities: dict[str, float]
    durability: float
    scale: int
    builder_id: int | None = None
    age: int = 0
    last_decay: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": dict(self.components),
            "properties": {key: round(value, 6) for key, value in self.properties.items()},
            "capabilities": {key: round(value, 6) for key, value in self.capabilities.items()},
            "durability": round(self.durability, 6),
            "scale": self.scale,
            "builder_id": self.builder_id,
            "age": self.age,
            "last_decay": {key: round(value, 6) for key, value in self.last_decay.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Structure":
        return cls(
            name=str(data["name"]),
            components={str(k): int(v) for k, v in data["components"].items()},
            properties={str(k): float(v) for k, v in data["properties"].items()},
            capabilities={str(k): float(v) for k, v in data["capabilities"].items()},
            durability=float(data["durability"]),
            scale=int(data.get("scale", sum(int(v) for v in data["components"].values()))),
            builder_id=None if data.get("builder_id") is None else int(data["builder_id"]),
            age=int(data.get("age", 0)),
            last_decay={str(k): float(v) for k, v in data.get("last_decay", {}).items()},
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
            "oxidizable": 0.24,
            "corrosion_resistance": 0.10,
            "biodegradable": 0.78,
            "water_soluble": 0.08,
            "thermal_stability": 0.34,
            "fatigue_resistance": 0.36,
            "abrasion_resistance": 0.30,
            "uv_sensitivity": 0.44,
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
            "oxidizable": 0.02,
            "corrosion_resistance": 0.88,
            "biodegradable": 0.01,
            "water_soluble": 0.02,
            "thermal_stability": 0.82,
            "fatigue_resistance": 0.80,
            "abrasion_resistance": 0.86,
            "uv_sensitivity": 0.01,
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
            "oxidizable": 0.22,
            "corrosion_resistance": 0.08,
            "biodegradable": 0.72,
            "water_soluble": 0.12,
            "thermal_stability": 0.24,
            "fatigue_resistance": 0.28,
            "abrasion_resistance": 0.18,
            "uv_sensitivity": 0.56,
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
            "oxidizable": 0.05,
            "corrosion_resistance": 0.54,
            "biodegradable": 0.12,
            "water_soluble": 0.10,
            "thermal_stability": 0.52,
            "fatigue_resistance": 0.46,
            "abrasion_resistance": 0.54,
            "uv_sensitivity": 0.08,
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
            "oxidizable": 0.46,
            "corrosion_resistance": 0.34,
            "biodegradable": 0.00,
            "water_soluble": 0.02,
            "thermal_stability": 0.44,
            "fatigue_resistance": 0.34,
            "abrasion_resistance": 0.64,
            "uv_sensitivity": 0.02,
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
            "oxidizable": 0.34,
            "corrosion_resistance": 0.70,
            "biodegradable": 0.24,
            "water_soluble": 0.03,
            "thermal_stability": 0.20,
            "fatigue_resistance": 0.30,
            "abrasion_resistance": 0.16,
            "uv_sensitivity": 0.66,
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
            "oxidizable": 0.18,
            "corrosion_resistance": 0.24,
            "biodegradable": 0.34,
            "water_soluble": 0.13,
            "thermal_stability": 0.36,
            "fatigue_resistance": 0.44,
            "abrasion_resistance": 0.46,
            "uv_sensitivity": 0.22,
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
    "carry",
    "protect",
    "record",
)
STRUCTURE_CAPABILITIES = (
    *ARTIFACT_CAPABILITIES,
    "channel",
    "enclose",
    "permeable",
    "shelter",
    "support",
    "gradient_harvest",
    "reaction_surface",
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
    fatigue_resistance = properties.get("fatigue_resistance", 0.0)
    abrasion_resistance = properties.get("abrasion_resistance", 0.0)
    corrosion_resistance = properties.get("corrosion_resistance", 0.0)
    sticky = properties.get("sticky", 0.0)
    porous = properties.get("porous", 0.0)
    absorbent = properties.get("absorbent", 0.0)
    buoyant = properties.get("buoyant", 0.0)
    density = properties.get("density", heavy)
    insulating = properties.get("insulating", 0.0)
    sealant = properties.get("sealant", 0.0)
    lightness = max(0.0, 1.0 - max(density, heavy * 0.85))
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
    capabilities["carry"] = min(
        1.0,
        container * 0.38
        + flexible * 0.22
        + bindable * 0.20
        + length * 0.08
        + lightness * 0.10
        + fatigue_resistance * 0.08
        - heavy * 0.12,
    )
    capabilities["protect"] = min(
        1.0,
        hard * 0.24
        + flexible * 0.12
        + abrasion_resistance * 0.18
        + thermal_capacity * 0.10
        + insulating * 0.12
        + sealant * 0.10
        + corrosion_resistance * 0.08
        + bindable * 0.06,
    )
    capabilities["record"] = min(
        1.0,
        porous * 0.20
        + absorbent * 0.18
        + flexible * 0.12
        + hard * 0.10
        + bindable * 0.12
        + sealant * 0.08
        + container * 0.06
        + lightness * 0.08,
    )
    return capabilities


def build_artifact(
    components: Mapping[str, int],
    method_quality: float = 0.0,
    target_affordance: str | None = None,
) -> Artifact:
    properties = component_properties(components)
    capabilities = derive_artifact_capabilities(properties)
    method_quality = _clamp01(method_quality)
    if method_quality > 0.0:
        properties["method_quality"] = method_quality
        for name, value in list(capabilities.items()):
            capabilities[name] = _clamp01(value * (1.0 + method_quality * 0.08) + method_quality * 0.006)
        if target_affordance in capabilities:
            target_value = capabilities[target_affordance]
            capabilities[target_affordance] = _clamp01(target_value + target_value * (1.0 - target_value) * method_quality * 0.70)
            properties["target_fit"] = capabilities[target_affordance]
    ranked = sorted(capabilities.items(), key=lambda item: item[1], reverse=True)
    dominant = [name for name, value in ranked[:2] if value > 0.25]
    name = "composite_" + ("_".join(dominant) if dominant else "object")
    durability = (
        35.0
        + properties.get("hard", 0.0) * 70.0
        + properties.get("flexible", 0.0) * 25.0
        + properties.get("bindable", 0.0) * 35.0
        + method_quality * (18.0 + properties.get("bindable", 0.0) * 20.0)
    )
    return Artifact(
        name=name,
        components=dict(components),
        properties=properties,
        capabilities=capabilities,
        durability=durability,
    )


def _scale_factor(scale: int) -> float:
    return max(0.0, min(1.0, math.log1p(max(1, scale)) / math.log(14.0)))


def derive_structure_capabilities(properties: Mapping[str, float], scale: int) -> dict[str, float]:
    base = derive_artifact_capabilities(properties)
    hard = properties.get("hard", 0.0)
    heavy = properties.get("heavy", 0.0)
    flexible = properties.get("flexible", 0.0)
    bindable = properties.get("bindable", 0.0)
    container = properties.get("container", 0.0)
    conductive = properties.get("conductive", 0.0)
    porous = properties.get("porous", 0.0)
    absorbent = properties.get("absorbent", 0.0)
    buoyant = properties.get("buoyant", 0.0)
    density = properties.get("density", heavy)
    sealant = properties.get("sealant", 0.0)
    insulating = properties.get("insulating", 0.0)
    sticky = properties.get("sticky", 0.0)
    length = properties.get("length", 0.0)
    thermal_capacity = max(properties.get("thermal_mass", 0.0), properties.get("thermal_capacity", 0.0))
    scale_gain = _scale_factor(scale)
    capabilities = {name: base.get(name, 0.0) for name in STRUCTURE_CAPABILITIES}
    capabilities["support"] = min(1.0, hard * 0.30 + density * 0.25 + bindable * 0.15 + length * 0.12 + sticky * 0.08 + scale_gain * 0.20)
    capabilities["channel"] = min(1.0, container * 0.28 + hard * 0.16 + length * 0.18 + bindable * 0.10 + sealant * 0.16 + scale_gain * 0.18)
    capabilities["enclose"] = min(1.0, container * 0.25 + sealant * 0.25 + hard * 0.16 + bindable * 0.14 + scale_gain * 0.20)
    capabilities["permeable"] = min(1.0, porous * 0.45 + absorbent * 0.16 + capabilities["filter"] * 0.24 + flexible * 0.08)
    capabilities["shelter"] = min(1.0, capabilities["enclose"] * 0.30 + capabilities["support"] * 0.20 + capabilities["insulate"] * 0.25 + capabilities["anchor"] * 0.15 + scale_gain * 0.10)
    capabilities["gradient_harvest"] = min(
        1.0,
        capabilities["anchor"] * 0.22
        + capabilities["channel"] * 0.25
        + capabilities["float"] * 0.10
        + capabilities["conduct"] * 0.15
        + capabilities["energy_storage"] * 0.10
        + length * 0.08
        + scale_gain * 0.18,
    )
    capabilities["reaction_surface"] = min(1.0, porous * 0.26 + conductive * 0.16 + thermal_capacity * 0.18 + container * 0.14 + absorbent * 0.10 + scale_gain * 0.16)
    capabilities["anchor"] = min(1.0, capabilities["anchor"] + capabilities["support"] * 0.15 + scale_gain * 0.08)
    capabilities["float"] = max(0.0, min(1.0, capabilities["float"] + buoyant * scale_gain * 0.06 - heavy * 0.04))
    for name in STRUCTURE_CAPABILITIES:
        capabilities[name] = min(1.0, max(0.0, capabilities.get(name, 0.0) * (0.72 + scale_gain * 0.38)))
    return capabilities


def build_structure(components: Mapping[str, int], builder_id: int | None = None) -> Structure:
    scale = sum(max(0, qty) for qty in components.values())
    properties = component_properties(components)
    capabilities = derive_structure_capabilities(properties, scale)
    ranked = sorted(capabilities.items(), key=lambda item: item[1], reverse=True)
    dominant = [name for name, value in ranked[:3] if value > 0.30 and name != "permeable"]
    name = "structure_" + ("_".join(dominant) if dominant else "assembly")
    durability = (
        70.0
        + properties.get("hard", 0.0) * 120.0
        + properties.get("bindable", 0.0) * 70.0
        + properties.get("sealant", 0.0) * 45.0
        + _scale_factor(scale) * 120.0
    )
    return Structure(
        name=name,
        components=dict(components),
        properties=properties,
        capabilities=capabilities,
        durability=durability,
        scale=scale,
        builder_id=builder_id,
    )


def extend_structure(structure: Structure, components: Mapping[str, int]) -> None:
    merged = dict(structure.components)
    for name, qty in components.items():
        merged[name] = merged.get(name, 0) + max(0, qty)
    updated = build_structure(merged, builder_id=structure.builder_id)
    structure.name = updated.name
    structure.components = updated.components
    structure.properties = updated.properties
    structure.capabilities = updated.capabilities
    structure.scale = updated.scale
    structure.durability = max(structure.durability * 0.82, updated.durability)


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


def structure_capability(structures: list[Structure], capability: str) -> float:
    best = 0.0
    for structure in structures:
        durability_factor = max(0.0, min(1.0, structure.durability / 180.0))
        scale_factor = _scale_factor(structure.scale)
        best = max(best, structure.capabilities.get(capability, 0.0) * (0.75 + scale_factor * 0.25) * durability_factor)
    return best


def structure_decay_channels(structure: Structure, environment: Mapping[str, float]) -> dict[str, float]:
    props = structure.properties
    caps = structure.capabilities
    scale = _scale_factor(structure.scale)
    hard = props.get("hard", 0.0)
    flexible = props.get("flexible", 0.0)
    brittle = props.get("brittle", 0.0)
    porous = props.get("porous", 0.0)
    absorbent = props.get("absorbent", 0.0)
    conductive = props.get("conductive", 0.0)
    combustible = props.get("combustible", 0.0)
    sealant = props.get("sealant", 0.0)
    density = props.get("density", props.get("heavy", 0.0))
    oxidizable = props.get("oxidizable", conductive * 0.45 + combustible * 0.18)
    corrosion_resistance = props.get("corrosion_resistance", sealant * 0.45 + hard * 0.22 + density * 0.14)
    biodegradable = props.get("biodegradable", combustible * 0.65 + porous * 0.12)
    water_soluble = props.get("water_soluble", max(0.0, absorbent * 0.12 - sealant * 0.08))
    thermal_stability = props.get("thermal_stability", props.get("thermal_capacity", 0.0) * 0.50 + hard * 0.22 + sealant * 0.10)
    fatigue_resistance = props.get("fatigue_resistance", hard * 0.35 + flexible * 0.22 + density * 0.18 - brittle * 0.18)
    abrasion_resistance = props.get("abrasion_resistance", hard * 0.48 + density * 0.24 + flexible * 0.08 - brittle * 0.14)
    uv_sensitivity = props.get("uv_sensitivity", combustible * 0.45 + flexible * 0.16 - hard * 0.12)

    temperature = _clamp01(environment.get("temperature", 0.5) / 1.35)
    heat_excess = max(0.0, environment.get("temperature", 0.5) - 0.68)
    cold_excess = max(0.0, 0.14 - environment.get("temperature", 0.5))
    fluid = _clamp01(environment.get("fluid_level", 0.0))
    humidity = _clamp01(environment.get("humidity", 0.5))
    salinity = _clamp01(environment.get("salinity", 0.0))
    oxygen = _clamp01(environment.get("oxygen", 0.35))
    acidity = _clamp01(environment.get("acidity", 0.10))
    biological_activity = _clamp01(environment.get("biological_activity", 0.0))
    abrasion = _clamp01(environment.get("abrasion", 0.0))
    wet_dry_cycle = _clamp01(environment.get("wet_dry_cycle", 0.0))
    current = _clamp01(environment.get("current_exposure", 0.0))
    pressure = _clamp01(environment.get("pressure", 0.0))
    light = _clamp01(environment.get("light", 0.0))
    flow_gradient = _clamp01(environment.get("flow_gradient", 0.0))

    enclose = caps.get("enclose", 0.0)
    permeable = caps.get("permeable", 0.0)
    shelter = caps.get("shelter", 0.0)
    support = caps.get("support", 0.0)
    anchor = caps.get("anchor", 0.0)
    channel = caps.get("channel", 0.0)
    gradient_harvest = caps.get("gradient_harvest", 0.0)
    reaction_surface = caps.get("reaction_surface", 0.0)
    filter_cap = caps.get("filter", 0.0)

    coating = _clamp01(sealant * 0.48 + enclose * 0.12 + shelter * 0.10)
    exposed_surface = _clamp01(0.42 + permeable * 0.22 + porous * 0.24 + absorbent * 0.12 - coating * 0.32)
    mechanical_resistance = _clamp01(abrasion_resistance * 0.55 + support * 0.22 + anchor * 0.16 + flexible * 0.10)
    chemical_resistance = _clamp01(corrosion_resistance * 0.62 + coating * 0.28 + density * 0.08)
    biological_resistance = _clamp01(coating * 0.40 + corrosion_resistance * 0.16 + hard * 0.16 + max(0.0, 1.0 - porous) * 0.10)
    thermal_resistance = _clamp01(thermal_stability * 0.62 + caps.get("insulate", 0.0) * 0.14 + density * 0.10)
    fatigue_resistance = _clamp01(fatigue_resistance * 0.58 + support * 0.18 + flexible * 0.10 + anchor * 0.08)

    wet_contact = _clamp01(fluid * 0.48 + humidity * 0.32 + wet_dry_cycle * 0.20)
    corrosion_env = _clamp01(salinity * 0.42 + acidity * 0.38 + oxygen * humidity * 0.25 + wet_dry_cycle * 0.12)
    biological_window = _clamp01(1.0 - abs(temperature - 0.46) * 1.65)
    thermal_env = _clamp01(heat_excess * 1.20 + cold_excess * 1.50 + wet_dry_cycle * 0.18 + light * 0.08)
    movement_env = _clamp01(current * 0.42 + pressure * 0.18 + abrasion * 0.34 + flow_gradient * 0.22)
    use_env = _clamp01(flow_gradient * (channel * 0.35 + gradient_harvest * 0.45) + reaction_surface * acidity * 0.20 + filter_cap * wet_contact * 0.10)
    size_load = 0.72 + scale * 0.42

    channels = {
        "baseline": 0.0010 + exposed_surface * 0.0009,
        "mechanical": movement_env * size_load * max(0.04, 1.0 - mechanical_resistance * 0.82) * 0.026,
        "chemical": corrosion_env * wet_contact * oxidizable * exposed_surface * max(0.03, 1.0 - chemical_resistance * 0.86) * 0.040,
        "biological": biological_activity * wet_contact * biodegradable * biological_window * max(0.04, 1.0 - biological_resistance * 0.78) * 0.030,
        "thermal": thermal_env * (combustible * 0.22 + brittle * 0.14 + 0.16) * max(0.04, 1.0 - thermal_resistance * 0.80) * 0.025,
        "solubility": fluid * (acidity * 0.42 + salinity * 0.22 + current * 0.18 + wet_dry_cycle * 0.18) * water_soluble * exposed_surface * max(0.05, 1.0 - coating * 0.72) * 0.034,
        "radiation": light * uv_sensitivity * max(0.04, 1.0 - shelter * 0.55 - enclose * 0.18) * 0.010,
        "fatigue": use_env * max(0.04, 1.0 - fatigue_resistance * 0.76) * 0.018,
    }
    return {name: max(0.0, value) for name, value in channels.items()}


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
