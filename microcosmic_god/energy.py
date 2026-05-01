from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
        },
    ),
}

AFFORDANCES = ("crack", "cut", "bind", "contain", "concentrate_heat", "conduct", "lever")


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


def derive_affordances(inventory: Mapping[str, int]) -> dict[str, float]:
    props = inventory_properties(inventory)
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
    }


def best_affordance(inventory: Mapping[str, int], skills: Mapping[str, float]) -> tuple[str, float]:
    potentials = derive_affordances(inventory)
    best_name = "crack"
    best_score = 0.0
    for name, potential in potentials.items():
        score = potential * (0.65 + 0.35 * skills.get(name, 0.0))
        if score > best_score:
            best_name = name
            best_score = score
    return best_name, best_score

