from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random
from typing import Any

from .config import RunConfig
from .energy import ENERGY_KINDS, MATERIALS, blank_energy


@dataclass(slots=True)
class Signal:
    source_id: int
    token: int
    intensity: float
    age: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "token": self.token, "intensity": self.intensity, "age": self.age}


@dataclass(slots=True)
class Mark:
    source_id: int
    token: int
    intensity: float
    durability: float
    age: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "token": self.token,
            "intensity": self.intensity,
            "durability": self.durability,
            "age": self.age,
        }


@dataclass(slots=True)
class Place:
    id: int
    name: str
    neighbors: list[int]
    resources: dict[str, float]
    materials: dict[str, int]
    locked_chemical: float
    capacity: int
    sun_exposure: float
    water_flow: float
    geothermal: float
    mineral_richness: float
    volatility: float
    obstacles: dict[str, float]
    habitat: dict[str, float]
    signals: list[Signal] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)

    def total_accessible_energy(self) -> float:
        return sum(self.resources.values())

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "neighbors": self.neighbors,
            "resources": {k: round(v, 4) for k, v in self.resources.items()},
            "locked_chemical": round(self.locked_chemical, 4),
            "materials": dict(self.materials),
            "capacity": self.capacity,
            "obstacles": {k: round(v, 4) for k, v in self.obstacles.items()},
            "habitat": {k: round(v, 4) for k, v in self.habitat.items()},
            "marks": [mark.to_dict() for mark in self.marks[-8:]],
        }


@dataclass(slots=True)
class World:
    places: list[Place]
    season_length: int
    tick: int = 0
    climate_drift: float = 0.0

    @classmethod
    def generate(cls, rng: Random, config: RunConfig) -> "World":
        names = [
            "sun shelf",
            "fungal hollow",
            "mineral vent",
            "reed basin",
            "dry ridge",
            "stone fall",
            "warm spring",
            "root maze",
            "ash flat",
            "glass scree",
            "moss sink",
            "wind throat",
            "brackish pool",
            "thorn gate",
            "blue clay",
            "salt pocket",
        ]
        places: list[Place] = []
        for i in range(config.places):
            resources = blank_energy()
            sun = rng.uniform(0.15, 1.0)
            water = rng.uniform(0.0, 1.0)
            geo = rng.random() ** 2.7
            mineral = rng.random() ** 1.3
            volatility = rng.uniform(0.03, 0.22)
            resources["radiant"] = 20.0 + sun * 70.0
            resources["chemical"] = rng.uniform(10.0, 40.0) + water * 15.0
            resources["biological_storage"] = rng.uniform(4.0, 20.0)
            resources["thermal"] = geo * 65.0 + sun * 10.0
            resources["mechanical"] = water * 45.0 + rng.uniform(0.0, 8.0)
            resources["electrical"] = mineral * rng.uniform(0.0, 4.0)
            resources["high_density"] = mineral * geo * rng.uniform(0.0, 1.2)

            materials = {name: 0 for name in MATERIALS}
            for name in MATERIALS:
                abundance = rng.random()
                if name in {"stone", "crystal"}:
                    abundance *= mineral + 0.25
                if name in {"branch", "fiber", "resin"}:
                    abundance *= water + sun + 0.20
                if name in {"shell", "bone"}:
                    abundance *= water + 0.15
                materials[name] = int(abundance * rng.randint(1, 8))
            obstacles = {
                "water": min(1.0, water * rng.uniform(0.15, 0.95)),
                "thorn": min(1.0, (sun + water) * rng.uniform(0.05, 0.55)),
                "height": min(1.0, rng.random() * rng.uniform(0.05, 0.70)),
                "heat": min(1.0, geo * rng.uniform(0.15, 0.90)),
            }
            aquatic = min(1.0, max(0.0, water * rng.uniform(0.15, 1.15)))
            habitat = {
                "aquatic": aquatic,
                "depth": aquatic * rng.uniform(0.05, 1.0),
                "salinity": aquatic * rng.random(),
                "humidity": min(1.0, water * 0.75 + sun * 0.10 + rng.random() * 0.15),
            }

            places.append(
                Place(
                    id=i,
                    name=f"{names[i % len(names)]} {i}",
                    neighbors=[],
                    resources=resources,
                    materials=materials,
                    locked_chemical=rng.uniform(8.0, 55.0) * (0.35 + mineral),
                    capacity=rng.randint(35, 95),
                    sun_exposure=sun,
                    water_flow=water,
                    geothermal=geo,
                    mineral_richness=mineral,
                    volatility=volatility,
                    obstacles=obstacles,
                    habitat=habitat,
                )
            )

        for i in range(config.places):
            cls._connect(places, i, (i + 1) % config.places)
            cls._connect(places, i, (i - 1) % config.places)
        extra_edges = max(1, config.places // 2)
        for _ in range(extra_edges):
            a = rng.randrange(config.places)
            b = rng.randrange(config.places)
            if a != b:
                cls._connect(places, a, b)

        return cls(places=places, season_length=config.season_length)

    @staticmethod
    def _connect(places: list[Place], a: int, b: int) -> None:
        if b not in places[a].neighbors:
            places[a].neighbors.append(b)
        if a not in places[b].neighbors:
            places[b].neighbors.append(a)

    def update_environment(self, rng: Random) -> None:
        self.tick += 1
        season = 0.5 + 0.5 * math.sin(2.0 * math.pi * self.tick / max(2, self.season_length))
        if self.tick % max(50, self.season_length // 8) == 0:
            self.climate_drift = max(-0.35, min(0.35, self.climate_drift + rng.gauss(0.0, 0.018)))

        for place in self.places:
            weather = 1.0 + rng.gauss(0.0, place.volatility * 0.02)
            radiant_target = (18.0 + place.sun_exposure * 85.0) * (0.35 + season * 0.85 + self.climate_drift * 0.25)
            place.resources["radiant"] += (radiant_target * weather - place.resources["radiant"]) * 0.08
            place.resources["chemical"] += 0.010 + place.water_flow * 0.030 + place.mineral_richness * 0.006
            place.resources["thermal"] += ((place.geothermal * 65.0 + place.sun_exposure * season * 15.0) - place.resources["thermal"]) * 0.015
            place.resources["mechanical"] += ((place.water_flow * 40.0 + place.volatility * 18.0) - place.resources["mechanical"]) * 0.010
            place.resources["electrical"] += place.mineral_richness * place.volatility * 0.006
            place.resources["biological_storage"] *= 0.9994
            place.locked_chemical += place.mineral_richness * 0.003

            if rng.random() < 0.002 + place.water_flow * 0.002:
                material = rng.choice(tuple(MATERIALS.keys()))
                place.materials[material] = min(99, place.materials.get(material, 0) + 1)

            for kind in ENERGY_KINDS:
                place.resources[kind] = max(0.0, min(180.0, place.resources[kind]))
            place.locked_chemical = max(0.0, min(260.0, place.locked_chemical))

            kept: list[Signal] = []
            for signal in place.signals:
                signal.age += 1
                signal.intensity *= 0.86
                if signal.age < 8 and signal.intensity > 0.015:
                    kept.append(signal)
            place.signals = kept

            kept_marks: list[Mark] = []
            for mark in place.marks:
                mark.age += 1
                mark.intensity *= 0.997
                mark.durability -= 0.10 + place.volatility * 0.06 + place.water_flow * 0.025
                if mark.durability > 0.0 and mark.intensity > 0.025:
                    kept_marks.append(mark)
            if len(kept_marks) > 32:
                kept_marks = sorted(kept_marks, key=lambda item: (item.intensity * item.durability, -item.age), reverse=True)[:32]
            place.marks = kept_marks

    def emit_signal(self, place_id: int, source_id: int, token: int, intensity: float) -> None:
        if 0 <= place_id < len(self.places):
            self.places[place_id].signals.append(Signal(source_id=source_id, token=token % 8, intensity=max(0.0, intensity)))

    def create_mark(self, place_id: int, source_id: int, token: int, intensity: float, durability: float) -> None:
        if 0 <= place_id < len(self.places):
            self.places[place_id].marks.append(
                Mark(
                    source_id=source_id,
                    token=token % 8,
                    intensity=max(0.0, intensity),
                    durability=max(1.0, durability),
                )
            )

    def to_summary(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "season_length": self.season_length,
            "climate_drift": round(self.climate_drift, 5),
            "places": [place.to_summary() for place in self.places],
        }
