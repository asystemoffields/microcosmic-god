from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Any

from .config import RunConfig
from .energy import ENERGY_KINDS, blank_energy


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


ENV_ARCHETYPES = (
    "pelagic",
    "reef",
    "trench",
    "hydrothermal_vent",
    "tidal_marsh",
    "high_ridge",
    "mineral_scree",
    "forest_edge",
    "desert_glass",
    "cavern",
)


@dataclass(slots=True)
class Signal:
    source_id: int
    token: int
    intensity: float
    age: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "token": self.token, "intensity": self.intensity, "age": self.age}


@dataclass(slots=True)
class Place:
    id: int
    name: str
    neighbors: list[int]
    resources: dict[str, float]
    sealed_essence: float
    capacity: int
    sun_exposure: float
    water_flow: float
    geothermal: float
    mineral_richness: float
    volatility: float
    obstacles: dict[str, float]
    terrain: dict[str, float]
    physics: dict[str, float]
    archetype: str = "mixed"
    signals: list[Signal] = field(default_factory=list)
    # Tick until which staple regeneration is suppressed (patch recovery).
    regen_recovery_until: int = 0

    def total_accessible_energy(self) -> float:
        return sum(self.resources.values())

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "neighbors": self.neighbors,
            "resources": {k: round(v, 4) for k, v in self.resources.items()},
            "sealed_essence": round(self.sealed_essence, 4),
            "capacity": self.capacity,
            "obstacles": {k: round(v, 4) for k, v in self.obstacles.items()},
            "terrain": {k: round(v, 4) for k, v in self.terrain.items()},
            "physics": {k: round(v, 4) for k, v in self.physics.items()},
            "archetype": self.archetype,
        }


@dataclass(slots=True)
class Edge:
    a: int
    b: int
    distance: float
    slope: float
    current: float
    permeability: float
    heat_conductance: float
    fluid_conductance: float
    danger: float
    traversal_required: float

    def other(self, place_id: int) -> int:
        if place_id == self.a:
            return self.b
        if place_id == self.b:
            return self.a
        raise ValueError(f"place {place_id} is not on edge {self.a}-{self.b}")

    def slope_from(self, place_id: int) -> float:
        return self.slope if place_id == self.a else -self.slope

    def current_from(self, place_id: int) -> float:
        return self.current if place_id == self.a else -self.current

    def to_summary(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "distance": round(self.distance, 4),
            "slope": round(self.slope, 4),
            "current": round(self.current, 4),
            "permeability": round(self.permeability, 4),
            "heat_conductance": round(self.heat_conductance, 4),
            "fluid_conductance": round(self.fluid_conductance, 4),
            "danger": round(self.danger, 4),
            "traversal_required": round(self.traversal_required, 4),
        }


@dataclass(slots=True)
class World:
    places: list[Place]
    season_length: int
    edges: list[Edge] = field(default_factory=list)
    edge_lookup: dict[tuple[int, int], Edge] = field(default_factory=dict, repr=False)
    edge_adjacency: dict[int, list[Edge]] = field(default_factory=dict, repr=False)
    tick: int = 0
    climate_drift: float = 0.0
    environment_harshness: float = 1.0
    patch_recovery_ticks: int = 0
    patch_recovery_floor: float = 0.0
    patch_recovery_jitter: float = 0.0

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
        harshness = max(0.2, float(getattr(config, "environment_harshness", 1.0)))
        hardship = max(0.0, harshness - 1.0)
        places: list[Place] = []
        for i in range(config.places):
            resources = blank_energy()
            sun = rng.uniform(0.15, 1.0)
            water = rng.uniform(0.0, 1.0)
            geo = rng.random() ** 2.7
            mineral = rng.random() ** 1.3
            volatility = rng.uniform(0.03, 0.22)
            elevation = _clamp(rng.random() ** 1.15)
            archetype = ENV_ARCHETYPES[i % len(ENV_ARCHETYPES)] if i < len(ENV_ARCHETYPES) else rng.choice(ENV_ARCHETYPES)
            if archetype == "pelagic":
                water = max(water, rng.uniform(0.74, 1.0))
                elevation = min(elevation, rng.uniform(0.02, 0.18))
                volatility = max(volatility, rng.uniform(0.18, 0.34))
            elif archetype == "reef":
                water = max(water, rng.uniform(0.55, 0.90))
                sun = max(sun, rng.uniform(0.55, 1.0))
                mineral = max(mineral, rng.uniform(0.35, 0.80))
            elif archetype == "trench":
                water = max(water, rng.uniform(0.82, 1.0))
                elevation = min(elevation, rng.uniform(0.0, 0.10))
                sun = min(sun, rng.uniform(0.05, 0.24))
                mineral = max(mineral, rng.uniform(0.45, 1.0))
                volatility = max(volatility, rng.uniform(0.12, 0.28))
            elif archetype == "hydrothermal_vent":
                water = max(water, rng.uniform(0.45, 0.92))
                geo = max(geo, rng.uniform(0.50, 1.0))
                mineral = max(mineral, rng.uniform(0.55, 1.0))
                volatility = max(volatility, rng.uniform(0.16, 0.36))
            elif archetype == "tidal_marsh":
                water = max(water, rng.uniform(0.45, 0.86))
                sun = max(sun, rng.uniform(0.45, 0.95))
                volatility = max(volatility, rng.uniform(0.16, 0.30))
            elif archetype == "high_ridge":
                elevation = max(elevation, rng.uniform(0.72, 1.0))
                water = min(water, rng.uniform(0.0, 0.28))
                sun = max(sun, rng.uniform(0.50, 1.0))
                volatility = max(volatility, rng.uniform(0.12, 0.26))
            elif archetype == "mineral_scree":
                mineral = max(mineral, rng.uniform(0.72, 1.0))
                elevation = max(elevation, rng.uniform(0.48, 0.92))
                water = min(water, rng.uniform(0.02, 0.36))
            elif archetype == "forest_edge":
                water = max(water, rng.uniform(0.34, 0.74))
                sun = max(sun, rng.uniform(0.40, 0.86))
            elif archetype == "desert_glass":
                water = min(water, rng.uniform(0.0, 0.18))
                sun = max(sun, rng.uniform(0.72, 1.0))
                mineral = max(mineral, rng.uniform(0.38, 0.88))
                volatility = max(volatility, rng.uniform(0.12, 0.30))
            elif archetype == "cavern":
                sun = min(sun, rng.uniform(0.04, 0.28))
                water = max(water, rng.uniform(0.18, 0.60))
                mineral = max(mineral, rng.uniform(0.45, 0.95))
                elevation = min(elevation, rng.uniform(0.08, 0.40))
            volatility = _clamp(volatility * (1.0 + hardship * 0.55), 0.03, 0.52)
            resources["solar"] = 20.0 + sun * 70.0
            resources["essence"] = rng.uniform(10.0, 40.0) + water * 15.0
            resources["residue_store"] = rng.uniform(4.0, 20.0)
            resources["thermal"] = geo * 65.0 + sun * 10.0
            resources["mechanical"] = water * 45.0 + rng.uniform(0.0, 8.0)
            resources["electrical"] = mineral * rng.uniform(0.0, 4.0)
            resources["dense_node"] = mineral * geo * rng.uniform(0.0, 1.2)
            if archetype in {"reef", "tidal_marsh", "forest_edge"}:
                resources["residue_store"] += rng.uniform(10.0, 36.0) * (0.55 + sun * water)
            if archetype in {"trench", "hydrothermal_vent"}:
                resources["thermal"] += rng.uniform(12.0, 52.0) * (0.40 + geo)
                resources["essence"] += rng.uniform(8.0, 28.0) * (0.40 + mineral)
                resources["dense_node"] += rng.uniform(0.5, 5.0) * geo * mineral
            if archetype in {"pelagic", "tidal_marsh"}:
                resources["mechanical"] += rng.uniform(8.0, 32.0) * water * volatility
            if archetype in {"high_ridge", "desert_glass"}:
                resources["solar"] += rng.uniform(8.0, 28.0) * sun
            if archetype in {"mineral_scree", "cavern"}:
                resources["electrical"] += rng.uniform(0.5, 8.0) * mineral
            sealed_essence = rng.uniform(8.0, 55.0) * (0.35 + mineral)
            if archetype in {"trench", "hydrothermal_vent", "mineral_scree", "cavern"}:
                sealed_essence *= rng.uniform(1.25, 2.15)
            accessible_factor = max(0.50, 1.0 - hardship * 0.24)
            residue_factor = max(0.38, 1.0 - hardship * 0.34)
            resources["solar"] *= max(0.70, 1.0 - hardship * 0.10)
            resources["essence"] *= accessible_factor
            resources["residue_store"] *= residue_factor
            resources["thermal"] *= 1.0 + hardship * 0.06
            resources["mechanical"] *= 1.0 + hardship * 0.08
            sealed_essence *= 1.0 + hardship * 0.30

            obstacles = {
                "water": min(1.0, water * rng.uniform(0.15, 0.95)),
                "thorn": min(1.0, (sun + water) * rng.uniform(0.05, 0.55)),
                "height": min(1.0, rng.random() * rng.uniform(0.05, 0.70)),
                "heat": min(1.0, geo * rng.uniform(0.15, 0.90)),
            }
            if archetype in {"pelagic", "trench"}:
                obstacles["water"] = max(obstacles["water"], rng.uniform(0.72, 1.0))
            if archetype in {"high_ridge", "mineral_scree"}:
                obstacles["height"] = max(obstacles["height"], rng.uniform(0.48, 0.92))
            if archetype == "hydrothermal_vent":
                obstacles["heat"] = max(obstacles["heat"], rng.uniform(0.45, 0.95))
            if archetype in {"forest_edge", "tidal_marsh"}:
                obstacles["thorn"] = max(obstacles["thorn"], rng.uniform(0.22, 0.72))
            obstacles["water"] = _clamp(obstacles["water"] * (1.0 + hardship * 0.20) + water * hardship * 0.05)
            obstacles["thorn"] = _clamp(obstacles["thorn"] * (1.0 + hardship * 0.18) + (sun + water) * hardship * 0.025)
            obstacles["height"] = _clamp(obstacles["height"] * (1.0 + hardship * 0.20) + elevation * hardship * 0.06)
            obstacles["heat"] = _clamp(obstacles["heat"] * (1.0 + hardship * 0.24) + max(0.0, geo - 0.30) * hardship * 0.10)
            aquatic = min(1.0, max(0.0, water * rng.uniform(0.15, 1.15)))
            depth = aquatic * rng.uniform(0.05, 1.0)
            if archetype in {"pelagic", "trench"}:
                depth = max(depth, rng.uniform(0.68, 1.0))
            elif archetype in {"reef", "tidal_marsh"}:
                depth = max(depth, rng.uniform(0.20, 0.56))
            salinity = aquatic * rng.random()
            if archetype in {"pelagic", "reef", "trench"}:
                salinity = max(salinity, rng.uniform(0.42, 0.96))
            humidity = min(1.0, water * 0.75 + sun * 0.10 + rng.random() * 0.15)
            terrain = {
                "aquatic": aquatic,
                "depth": depth,
                "salinity": salinity,
                "humidity": humidity,
            }
            temperature = _clamp(0.08 + sun * 0.33 + geo * 0.42 + rng.gauss(0.0, 0.04), 0.0, 1.25)
            temperature = _clamp(0.50 + (temperature - 0.50) * (1.0 + hardship * 0.48), 0.0, 1.45)
            physics = {
                "temperature": temperature,
                "fluid_level": aquatic,
                "pressure": _clamp(depth * (0.55 + aquatic * 0.65), 0.0, 1.35),
                "humidity": humidity,
                "salinity": salinity,
                "elevation": elevation,
                "current_exposure": water * volatility,
                "thermal_mass": _clamp(0.18 + mineral * 0.35 + water * 0.18 + rng.random() * 0.22),
                "light": sun,
                "oxygen": _clamp(0.18 + sun * 0.32 + (1.0 - depth) * 0.18 + water * volatility * 0.10),
                "acidity": _clamp(0.04 + geo * 0.20 + volatility * 0.10 + rng.random() * 0.08),
                "residue_activity": _clamp(water * 0.28 + sun * 0.16 + resources["residue_store"] / 220.0),
                "abrasion": _clamp(volatility * 0.34 + water * 0.14 + abs(elevation - 0.5) * 0.10),
                "wet_dry_cycle": _clamp(water * (1.0 - water) * 0.70 + volatility * 0.22),
                "resource_gradient": 0.0,
                "terrain_richness": 0.0,
            }
            if archetype == "trench":
                physics["pressure"] = _clamp(physics["pressure"] + rng.uniform(0.22, 0.55), 0.0, 1.35)
                physics["oxygen"] = _clamp(physics["oxygen"] - rng.uniform(0.05, 0.18))
            elif archetype == "hydrothermal_vent":
                physics["temperature"] = _clamp(physics["temperature"] + rng.uniform(0.18, 0.42), 0.0, 1.45)
                physics["acidity"] = _clamp(physics["acidity"] + rng.uniform(0.10, 0.28))
            elif archetype == "cavern":
                physics["interiority"] = rng.uniform(0.36, 0.82)
                physics["boundary_permeability"] = rng.uniform(0.08, 0.36)
                physics["shelter"] = rng.uniform(0.08, 0.28)
            physics["pressure"] = _clamp(physics["pressure"] * (1.0 + hardship * 0.18) + aquatic * hardship * 0.05, 0.0, 1.60)
            physics["current_exposure"] = _clamp(
                physics["current_exposure"] * (1.0 + hardship * 0.55) + water * volatility * hardship * 0.10,
                0.0,
                1.45,
            )
            physics["abrasion"] = _clamp(physics["abrasion"] * (1.0 + hardship * 0.55) + abs(elevation - 0.5) * hardship * 0.05)
            physics["wet_dry_cycle"] = _clamp(physics["wet_dry_cycle"] * (1.0 + hardship * 0.40) + volatility * hardship * 0.08)
            physics["oxygen"] = _clamp(physics["oxygen"] - hardship * (0.015 + aquatic * 0.030 + depth * 0.020))
            physics["acidity"] = _clamp(physics["acidity"] * (1.0 + hardship * 0.12) + geo * hardship * 0.025)
            physics["resource_gradient"] = _clamp(
                resources["dense_node"] / 8.0
                + resources["thermal"] / 220.0
                + resources["mechanical"] / 240.0
                + sealed_essence / 160.0
            )
            physics["terrain_richness"] = _clamp(0.05 + mineral * 0.42 + water * 0.29 + sun * 0.10)
            base_capacity = rng.randint(35, 95)
            capacity = max(12, int(round(base_capacity * max(0.52, 1.0 - hardship * 0.26))))

            places.append(
                Place(
                    id=i,
                    name=f"{names[i % len(names)]} {i}",
                    neighbors=[],
                    resources=resources,
                    sealed_essence=sealed_essence,
                    capacity=capacity,
                    sun_exposure=sun,
                    water_flow=water,
                    geothermal=geo,
                    mineral_richness=mineral,
                    volatility=volatility,
                    obstacles=obstacles,
                    terrain=terrain,
                    physics=physics,
                    archetype=archetype,
                )
            )

        edges: list[Edge] = []
        edge_lookup: dict[tuple[int, int], Edge] = {}
        edge_adjacency: dict[int, list[Edge]] = {place.id: [] for place in places}
        for i in range(config.places):
            cls._connect(places, edges, edge_lookup, edge_adjacency, i, (i + 1) % config.places, rng)
            cls._connect(places, edges, edge_lookup, edge_adjacency, i, (i - 1) % config.places, rng)
        extra_edges = max(1, config.places // 2)
        for _ in range(extra_edges):
            a = rng.randrange(config.places)
            b = rng.randrange(config.places)
            if a != b:
                cls._connect(places, edges, edge_lookup, edge_adjacency, a, b, rng)

        return cls(
            places=places,
            season_length=config.season_length,
            edges=edges,
            edge_lookup=edge_lookup,
            edge_adjacency=edge_adjacency,
            environment_harshness=harshness,
            patch_recovery_ticks=int(getattr(config, "patch_recovery_ticks", 0)),
            patch_recovery_floor=float(getattr(config, "patch_recovery_floor", 0.0)),
            patch_recovery_jitter=float(getattr(config, "patch_recovery_jitter", 0.0)),
        )

    @staticmethod
    def _connect(
        places: list[Place],
        edges: list[Edge],
        edge_lookup: dict[tuple[int, int], Edge],
        edge_adjacency: dict[int, list[Edge]],
        a: int,
        b: int,
        rng: Random,
    ) -> None:
        key = (min(a, b), max(a, b))
        if key in edge_lookup:
            return
        if b not in places[a].neighbors:
            places[a].neighbors.append(b)
        if a not in places[b].neighbors:
            places[b].neighbors.append(a)
        elevation_a = places[a].physics["elevation"]
        elevation_b = places[b].physics["elevation"]
        fluid_a = places[a].physics["fluid_level"]
        fluid_b = places[b].physics["fluid_level"]
        slope = elevation_b - elevation_a
        current = _clamp((elevation_a - elevation_b) * 0.55 + (fluid_a - fluid_b) * 0.20 + rng.gauss(0.0, 0.12), -1.0, 1.0)
        permeability = _clamp(rng.uniform(0.25, 0.95) - abs(slope) * 0.12, 0.05, 1.0)
        wateriness = (fluid_a + fluid_b) * 0.5
        edge = Edge(
            a=a,
            b=b,
            distance=rng.uniform(0.65, 1.85),
            slope=slope,
            current=current,
            permeability=permeability,
            heat_conductance=_clamp(rng.uniform(0.08, 0.55) + (places[a].mineral_richness + places[b].mineral_richness) * 0.10),
            fluid_conductance=_clamp(rng.uniform(0.05, 0.55) + wateriness * 0.35),
            danger=_clamp(abs(slope) * 0.25 + wateriness * 0.10 + rng.random() * 0.08),
            traversal_required=_clamp(abs(slope) * 0.45 + wateriness * 0.30 + rng.random() * 0.10),
        )
        edges.append(edge)
        edge_lookup[key] = edge
        edge_adjacency[a].append(edge)
        edge_adjacency[b].append(edge)

    def edge_between(self, a: int, b: int) -> Edge | None:
        return self.edge_lookup.get((min(a, b), max(a, b)))

    def edges_from(self, place_id: int) -> list[Edge]:
        return self.edge_adjacency.get(place_id, [])

    def downstream_neighbor(self, place_id: int) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        for edge in self.edges_from(place_id):
            outward = edge.current_from(place_id) * edge.fluid_conductance * edge.permeability
            if outward <= 0.02:
                continue
            if best is None or outward > best[1]:
                best = (edge.other(place_id), outward)
        return best

    def update_environment(self, rng: Random) -> dict[str, int]:
        events: Counter[str] = Counter()
        self.tick += 1
        season = 0.5 + 0.5 * math.sin(2.0 * math.pi * self.tick / max(2, self.season_length))
        harshness = max(0.2, self.environment_harshness)
        hardship = max(0.0, harshness - 1.0)
        if self.tick % max(50, self.season_length // 8) == 0:
            drift_limit = 0.35 + hardship * 0.10
            self.climate_drift = max(
                -drift_limit,
                min(drift_limit, self.climate_drift + rng.gauss(0.0, 0.018 * (1.0 + hardship * 0.55))),
            )

        old_temperature = [place.physics.get("temperature", 0.5) for place in self.places]
        old_fluid = [place.physics.get("fluid_level", 0.0) for place in self.places]
        old_salinity = [place.physics.get("salinity", 0.0) for place in self.places]
        temperature_delta = [0.0 for _ in self.places]
        fluid_delta = [0.0 for _ in self.places]
        salinity_delta = [0.0 for _ in self.places]

        for place in self.places:
            physics = place.physics
            weather = 1.0 + rng.gauss(0.0, place.volatility * 0.02 * (1.0 + hardship * 0.50))
            solar_target = (18.0 + place.sun_exposure * 85.0) * (0.35 + season * 0.85 + self.climate_drift * 0.25)
            place.resources["solar"] += (solar_target * weather - place.resources["solar"]) * 0.08
            essence_regen = (0.010 + place.water_flow * 0.030 + place.mineral_richness * 0.006) * max(0.45, 1.0 - hardship * 0.30)
            if place.regen_recovery_until >= self.tick:
                essence_regen *= _clamp(self.patch_recovery_floor)
                events["patch_recovering"] += 1
            place.resources["essence"] += essence_regen
            thermal_mass = _clamp(physics.get("thermal_mass", 0.4), 0.05, 1.0)
            temperature_target = _clamp(
                0.06
                + place.sun_exposure * (0.20 + season * 0.30)
                + place.geothermal * 0.40
                + place.resources["thermal"] / 520.0
                + self.climate_drift * 0.18,
                0.0,
                1.35,
            )
            temperature_target = _clamp(0.50 + (temperature_target - 0.50) * (1.0 + hardship * 0.25), 0.0, 1.45)
            thermal_rate = (0.012 + (1.0 - thermal_mass) * 0.040) * (1.0 + hardship * 0.20)
            physics["temperature"] += (temperature_target - physics.get("temperature", 0.5)) * thermal_rate
            evaporation = max(0.0, physics["temperature"] - 0.55) * physics.get("fluid_level", 0.0) * 0.004 * (1.0 + hardship * 0.25)
            fluid_target = _clamp(place.water_flow * (0.45 + season * 0.30) + rng.gauss(0.0, place.volatility * 0.004 * (1.0 + hardship * 0.50)))
            physics["fluid_level"] += (fluid_target - physics.get("fluid_level", 0.0)) * 0.006 * (1.0 + hardship * 0.12) - evaporation
            physics["humidity"] += (physics["fluid_level"] * 0.70 + evaporation * 10.0 - physics.get("humidity", 0.5)) * 0.020
            viability_window = max(0.0, 1.0 - abs(physics["temperature"] - 0.52) * 1.65)
            oxygen_target = _clamp(0.16 + place.sun_exposure * 0.32 + physics["fluid_level"] * physics.get("current_exposure", 0.0) * 0.35 + (1.0 - physics.get("pressure", 0.0)) * 0.12)
            acidity_target = _clamp(0.04 + place.geothermal * 0.22 + place.volatility * 0.08 + place.resources["essence"] / 2100.0 + physics.get("salinity", 0.0) * 0.06)
            residue_target = _clamp(
                (physics["humidity"] * 0.26 + physics["fluid_level"] * 0.24 + place.resources["residue_store"] / 240.0 + viability_window * place.sun_exposure * 0.12)
                * max(0.75, 1.0 - hardship * 0.10)
            )
            abrasion_target = _clamp(
                (physics.get("current_exposure", 0.0) * 0.48 + place.volatility * 0.20 + physics["fluid_level"] * 0.12 + abs(physics["temperature"] - 0.5) * 0.06)
                * (1.0 + hardship * 0.35)
                + place.volatility * hardship * 0.04
            )
            wet_dry_target = _clamp(
                (place.volatility * 0.22 + physics["fluid_level"] * (1.0 - physics["fluid_level"]) * 0.56 + evaporation * 14.0)
                * (1.0 + hardship * 0.35)
                + place.volatility * hardship * 0.04
            )
            physics["oxygen"] += (oxygen_target - physics.get("oxygen", 0.35)) * 0.018
            physics["acidity"] += (acidity_target - physics.get("acidity", 0.10)) * 0.012
            physics["residue_activity"] += (residue_target - physics.get("residue_activity", 0.0)) * 0.016
            physics["abrasion"] += (abrasion_target - physics.get("abrasion", 0.0)) * 0.018
            physics["wet_dry_cycle"] += (wet_dry_target - physics.get("wet_dry_cycle", 0.0)) * 0.014
            place.resources["thermal"] += ((physics["temperature"] * 150.0 + place.geothermal * 30.0) - place.resources["thermal"]) * 0.012
            place.resources["mechanical"] += ((place.water_flow * 25.0 + physics.get("current_exposure", 0.0) * 75.0 + place.volatility * 18.0) - place.resources["mechanical"]) * 0.010
            place.resources["electrical"] += place.mineral_richness * place.volatility * 0.006
            place.resources["residue_store"] *= max(0.9965, 0.9994 - hardship * 0.0008)
            place.sealed_essence += place.mineral_richness * (0.003 + hardship * 0.001)

            for kind in ENERGY_KINDS:
                place.resources[kind] = max(0.0, min(180.0, place.resources[kind]))
            place.sealed_essence = max(0.0, min(260.0, place.sealed_essence))

        for edge in self.edges:
            a = edge.a
            b = edge.b
            heat_flow = (old_temperature[a] - old_temperature[b]) * edge.heat_conductance * 0.022
            temperature_delta[a] -= heat_flow / max(0.25, self.places[a].physics.get("thermal_mass", 0.5))
            temperature_delta[b] += heat_flow / max(0.25, self.places[b].physics.get("thermal_mass", 0.5))
            fluid_flow = ((old_fluid[a] - old_fluid[b]) * 0.035 + edge.current * 0.018) * edge.fluid_conductance * edge.permeability
            fluid_flow = _clamp(fluid_flow, -0.045, 0.045)
            fluid_delta[a] -= fluid_flow
            fluid_delta[b] += fluid_flow
            salinity_mix = (old_salinity[a] - old_salinity[b]) * abs(fluid_flow) * 0.45
            if fluid_flow > 0.0:
                salinity_delta[a] -= salinity_mix * 0.25
                salinity_delta[b] += salinity_mix
                essence = min(self.places[a].resources["essence"], fluid_flow * 12.0)
                if essence > 0.03:
                    self.places[a].resources["essence"] -= essence
                    self.places[b].resources["essence"] = min(180.0, self.places[b].resources["essence"] + essence * 0.92)
                    events["essence_advection"] += 1
            elif fluid_flow < 0.0:
                salinity_delta[b] -= salinity_mix * 0.25
                salinity_delta[a] += salinity_mix
                essence = min(self.places[b].resources["essence"], -fluid_flow * 12.0)
                if essence > 0.03:
                    self.places[b].resources["essence"] -= essence
                    self.places[a].resources["essence"] = min(180.0, self.places[a].resources["essence"] + essence * 0.92)
                    events["essence_advection"] += 1

        for index, place in enumerate(self.places):
            physics = place.physics
            physics["temperature"] = _clamp(physics.get("temperature", 0.5) + temperature_delta[index], 0.0, 1.45)
            physics["fluid_level"] = _clamp(physics.get("fluid_level", 0.0) + fluid_delta[index], 0.0, 1.25)
            physics["salinity"] = _clamp(physics.get("salinity", 0.0) + salinity_delta[index], 0.0, 1.25)
            physics["humidity"] = _clamp(physics.get("humidity", 0.5), 0.0, 1.15)
            physics["pressure"] = _clamp(
                physics["fluid_level"] * (0.35 + physics["fluid_level"] * 0.65)
                + max(0.0, 0.35 - physics["elevation"]) * 0.18
                + hardship * (physics["fluid_level"] * 0.06 + max(0.0, 0.45 - physics["elevation"]) * 0.04),
                0.0,
                1.7,
            )
            current_values = [abs(edge.current_from(place.id)) * edge.fluid_conductance * edge.permeability for edge in self.edges_from(place.id)]
            physics["current_exposure"] = _clamp(
                sum(current_values) / max(1, len(current_values)) + physics["fluid_level"] * 0.12 + place.volatility * hardship * 0.08,
                0.0,
                1.45,
            )
            physics["light"] = _clamp(place.sun_exposure * (0.35 + season * 0.85), 0.0, 1.25)
            physics["oxygen"] = _clamp(physics.get("oxygen", 0.35) - hardship * (0.002 + physics["pressure"] * 0.003))
            physics["acidity"] = _clamp(physics.get("acidity", 0.10))
            physics["residue_activity"] = _clamp(physics.get("residue_activity", 0.0))
            physics["abrasion"] = _clamp(physics.get("abrasion", 0.0))
            physics["wet_dry_cycle"] = _clamp(physics.get("wet_dry_cycle", 0.0))
            place.terrain["aquatic"] = _clamp(physics["fluid_level"])
            place.terrain["depth"] = _clamp(physics["pressure"] * 0.78)
            place.terrain["salinity"] = _clamp(physics["salinity"])
            place.terrain["humidity"] = _clamp(physics["humidity"])
            place.obstacles["water"] = _clamp(physics["fluid_level"])
            place.obstacles["height"] = _clamp(max(place.obstacles.get("height", 0.0) * 0.96, physics["elevation"] * 0.22))
            place.obstacles["heat"] = _clamp(max(0.0, physics["temperature"] - 0.55) * 1.4 + place.geothermal * 0.12)
            place.resources["thermal"] = max(0.0, min(180.0, place.resources["thermal"]))
            place.resources["mechanical"] = max(0.0, min(180.0, place.resources["mechanical"]))

        signal_transfers: list[tuple[int, Signal]] = []
        for place in self.places:
            current = place.physics.get("current_exposure", 0.0)
            downstream = self.downstream_neighbor(place.id)
            kept: list[Signal] = []
            for signal in place.signals:
                signal.age += 1
                signal.intensity *= max(0.70, 0.87 - current * 0.05)
                if downstream and rng.random() < current * 0.035:
                    signal_transfers.append((downstream[0], signal))
                    events["signal_advection"] += 1
                    continue
                if signal.age < 8 and signal.intensity > 0.015:
                    kept.append(signal)
            place.signals = kept
        for target_id, signal in signal_transfers:
            self.places[target_id].signals.append(signal)
        return dict(events)

    def emit_signal(self, place_id: int, source_id: int, token: int, intensity: float) -> None:
        if 0 <= place_id < len(self.places):
            self.places[place_id].signals.append(Signal(source_id=source_id, token=token % 8, intensity=max(0.0, intensity)))

    def note_patch_depletion(self, place_id: int, rng: Random) -> bool:
        """Start (or extend) a regen-recovery window after a substantial feed.

        Returns True if a window was started/extended. With jitter 0 the window
        is exactly `patch_recovery_ticks`; jitter blends toward a same-mean
        exponential draw (the scrambled control), so timing a return visit
        stops being predictable but mean downtime is unchanged.
        """
        if self.patch_recovery_ticks <= 0 or not 0 <= place_id < len(self.places):
            return False
        base = float(self.patch_recovery_ticks)
        jitter = _clamp(self.patch_recovery_jitter)
        duration = base if jitter <= 0.0 else (1.0 - jitter) * base + jitter * rng.expovariate(1.0 / base)
        until = self.tick + max(1, int(round(duration)))
        place = self.places[place_id]
        place.regen_recovery_until = max(place.regen_recovery_until, until)
        return True

    def to_summary(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "season_length": self.season_length,
            "climate_drift": round(self.climate_drift, 5),
            "environment_harshness": round(self.environment_harshness, 4),
            "places": [place.to_summary() for place in self.places],
            "edges": [edge.to_summary() for edge in self.edges],
        }
