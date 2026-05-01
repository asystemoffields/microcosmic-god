from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Any

from .config import RunConfig
from .energy import ENERGY_KINDS, MATERIALS, Structure, blank_energy


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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
    physics: dict[str, float]
    structures: list[Structure] = field(default_factory=list)
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
            "physics": {k: round(v, 4) for k, v in self.physics.items()},
            "structures": [structure.to_dict() for structure in self.structures[-8:]],
            "marks": [mark.to_dict() for mark in self.marks[-8:]],
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
            elevation = _clamp(rng.random() ** 1.15)
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
            depth = aquatic * rng.uniform(0.05, 1.0)
            salinity = aquatic * rng.random()
            humidity = min(1.0, water * 0.75 + sun * 0.10 + rng.random() * 0.15)
            habitat = {
                "aquatic": aquatic,
                "depth": depth,
                "salinity": salinity,
                "humidity": humidity,
            }
            temperature = _clamp(0.08 + sun * 0.33 + geo * 0.42 + rng.gauss(0.0, 0.04), 0.0, 1.25)
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
                    physics=physics,
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

        return cls(places=places, season_length=config.season_length, edges=edges, edge_lookup=edge_lookup, edge_adjacency=edge_adjacency)

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

    def _apply_structures(self, place: Place, events: Counter[str]) -> None:
        if not place.structures:
            place.physics["interiority"] = max(0.0, place.physics.get("interiority", 0.0) * 0.98)
            place.physics["boundary_permeability"] = max(0.0, place.physics.get("boundary_permeability", 0.0) * 0.98)
            place.physics["shelter"] = max(0.0, place.physics.get("shelter", 0.0) * 0.98)
            return
        physics = place.physics
        edges = self.edges_from(place.id)
        edge_current = max((abs(edge.current_from(place.id)) for edge in edges), default=0.0)
        edge_slope = max((abs(edge.slope_from(place.id)) for edge in edges), default=0.0)
        flow_gradient = _clamp(physics.get("fluid_level", 0.0) * physics.get("current_exposure", 0.0) + edge_current * 0.35 + edge_slope * physics.get("fluid_level", 0.0) * 0.08)
        interiority = 0.0
        boundary_permeability = 0.0
        shelter = 0.0
        kept: list[Structure] = []
        for structure in place.structures:
            structure.age += 1
            scale = min(1.0, math.log1p(max(1, structure.scale)) / math.log(14.0))
            enclose = structure.capabilities.get("enclose", 0.0)
            permeable = structure.capabilities.get("permeable", 0.0)
            channel = structure.capabilities.get("channel", 0.0)
            gradient_harvest = structure.capabilities.get("gradient_harvest", 0.0)
            conduct = structure.capabilities.get("conduct", 0.0)
            storage = structure.capabilities.get("energy_storage", 0.0)
            filter_cap = structure.capabilities.get("filter", 0.0)
            reaction_surface = structure.capabilities.get("reaction_surface", 0.0)
            support = structure.capabilities.get("support", 0.0)
            anchor = structure.capabilities.get("anchor", 0.0)
            interiority = max(interiority, enclose * (0.55 + scale * 0.45))
            boundary_permeability = max(boundary_permeability, permeable)
            shelter = max(shelter, structure.capabilities.get("shelter", 0.0))

            if gradient_harvest > 0.08 and flow_gradient > 0.03:
                mechanical = gradient_harvest * flow_gradient * (0.06 + scale * 0.10)
                place.resources["mechanical"] = min(180.0, place.resources["mechanical"] + mechanical)
                events["structure_gradient_harvest"] += 1
                if conduct > 0.08:
                    electrical = mechanical * conduct * (0.10 + storage * 0.10)
                    place.resources["electrical"] = min(180.0, place.resources["electrical"] + electrical)
                    events["structure_gradient_conversion"] += 1

            if filter_cap > 0.08 and (flow_gradient > 0.02 or permeable > 0.25):
                captured = filter_cap * (flow_gradient + permeable * 0.15) * (0.03 + scale * 0.04)
                place.resources["chemical"] = min(180.0, place.resources["chemical"] + captured)
                place.resources["biological_storage"] = min(180.0, place.resources["biological_storage"] + captured * 0.35)
                events["structure_filtration"] += 1

            if reaction_surface > 0.08:
                heat_gradient = abs(physics.get("temperature", 0.5) - 0.48) + place.geothermal * 0.10
                reaction = reaction_surface * heat_gradient * (0.01 + scale * 0.025)
                place.resources["chemical"] = min(180.0, place.resources["chemical"] + reaction)
                events["structure_reaction_surface"] += 1

            if channel > 0.05:
                place.resources["mechanical"] = min(180.0, place.resources["mechanical"] + channel * flow_gradient * 0.025)

            if enclose > 0.05:
                exchange_block = enclose * max(0.0, 1.0 - permeable * 0.70)
                physics["humidity"] = _clamp(physics.get("humidity", 0.5) + exchange_block * 0.002 - max(0.0, physics.get("temperature", 0.5) - 0.65) * permeable * 0.001)
                physics["current_exposure"] = _clamp(physics.get("current_exposure", 0.0) * (1.0 - min(0.12, exchange_block * anchor * 0.035)))

            wear = 0.012 + physics.get("current_exposure", 0.0) * 0.018 + physics.get("pressure", 0.0) * 0.010 + max(0.0, physics.get("temperature", 0.5) - 0.75) * 0.030
            structure.durability -= max(0.002, wear * (1.0 - min(0.65, support * 0.40 + anchor * 0.25)))
            if structure.durability > 0.0:
                kept.append(structure)
            else:
                for name, qty in structure.components.items():
                    if qty > 0:
                        place.materials[name] = min(99, place.materials.get(name, 0) + max(1, qty // 3))
                events["structure_decay"] += 1
        place.structures = kept[-16:]
        physics["interiority"] = _clamp(interiority)
        physics["boundary_permeability"] = _clamp(boundary_permeability)
        physics["shelter"] = _clamp(shelter)

    def update_environment(self, rng: Random) -> dict[str, int]:
        events: Counter[str] = Counter()
        self.tick += 1
        season = 0.5 + 0.5 * math.sin(2.0 * math.pi * self.tick / max(2, self.season_length))
        if self.tick % max(50, self.season_length // 8) == 0:
            self.climate_drift = max(-0.35, min(0.35, self.climate_drift + rng.gauss(0.0, 0.018)))

        old_temperature = [place.physics.get("temperature", 0.5) for place in self.places]
        old_fluid = [place.physics.get("fluid_level", 0.0) for place in self.places]
        old_salinity = [place.physics.get("salinity", 0.0) for place in self.places]
        temperature_delta = [0.0 for _ in self.places]
        fluid_delta = [0.0 for _ in self.places]
        salinity_delta = [0.0 for _ in self.places]

        for place in self.places:
            physics = place.physics
            weather = 1.0 + rng.gauss(0.0, place.volatility * 0.02)
            radiant_target = (18.0 + place.sun_exposure * 85.0) * (0.35 + season * 0.85 + self.climate_drift * 0.25)
            place.resources["radiant"] += (radiant_target * weather - place.resources["radiant"]) * 0.08
            place.resources["chemical"] += 0.010 + place.water_flow * 0.030 + place.mineral_richness * 0.006
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
            physics["temperature"] += (temperature_target - physics.get("temperature", 0.5)) * (0.012 + (1.0 - thermal_mass) * 0.040)
            evaporation = max(0.0, physics["temperature"] - 0.55) * physics.get("fluid_level", 0.0) * 0.004
            fluid_target = _clamp(place.water_flow * (0.45 + season * 0.30) + rng.gauss(0.0, place.volatility * 0.004))
            physics["fluid_level"] += (fluid_target - physics.get("fluid_level", 0.0)) * 0.006 - evaporation
            physics["humidity"] += (physics["fluid_level"] * 0.70 + evaporation * 10.0 - physics.get("humidity", 0.5)) * 0.020
            place.resources["thermal"] += ((physics["temperature"] * 150.0 + place.geothermal * 30.0) - place.resources["thermal"]) * 0.012
            place.resources["mechanical"] += ((place.water_flow * 25.0 + physics.get("current_exposure", 0.0) * 75.0 + place.volatility * 18.0) - place.resources["mechanical"]) * 0.010
            place.resources["electrical"] += place.mineral_richness * place.volatility * 0.006
            place.resources["biological_storage"] *= 0.9994
            place.locked_chemical += place.mineral_richness * 0.003

            if rng.random() < 0.002 + place.water_flow * 0.002:
                material = rng.choice(tuple(MATERIALS.keys()))
                place.materials[material] = min(99, place.materials.get(material, 0) + 1)

            for kind in ENERGY_KINDS:
                place.resources[kind] = max(0.0, min(180.0, place.resources[kind]))
            place.locked_chemical = max(0.0, min(260.0, place.locked_chemical))

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
                chemical = min(self.places[a].resources["chemical"], fluid_flow * 12.0)
                if chemical > 0.03:
                    self.places[a].resources["chemical"] -= chemical
                    self.places[b].resources["chemical"] = min(180.0, self.places[b].resources["chemical"] + chemical * 0.92)
                    events["chemical_advection"] += 1
            elif fluid_flow < 0.0:
                salinity_delta[b] -= salinity_mix * 0.25
                salinity_delta[a] += salinity_mix
                chemical = min(self.places[b].resources["chemical"], -fluid_flow * 12.0)
                if chemical > 0.03:
                    self.places[b].resources["chemical"] -= chemical
                    self.places[a].resources["chemical"] = min(180.0, self.places[a].resources["chemical"] + chemical * 0.92)
                    events["chemical_advection"] += 1

        for index, place in enumerate(self.places):
            physics = place.physics
            physics["temperature"] = _clamp(physics.get("temperature", 0.5) + temperature_delta[index], 0.0, 1.45)
            physics["fluid_level"] = _clamp(physics.get("fluid_level", 0.0) + fluid_delta[index], 0.0, 1.25)
            physics["salinity"] = _clamp(physics.get("salinity", 0.0) + salinity_delta[index], 0.0, 1.25)
            physics["humidity"] = _clamp(physics.get("humidity", 0.5), 0.0, 1.15)
            physics["pressure"] = _clamp(physics["fluid_level"] * (0.35 + physics["fluid_level"] * 0.65) + max(0.0, 0.35 - physics["elevation"]) * 0.18, 0.0, 1.6)
            current_values = [abs(edge.current_from(place.id)) * edge.fluid_conductance * edge.permeability for edge in self.edges_from(place.id)]
            physics["current_exposure"] = _clamp(sum(current_values) / max(1, len(current_values)) + physics["fluid_level"] * 0.12, 0.0, 1.3)
            physics["light"] = _clamp(place.sun_exposure * (0.35 + season * 0.85), 0.0, 1.25)
            place.habitat["aquatic"] = _clamp(physics["fluid_level"])
            place.habitat["depth"] = _clamp(physics["pressure"] * 0.78)
            place.habitat["salinity"] = _clamp(physics["salinity"])
            place.habitat["humidity"] = _clamp(physics["humidity"])
            place.obstacles["water"] = _clamp(physics["fluid_level"])
            place.obstacles["height"] = _clamp(max(place.obstacles.get("height", 0.0) * 0.96, physics["elevation"] * 0.22))
            place.obstacles["heat"] = _clamp(max(0.0, physics["temperature"] - 0.55) * 1.4 + place.geothermal * 0.12)
            place.resources["thermal"] = max(0.0, min(180.0, place.resources["thermal"]))
            place.resources["mechanical"] = max(0.0, min(180.0, place.resources["mechanical"]))

        for place in self.places:
            self._apply_structures(place, events)

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

            kept_marks: list[Mark] = []
            for mark in place.marks:
                mark.age += 1
                mark.intensity *= 0.997
                mark.durability -= 0.10 + place.volatility * 0.06 + place.physics.get("fluid_level", 0.0) * 0.035 + current * 0.020 + place.physics.get("temperature", 0.5) * 0.010
                if mark.durability > 0.0 and mark.intensity > 0.025:
                    kept_marks.append(mark)
                else:
                    events["mark_eroded"] += 1
            if len(kept_marks) > 32:
                kept_marks = sorted(kept_marks, key=lambda item: (item.intensity * item.durability, -item.age), reverse=True)[:32]
            place.marks = kept_marks
        for target_id, signal in signal_transfers:
            self.places[target_id].signals.append(signal)
        return dict(events)

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
            "edges": [edge.to_summary() for edge in self.edges],
        }
