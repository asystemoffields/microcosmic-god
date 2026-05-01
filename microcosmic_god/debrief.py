from __future__ import annotations

from collections import Counter
from typing import Any

from .energy import ENERGY_KINDS
from .organisms import Organism
from .world import World


def population_counts(organisms: dict[int, Organism]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for organism in organisms.values():
        if organism.alive:
            counts[organism.kind] += 1
            if organism.neural:
                counts["neural"] += 1
    counts["total"] = sum(1 for organism in organisms.values() if organism.alive)
    return dict(counts)


def world_energy_summary(world: World) -> dict[str, float]:
    totals = {kind: 0.0 for kind in ENERGY_KINDS}
    locked = 0.0
    for place in world.places:
        for kind in ENERGY_KINDS:
            totals[kind] += place.resources[kind]
        locked += place.locked_chemical
    totals = {kind: round(value, 4) for kind, value in totals.items()}
    totals["locked_chemical"] = round(locked, 4)
    return totals


def world_physics_summary(world: World) -> dict[str, float]:
    keys = (
        "temperature",
        "fluid_level",
        "pressure",
        "humidity",
        "salinity",
        "elevation",
        "current_exposure",
        "oxygen",
        "acidity",
        "biological_activity",
        "abrasion",
        "wet_dry_cycle",
        "interiority",
        "boundary_permeability",
        "shelter",
    )
    summary: dict[str, float] = {}
    for key in keys:
        values = [place.physics.get(key, 0.0) for place in world.places]
        summary[f"avg_{key}"] = round(sum(values) / max(1, len(values)), 5)
        summary[f"max_{key}"] = round(max(values, default=0.0), 5)
    edge_currents = [abs(edge.current) for edge in world.edges]
    edge_slopes = [abs(edge.slope) for edge in world.edges]
    summary["avg_edge_current"] = round(sum(edge_currents) / max(1, len(edge_currents)), 5)
    summary["max_edge_current"] = round(max(edge_currents, default=0.0), 5)
    summary["avg_edge_slope"] = round(sum(edge_slopes) / max(1, len(edge_slopes)), 5)
    summary["max_edge_slope"] = round(max(edge_slopes, default=0.0), 5)
    structures = [structure for place in world.places for structure in place.structures]
    summary["structure_count"] = float(len(structures))
    summary["avg_structure_scale"] = round(sum(structure.scale for structure in structures) / max(1, len(structures)), 5)
    summary["max_structure_scale"] = round(max((structure.scale for structure in structures), default=0.0), 5)
    return summary


def top_organisms(organisms: dict[int, Organism], limit: int = 10) -> list[dict[str, Any]]:
    living = [organism for organism in organisms.values() if organism.alive]
    living.sort(key=lambda item: (item.offspring_count, item.successful_tools, item.energy, item.age), reverse=True)
    return [organism.to_summary() for organism in living[:limit]]


def build_debrief(sim: Any, reason: str, elapsed_seconds: float) -> dict[str, Any]:
    counts = population_counts(sim.organisms)
    energy = world_energy_summary(sim.world)
    physics = world_physics_summary(sim.world)
    likely_causes: list[str] = []
    if counts.get("total", 0) == 0:
        likely_causes.append("full extinction")
    if counts.get("neural", 0) == 0:
        likely_causes.append("neural lineage extinction")
    if energy.get("biological_storage", 0.0) < len(sim.world.places) * 2.0:
        likely_causes.append("low accessible biological storage")
    if energy.get("chemical", 0.0) < len(sim.world.places) * 4.0:
        likely_causes.append("low accessible chemical energy")
    if sim.deaths_by_cause:
        likely_causes.append(f"dominant death cause: {sim.deaths_by_cause.most_common(1)[0][0]}")
    if not likely_causes:
        likely_causes.append("run stopped by configured limit, not collapse")

    return {
        "reason": reason,
        "tick": sim.tick,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "population": counts,
        "births_by_mode": dict(sim.births_by_mode),
        "deaths_by_cause": dict(sim.deaths_by_cause),
        "deaths_by_kind_cause": dict(sim.deaths_by_kind_cause),
        "tool_successes": dict(sim.tool_successes),
        "marks_created": dict(sim.marks_created),
        "artifacts_created": dict(sim.artifacts_created),
        "artifacts_broken": dict(sim.artifacts_broken),
        "structures_built": dict(getattr(sim, "structures_built", {})),
        "structures_extended": dict(getattr(sim, "structures_extended", {})),
        "reproduction_attempts": dict(sim.reproduction_attempts),
        "reproduction_failures": dict(sim.reproduction_failures),
        "action_counts": dict(sim.action_counts),
        "action_energy_delta": {key: round(value, 6) for key, value in sim.action_energy_delta.items()},
        "action_avg_energy_delta": {
            key: round(sim.action_energy_delta[key] / max(1, sim.action_counts[key]), 6)
            for key in sim.action_counts
        },
        "checkpointing": sim.checkpoints.to_summary(),
        "world_energy": energy,
        "world_physics": physics,
        "physics_events": dict(sim.physics_events),
        "climate_drift": round(sim.world.climate_drift, 6),
        "top_living_organisms": top_organisms(sim.organisms),
        "likely_causes": likely_causes,
        "last_aggregates": sim.aggregate_history[-10:],
        "interventions_applied": sim.interventions_applied,
    }
