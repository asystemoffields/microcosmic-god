from __future__ import annotations

from collections import Counter
import math
from typing import Any

from .energy import ENERGY_KINDS
from .individuals import Individual
from .world import World


def pool_counts(individuals: dict[int, Individual]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for individual in individuals.values():
        if individual.alive:
            counts[individual.kind] += 1
            if individual.neural:
                counts["neural"] += 1
    counts["total"] = sum(1 for individual in individuals.values() if individual.alive)
    return dict(counts)


def world_energy_summary(world: World) -> dict[str, float]:
    totals = {kind: 0.0 for kind in ENERGY_KINDS}
    locked = 0.0
    for place in world.places:
        for kind in ENERGY_KINDS:
            totals[kind] += place.resources[kind]
        locked += place.sealed_essence
    totals = {kind: round(value, 4) for kind, value in totals.items()}
    totals["sealed_essence"] = round(locked, 4)
    return totals


def world_physics_summary(world: World) -> dict[str, Any]:
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
        "residue_activity",
        "abrasion",
        "wet_dry_cycle",
        "interiority",
        "boundary_permeability",
        "shelter",
        "resource_gradient",
        "terrain_richness",
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
    archetypes: Counter[str] = Counter(place.archetype for place in world.places)
    summary["archetype_counts"] = dict(archetypes)  # type: ignore[assignment]
    return summary


def _round_profile(profile: Counter[str]) -> dict[str, float]:
    return {key: round(value, 6) for key, value in sorted(profile.items()) if value > 0.0}


def success_profile_summary(individuals: dict[int, Individual]) -> dict[str, dict[str, float]]:
    all_totals: Counter[str] = Counter()
    active_totals: Counter[str] = Counter()
    neural_totals: Counter[str] = Counter()
    for individual in individuals.values():
        for key, value in individual.success_profile.items():
            all_totals[key] += value
            if individual.alive:
                active_totals[key] += value
            if individual.neural:
                neural_totals[key] += value
    return {
        "all": _round_profile(all_totals),
        "active": _round_profile(active_totals),
        "neural": _round_profile(neural_totals),
    }


def individual_success_score(individual: Individual) -> float:
    profile = individual.success_profile
    return (
        individual.child_count * 4.0
        + individual.successful_taps * 1.5
        + math.log1p(individual.successful_taps + individual.mistap_count) * 0.5
        + profile.get("prediction_fit", 0.0) * 1.2
        + profile.get("tap", 0.0)
        + individual.energy / max(1.0, individual.storage_limit())
    )


def top_individuals(individuals: dict[int, Individual], limit: int = 10) -> list[dict[str, Any]]:
    active = [individual for individual in individuals.values() if individual.alive]
    active.sort(key=lambda item: (individual_success_score(item), item.child_count, item.successful_taps, item.energy, item.age), reverse=True)
    return [individual.to_summary() for individual in active[:limit]]


def build_debrief(sim: Any, reason: str, elapsed_seconds: float) -> dict[str, Any]:
    counts = pool_counts(sim.individuals)
    energy = world_energy_summary(sim.world)
    physics = world_physics_summary(sim.world)
    likely_causes: list[str] = []
    if counts.get("total", 0) == 0:
        likely_causes.append("full washout")
    if counts.get("neural", 0) == 0:
        likely_causes.append("neural line washout")
    if energy.get("residue_store", 0.0) < len(sim.world.places) * 2.0:
        likely_causes.append("low accessible residue storage")
    if energy.get("essence", 0.0) < len(sim.world.places) * 4.0:
        likely_causes.append("low accessible essence energy")
    if sim.deaths_by_cause:
        likely_causes.append(f"dominant death cause: {sim.deaths_by_cause.most_common(1)[0][0]}")
    if not likely_causes:
        likely_causes.append("run stopped by configured limit, not collapse")

    return {
        "reason": reason,
        "tick": sim.tick,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "pool": counts,
        "births_by_mode": dict(sim.births_by_mode),
        "deaths_by_cause": dict(sim.deaths_by_cause),
        "deaths_by_kind_cause": dict(sim.deaths_by_kind_cause),
        "tap_outcomes": dict(getattr(sim, "tap_outcomes", {})),
        "tap_cue_channel": getattr(sim, "tap_cue_channel", ""),
        "patch_recovery_triggers": int(getattr(sim, "patch_recovery_triggers", 0)),
        "structural_steps": dict(getattr(sim, "structural_steps", {})),
        "success_profile": success_profile_summary(sim.individuals),
        "lines": sim._line_summary() if hasattr(sim, "_line_summary") else {},
        "spawn_attempts": dict(sim.spawn_attempts),
        "spawn_failures": dict(sim.spawn_failures),
        "evolution_policy": sim.optimization.to_summary(),
        "action_counts": dict(sim.action_counts),
        "infeasible_commits": dict(getattr(sim, "infeasible_commits", {})),
        "action_energy_delta": {key: round(value, 6) for key, value in sim.action_energy_delta.items()},
        "action_avg_energy_delta": {
            key: round(sim.action_energy_delta[key] / max(1, sim.action_counts[key]), 6)
            for key in sim.action_counts
        },
        "checkpointing": sim.checkpoints.to_summary(),
        "observer": sim.observer.to_summary() if hasattr(sim, "observer") else {},
        "world_energy": energy,
        "world_physics": physics,
        "physics_events": dict(sim.physics_events),
        "climate_drift": round(sim.world.climate_drift, 6),
        "top_active_individuals": top_individuals(sim.individuals),
        "likely_causes": likely_causes,
        "last_aggregates": sim.aggregate_history[-10:],
        "interventions_applied": sim.interventions_applied,
    }
