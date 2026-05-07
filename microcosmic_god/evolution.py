from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .brain import TinyBrain
from .config import RunConfig
from .genome import Genome
from .organisms import Organism


@dataclass(slots=True)
class EvolutionDecision:
    plan: "OffspringPlan | None" = None
    failure: str | None = None
    energy_penalty: float = 0.0


@dataclass(slots=True)
class OffspringPlan:
    operator: str
    child_kind: str
    child_genome: Genome
    location: int
    child_energy: float
    generation: int
    parent_ids: tuple[int, ...]
    brain_template: TinyBrain | None
    parent_costs: dict[int, float]


class EvolutionEngine:
    """Variation operators for producing new organisms.

    The simulation handles world constraints such as local capacity and death.
    This class owns the search operators: which inherited material is copied,
    mutated, recombined, or eventually selected by non-biological farm policies.
    """

    def __init__(self, rng: Random, config: RunConfig):
        self.rng = rng
        self.config = config

    def clone_mutate_reserve_threshold(self, parent: Organism) -> float:
        return parent.clone_mutate_energy_threshold()

    def recombine_reserve_threshold(self, parent: Organism) -> float:
        return parent.recombine_energy_threshold() * (0.34 + parent.genome.offspring_investment * 0.10)

    def compatible_for_recombine(self, a: Organism, b: Organism) -> bool:
        return a.genome.distance(b.genome) < 0.50

    def plan_clone_mutate(self, parent: Organism) -> EvolutionDecision:
        threshold = self.clone_mutate_reserve_threshold(parent)
        strain = self._complexity_strain(parent)
        cost = threshold * (0.32 + parent.genome.offspring_investment * 0.28) * (1.0 + strain * 0.18)
        reserve = max(threshold, cost * 1.04)
        if parent.energy < reserve:
            return EvolutionDecision(failure="clone_mutate_low_energy", energy_penalty=0.03 + strain * 0.02)

        mutation_strength = 0.055 + strain * 0.025
        child_genome = parent.genome.mutate(self.rng, strength=mutation_strength)
        child_energy = cost * max(0.32, 0.42 - strain * 0.035)
        return EvolutionDecision(
            plan=OffspringPlan(
                operator="clone_mutate",
                child_kind=parent.kind,
                child_genome=child_genome,
                location=parent.location,
                child_energy=child_energy,
                generation=parent.generation + 1,
                parent_ids=(parent.id,),
                brain_template=self._inherit_template_clone_mutate(parent, child_genome, strain),
                parent_costs={parent.id: cost},
            )
        )

    def plan_recombine(self, a: Organism, b: Organism) -> EvolutionDecision:
        if not self.compatible_for_recombine(a, b):
            return EvolutionDecision(failure="recombine_incompatible", energy_penalty=0.0)
        cost_a = self._recombine_cost(a)
        cost_b = self._recombine_cost(b)
        if a.energy < cost_a or b.energy < cost_b:
            return EvolutionDecision(failure="recombine_cost_energy", energy_penalty=0.0)

        child_genome = Genome.recombine(self.rng, a.genome, b.genome)
        child_genome.developmental_complexity = min(1.0, child_genome.developmental_complexity + self.rng.uniform(0.00, 0.04))
        child_energy = 4.0 + (cost_a + cost_b) * 0.85
        return EvolutionDecision(
            plan=OffspringPlan(
                operator="recombine",
                child_kind="agent",
                child_genome=child_genome,
                location=a.location,
                child_energy=child_energy,
                generation=max(a.generation, b.generation) + 1,
                parent_ids=(a.id, b.id),
                brain_template=self._inherit_template_recombine(a, b, child_genome),
                parent_costs={a.id: cost_a, b.id: cost_b},
            )
        )

    def _complexity_strain(self, parent: Organism) -> float:
        soft_limit = getattr(self.config, "clone_complexity_soft_limit", self.config.asexual_complexity_ceiling)
        return max(0.0, parent.genome.complexity() - soft_limit)

    def _recombine_cost(self, parent: Organism) -> float:
        return parent.recombine_energy_threshold() * (0.035 + parent.genome.offspring_investment * 0.050)

    def _inherit_template_clone_mutate(self, parent: Organism, child_genome: Genome, strain: float) -> TinyBrain | None:
        if parent.brain_template is None or child_genome.neural_budget < 2.0:
            return None
        target_hidden = int(round(child_genome.neural_budget))
        mutation_scale = 0.025 + child_genome.mutation_rate * 0.25 + strain * 0.010
        # When child genome calls for a different brain size, clone_for_offspring
        # resizes the inherited template instead of returning None - the parent's
        # learned function is preserved across size changes.
        return parent.brain_template.clone_for_offspring(
            self.rng, mutation_scale=mutation_scale, target_hidden_size=target_hidden
        )

    def _inherit_template_recombine(self, a: Organism, b: Organism, child_genome: Genome) -> TinyBrain | None:
        target_hidden = int(round(child_genome.neural_budget))
        if target_hidden < 2:
            return None
        # Prefer parents whose template size already matches; fall back to either
        # parent (size will be reconciled via resize during cloning).
        exact = [parent.brain_template for parent in (a, b) if parent.brain_template and parent.brain_template.hidden_size == target_hidden]
        any_template = [parent.brain_template for parent in (a, b) if parent.brain_template]
        templates = exact or any_template
        if not templates:
            return None
        chosen = self.rng.choice(templates)
        return chosen.clone_for_offspring(
            self.rng,
            mutation_scale=0.035 + child_genome.mutation_rate * 0.20,
            target_hidden_size=target_hidden,
        )

    def to_summary(self) -> dict[str, object]:
        return {
            "selection_frame": "make_more_like_effective_operators_and_learners",
            "operators": ["clone_mutate", "recombine"],
            "clone_complexity_soft_limit": getattr(self.config, "clone_complexity_soft_limit", self.config.asexual_complexity_ceiling),
            "recombine_genome_distance_limit": 0.50,
            "sealed_run_policy": "operators are triggered by in-world action and interaction",
            "future_farm_policy": "archive-driven selection can add non-biological operators without changing world physics",
        }
