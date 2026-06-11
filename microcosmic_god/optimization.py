from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .brain import TinyController
from .config import RunConfig
from .params import ParamVector
from .organisms import Individual


@dataclass(slots=True)
class OptimizerDecision:
    plan: "OffspringPlan | None" = None
    failure: str | None = None
    energy_penalty: float = 0.0


@dataclass(slots=True)
class OffspringPlan:
    operator: str
    child_kind: str
    child_genome: ParamVector
    location: int
    child_energy: float
    cycle: int
    parent_ids: tuple[int, ...]
    controller_template: TinyController | None
    parent_costs: dict[int, float]


class Optimizer:
    """Variation operators for producing new individuals.

    The simulation handles world constraints such as local capacity and deactivation.
    This class owns the search operators: which inherited material is copied,
    mutated, recombined, or eventually selected by non-biological farm policies.
    """

    def __init__(self, rng: Random, config: RunConfig):
        self.rng = rng
        self.config = config

    def clone_mutate_reserve_threshold(self, parent: Individual) -> float:
        return parent.clone_mutate_energy_threshold()

    def recombine_reserve_threshold(self, parent: Individual) -> float:
        return parent.recombine_energy_threshold() * (0.34 + parent.params.offspring_investment * 0.10)

    def compatible_for_recombine(self, a: Individual, b: Individual) -> bool:
        return a.params.distance(b.params) < 0.50

    def plan_clone_mutate(self, parent: Individual) -> OptimizerDecision:
        threshold = self.clone_mutate_reserve_threshold(parent)
        strain = self._complexity_strain(parent)
        cost = threshold * (0.32 + parent.params.offspring_investment * 0.28) * (1.0 + strain * 0.18)
        reserve = max(threshold, cost * 1.04)
        if parent.energy < reserve:
            return OptimizerDecision(failure="clone_mutate_low_energy", energy_penalty=0.03 + strain * 0.02)

        mutation_strength = 0.055 + strain * 0.025
        child_genome = parent.params.perturb(self.rng, strength=mutation_strength)
        child_energy = cost * max(0.32, 0.42 - strain * 0.035)
        return OptimizerDecision(
            plan=OffspringPlan(
                operator="clone_mutate",
                child_kind=parent.kind,
                child_genome=child_genome,
                location=parent.location,
                child_energy=child_energy,
                cycle=parent.cycle + 1,
                parent_ids=(parent.id,),
                controller_template=self._inherit_template_clone_mutate(parent, child_genome, strain),
                parent_costs={parent.id: cost},
            )
        )

    def plan_recombine(self, a: Individual, b: Individual) -> OptimizerDecision:
        if not self.compatible_for_recombine(a, b):
            return OptimizerDecision(failure="recombine_incompatible", energy_penalty=0.0)
        cost_a = self._recombine_cost(a)
        cost_b = self._recombine_cost(b)
        if a.energy < cost_a or b.energy < cost_b:
            return OptimizerDecision(failure="recombine_cost_energy", energy_penalty=0.0)

        child_genome = ParamVector.combine(self.rng, a.params, b.params)
        child_genome.developmental_complexity = min(1.0, child_genome.developmental_complexity + self.rng.uniform(0.00, 0.04))
        child_energy = 4.0 + (cost_a + cost_b) * 0.85
        return OptimizerDecision(
            plan=OffspringPlan(
                operator="recombine",
                child_kind="agent",
                child_genome=child_genome,
                location=a.location,
                child_energy=child_energy,
                cycle=max(a.cycle, b.cycle) + 1,
                parent_ids=(a.id, b.id),
                controller_template=self._inherit_template_recombine(a, b, child_genome),
                parent_costs={a.id: cost_a, b.id: cost_b},
            )
        )

    def _complexity_strain(self, parent: Individual) -> float:
        soft_limit = getattr(self.config, "clone_complexity_soft_limit", self.config.asexual_complexity_ceiling)
        return max(0.0, parent.params.complexity() - soft_limit)

    def _recombine_cost(self, parent: Individual) -> float:
        return parent.recombine_energy_threshold() * (0.035 + parent.params.offspring_investment * 0.050)

    def _inherit_template_clone_mutate(self, parent: Individual, child_genome: ParamVector, strain: float) -> TinyController | None:
        if parent.controller_template is None or child_genome.neural_budget < 2.0:
            return None
        target_hidden = int(round(child_genome.neural_budget))
        mutation_scale = 0.025 + child_genome.perturbation_rate * 0.25 + strain * 0.010
        # When child params calls for a different controller size, clone_for_offspring
        # resizes the inherited template instead of returning None - the parent's
        # learned function is preserved across size changes.
        kwargs = {}
        if hasattr(parent.controller_template, "blocks"):
            kwargs["structural_rate"] = self.config.structural_mutation_rate
        child_template = parent.controller_template.clone_for_offspring(
            self.rng, mutation_scale=mutation_scale, target_hidden_size=target_hidden, **kwargs
        )
        self._sync_genome_to_structure(child_genome, child_template)
        return child_template

    @staticmethod
    def _sync_genome_to_structure(child_genome: ParamVector, template: TinyController | None) -> None:
        # Modular controllers own their capacity structurally (blocks can be
        # added/duplicated/pruned at cloning); the genome budget FOLLOWS the
        # structure so upkeep economics price actual capacity. The legacy
        # arrangement (budget leads, controller resizes) is unchanged.
        if template is not None and getattr(template, "capacity", None) is not None:
            child_genome.neural_budget = float(template.capacity)

    def _inherit_template_recombine(self, a: Individual, b: Individual, child_genome: ParamVector) -> TinyController | None:
        target_hidden = int(round(child_genome.neural_budget))
        if target_hidden < 2:
            return None
        # Prefer parents whose template size already matches; fall back to either
        # parent (size will be reconciled via resize during cloning).
        exact = [parent.controller_template for parent in (a, b) if parent.controller_template and parent.controller_template.hidden_size == target_hidden]
        any_template = [parent.controller_template for parent in (a, b) if parent.controller_template]
        templates = exact or any_template
        if not templates:
            return None
        chosen = self.rng.choice(templates)
        kwargs = {}
        if hasattr(chosen, "blocks"):
            kwargs["structural_rate"] = self.config.structural_mutation_rate
        child_template = chosen.clone_for_offspring(
            self.rng,
            mutation_scale=0.035 + child_genome.perturbation_rate * 0.20,
            target_hidden_size=target_hidden,
            **kwargs,
        )
        self._sync_genome_to_structure(child_genome, child_template)
        return child_template

    def to_summary(self) -> dict[str, object]:
        return {
            "selection_frame": "make_more_like_effective_operators_and_learners",
            "operators": ["clone_mutate", "recombine"],
            "clone_complexity_soft_limit": getattr(self.config, "clone_complexity_soft_limit", self.config.asexual_complexity_ceiling),
            "recombine_genome_distance_limit": 0.50,
            "sealed_run_policy": "operators are triggered by in-world action and interaction",
            "future_farm_policy": "archive-driven ranking can add non-biological operators without changing world physics",
        }
