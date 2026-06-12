from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .controller import TinyController
from .config import RunConfig
from .params import ParamVector
from .individuals import Individual


@dataclass(slots=True)
class OptimizerDecision:
    plan: "ChildPlan | None" = None
    failure: str | None = None
    energy_penalty: float = 0.0


@dataclass(slots=True)
class ChildPlan:
    operator: str
    child_kind: str
    child_params: ParamVector
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
    perturbed, combined, or eventually selected by non-residue farm policies.
    """

    def __init__(self, rng: Random, config: RunConfig):
        self.rng = rng
        self.config = config

    def clone_perturb_reserve_threshold(self, parent: Individual) -> float:
        return parent.clone_perturb_energy_threshold()

    def combine_reserve_threshold(self, parent: Individual) -> float:
        return parent.combine_energy_threshold() * (0.34 + parent.params.child_investment * 0.10)

    def compatible_for_combine(self, a: Individual, b: Individual) -> bool:
        return a.params.distance(b.params) < 0.50

    def plan_clone_perturb(self, parent: Individual) -> OptimizerDecision:
        threshold = self.clone_perturb_reserve_threshold(parent)
        strain = self._complexity_strain(parent)
        cost = threshold * (0.32 + parent.params.child_investment * 0.28) * (1.0 + strain * 0.18)
        reserve = max(threshold, cost * 1.04)
        if parent.energy < reserve:
            return OptimizerDecision(failure="clone_perturb_low_energy", energy_penalty=0.03 + strain * 0.02)

        perturbation_strength = 0.055 + strain * 0.025
        child_params = parent.params.perturb(self.rng, strength=perturbation_strength)
        child_energy = cost * max(0.32, 0.42 - strain * 0.035)
        return OptimizerDecision(
            plan=ChildPlan(
                operator="clone_perturb",
                child_kind=parent.kind,
                child_params=child_params,
                location=parent.location,
                child_energy=child_energy,
                cycle=parent.cycle + 1,
                parent_ids=(parent.id,),
                controller_template=self._inherit_template_clone_perturb(parent, child_params, strain),
                parent_costs={parent.id: cost},
            )
        )

    def plan_combine(self, a: Individual, b: Individual) -> OptimizerDecision:
        if not self.compatible_for_combine(a, b):
            return OptimizerDecision(failure="combine_incompatible", energy_penalty=0.0)
        cost_a = self._combine_cost(a)
        cost_b = self._combine_cost(b)
        if a.energy < cost_a or b.energy < cost_b:
            return OptimizerDecision(failure="combine_cost_energy", energy_penalty=0.0)

        child_params = ParamVector.combine(self.rng, a.params, b.params)
        child_params.developmental_complexity = min(1.0, child_params.developmental_complexity + self.rng.uniform(0.00, 0.04))
        child_energy = 4.0 + (cost_a + cost_b) * 0.85
        return OptimizerDecision(
            plan=ChildPlan(
                operator="combine",
                child_kind="agent",
                child_params=child_params,
                location=a.location,
                child_energy=child_energy,
                cycle=max(a.cycle, b.cycle) + 1,
                parent_ids=(a.id, b.id),
                controller_template=self._inherit_template_combine(a, b, child_params),
                parent_costs={a.id: cost_a, b.id: cost_b},
            )
        )

    def _complexity_strain(self, parent: Individual) -> float:
        soft_limit = getattr(self.config, "clone_complexity_soft_limit", self.config.solo_complexity_ceiling)
        return max(0.0, parent.params.complexity() - soft_limit)

    def _combine_cost(self, parent: Individual) -> float:
        return parent.combine_energy_threshold() * (0.035 + parent.params.child_investment * 0.050)

    def _inherit_template_clone_perturb(self, parent: Individual, child_params: ParamVector, strain: float) -> TinyController | None:
        if parent.controller_template is None or child_params.neural_budget < 2.0:
            return None
        target_hidden = int(round(child_params.neural_budget))
        perturbation_scale = 0.025 + child_params.perturbation_rate * 0.25 + strain * 0.010
        # When child params calls for a different controller size, clone_for_child
        # resizes the inherited template instead of returning None - the parent's
        # learned function is preserved across size changes.
        kwargs = {}
        if hasattr(parent.controller_template, "blocks"):
            kwargs["structural_rate"] = self.config.structural_perturbation_rate
        child_template = parent.controller_template.clone_for_child(
            self.rng, perturbation_scale=perturbation_scale, target_hidden_size=target_hidden, **kwargs
        )
        self._sync_params_to_structure(child_params, child_template)
        return child_template

    @staticmethod
    def _sync_params_to_structure(child_params: ParamVector, template: TinyController | None) -> None:
        # Modular controllers own their capacity structurally (blocks can be
        # added/duplicated/pruned at cloning); the params budget FOLLOWS the
        # structure so upkeep economics price actual capacity. The legacy
        # arrangement (budget leads, controller resizes) is unchanged.
        if template is not None and getattr(template, "capacity", None) is not None:
            child_params.neural_budget = float(template.capacity)

    def _inherit_template_combine(self, a: Individual, b: Individual, child_params: ParamVector) -> TinyController | None:
        target_hidden = int(round(child_params.neural_budget))
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
            kwargs["structural_rate"] = self.config.structural_perturbation_rate
        child_template = chosen.clone_for_child(
            self.rng,
            perturbation_scale=0.035 + child_params.perturbation_rate * 0.20,
            target_hidden_size=target_hidden,
            **kwargs,
        )
        self._sync_params_to_structure(child_params, child_template)
        return child_template

    def to_summary(self) -> dict[str, object]:
        return {
            "selection_frame": "make_more_like_effective_operators_and_learners",
            "operators": ["clone_perturb", "combine"],
            "clone_complexity_soft_limit": getattr(self.config, "clone_complexity_soft_limit", self.config.solo_complexity_ceiling),
            "combine_param_distance_limit": 0.50,
            "sealed_run_policy": "operators are triggered by in-world action and interaction",
            "future_farm_policy": "archive-driven ranking can add non-residue operators without changing world physics",
        }
