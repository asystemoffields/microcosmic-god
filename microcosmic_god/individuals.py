from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from .controller import PREDICTION_HEADS, TinyController
from .params import ParamVector

SIGNAL_VALUE_SIZE = 8
RECENT_TRACE_LABELS = (
    "action",
    "energy_delta",
    "health_delta",
    "damage",
    "prediction_error",
    "spawning",
    "social",
    "tap",
)
EVENT_MEMORY_LABELS = (
    "energy_gain",
    "energy_loss",
    "health_gain",
    "damage",
    "spawning",
    "social",
    "tap",
    "surprise",
)
SUCCESS_PROFILE_LABELS = (
    "energy_gain",
    "prediction_fit",
    "tap",
    "spawning",
)
RECENT_TRACE_SIZE = len(RECENT_TRACE_LABELS)
PREDICTION_ERROR_SIZE = len(PREDICTION_HEADS)
EVENT_MEMORY_SIZE = len(EVENT_MEMORY_LABELS)
# Era 2 base: era 1's 42 minus the inventory-ratio and skill-breadth dims
# (their subsystems were cut; docs/ENV_AXIS_REVIEW.md).
OBSERVATION_SIZE = 40 + RECENT_TRACE_SIZE + PREDICTION_ERROR_SIZE + EVENT_MEMORY_SIZE + SIGNAL_VALUE_SIZE

ACTIONS = (
    "rest",
    "move",
    "eat",
    "absorb_solar",
    "forage",
    "tap",
    "drain",
    "signal",
    "coordinate",
    "clone_perturb",
)

ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}


def _clip(value: float, low: float = -1.0, high: float = 1.5) -> float:
    return max(low, min(high, float(value)))


@dataclass(slots=True)
class Individual:
    id: int
    kind: str
    params: ParamVector
    location: int
    energy: float
    health: float = 1.0
    age: int = 0
    cycle: int = 0
    parent_ids: tuple[int, ...] = ()
    line_root_id: int = 0
    parent_line_ids: tuple[int, ...] = ()
    inherited_controller_template: bool = False
    controller: TinyController | None = None
    controller_template: TinyController | None = None
    signal_values: list[float] = field(default_factory=lambda: [0.0 for _ in range(SIGNAL_VALUE_SIZE)])
    prediction_error_profile: list[float] = field(default_factory=lambda: [0.0 for _ in PREDICTION_HEADS])
    event_memory: list[float] = field(default_factory=lambda: [0.0 for _ in range(EVENT_MEMORY_SIZE)])
    alive: bool = True
    last_action: str = "rest"
    last_valence: float = 0.0
    last_energy_delta: float = 0.0
    recent_action_index: int = 0
    recent_health_delta: float = 0.0
    recent_damage: float = 0.0
    recent_prediction_error: float = 0.0
    recent_spawn_feedback: float = 0.0
    recent_social_feedback: float = 0.0
    recent_tap_feedback: float = 0.0
    combine_intent_until: int = -1
    coordination_token: int = 0
    successful_taps: int = 0
    mistap_count: int = 0
    child_count: int = 0
    success_profile: dict[str, float] = field(default_factory=lambda: {label: 0.0 for label in SUCCESS_PROFILE_LABELS})
    # Developmental subsidy on the neural component of upkeep (set from
    # RunConfig by the simulation at creation; 0 ticks = legacy full price).
    neural_upkeep_grace_ticks: int = 0
    neural_upkeep_grace_floor: float = 0.35

    @property
    def neural(self) -> bool:
        return self.controller is not None

    def hidden_size(self) -> int:
        return max(0, int(round(self.params.neural_budget)))

    def storage_limit(self) -> float:
        return 24.0 + self.params.storage_capacity * 95.0 + self.params.developmental_complexity * 25.0

    def upkeep_cost(self) -> float:
        base = 0.018
        body = (
            self.params.mobility * 0.030
            + self.params.manipulator * 0.020
            + self.params.resilience * 0.015
            + self.params.sensor_range * 0.010
            + self.params.developmental_complexity * 0.020
        )
        # neural_budget coefficient lowered 0.0045 -> 0.0030 so a 64-unit controller
        # costs 0.192/tick instead of 0.288. Bigger controllers need a long enough
        # active span to bootstrap their learned representations; if they run out of
        # energy before cognition pays off, ranking drives capacity down regardless of
        # the world's puzzle complexity. This relief lets the experiment run.
        # Episodic capacity at 0.0035/slot - moderate cost since each slot is
        # a hidden_size-dim vector (a controller at neural_budget=8 with
        # episodic_capacity=8 holds 64 floats of episodic memory).
        episodic = max(0.0, getattr(self.params, "episodic_capacity", 0.0)) * 0.0035
        neural = (
            self.params.neural_budget * 0.0030
            + self.params.memory_budget * 0.0028
            + self.params.prediction_weight * 0.018
            + self.params.plasticity_rate * 0.010
            + episodic
        )
        if self.kind in {"collector", "converter"}:
            base *= 0.55
            body *= 0.35
        # Developmental subsidy: capacity's benefit arrives only after lifetime
        # learning fills it, so the neural term ramps in over the grace window
        # rather than charging full price from tick zero.
        if self.neural_upkeep_grace_ticks > 0 and self.age < self.neural_upkeep_grace_ticks:
            floor = min(1.0, max(0.0, self.neural_upkeep_grace_floor))
            ramp = floor + (1.0 - floor) * (self.age / self.neural_upkeep_grace_ticks)
            neural *= ramp
        return base + body + neural

    def adult(self) -> bool:
        return self.age >= 25 and self.health > 0.35

    def clone_perturb_energy_threshold(self) -> float:
        return 22.0 + self.params.single_parent_threshold * 45.0 + self.params.complexity() * 7.0

    def combine_energy_threshold(self) -> float:
        return 26.0 + self.params.two_parent_threshold * 55.0 + self.params.complexity() * 8.0

    def solo_energy_threshold(self) -> float:
        return self.clone_perturb_energy_threshold()

    def paired_energy_threshold(self) -> float:
        return self.combine_energy_threshold()

    def choose_signal_token(self) -> int:
        if self.controller is not None and self.controller.last_outputs.size:
            return int(max(range(min(8, len(self.controller.last_outputs))), key=lambda i: self.controller.last_outputs[i])) % 8
        return (self.id + self.age + int(self.energy)) % 8

    def learn_signal_value(self, token: int, valence: float) -> None:
        if 0 <= token < len(self.signal_values):
            self.signal_values[token] = self.signal_values[token] * 0.96 + valence * 0.04

    def recent_trace(self) -> list[float]:
        return [
            self.recent_action_index / max(1.0, float(len(ACTIONS) - 1)),
            _clip(self.last_energy_delta / 10.0),
            _clip(self.recent_health_delta * 4.0),
            _clip(self.recent_damage * 4.0, 0.0, 1.5),
            _clip(self.recent_prediction_error),
            _clip(self.recent_spawn_feedback, 0.0, 1.5),
            _clip(self.recent_social_feedback),
            _clip(self.recent_tap_feedback, 0.0, 1.5),
        ]

    def record_action_result(
        self,
        action_index: int,
        energy_delta: float,
        health_delta: float,
        damage: float,
        prediction_error: float,
        spawn_feedback: float,
        social_feedback: float,
        tap_feedback: float,
        prediction_errors: dict[str, float] | None = None,
    ) -> None:
        prediction_errors = prediction_errors or {"energy": prediction_error}
        self.recent_action_index = max(0, min(len(ACTIONS) - 1, int(action_index)))
        self.last_action = ACTIONS[self.recent_action_index]
        self.last_energy_delta = energy_delta
        self.recent_health_delta = health_delta
        self.recent_damage = damage
        self.recent_prediction_error = prediction_errors.get("energy", prediction_error)
        self.prediction_error_profile = [_clip(prediction_errors.get(head, 0.0)) for head in PREDICTION_HEADS]
        self.recent_spawn_feedback = spawn_feedback
        self.recent_social_feedback = social_feedback
        self.recent_tap_feedback = tap_feedback
        self._write_event_memory(
            energy_delta=energy_delta,
            health_delta=health_delta,
            damage=damage,
            spawn_feedback=spawn_feedback,
            social_feedback=social_feedback,
            tap_feedback=tap_feedback,
            prediction_errors=prediction_errors,
        )
        if energy_delta > 0.0:
            self.record_success("energy_gain", min(3.0, energy_delta / 8.0))
        average_error = sum(abs(prediction_errors.get(head, 0.0)) for head in PREDICTION_HEADS) / max(1, len(PREDICTION_HEADS))
        if average_error < 1.0 and (abs(energy_delta) + abs(health_delta) + spawn_feedback + social_feedback + tap_feedback) > 0.0:
            self.record_success("prediction_fit", (1.0 - average_error) * 0.04)

    def _write_event_memory(
        self,
        energy_delta: float,
        health_delta: float,
        damage: float,
        spawn_feedback: float,
        social_feedback: float,
        tap_feedback: float,
        prediction_errors: dict[str, float],
    ) -> None:
        if len(self.event_memory) != EVENT_MEMORY_SIZE:
            self.event_memory = [0.0 for _ in range(EVENT_MEMORY_SIZE)]
        memory_gate = _clip(self.params.memory_budget / 12.0, 0.0, 1.0)
        decay = 0.86 + memory_gate * 0.10
        write = (0.03 + memory_gate * 0.12) * (0.75 + min(1.0, self.params.plasticity_rate * 2.5) * 0.25)
        surprise = sum(abs(prediction_errors.get(head, 0.0)) for head in PREDICTION_HEADS) / max(1, len(PREDICTION_HEADS))
        event = [
            _clip(max(0.0, energy_delta) / 10.0, 0.0, 1.5),
            _clip(max(0.0, -energy_delta) / 10.0, 0.0, 1.5),
            _clip(max(0.0, health_delta) * 4.0, 0.0, 1.5),
            _clip(damage * 4.0, 0.0, 1.5),
            _clip(spawn_feedback, 0.0, 1.5),
            _clip(social_feedback),
            _clip(tap_feedback, 0.0, 1.5),
            _clip(surprise, 0.0, 1.5),
        ]
        self.event_memory = [_clip(old * decay + value * write) for old, value in zip(self.event_memory, event)]

    def record_success(self, label: str, amount: float = 1.0) -> None:
        if not self.success_profile:
            self.success_profile = {name: 0.0 for name in SUCCESS_PROFILE_LABELS}
        if label not in self.success_profile:
            self.success_profile[label] = 0.0
        self.success_profile[label] = max(0.0, min(1_000_000.0, self.success_profile[label] + max(0.0, amount)))

    def record_tap(self, success: bool) -> None:
        if success:
            self.successful_taps += 1
        else:
            self.mistap_count += 1

    def repair_or_decay(self) -> None:
        if self.energy > self.storage_limit():
            self.energy = self.storage_limit()
        if self.energy > self.upkeep_cost() * 20.0 and self.health < 1.0:
            repair = min(1.0 - self.health, 0.003 + self.params.storage_capacity * 0.004)
            self.health += repair
            self.energy -= repair * 5.0

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "location": self.location,
            "age": self.age,
            "cycle": self.cycle,
            "line_root_id": self.line_root_id,
            "parent_line_ids": list(self.parent_line_ids),
            "inherited_controller_template": self.inherited_controller_template,
            "energy": round(self.energy, 4),
            "health": round(self.health, 4),
            "neural": self.neural,
            "child_count": self.child_count,
            "successful_taps": self.successful_taps,
            "mistap_count": self.mistap_count,
            "success_profile": {key: round(value, 4) for key, value in sorted(self.success_profile.items()) if value > 0.0},
            "last_action": self.last_action,
            "last_valence": round(self.last_valence, 4),
            "last_energy_delta": round(self.last_energy_delta, 4),
            "complexity": round(self.params.complexity(), 4),
            "parents": list(self.parent_ids),
        }

    def cognitive_snapshot(self) -> dict[str, Any]:
        return {
            "line": {
                "root_id": self.line_root_id,
                "parents": list(self.parent_ids),
                "parent_lines": list(self.parent_line_ids),
                "inherited_controller_template": self.inherited_controller_template,
            },
            "last_action": self.last_action,
            "last_valence": round(self.last_valence, 6),
            "recent_trace": {label: round(value, 6) for label, value in zip(RECENT_TRACE_LABELS, self.recent_trace())},
            "prediction_errors": {label: round(value, 6) for label, value in zip(PREDICTION_HEADS, self.prediction_error_profile)},
            "event_memory": {label: round(value, 6) for label, value in zip(EVENT_MEMORY_LABELS, self.event_memory)},
            "success_profile": {key: round(value, 6) for key, value in sorted(self.success_profile.items()) if value > 0.0},
            "signal_values": [round(value, 6) for value in self.signal_values],
        }


def controller_from_dict(data: dict) -> "TinyController":
    """Deserialize a controller by architecture marker (legacy dicts lack one)."""
    if data.get("architecture") == "modular_v1":
        from .modular import ModularController

        return ModularController.from_dict(data)  # type: ignore[return-value]
    return TinyController.from_dict(data)


def make_controller_for_params(rng: Random, params: ParamVector) -> tuple[TinyController | None, TinyController | None]:
    hidden = int(round(params.neural_budget))
    if hidden < 2:
        return None, None
    episodic_capacity = int(round(max(0.0, getattr(params, "episodic_capacity", 0.0))))
    template = TinyController.random(rng, OBSERVATION_SIZE, hidden, len(ACTIONS), episodic_capacity=episodic_capacity)
    controller = TinyController.from_dict(template.to_dict(include_state=False))
    return controller, template


def make_modular_controller_for_params(rng: Random, params: ParamVector, n_blocks: int = 1):
    """Modular counterpart of make_controller_for_params (Phase 2 wire-in).

    The params's neural_budget seeds per-block size; afterwards structure owns
    capacity and the params budget follows it (synced at spawning by the
    optimizer, and at seeding by the caller for n_blocks > 1).
    """
    from .modular import ModularController

    hidden = max(2, int(round(params.neural_budget)))
    template = ModularController.random(rng, OBSERVATION_SIZE, len(ACTIONS), n_blocks=max(1, n_blocks), block_hidden=hidden)
    params.neural_budget = float(template.capacity)
    controller = ModularController.from_dict(template.to_dict(include_state=False))
    return controller, template


def individual_from_params(
    rng: Random,
    id_: int,
    kind: str,
    params: ParamVector,
    location: int,
    energy: float,
    cycle: int = 0,
    parent_ids: tuple[int, ...] = (),
    controller_template: TinyController | None = None,
) -> Individual:
    controller: TinyController | None = None
    template: TinyController | None = None
    if controller_template is not None:
        template = controller_template
        controller = controller_from_dict(template.to_dict(include_state=False))
    elif params.neural_budget >= 2.0 and kind == "agent":
        controller, template = make_controller_for_params(rng, params)
    return Individual(
        id=id_,
        kind=kind,
        params=params,
        location=location,
        energy=energy,
        cycle=cycle,
        parent_ids=parent_ids,
        inherited_controller_template=controller_template is not None,
        controller=controller,
        controller_template=template,
    )
