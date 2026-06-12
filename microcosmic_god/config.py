from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunConfig:
    seed: int = 1
    profile: str = "minute"
    max_ticks: int = 8_000
    max_wall_seconds: float = 300.0
    places: int = 36
    initial_collectors: int = 320
    initial_converters: int = 90
    initial_agents: int = 80
    max_pool: int = 4_000
    season_length: int = 2_000
    log_every: int = 100
    checkpoint_every: int = 1_000
    output_dir: str = "runs"
    run_mode: str = "sealed"
    interventions_path: str | None = None
    stop_on_neural_washout: bool = True
    stop_on_full_washout: bool = True
    event_detail: bool = True
    clone_complexity_soft_limit: float = 4.8
    solo_complexity_ceiling: float = 4.8
    neural_checkpoint_limit: int = 64
    compute_backend: str = "cpu"
    device: str = "auto"
    environment_harshness: float = 1.0
    # Multi-world ranking: every N ticks, regenerate the world with a new
    # seed (new physics, new puzzles). Controllers that memorized the specific
    # world are deactivated when it changes; controllers that abstracted the
    # underlying rule persist. 0 = disabled (legacy single-world behavior).
    world_refresh_every: int = 0
    # Developmental subsidy: for an individual's first N ticks, the neural
    # component of upkeep ramps from `floor` x cost up to full cost. Capacity's
    # benefit arrives only after lifetime learning fills it, but its upkeep is
    # charged immediately - a quality valley at every rung of growth that has
    # kept every observed champion inside the initialization size range
    # (docs/CONTROLLER_ADAPTABILITY.md). The subsidy gives capacity a window
    # to pay for itself. 0 = disabled (legacy behavior).
    neural_upkeep_grace_ticks: int = 0
    neural_upkeep_grace_floor: float = 0.35
    # Fraction of initial agents seeded with the typed modular controller
    # (microcosmic_god/modular.py) instead of the legacy single-layer one.
    # Modular lines can grow/duplicate/prune blocks at spawning;
    # capacity is structure-owned and the params budget follows it.
    initial_modular_fraction: float = 0.0
    # Block count for seeded modular controllers: each founder draws uniformly
    # from [1, initial_modular_max_blocks]. >1 lets head-start experiments ask
    # whether multi-block bodies pay before perturbation has to discover them.
    initial_modular_max_blocks: int = 1
    # Probability that a modular clone_perturb child takes a structural perturbation
    # (duplicate / neutral add / prune). The Phase 2 default was a hunch;
    # the E2 sweep picks the real value.
    structural_perturbation_rate: float = 0.06
    # Patch recovery: a substantial feeding event at a place suppresses that
    # place's staple regeneration (regen x floor) for a recovery window. With
    # jitter 0 the window length is fixed, so remembering where/when you fed
    # predicts when a patch is worth revisiting — route rotation becomes a
    # learnable competence. jitter 1 draws the window from a same-mean
    # exponential instead, severing that predictability while keeping mean
    # energetics fixed (the scrambled-cue control for attribution). 0 = off.
    patch_recovery_ticks: int = 0
    patch_recovery_floor: float = 0.0
    patch_recovery_jitter: float = 0.0
    # Action-resolution search depth. Legacy behavior (0) walks the controller's
    # full ranked action list and executes the first FEASIBLE action — the world
    # silently filters by feasibility, so a fixed priority ordering plus this
    # walk is already a context-sensitive policy with the context supplied free
    # by the harness (docs/TRANSFER_BARRIER.md: this is the leak that let the
    # #2867 champion be a blind priority program). When k>=1, only the top-k
    # ranked actions are checked for feasibility; if none is feasible the
    # individual commits to its top-ranked choice and the world adjudicates
    # it (an infeasible action no-ops while upkeep still drains). k=1 removes
    # the search entirely: the policy must rank a currently-executable action
    # first, which it can only do by reading state — i.e. perception must pay.
    action_search_depth: int = 0
    # Scale on the harness-side output boost for coordinate/clone_perturb when an
    # individual is adult and energy-rich. This is the second free-state channel
    # (after the feasibility walk): it conditions spawning timing on state
    # the controller never has to perceive — it is what gave the #2867 champion
    # its realized energy-conditional behavior despite a constant ranking
    # (docs/TRANSFER_BARRIER.md). 1.0 = legacy; 0.0 removes the injection so
    # spawning timing must come from the controller's own outputs.
    drive_injection_scale: float = 1.0
    # Era 2 (docs/ENV_AXIS_REVIEW.md): tap — gate-free release of a place's
    # sealed reserve, keyed to an observable cue channel. The gate level is
    # the tap_cue_threshold percentile of the active channel's distribution
    # across places (recomputed each refresh), so every cue contract keeps a
    # comparable fraction of places tappable. Above the level a tap releases
    # energy (actor share + place spill); below it the tap misfires at an
    # energy cost and a small health hit. Both inputs are existing
    # observation dims, so discriminating taps are a pure perception demand.
    tap_cue_threshold: float = 0.70
    # Cue contract drift: 0 = the cue channel is fixed (era 2.0). 1 = the
    # cue channel identity is re-drawn from TAP_CUE_CHANNELS at every world
    # refresh (era 2.1) — the contract is then discoverable only through tap
    # outcomes and prediction errors, selecting for in-lifetime re-mapping.
    tap_cue_drift: int = 0
    # Scale on the combine-intent window (one coordinate action holds pairing
    # intent open for 6-19 ticks at 1.0). The window is an ungated
    # spawn-timing channel — a blind policy reproduces through it at
    # exploration-floor rates — so scaling it down makes pairing demand
    # repeated deliberate coordination. 1.0 = legacy.
    combine_intent_window_scale: float = 1.0
    # Constant part of the chooser's random-action rate (param terms add to
    # it). Legacy 0.025. Annealable in later stages.
    exploration_floor: float = 0.025

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_profile(cls, profile: str, **overrides: Any) -> "RunConfig":
        config = cls(profile=profile)
        if profile == "smoke":
            config.max_ticks = 300
            config.max_wall_seconds = 20.0
            config.places = 12
            config.initial_collectors = 80
            config.initial_converters = 20
            config.initial_agents = 18
            config.max_pool = 600
            config.log_every = 25
            config.checkpoint_every = 150
        elif profile == "minute":
            config.environment_harshness = 1.35
        elif profile == "long":
            config.max_ticks = 1_000_000
            config.max_wall_seconds = 86_400.0
            config.places = 96
            config.initial_collectors = 2_000
            config.initial_converters = 600
            config.initial_agents = 400
            config.max_pool = 30_000
            config.season_length = 12_000
            config.log_every = 2_500
            config.checkpoint_every = 20_000
            config.event_detail = False
            config.environment_harshness = 1.35
        elif profile == "modal":
            config.max_ticks = 10_000_000
            config.max_wall_seconds = 259_200.0
            config.places = 256
            config.initial_collectors = 4_000
            config.initial_converters = 1_000
            config.initial_agents = 800
            config.max_pool = 80_000
            config.season_length = 40_000
            config.log_every = 5_000
            config.checkpoint_every = 50_000
            config.event_detail = False
            config.environment_harshness = 1.45
        else:
            raise ValueError(f"unknown profile: {profile}")

        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        return config

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
