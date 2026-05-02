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
    initial_plants: int = 320
    initial_fungi: int = 90
    initial_agents: int = 80
    max_population: int = 4_000
    season_length: int = 2_000
    log_every: int = 100
    checkpoint_every: int = 1_000
    output_dir: str = "runs"
    run_mode: str = "sealed"
    interventions_path: str | None = None
    stop_on_neural_extinction: bool = True
    stop_on_full_extinction: bool = True
    event_detail: bool = True
    clone_complexity_soft_limit: float = 4.8
    asexual_complexity_ceiling: float = 4.8
    neural_checkpoint_limit: int = 64
    compute_backend: str = "cpu"
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_profile(cls, profile: str, **overrides: Any) -> "RunConfig":
        config = cls(profile=profile)
        if profile == "smoke":
            config.max_ticks = 300
            config.max_wall_seconds = 20.0
            config.places = 12
            config.initial_plants = 80
            config.initial_fungi = 20
            config.initial_agents = 18
            config.max_population = 600
            config.log_every = 25
            config.checkpoint_every = 150
        elif profile == "minute":
            pass
        elif profile == "long":
            config.max_ticks = 1_000_000
            config.max_wall_seconds = 86_400.0
            config.places = 96
            config.initial_plants = 2_000
            config.initial_fungi = 600
            config.initial_agents = 400
            config.max_population = 30_000
            config.season_length = 12_000
            config.log_every = 2_500
            config.checkpoint_every = 20_000
            config.event_detail = False
        elif profile == "modal":
            config.max_ticks = 10_000_000
            config.max_wall_seconds = 259_200.0
            config.places = 256
            config.initial_plants = 4_000
            config.initial_fungi = 1_000
            config.initial_agents = 800
            config.max_population = 80_000
            config.season_length = 40_000
            config.log_every = 5_000
            config.checkpoint_every = 50_000
            config.event_detail = False
        else:
            raise ValueError(f"unknown profile: {profile}")

        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        return config

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
