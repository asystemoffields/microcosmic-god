from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .config import RunConfig
from .simulation import Simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Microcosmic God artificial-life experiments.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a headless simulation")
    run.add_argument("--profile", choices=["smoke", "minute", "long", "modal"], default="minute")
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--ticks", type=int, default=None, help="maximum simulation ticks")
    run.add_argument("--wall-seconds", type=float, default=None, help="maximum wall-clock seconds; 0 disables wall limit")
    run.add_argument("--places", type=int, default=None)
    run.add_argument("--plants", type=int, default=None)
    run.add_argument("--fungi", type=int, default=None)
    run.add_argument("--agents", type=int, default=None)
    run.add_argument("--max-population", type=int, default=None)
    run.add_argument("--output-dir", default=None)
    run.add_argument("--log-every", type=int, default=None)
    run.add_argument("--checkpoint-every", type=int, default=None)
    run.add_argument("--checkpoint-limit", type=int, default=None)
    run.add_argument(
        "--harshness",
        "--environment-harshness",
        dest="environment_harshness",
        type=float,
        default=None,
        help="environment pressure multiplier; 1.0 is baseline, higher values reduce easy survival",
    )
    run.add_argument(
        "--world-refresh-every",
        dest="world_refresh_every",
        type=int,
        default=None,
        help="regenerate the world (new seed, new physics) every N ticks; 0 disables (legacy single-world)",
    )
    run.add_argument(
        "--neural-grace-ticks",
        dest="neural_upkeep_grace_ticks",
        type=int,
        default=None,
        help="developmental subsidy: neural upkeep ramps from a floor to full price over an individual's first N ticks; 0 disables (legacy)",
    )
    run.add_argument(
        "--neural-grace-floor",
        dest="neural_upkeep_grace_floor",
        type=float,
        default=None,
        help="starting fraction of neural upkeep at age 0 when --neural-grace-ticks is set (default 0.35)",
    )
    run.add_argument(
        "--modular-fraction",
        dest="initial_modular_fraction",
        type=float,
        default=None,
        help="fraction of initial agents seeded with the typed modular controller (cpu backend only)",
    )
    run.add_argument(
        "--modular-max-blocks",
        dest="initial_modular_max_blocks",
        type=int,
        default=None,
        help="seeded modular founders draw block count uniformly from [1, N]",
    )
    run.add_argument(
        "--structural-rate",
        dest="structural_mutation_rate",
        type=float,
        default=None,
        help="probability a modular clone takes a structural mutation (duplicate/add/prune)",
    )
    run.add_argument(
        "--patch-recovery-ticks",
        dest="patch_recovery_ticks",
        type=int,
        default=None,
        help="suppress a place's staple regen for N ticks after a substantial feed (0 disables)",
    )
    run.add_argument(
        "--patch-recovery-floor",
        dest="patch_recovery_floor",
        type=float,
        default=None,
        help="regen multiplier while a patch recovers (default 0.0)",
    )
    run.add_argument(
        "--patch-recovery-jitter",
        dest="patch_recovery_jitter",
        type=float,
        default=None,
        help="0 = fixed recovery window (predictable); 1 = same-mean exponential draw (scrambled control)",
    )
    run.add_argument(
        "--action-search-depth",
        dest="action_search_depth",
        type=int,
        default=None,
        help="check only the top-k ranked actions for feasibility; past k the individual commits to its top choice and infeasible attempts no-op with an energy cost (0 = legacy full walk)",
    )
    run.add_argument("--backend", choices=["cpu", "torch"], default=None, help="controller compute backend")
    run.add_argument("--device", default=None, help="compute device for --backend torch, such as auto, cpu, cuda, or cuda:0")
    run.add_argument("--garden", action="store_true", help="allow logged interventions")
    run.add_argument("--interventions", default=None, help="path to an interventions JSON file")
    run.add_argument("--no-stop-on-neural-extinction", action="store_true")
    run.add_argument("--quiet-events", action="store_true", help="only write aggregate events, not births/deaths/tool details")
    run.add_argument("--dry-run", action="store_true", help="print resolved config and exit")

    spec = sub.add_parser("specs", help="print local machine specs relevant to simulation sizing")
    spec.add_argument("--json", action="store_true")

    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    overrides = {
        "seed": args.seed,
        "max_ticks": args.ticks,
        "max_wall_seconds": args.wall_seconds,
        "places": args.places,
        "initial_plants": args.plants,
        "initial_fungi": args.fungi,
        "initial_agents": args.agents,
        "max_population": args.max_population,
        "output_dir": args.output_dir,
        "log_every": args.log_every,
        "checkpoint_every": args.checkpoint_every,
        "neural_checkpoint_limit": args.checkpoint_limit,
        "environment_harshness": args.environment_harshness,
        "world_refresh_every": args.world_refresh_every,
        "neural_upkeep_grace_ticks": args.neural_upkeep_grace_ticks,
        "neural_upkeep_grace_floor": args.neural_upkeep_grace_floor,
        "initial_modular_fraction": args.initial_modular_fraction,
        "initial_modular_max_blocks": args.initial_modular_max_blocks,
        "structural_mutation_rate": args.structural_mutation_rate,
        "patch_recovery_ticks": args.patch_recovery_ticks,
        "patch_recovery_floor": args.patch_recovery_floor,
        "patch_recovery_jitter": args.patch_recovery_jitter,
        "action_search_depth": args.action_search_depth,
        "compute_backend": args.backend,
        "device": args.device,
        "run_mode": "garden" if args.garden else "sealed",
        "interventions_path": args.interventions,
        "stop_on_neural_extinction": not args.no_stop_on_neural_extinction,
        "event_detail": not args.quiet_events,
    }
    return RunConfig.from_profile(args.profile, **overrides)


def machine_specs() -> dict[str, object]:
    specs: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(Path.cwd().anchor or Path.cwd()))
        specs.update(
            {
                "cpu_logical": psutil.cpu_count(logical=True),
                "cpu_physical": psutil.cpu_count(logical=False),
                "memory_total_gib": round(memory.total / (1024**3), 3),
                "memory_available_gib": round(memory.available / (1024**3), 3),
                "disk_free_gib": round(disk.free / (1024**3), 3),
            }
        )
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        specs["psutil_error"] = str(exc)
    return specs


def print_run_card(config: RunConfig) -> None:
    print("Microcosmic God run card")
    print(f"  profile: {config.profile}")
    print(f"  seed: {config.seed}")
    print(f"  mode: {config.run_mode}")
    print(f"  max ticks: {config.max_ticks:,}")
    print(f"  wall limit: {'none' if config.max_wall_seconds == 0 else f'{config.max_wall_seconds:.1f}s'}")
    print(f"  places: {config.places}")
    print(f"  initial pool: plants={config.initial_plants}, fungi={config.initial_fungi}, neural_agents={config.initial_agents}")
    print(f"  max pool: {config.max_population}")
    print(f"  stop on neural extinction: {config.stop_on_neural_extinction}")
    print(f"  stop on full extinction: {config.stop_on_full_extinction}")
    print(f"  log every: {config.log_every} ticks")
    print(f"  checkpoint every: {config.checkpoint_every} ticks")
    print(f"  checkpoint limit: {config.neural_checkpoint_limit}")
    print(f"  environment harshness: {config.environment_harshness:.2f}")
    print(f"  world refresh every: {'never' if config.world_refresh_every <= 0 else f'{config.world_refresh_every} ticks'}")
    print(f"  compute backend: {config.compute_backend}")
    print(f"  device: {config.device}")
    print(f"  output dir: {config.output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "specs":
        specs = machine_specs()
        if args.json:
            print(json.dumps(specs, indent=2, sort_keys=True))
        else:
            print("Machine specs")
            for key, value in specs.items():
                print(f"  {key}: {value}")
        return

    if args.command in {None, "run"}:
        if args.command is None:
            args = parser.parse_args(["run"])
        config = config_from_args(args)
        print_run_card(config)
        if args.dry_run:
            print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
            return
        sim = Simulation(config)
        debrief = sim.run()
        print("Run complete")
        print(f"  reason: {debrief['reason']}")
        print(f"  tick: {debrief['tick']:,}")
        print(f"  elapsed seconds: {debrief['elapsed_seconds']}")
        print(f"  pool: {debrief['population']}")
        print(f"  births: {debrief['births_by_mode']}")
        print(f"  deaths: {debrief['deaths_by_cause']}")
        print(f"  tool successes: {debrief['tool_successes']}")
        print(f"  structures built: {debrief.get('structures_built', {})}")
        print(f"  structures extended: {debrief.get('structures_extended', {})}")
        print(f"  likely causes: {', '.join(debrief['likely_causes'])}")
        print(f"  run directory: {sim.logger.run_dir}")
        return

    parser.error("unknown command")


if __name__ == "__main__":
    main()
