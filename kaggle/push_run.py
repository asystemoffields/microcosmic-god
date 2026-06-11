#!/usr/bin/env python3
"""Push a Microcosmic God experiment to a free Kaggle CPU kernel.

Materializes kernel_template.py with the requested CONFIG, writes
kernel-metadata.json, pushes via the kaggle CLI, and (with --wait) polls
until completion and downloads outputs.

Examples:
  # 10-minute pipeline-validation run
  python kaggle/push_run.py --name mg-smoke1 --wall-seconds 600 --seed 7 --wait

  # 12-hour structural-evolution run (the big one)
  python kaggle/push_run.py --name mg-evolve-s1 --wall-seconds 42000 \
      --ticks 2000000 --seed 11 --modular-fraction 0.5

Notes:
  - CPU kernels are quota-free; sessions cap at ~12h, so keep wall_seconds
    under ~42000 to leave room for clone + collection.
  - The kaggle CLI process itself can balloon when DOWNLOADING large outputs;
    mg outputs are small (aggregates + checkpoints), but run collection under
    a memory-limited scope on a small box anyway.
"""

from __future__ import annotations

import argparse
import json
import pprint
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
USER = "asystemoffields"
KAGGLE = "/data/kagglecli-venv/bin/kaggle"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="kernel slug, e.g. mg-evolve-s1")
    parser.add_argument("--branch", default="fable-working")
    parser.add_argument("--profile", default="minute")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=8000)
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    parser.add_argument("--harshness", type=float, default=None)
    parser.add_argument("--neural-grace-ticks", type=int, default=150)
    parser.add_argument("--neural-grace-floor", type=float, default=0.35)
    parser.add_argument("--modular-fraction", type=float, default=0.5)
    parser.add_argument("--modular-max-blocks", type=int, default=1)
    parser.add_argument("--structural-rate", type=float, default=0.06)
    parser.add_argument("--world-refresh-every", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-limit", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wait", action="store_true", help="poll status and download output when complete")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()

    config = {
        "branch": args.branch,
        "profile": args.profile,
        "seed": args.seed,
        "ticks": args.ticks,
        "wall_seconds": args.wall_seconds,
        "harshness": args.harshness,
        "neural_grace_ticks": args.neural_grace_ticks,
        "neural_grace_floor": args.neural_grace_floor,
        "modular_fraction": args.modular_fraction,
        "modular_max_blocks": args.modular_max_blocks,
        "structural_rate": args.structural_rate,
        "world_refresh_every": args.world_refresh_every,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_limit": args.checkpoint_limit,
        "quiet_events": True,
        "log_every": args.log_every,
    }

    package = HERE / "_packages" / args.name
    package.mkdir(parents=True, exist_ok=True)
    template = (HERE / "kernel_template.py").read_text()
    body = re.sub(
        r"CONFIG = \{.*?\n\}",
        # pformat, not json.dumps: the kernel is Python source (None, not null).
        "CONFIG = " + pprint.pformat(config, indent=4, sort_dicts=False),
        template,
        count=1,
        flags=re.DOTALL,
    )
    (package / "kernel.py").write_text(body)
    (package / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{USER}/{args.name}",
                "title": args.name,
                "code_file": "kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            indent=2,
        )
    )
    subprocess.run([KAGGLE, "kernels", "push", "-p", str(package)], check=True)
    print(f"pushed {USER}/{args.name}")

    if not args.wait:
        return
    while True:
        time.sleep(args.poll_seconds)
        out = subprocess.run(
            [KAGGLE, "kernels", "status", f"{USER}/{args.name}"],
            capture_output=True, text=True, check=False,
        )
        status = (out.stdout + out.stderr).strip()
        print(time.strftime("%H:%M:%S"), status, flush=True)
        if "complete" in status.lower():
            break
        if "error" in status.lower() or "cancel" in status.lower():
            sys.exit(f"kernel did not complete: {status}")
    dest = Path(args.out) / args.name
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([KAGGLE, "kernels", "output", f"{USER}/{args.name}", "-p", str(dest)], check=True)
    print(f"output -> {dest}")


if __name__ == "__main__":
    main()
