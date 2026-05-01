from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python analysis/scripts/summarize_run.py runs/<run_dir>")
        raise SystemExit(2)
    run_dir = Path(sys.argv[1])
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}")
        raise SystemExit(1)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"run: {run_dir}")
    print(f"reason: {summary['reason']}")
    print(f"tick: {summary['tick']}")
    print(f"elapsed_seconds: {summary['elapsed_seconds']}")
    print(f"population: {summary['population']}")
    print(f"births: {summary['births_by_mode']}")
    print(f"deaths: {summary['deaths_by_cause']}")
    print(f"deaths_by_kind_cause: {summary.get('deaths_by_kind_cause', {})}")
    print(f"tool_successes: {summary['tool_successes']}")
    print(f"marks_created: {summary.get('marks_created', {})}")
    print(f"reproduction_attempts: {summary.get('reproduction_attempts', {})}")
    print(f"reproduction_failures: {summary.get('reproduction_failures', {})}")
    print(f"action_avg_energy_delta: {summary.get('action_avg_energy_delta', {})}")
    print(f"checkpoints: {summary['checkpointing']}")
    print("likely_causes:")
    for cause in summary["likely_causes"]:
        print(f"  - {cause}")


if __name__ == "__main__":
    main()
