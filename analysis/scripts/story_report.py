from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_aggregates(run_dir: Path) -> list[dict[str, Any]]:
    events = run_dir / "events.jsonl"
    if not events.exists():
        return []
    aggregates: list[dict[str, Any]] = []
    for line in events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("kind") == "aggregate":
            aggregates.append(item)
    return aggregates


def checkpoint_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "checkpoints").glob("*.json")):
        data = load_json(path)
        organism = data["organism"]
        reason = data["reason"]
        score = (
            organism["offspring_count"] * 5.0
            + organism["successful_tools"] * 2.0
            + organism["age"] / 500.0
            + organism["generation"] * 0.5
            + (3.0 if reason.startswith("first_") else 0.0)
            + (2.0 if reason == "interval_champion" else 0.0)
            + (1.5 if reason.startswith("death_") else 0.0)
        )
        rows.append(
            {
                "score": round(score, 3),
                "tick": data["tick"],
                "reason": reason,
                "id": organism["id"],
                "age": organism["age"],
                "generation": organism["generation"],
                "energy": organism["energy"],
                "offspring": organism["offspring_count"],
                "tools": organism["successful_tools"],
                "complexity": organism["complexity"],
                "file": path.name,
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def print_timeline(aggregates: list[dict[str, Any]]) -> None:
    if not aggregates:
        print("No aggregate events found.")
        return
    print("Timeline")
    stride = max(1, len(aggregates) // 8)
    for item in aggregates[::stride]:
        pop = item["population"]
        tools = item.get("tool_successes", {})
        print(
            f"  tick {item['tick']:>6}: total={pop.get('total', 0):>5} "
            f"neural={pop.get('neural', 0):>4} "
            f"neural_energy={item.get('neural_avg_energy', 0):>8.3f} "
            f"tools={sum(tools.values()) if tools else 0:>5} "
            f"marks={sum(item.get('marks_created', {}).values()):>5} "
            f"fluid={item.get('world_physics', {}).get('avg_fluid_level', 0):>5.2f}"
        )
    last = aggregates[-1]
    pop = last["population"]
    print(
        f"  tick {last['tick']:>6}: total={pop.get('total', 0):>5} "
        f"neural={pop.get('neural', 0):>4} "
        f"neural_energy={last.get('neural_avg_energy', 0):>8.3f} "
        f"tools={sum(last.get('tool_successes', {}).values()):>5} "
        f"marks={sum(last.get('marks_created', {}).values()):>5} "
        f"fluid={last.get('world_physics', {}).get('avg_fluid_level', 0):>5.2f}"
    )


def print_notes(summary: dict[str, Any], aggregates: list[dict[str, Any]]) -> None:
    print("Run Notes")
    print(f"  reason: {summary['reason']}")
    print(f"  tick: {summary['tick']}")
    print(f"  elapsed_seconds: {summary['elapsed_seconds']}")
    print(f"  final_population: {summary['population']}")
    print(f"  births_by_mode: {summary['births_by_mode']}")
    print(f"  deaths_by_cause: {summary['deaths_by_cause']}")
    print(f"  deaths_by_kind_cause: {summary.get('deaths_by_kind_cause', {})}")
    print(f"  tool_successes: {summary['tool_successes']}")
    print(f"  marks_created: {summary.get('marks_created', {})}")
    print(f"  artifacts_created: {summary.get('artifacts_created', {})}")
    print(f"  artifacts_broken: {summary.get('artifacts_broken', {})}")
    print(f"  structures_built: {summary.get('structures_built', {})}")
    print(f"  structures_extended: {summary.get('structures_extended', {})}")
    print(f"  physics_events: {summary.get('physics_events', {})}")
    print(f"  world_physics: {summary.get('world_physics', {})}")
    print(f"  reproduction_attempts: {summary.get('reproduction_attempts', {})}")
    print(f"  reproduction_failures: {summary.get('reproduction_failures', {})}")
    print(f"  action_avg_energy_delta: {summary.get('action_avg_energy_delta', {})}")
    if "sexual" not in summary.get("births_by_mode", {}):
        print("  observation: no sexual reproduction occurred in this run.")
    if summary["reason"] == "neural_extinction":
        print("  observation: neural agents vanished while non-neural ecology persisted.")
    if aggregates:
        first = aggregates[0]["population"].get("neural", 0)
        last_neural = aggregates[-1]["population"].get("neural", 0)
        print(f"  neural_trajectory: {first} at first aggregate -> {last_neural} at last aggregate")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python analysis/scripts/story_report.py runs/<run_dir>")
        raise SystemExit(2)
    run_dir = Path(sys.argv[1])
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path}")
        raise SystemExit(1)

    summary = load_json(summary_path)
    aggregates = load_aggregates(run_dir)
    rows = checkpoint_rows(run_dir)

    print(f"Story report: {run_dir}")
    print_notes(summary, aggregates)
    print()
    print_timeline(aggregates)
    print()
    print("Top Checkpoint Candidates")
    for row in rows[:10]:
        print(
            f"  score={row['score']:>8} tick={row['tick']:>6} id={row['id']:>5} "
            f"reason={row['reason']:<34} age={row['age']:>5} gen={row['generation']:>2} "
            f"offspring={row['offspring']:>2} tools={row['tools']:>4} energy={row['energy']:>8} "
            f"file={row['file']}"
        )


if __name__ == "__main__":
    main()
