"""Render a run directory's streams into a readable digest.

Usage: python analysis/run_digest.py <run_dir> [run_dir ...]
Reads status.json / summary.json, structure_events.jsonl, story_events.jsonl.
Cheap: streams JSONL once, holds only counters and small samples.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def digest(run_dir: Path) -> None:
    print(f"\n=== {run_dir} ===")
    head = None
    for name in ("summary.json", "status.json"):
        path = run_dir / name
        if path.exists():
            head = json.loads(path.read_text())
            print(f"[{name}]", end=" ")
            break
    if head:
        tick = head.get("tick", "?")
        reason = head.get("reason", "running")
        pop = head.get("population", {})
        arch = head.get("architecture") or (head.get("last_aggregates") or [{}])[-1].get("architecture", {})
        print(f"tick {tick}  ({reason})  pool {pop}")
        if arch:
            print(
                f"    census: modular={arch.get('modular', 0)} legacy={arch.get('legacy', 0)} "
                f"blocks_mean={arch.get('blocks_mean', '-')} blocks_max={arch.get('blocks_max', '-')} "
                f"capacity_mean={arch.get('capacity_mean', '-')}"
            )
        steps = head.get("structural_steps", {})
        if steps:
            print(f"    structural steps: {steps}")
        if head.get("patch_recovery_triggers"):
            print(f"    patch recovery triggers: {head['patch_recovery_triggers']}")

    # Structural genealogy: who took steps, and did stepped lineages persist
    # (proxy: a stepped child that later appears as a parent of another step).
    ops = Counter()
    steppers: set[int] = set()
    step_parents: set[int] = set()
    by_lineage = Counter()
    transitions = Counter()
    for record in _iter_jsonl(run_dir / "structure_events.jsonl"):
        ops[record["op"]] += 1
        steppers.add(record["child_id"])
        step_parents.update(record.get("parent_ids", []))
        if record.get("lineage_root_id") is not None:
            by_lineage[record["lineage_root_id"]] += 1
        transitions[f"{record['blocks_before']}->{record['blocks_after']}"] += 1
    if ops:
        chained = len(steppers & step_parents)
        print(f"    genealogy: {dict(ops)}  | stepped children that later stepped again: {chained}")
        print(f"    block transitions: {dict(transitions.most_common(8))}")
        top = by_lineage.most_common(3)
        print(f"    most structurally active lineages: {top}")

    # Story stream: firsts timeline + notable lives.
    firsts: list[tuple[int, str]] = []
    seen_kinds: set[str] = set()
    notable: list[dict] = []
    promoted = 0
    for record in _iter_jsonl(run_dir / "story_events.jsonl"):
        promoted += 1
        kind = record.get("kind", "?")
        rarity = record.get("rarity_key")
        if rarity and rarity not in seen_kinds:
            seen_kinds.add(rarity)
            firsts.append((record.get("tick", 0), rarity))
        if kind == "notable_death":
            notable.append(record)
    if promoted:
        print(f"    stories promoted: {promoted}")
        print("    firsts:")
        for tick, rarity in firsts[:20]:
            print(f"      t{tick:>6}  {rarity}")
        if len(firsts) > 20:
            print(f"      ... and {len(firsts) - 20} more")
    if notable:
        notable.sort(key=lambda r: r.get("payload", {}).get("score", 0.0), reverse=True)
        print("    notable lives (top 5 by checkpoint score):")
        for record in notable[:5]:
            p = record.get("payload", {})
            arch = p.get("architecture") or {}
            arch_str = f" blocks={arch.get('blocks')} cap={arch.get('capacity')}" if arch else ""
            print(
                f"      #{p.get('organism_id')} died t{record.get('tick')} of {p.get('cause')}"
                f" age={p.get('age', '?')} offspring={p.get('offspring_count')}"
                f" tools={p.get('successful_tools')}{arch_str}"
            )


def biography(run_dir: Path, organism_id: int) -> None:
    """Print one individual's arc: every story/structure event that touches it."""
    print(f"\n=== {run_dir} — individual #{organism_id} ===")
    needle = f"organism:{organism_id}"
    timeline: list[tuple[int, str]] = []
    for record in _iter_jsonl(run_dir / "story_events.jsonl"):
        payload = record.get("payload", {})
        touched = (
            any(needle == s for s in record.get("subjects") or [])
            or payload.get("organism_id") == organism_id
            or payload.get("child_id") == organism_id
            or organism_id in (payload.get("parent_ids") or [])
        )
        if not touched:
            continue
        keep = {
            k: payload[k]
            for k in (
                "mode", "cause", "age", "offspring_count", "successful_tools",
                "child_id", "parent_ids", "generation", "complexity", "place",
                "affordance", "gain", "architecture",
            )
            if k in payload and payload[k] not in (None, [], {})
        }
        timeline.append((record.get("tick", 0), f"{record.get('kind', '?'):<18} {keep}"))
    for record in _iter_jsonl(run_dir / "structure_events.jsonl"):
        if record.get("child_id") == organism_id or organism_id in (record.get("parent_ids") or []):
            role = "child" if record.get("child_id") == organism_id else "parent"
            timeline.append(
                (
                    record.get("tick", 0),
                    f"structural_step    as {role}: {record['op']} blocks "
                    f"{record['blocks_before']}->{record['blocks_after']} "
                    f"cap {record['capacity_after']} child {record['child_id']}",
                )
            )
    timeline.sort(key=lambda item: item[0])
    if not timeline:
        print("    no recorded events (unpromoted life, or wrong id)")
    for tick, line in timeline:
        print(f"  t{tick:>6}  {line}")


def main() -> None:
    args = sys.argv[1:]
    organism_id = None
    if "--organism" in args:
        index = args.index("--organism")
        organism_id = int(args[index + 1])
        del args[index : index + 2]
    for arg in args:
        root = Path(arg)
        # Accept either a run dir or a parent of run dirs.
        if (root / "config.json").exists():
            runs = [root]
        else:
            runs = [config.parent for config in sorted(root.glob("**/config.json"))]
        for run in runs:
            if organism_id is not None:
                biography(run, organism_id)
            else:
                digest(run)


if __name__ == "__main__":
    main()
