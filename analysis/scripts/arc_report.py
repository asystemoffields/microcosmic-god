from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ANCHOR_KINDS = {
    "causal_unlock",
    "structure_built",
    "crafted_tool",
    "mark_lesson_written",
    "notable_death",
    "checkpoint_saved",
}

WALK_BACK = 80
WALK_FORWARD = 20

ANCHOR_BASE_WEIGHT = {
    "causal_unlock": 12.0,
    "crafted_tool": 4.0,
    "mark_lesson_written": 4.0,
    "structure_built": 1.0,
    "notable_death": 1.5,
    "checkpoint_saved": 1.0,
}


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def organism_id(event: dict[str, Any]) -> int | None:
    payload = event.get("payload", {})
    org = payload.get("organism_id")
    if org is None:
        org = payload.get("child_id")
    return org


def lineage_id(event: dict[str, Any]) -> int | None:
    payload = event.get("payload", {})
    if "lineage_root_id" in payload:
        return payload["lineage_root_id"]
    for subj in event.get("subjects", []):
        if isinstance(subj, str) and subj.startswith("lineage:"):
            try:
                return int(subj.split(":", 1)[1])
            except ValueError:
                continue
    return None


def is_structure_anchor(event: dict[str, Any]) -> bool:
    payload = event.get("payload", {})
    if event.get("kind") != "structure_built":
        return False
    return bool(payload.get("extended")) or payload.get("scale", 0) >= 10 or len(payload.get("helpers", []) or []) >= 3


def is_anchor(event: dict[str, Any]) -> bool:
    """Collaboration is intentionally excluded — protect-style collabs re-emit every
    tick and would swamp arcs. The helper count of one-shot collabs is already
    captured by the resulting structure_built/tool_success event."""
    kind = event.get("kind")
    if kind == "structure_built":
        return is_structure_anchor(event)
    return kind in ANCHOR_KINDS


AGENTIC_PROFILE_KEYS = (
    "causal_unlock",
    "structure",
    "reproduction",
    "tool_make",
    "knowledge_transmitted",
    "social_learning",
    "collaboration",
    "causal_step",
    "written_learning",
)


def diversity_factor(profile: dict[str, float]) -> float:
    """Discount post-hoc tool_use credit when the organism showed no other agentic signals.

    A pure tool_use spike with zero across causal/structure/reproduction/social
    is the specialist-trap signature: 328 lever-pulls at one place add up to a
    huge tool_use score but reflect rote memory loops, not adaptation.
    Returns ~0.25 for pure trap, 1.0 for balanced, capped at 1.0.
    """
    tool_use = profile.get("tool_use", 0.0) or 0.0
    if tool_use < 1.0:
        return 1.0
    other = sum((profile.get(k, 0.0) or 0.0) for k in AGENTIC_PROFILE_KEYS)
    ratio = other / (other + tool_use)
    return max(0.2, min(1.0, 0.25 + ratio * 1.5))


def anchor_weight(event: dict[str, Any]) -> float:
    kind = event.get("kind")
    payload = event.get("payload", {})
    if kind == "structure_built":
        scale = payload.get("scale", 0) or 0
        helpers = payload.get("helper_components", 0) or 0
        extended = 0.5 if payload.get("extended") else 0.0
        return 0.4 + scale * 0.08 + helpers * 0.4 + extended
    if kind == "notable_death":
        profile = payload.get("success_profile", {}) or {}
        tool_use = profile.get("tool_use", 0.0) or 0.0
        unlocks = profile.get("causal_unlock", 0.0) or 0.0
        score = payload.get("score", 0.0) or 0.0
        return 0.3 + min(score, 100.0) * 0.02 + tool_use * 0.2 * diversity_factor(profile) + unlocks * 2.0
    return ANCHOR_BASE_WEIGHT.get(kind, 0.0)


def collect_subjects(event: dict[str, Any]) -> set[str]:
    return set(event.get("subjects", []) or [])


def build_arcs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One arc per organism. Headline anchor = strongest anchor involving them."""

    by_organism: dict[int, dict[str, Any]] = {}

    for event in events:
        if not is_anchor(event):
            continue
        org = organism_id(event)
        if org is None:
            continue
        weight = anchor_weight(event)
        record = by_organism.setdefault(
            org,
            {
                "organism_id": org,
                "anchors": [],
                "headline_weight": 0.0,
                "headline": None,
            },
        )
        record["anchors"].append(event)
        if weight > record["headline_weight"]:
            record["headline_weight"] = weight
            record["headline"] = event

    arcs: list[dict[str, Any]] = []
    for org, record in by_organism.items():
        anchor_ticks = [a["tick"] for a in record["anchors"]]
        tick_lo = min(anchor_ticks) - WALK_BACK
        tick_hi = max(anchor_ticks) + WALK_FORWARD

        relevant: list[dict[str, Any]] = []
        for event in events:
            tick = event.get("tick", 0)
            if tick < tick_lo or tick > tick_hi:
                continue
            if organism_id(event) == org or f"organism:{org}" in collect_subjects(event):
                relevant.append(event)

        if not relevant:
            continue

        places = sorted({e.get("payload", {}).get("place")
                         for e in relevant
                         if e.get("payload", {}).get("place") is not None})
        lineage = lineage_id(record["headline"]) or next(
            (lineage_id(e) for e in relevant if lineage_id(e) is not None), None
        )

        anchor_weights = sorted((anchor_weight(a) for a in record["anchors"]), reverse=True)
        arc_score = sum(anchor_weights[:6]) + 0.2 * sum(anchor_weights[6:])
        arc_score += 1.5 * max(0, len(places) - 1)
        anchor_kinds = {a.get("kind") for a in record["anchors"]}
        if {"structure_built", "causal_unlock"}.issubset(anchor_kinds):
            arc_score += 4.0
        if "crafted_tool" in anchor_kinds and any(k in anchor_kinds for k in ("tool_success", "structure_built", "causal_unlock")):
            arc_score += 2.0

        arcs.append(
            {
                "organism_id": org,
                "lineage_id": lineage,
                "headline": record["headline"],
                "anchors": record["anchors"],
                "events": relevant,
                "tick_lo": min(e.get("tick", 0) for e in relevant),
                "tick_hi": max(e.get("tick", 0) for e in relevant),
                "places": places,
                "score": arc_score,
            }
        )

    arcs.sort(key=lambda a: a["score"], reverse=True)
    return arcs


COLLAPSIBLE_KINDS = {"movement_attempt", "collaboration"}


def _collapse_key(event: dict[str, Any]) -> tuple | None:
    kind = event.get("kind")
    payload = event.get("payload", {})
    if kind == "movement_attempt":
        return ("move", payload.get("organism_id"), payload.get("dominant_motive"))
    if kind == "collaboration":
        return ("collab", payload.get("organism_id"), payload.get("focus"), payload.get("place"))
    return None


def collapse_steady_state(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse adjacent repeats of routine event kinds into a single count event."""
    out: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []
    run_key: tuple | None = None

    def flush() -> None:
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
            return
        first, last = run[0], run[-1]
        kind = first["kind"]
        first_payload = first.get("payload", {})
        if kind == "movement_attempt":
            out.append({
                "kind": "movement_run",
                "tick": first["tick"],
                "tick_end": last["tick"],
                "count": len(run),
                "payload": {
                    "organism_id": first_payload.get("organism_id"),
                    "dominant_motive": first_payload.get("dominant_motive"),
                    "places": sorted({e["payload"].get("destination") for e in run if e["payload"].get("destination") is not None}),
                },
            })
        elif kind == "collaboration":
            helpers_max = max(len(e["payload"].get("helpers", []) or []) for e in run)
            out.append({
                "kind": "collaboration_run",
                "tick": first["tick"],
                "tick_end": last["tick"],
                "count": len(run),
                "payload": {
                    "organism_id": first_payload.get("organism_id"),
                    "focus": first_payload.get("focus"),
                    "place": first_payload.get("place"),
                    "helpers_max": helpers_max,
                },
            })
        else:
            out.extend(run)

    for event in events:
        if event.get("kind") in COLLAPSIBLE_KINDS:
            key = _collapse_key(event)
            if run_key == key:
                run.append(event)
                continue
            flush()
            run = [event]
            run_key = key
        else:
            flush()
            run = []
            run_key = None
            out.append(event)
    flush()
    return out


def render_event(event: dict[str, Any]) -> str:
    tick = event.get("tick")
    kind = event.get("kind")
    p = event.get("payload", {})
    if kind == "birth":
        parents = p.get("parent_ids") or p.get("parent_lineage_ids") or []
        gen = p.get("generation")
        mode = p.get("mode")
        lin = p.get("lineage_root_id")
        return f"  t={tick:>5}  born     gen={gen} lineage={lin} mode={mode} parents={parents}"
    if kind == "structure_built":
        helpers = len(p.get("helpers", []) or [])
        ext = "extended" if p.get("extended") else "built"
        return (f"  t={tick:>5}  {ext:8s} {p.get('structure', '?')} "
                f"scale={p.get('scale', '?')} place={p.get('place', '?')} helpers={helpers}")
    if kind == "causal_unlock":
        seq = ">".join(p.get("sequence", []))
        return (f"  t={tick:>5}  unlock   {p.get('energy', '?')} via {seq} "
                f"place={p.get('place', '?')} released={p.get('released', 0):.1f}")
    if kind == "collaboration":
        helpers = p.get("helpers", []) or []
        return (f"  t={tick:>5}  collab   focus={p.get('focus', '?')} place={p.get('place', '?')} "
                f"helpers={len(helpers)}")
    if kind == "tool_success":
        return (f"  t={tick:>5}  tool     {p.get('affordance', '?')} place={p.get('place', '?')} "
                f"score={p.get('score', 0):.2f} fit={p.get('situation_fit', 0):.2f}")
    if kind == "crafted_tool":
        return (f"  t={tick:>5}  craft    {p.get('artifact', '?')} target={p.get('target', '?')} "
                f"place={p.get('place', '?')} fit={p.get('target_fit', 0):.2f}")
    if kind == "mark_lesson_written":
        return (f"  t={tick:>5}  inscribe affordance={p.get('affordance', '?')} "
                f"problem={p.get('problem_kind', '?')} place={p.get('place', '?')} "
                f"clarity={p.get('clarity', 0):.2f}")
    if kind == "notable_death":
        prof = p.get("success_profile", {}) or {}
        return (f"  t={tick:>5}  died     cause={p.get('cause', '?')} kind={p.get('kind', '?')} "
                f"score={p.get('score', 0):.1f} "
                f"tool_use={prof.get('tool_use', 0):.1f} unlock={prof.get('causal_unlock', 0):.1f} "
                f"struct={prof.get('structure', 0):.1f} repro={prof.get('reproduction', 0):.1f} "
                f"offspring={p.get('offspring_count', 0)}")
    if kind == "checkpoint_saved":
        return f"  t={tick:>5}  ckpt     reason={p.get('reason', '?')}"
    if kind == "movement_run":
        places = p.get("places", [])
        return (f"  t={tick:>5}  ...     {event['count']:>3} moves "
                f"motive={p.get('dominant_motive', '?')} dests={places} "
                f"(through t={event['tick_end']})")
    if kind == "collaboration_run":
        return (f"  t={tick:>5}  ...     {event['count']:>3} collabs "
                f"focus={p.get('focus', '?')} place={p.get('place', '?')} "
                f"helpers_max={p.get('helpers_max', 0)} (through t={event['tick_end']})")
    if kind == "movement_attempt":
        return (f"  t={tick:>5}  move     {p.get('origin', '?')}>{p.get('destination', '?')} "
                f"motive={p.get('dominant_motive', '?')}")
    return f"  t={tick:>5}  {kind}"


def render_arc(arc: dict[str, Any], collapse: bool = True) -> str:
    org = arc["organism_id"]
    lineage = arc["lineage_id"]
    span = f"t={arc['tick_lo']}-{arc['tick_hi']}"
    headline = arc["headline"]
    anchor_kind_counts = defaultdict(int)
    for a in arc["anchors"]:
        anchor_kind_counts[a.get("kind")] += 1
    anchor_summary = ", ".join(f"{k}x{v}" for k, v in sorted(anchor_kind_counts.items()))

    lines = [
        f"Organism {org}  lineage={lineage}  places={arc['places']}  {span}  score={arc['score']:.1f}",
        f"  headline:  {render_event(headline).strip()}",
        f"  anchors:   {anchor_summary}",
    ]

    events = collapse_steady_state(arc["events"]) if collapse else arc["events"]
    for event in events:
        lines.append(render_event(event))
    return "\n".join(lines)


def find_specialist_traps(arcs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Surface organisms whose 'success' came from a single repetitive loop.

    Signature: notable_death anchor with high tool_use, near-zero across the
    other agentic categories, no offspring, and the arc visited at most one
    place. These are the 2025-class organisms that the raw success_profile
    score would mistake for high achievers.
    """
    traps: list[dict[str, Any]] = []
    for arc in arcs:
        death_anchor = next(
            (a for a in arc["anchors"] if a.get("kind") == "notable_death"),
            None,
        )
        if death_anchor is None:
            continue
        payload = death_anchor.get("payload", {})
        profile = payload.get("success_profile", {}) or {}
        tool_use = profile.get("tool_use", 0.0) or 0.0
        if tool_use < 50.0:
            continue
        factor = diversity_factor(profile)
        offspring = payload.get("offspring_count", 0) or 0
        if factor > 0.4 or offspring > 0 or len(arc["places"]) > 1:
            continue
        traps.append({
            "organism_id": arc["organism_id"],
            "lineage_id": arc["lineage_id"],
            "place": arc["places"][0] if arc["places"] else None,
            "tool_use": tool_use,
            "diversity_factor": factor,
            "death_score": payload.get("score", 0.0) or 0.0,
            "cause": payload.get("cause"),
            "tick_lo": arc["tick_lo"],
            "tick_hi": arc["tick_hi"],
        })
    traps.sort(key=lambda t: t["tool_use"], reverse=True)
    return traps


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python analysis/scripts/arc_report.py runs/<run_dir> [--top N] [--write-jsonl]")
        raise SystemExit(2)
    run_dir = Path(sys.argv[1])
    args = sys.argv[2:]
    top_n = 10
    write_jsonl = False
    i = 0
    while i < len(args):
        if args[i] == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1])
            i += 2
        elif args[i] == "--write-jsonl":
            write_jsonl = True
            i += 1
        else:
            print(f"unknown arg: {args[i]}")
            raise SystemExit(2)

    story_path = run_dir / "story_events.jsonl"
    if not story_path.exists():
        print(f"missing {story_path}")
        raise SystemExit(1)

    events = load_events(story_path)
    arcs = build_arcs(events)
    traps = find_specialist_traps(arcs)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    print(f"Arc report: {run_dir}")
    print(f"  raw events: {len(events)}  arcs detected: {len(arcs)}  showing top: {min(top_n, len(arcs))}")
    print()
    for arc in arcs[:top_n]:
        print(render_arc(arc))
        print()

    # Lineage civilizations: summary.json already aggregates these per run end.
    # Render the top scoring lineages so the run output tells a complete story
    # without requiring a separate grep through events.jsonl.
    lineages = (summary.get("lineages") or {}).get("top_living") or []
    if not lineages:
        # Fallback: lineage_summary may emit "by_living" or just a flat list.
        candidate = summary.get("lineages") or {}
        for key in ("top_living", "by_living", "top"):
            if isinstance(candidate.get(key), list):
                lineages = candidate[key]
                break
        if not lineages and isinstance(candidate, list):
            lineages = candidate
    if lineages:
        print("Top Lineages")
        for row in lineages[:8]:
            profile = row.get("profile", {}) or {}
            tools = row.get("tool_use_counts", {}) or {}
            tool_summary = ", ".join(f"{k}:{v}" for k, v in list(tools.items())[:5])
            profile_summary = ", ".join(
                f"{k}={v:.1f}" for k, v in sorted(profile.items(), key=lambda kv: kv[1], reverse=True)[:4]
            )
            print(
                f"  lineage={row.get('lineage_root_id'):>4}  "
                f"living={row.get('living', 0):>3} ({row.get('living_neural', 0)} neural)  "
                f"max_gen={row.get('max_generation', 0):>2}  "
                f"born={row.get('born', 0):>3}  "
                f"offspring={row.get('offspring_total', 0):>3}  "
                f"tools={row.get('successful_tools_total', 0):>3}  "
                f"score={row.get('score', 0):.1f}"
            )
            if profile_summary:
                print(f"      profile: {profile_summary}")
            if tool_summary:
                print(f"      tools:   {tool_summary}")
            top_ids = row.get("top_living_ids", []) or []
            if top_ids:
                print(f"      top organisms: {top_ids}")
        print()

    # Brain capacity & attention trajectory across the run, sampled from
    # aggregates. Tells you whether brains grew during the run and whether
    # attention concentrated as the population learned.
    aggregates: list[dict[str, Any]] = []
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") == "aggregate":
                aggregates.append(item)
    if aggregates and any("brain_capacity" in a for a in aggregates):
        print("Brain Capacity Trajectory")
        stride = max(1, len(aggregates) // 8)
        for item in aggregates[::stride]:
            cap = item.get("brain_capacity") or {}
            attn = item.get("attention_stats") or {}
            print(
                f"  tick {item['tick']:>6}: brains={cap.get('count', 0):>4}  "
                f"hidden mean={cap.get('mean', 0):>5.1f}  max={cap.get('max', 0):>3}  "
                f"p90={cap.get('p90', 0):>3}  "
                f"attn max_fid={attn.get('mean_max_fidelity', 0):>4.2f}  "
                f"concentration={attn.get('mean_concentration', 0):>4.2f}"
            )
        print()

    if traps:
        print("Specialist Traps")
        print(f"  {len(traps)} organism(s) accumulated high tool_use with zero diversification, no offspring, single place.")
        print(f"  These look like big achievers by raw score but actually got stuck in a memory loop.")
        for t in traps:
            print(f"  - org={t['organism_id']:>4} lineage={t['lineage_id']} place={t['place']} "
                  f"t={t['tick_lo']}-{t['tick_hi']} tool_use={t['tool_use']:.0f} "
                  f"divfactor={t['diversity_factor']:.2f} death_score={t['death_score']:.0f} "
                  f"cause={t['cause']}")
        print()

    if write_jsonl:
        out_path = run_dir / "arcs.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for arc in arcs:
                f.write(json.dumps({
                    "organism_id": arc["organism_id"],
                    "lineage_id": arc["lineage_id"],
                    "tick_lo": arc["tick_lo"],
                    "tick_hi": arc["tick_hi"],
                    "places": arc["places"],
                    "score": arc["score"],
                    "headline_kind": arc["headline"].get("kind"),
                    "anchor_kinds": [a.get("kind") for a in arc["anchors"]],
                    "event_count": len(arc["events"]),
                }) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
