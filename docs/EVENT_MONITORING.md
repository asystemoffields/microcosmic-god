# Event Monitoring Strategy

Microcosmic God should not save everything. It should make everything monitorable, then persist only what becomes important.

## Principle

Most substrate dynamics should be counted, summarized, or sampled. Full event records are for moments that are rare, consequential, surprising, or attached to organisms/artifacts/places that later become important.

The simulator needs three memory layers:

- Hot counters: cheap aggregate counts for routine substrate events such as wear, drift, decay, failed actions, resource changes, and common movement.
- Rolling context: bounded recent histories around organisms, places, structures, artifacts, marks, and lineages.
- Promoted records: durable story events saved when an event crosses an interestingness threshold or becomes relevant through later success.

## What To Keep Cheap

Do not emit per-tick reports for every object, material, structure, mark, field, or organism. Prefer counters and summaries for:

- routine material decay
- ordinary metabolic changes
- background climate and field drift
- common failed attempts
- unremarkable movement
- tiny resource diffusion or advection
- marks that never get read
- tools that never matter after creation

These should remain inspectable through aggregate histories and final summaries, but not dominate run storage.

## What To Promote

Promote richer event records when something becomes causally interesting:

- first successful affordance, structure type, causal unlock, or mark-read lesson
- rare or high-payoff energy unlocks
- a tool/structure/mark that is reused, copied, teaches another agent, or changes survival
- sudden lineage expansion or collapse
- unexpected death of a high-scoring learner/operator
- cross-place knowledge movement, such as reading a mark then making/marking elsewhere
- sharp shifts in prediction error, tool skill, reproductive success, or habitat mastery
- events involving agents later saved as checkpoints

## Retrospective Promotion

Some events only become interesting later. To support that without saving everything:

- Keep bounded rolling traces per agent and place.
- When a brain is checkpointed, include the recent local trace, relevant marks, tools, structures, and causal challenge state.
- When a lineage becomes standout, promote a compact lineage story from recent parent/child/operator records.
- When a mark/tool/structure is reused often, start saving richer events for that object from that point onward.

## Interestingness Heuristic

A future `EventObserver` should score candidate events roughly as:

```text
interestingness =
  rarity
  + consequence magnitude
  + surprise/prediction error
  + energy unlocked
  + tool/structure/mark reuse
  + lineage/checkpoint relevance
  + cross-place or cross-agent transmission
  - routine background frequency
```

This score is for logging and inspection only. It must not feed back into sealed-run fitness.

## Storage Contract

Each run should produce:

- compact aggregate counters
- bounded recent traces
- promoted story events
- checkpoint payloads for selected neural agents
- final summaries with enough context to explain extinctions, standouts, and world changes

The goal is to find the stories without drowning in the substrate.

## Current Implementation

Runs now include `story_events.jsonl` alongside `events.jsonl`. Routine events still flow into counters and aggregates, while an observer promotes rare or consequential records such as causal unlocks, first/strong tool events, structures, intentional lesson inscriptions, successful mark reads, notable births/deaths, and checkpoint saves.

The observer keeps bounded recent context by subject (`organism:*`, `place:*`, `affordance:*`, etc.) and writes only compact payloads plus nearby context. It is descriptive only: story promotion does not alter fitness, action selection, reproduction, learning, or world physics.

Plain marks remain cheap telemetry. Intentional lesson writes and successful reads can become story events because they may connect tool knowledge across agents, places, and time.
