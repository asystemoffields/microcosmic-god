# Growing perception-coupled competence — setup audit & program (2026-06-12)

The transfer-barrier diagnosis (docs/TRANSFER_BARRIER.md) ended with one
sentence of consequence: *the world asks no question that only observation can
answer.* This document is the systematic answer to "what would it require for
the answer to be yes" — an audit of every place the harness or environment was
quietly footgunning that goal, what was built to fix it, what is deliberately
deferred, and the pre-registered predictions for the first runs. The point of
growing it: once a perception-coupled, portable competence *evolves* here, we
can take it apart and ask what made it portable — which is the question the
whole program is for.

## Requirement A — blindness must be unprofitable

Every channel through which the harness hands a controller state it never
perceived is a subsidy for blindness. Ledger:

| # | Channel | Status |
|---|---------|--------|
| A1 | **Feasibility walk.** Action resolution walked the full ranked list to the first feasible action; the gates (adult? inventory? params?) computed context for free. | **Closed** — `action_search_depth` (k≥1 bounds the walk; k=1 removes it; commit past the bound no-ops at cost). Default 0 = legacy. |
| A2 | **Drive injection.** The chooser itself boosts coordinate/clone_mutate outputs exactly when adult ∧ energy>0.62. Trace analysis shows this is what produced #2867's realized energy-conditional behavior (84% coordinate when rich, 11% drain when poor) on top of a constant ranking — the injection, not perception, was the policy's context sensitivity. | **Closed** — `drive_injection_scale` (0 removes it). Default 1.0 = legacy. |
| A3 | **World-side targeting.** Handlers micro-optimize within a macro-action: `_drain` picks the weakest adjacent target, `_eat` the best local food. The controller chooses *that* it acts, the world chooses *how well*. | **Open, second-order.** Becomes the binding subsidy only after A1/A2 pressure produces controllers that read state at all. Next rung, not now. |
| A4 | **Ungated actions.** k=1 taxes only gate-failures. `eat`, `forage`, `drain`, `signal`, `observe`, `rest`, `absorb_solar` pass the gate unconditionally and are adjudicated by their handlers — so a blind constant program (e.g. "eat forever") stays viable wherever one action is reliably profitable. | **Open — the sufficiency lever is environmental.** k=1 is necessary, not sufficient. What makes no constant action reliably profitable: cue-dependent payoffs (the planned cue-reliability feature; recalibrated patch recovery; a resource variant that harms unless a cue channel distinguishes it). Stage 2 below. |

## Requirement B — perception must be *possible*

If the observation vector doesn't carry the state the gates read, k=1 creates
an unwinnable task and pools collapse into constant-action fallbacks — the
opposite footgun. Audit of `_observe` against the gate inputs:

**Perceivable today:** energy ratio (channel 0), health (1), age (2), local
occupancy / capacity (drain & coordinate adjacency), neural fraction,
inventory ratio (pickup/craft/build counts), own params incl. mobility &
manipulator (the param-threshold gates), 7 resource channels + place physics
(eat/forage profitability). The first-order task — rank a currently-executable
action first — is expressible from the existing observation. No blocker.

**Gaps (noted, deliberately deferred):** artifact count is not observable
(half of the use_tool gate, part of craft); collective material count is not
observable (build's alternative gate); the adult threshold sits at 0.0125 on
the age channel (age/2000, threshold 25) — perceivable but low-salience.
**Why deferred:** any new channel changes `OBSERVATION_SIZE`, which changes
controller input width and breaks every existing checkpoint and probe
baseline. None of these gaps is first-order for boot viability. If a gap
proves binding, batch all channel additions into one schema change at a
new-era boundary and keep the old branch for legacy probes.

## Requirement C — we must be able to *see* it happening

Built today, both validated:

- **`analysis/coupling_probe.py`** — checkpoint-only (no world): streams
  seeded random inputs through a reconstructed controller and reports distinct
  rankings/heads, per-action output std vs inter-action gaps (coupling ratio),
  and the zero-input head. Baseline signature of the blind champion #2867:
  `heads=1, coupling=0.155, zero_head_matches_stream=True`. Run it over a
  checkpoints dir to get coupling-over-cycles. This is the dependent variable
  for "perception became selectable."
- **`infeasible_commits`** counter — per-action count of bounded-search
  commits, in every aggregate log and debrief. Reads the k-pressure directly:
  how often controllers pay for wrong rankings, and whether that declines
  across cycles.
- Existing: architecture census, structure_events genealogy, run_digest.

## Requirement D — no compatibility footguns

- Both new knobs default to byte-identical legacy behavior (k=0, scale 1.0);
  every old checkpoint, probe arm, and result stays valid and comparable.
- Old-config checkpoints load against new code (`getattr` defaults).
- Checkpoint/on-disk formats untouched. 118 tests green; smoke run at k=1
  verified end-to-end.

## Pre-registered: the k-sweep pre-flight (running now, local, minute-scale)

Three arms, seed 7, minute profile: **A** k=0/drive 1.0 (legacy control),
**B** k=1/drive 1.0, **C** k=1/drive 0.0 (full pressure).

- **P1 (boot):** B and C still boot at minute scale. Plants/fungi take the
  non-neural heuristic branch, untouched by k, so the ecology stands; the
  neural cohort pays wasted ticks but the developmental subsidy and
  exploration floor should carry founders.
  *If C goes neural-extinct at minute scale:* PARK, predicted
  scale-to-signal = anneal (boot at k=3, step to 1) or extend grace — per the
  standing scale-discipline rule, kill nothing below pre-registered scale.
- **P2 (pressure is real):** infeasible_commits > 0 in B/C and the
  action mix shifts toward always-feasible actions relative to A.
- **P3 (the long question, Kaggle tier):** across cycles at k=1, per-champion
  coupling ratio and distinct-head count rise above the #2867 baseline, and
  infeasible_commits per neural choice falls — perception being selected.
- **P4:** only after P3 lands does the interface ladder (R1-R4) measure
  anything; re-run the gauntlet on new-era champions (blind arm should now
  fall below trained) and only then re-ask Catch.
- **P5 (unification):** capacity-erosion slope shallows under k=1 — same
  phenomenon as the transfer barrier, so the same lever should move both.

## Staging

- **Stage 0** — this pre-flight (boot + counters). Local, short.
- **Stage 1** — Kaggle 6-h tier at the validated campaign config (mixed boot
  0.5, r1500, h1.6, rate 0.30, max-blocks 3, grace 150/0.35) × {k=1, drive 0},
  seeds paired with the legacy 6-h runs. Read P3/P5.
- **Stage 2** — environmental sufficiency: cue-reliability / recalibrated
  patch recovery, so ungated constant programs (A4) stop being viable.
- **Stage 3** — re-measure the ladder; reverse-engineer the grown perceivers
  (weights_in/attention structure of high-coupling champions vs #2867's
  attractor program).
