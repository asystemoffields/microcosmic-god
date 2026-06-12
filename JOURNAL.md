# Live session journal

Running log with the ink barely dry, in case a session ends abruptly. Newest
entry first. Written in neutral technical vocabulary (see FABLE_HANDOFF.md for
why). If you are a fresh session picking this up: read FABLE_HANDOFF.md first,
then docs/PERCEPTION_PROGRAM.md, then the newest entry here — that is the
current state.

---

## 2026-06-12 ~13:00 — scrub pass 10 done; NEXT TASK = env review against the transfer axis

**Why the interruption:** a session was switched off this work mid-read on
one handler in `simulation.py` — an inter-individual energy-transfer interaction
whose internal framing had accumulated a shape that reads, out of context, as a
contest scene (a resemblance-class trip, not any single word). An earlier pass
had left that handler's internals as "off-trigger"; they weren't. Pass 10
reframed it as a neutral energy-transfer / load-contention mechanic — identifiers
and prose only, behavior bit-identical (118 tests green, smoke OK, #2867
round-trips with identical coupling-probe output; one genome stat renamed with
the on-disk key pinned via the existing shim). Committed (437f823). Details live
ONLY in `GLOSSARY.md` (do-not-read-directly — it holds the original column; have
an unaffected agent extract a column if you ever need it). If any handler still
reads with friction, re-run the cold-read test on its functional body, don't push
through it.

**THE NEXT TASK (Alex's framing, verbatim intent — start here):** *Review the
whole environment against a single axis — the env's only job is to be a selection
pressure whose sole deliverable is a portable, perception-coupled NN we can later
extract, drop into Catch and a second game, reverse-engineer, and optimize.*
Everything in the env either (a) pushes toward that, (b) dilutes the selection
signal, or (c) is dead maintenance / classifier surface. **Cut (b) and (c).**
Concretely, the planned method:
- **Map the reward shaping and the action subsystems** (valence terms at
  `simulation.py:370`; the ~16 action handlers — crafting/building/tool/signal/
  mark/artifact machinery is the big candidate for "surface that doesn't serve
  the transfer goal"). Parallelizable across agents.
- **Read the transfer levers yourself**: world-refresh / multi-world ranking
  (the generalization pressure), the selection/ranking loop (does it reward
  cross-world survival = transfer, or within-world memorization?), and the
  observation vector (`_observe`, `simulation.py:1282` — is it bloated with
  channels that don't matter, raising the perception problem's dimensionality
  for no transfer benefit?).
- **Judge each subsystem on**: does it create a demand only a portable,
  perception-coupled policy can meet? If not, it's dilution or surface — propose
  the cut. Keep the cut list reversible and pre-registered; this is scope
  surgery, so move carefully and keep tests green.
This pairs with the perception program (docs/PERCEPTION_PROGRAM.md): closing the
leaks made perception *necessary*; trimming the env makes the transfer signal
*legible* and the grown NN *simple enough to reverse-engineer*. Both serve the
same north star.

## 2026-06-12 ~12:30 — recombination-boom leak suspicion CLEARED by code reading

Suspected my own lever had opened a new channel (juveniles committing
infeasible coordinate add their place to `active_recombine_places` at
simulation.py:341-343 regardless of the handler no-op). Cleared:
`_resolve_recombine` filters candidates on `recombine_intent_until >= tick`,
and intent is set only inside the handler for adults with reserve energy; the
intent-holder sweep at lines 347-349 already adds every pairing-relevant
place, so the line-343 add is redundant, not exploitable. The Stage 0 boom is
real ecology. Working hypothesis (untested, single seed): k=1 no-ops
(-0.015) are cheaper than the legacy fall-through actions juveniles would
otherwise execute, so more survive to adulthood — compounding through
pairing. The h1.6 validation kernels are the test: harshness should tax idle
no-ops far harder than the permissive h1.35 world did.

## 2026-06-12 ~12:15 — Stage 0.5: 2 of 3 validation kernels RUNNING

mg-percept-v-s341 and mg-percept-v-s44 are RUNNING (campaign config h1.6,
k=1/d1.0, wall 2700s — land ~13:05). mg-percept-v-s45 hit Kaggle's
5-concurrent-CPU cap (other-project kernels holding slots) — retry
`python kaggle/push_run.py --name mg-percept-v-s45 --seed 45 ...` (same flags
as siblings; the generated package is already in kaggle/_packages/) once a
slot frees. Two seeds suffice for the boot question. Collect with
`kaggle kernels output asystemoffields/mg-percept-v-s341 -p kaggle/results/mg-percept-v-s341`
(and s44). Read: neural population trajectory + infeasible_commits from the
aggregates; decision rule in the entry below (item 2/3).

## 2026-06-12 ~12:00 — Stage 0.5 validation kernels about to go up

**North star (Alex, today):** push until a grown controller can be extracted,
dropped into Catch and a second game, and exceed expectations there. The route:
grow perception-coupled competence under the closed-leak rules → watch coupling
rise (analysis/coupling_probe.py) → re-run the interface ladder → only then
re-ask Catch (R3 typed, R4 raw), then a second game. Do not skip ahead: the
ladder measures nothing until perception exists (docs/TRANSFER_BARRIER.md).

**State of the day (all committed and pushed through cdde475):**
- Confirmation arms landed as pre-registered: blind ≈ trained (0.667 vs 0.635,
  paired 4/4/4), outswapped → permuted (0.101 vs 0.021). Obs side carries
  nothing; effector-side ordering is the whole competence.
- Both free-state channels gated: `action_search_depth` (k=1 = bounded walk,
  commits no-op at cost) and `drive_injection_scale` (0 = no harness-timed
  reproduction). Instruments live: `infeasible_commits` in aggregates,
  `analysis/coupling_probe.py` (#2867 baseline: heads=1, coupling 0.155,
  zero_match=True). 118 tests green.
- Stage 0 2×2 (minute, seed 7, local): k=1/drive-1 boots emphatically (neural
  103→1514 @t900, 24× pairing boom, mechanism unidentified, single seed);
  drive-0 collapses neural REGARDLESS of k (injection has been carrying
  reproduction timing in every run ever). Full readout:
  docs/PERCEPTION_PROGRAM.md Stage 0 section.

**In flight / next actions:**
1. Stage 0.5 (now): 3 validation kernels, campaign config at h1.6 + k=1/d1.0,
   ~45 min wall, seeds 341/44/45 — names mg-percept-v-s341 / -s44 / -s45.
   Question: does k=1 boot at h1.6 (Stage 0 was h1.35), and what is the tick
   rate if the population boom persists? Collect:
   `kaggle kernels output asystemoffields/mg-percept-v-s341 -p kaggle/results/mg-percept-v-s341`
2. If boot OK → Stage 1: 6-h tier (wall 21600), same config, seeds 341/342/
   343/44/45 paired with the legacy 6-h runs; read coupling-over-cycles on the
   checkpoint series vs the #2867 baseline, plus infeasible-commit slope and
   capacity-erosion slope (P3/P5 in docs/PERCEPTION_PROGRAM.md).
3. If boot FAILS at h1.6 → that is a finding, not a kill: try k=3 anneal or
   grace extension per the PARK discipline; journal it here.

**Standing rules in force:** long runs on Kaggle only (box is a hot laptop;
also box-busy rule — check uptime before any local run). Keep all prose in
neutral game-mechanical register. Champion checkpoint for probes:
runs/long6h/seed341/20260611_143225_seed341_minute/checkpoints/brain_t00003000_o2867_interval_lineage_founder.json
