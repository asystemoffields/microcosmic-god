# Live session journal

Running log with the ink barely dry, in case a session ends abruptly. Newest
entry first. Written in neutral technical vocabulary (see FABLE_HANDOFF.md for
why). If you are a fresh session picking this up: read FABLE_HANDOFF.md first,
then docs/PERCEPTION_PROGRAM.md, then the newest entry here — that is the
current state.

---

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
