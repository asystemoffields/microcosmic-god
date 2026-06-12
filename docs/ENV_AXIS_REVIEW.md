# Environment review against the single axis (2026-06-12)

**The axis.** The environment's only job is to be a selection pressure whose
sole deliverable is a **portable, perception-coupled neural-network policy**
we can extract, drop into other games, reverse-engineer, and optimize.
Everything in the env either (a) pushes toward that, (b) dilutes the
selection signal, or (c) is dead maintenance/surface. This review cuts (b)
and (c). Evidence base: `review/ACTION_MAP.md` (per-action + subsystem map),
`review/SHAPING_MAP.md` (shaping, chooser bypasses, champion metrics,
spawning economics), the direct lever read in `JOURNAL.md` (2026-06-12
~17:50), and the Stage 0.5 tax data in `PERCEPTION_PROGRAM.md`.

**The bar (Alex, 2026-06-12).** "Portable" is to be read broadly: a
competence that transfers to Catch but not to a qualitatively different game
is a no-go; easy targets are signal, not wins. The deliverable is a
*pocketknife*, not a wrench — which means the thing selection must grow is
not a fixed cue→action mapping (a wrench: works only where channel 30 means
what it meant at home) but **general perception-action machinery: the
ability to discover and re-map cue→consequence contracts quickly**, carried
by the controller's own plasticity, prediction heads, and replay. The
extraction interface is adapter-mediated either way (the Catch harness
learns linear maps in/out), so what actually rides through extraction is the
internal dynamics — exactly the part "contract drift" selects for and a
fixed mapping does not.

## 1. Unified diagnosis

Four levels of harness subsidy answered the perception question for the
policy. Two are closed; two were open and are the subject of this review:

1. **Feasibility walk** — closed (`action_search_depth`, k=1).
2. **Drive injection** — closed (`drive_injection_scale`; removal is its own
   staged rung, Stage 1.5).
3. **The verb/argument split (systemic, was open).** The controller emits
   one of 15 verbs; scripted handler code picks every argument and answers
   every situation question (~210 lines of situation cognition for tools,
   craft target scoring that reads the causal challenge's expected step
   directly, memory-greedy move destinations, weakest-target selection,
   helper recruitment, a scripted partner-quality score inside reproduction
   itself, a non-heritable skill table that improves outcomes with no
   controller involvement). Under this split, k=1 can only force "rank an
   executable verb" — never "perceive what to act on."
4. **The gate-blind tax (was open).** 83% pooled (86/86/78% per seed) of the
   k=1 infeasible tax lands on use_tool/craft/build/pickup, whose gate
   inputs (artifact state, collective materials, exact inventory counts) have
   zero observation dims. For three seeds running this was an unwinnable
   pressure component — plausibly what kills marginal boots (s341).

Secondary findings that compound the case:
- **The refresh persists structures/marks/signals/materials** while wiping
  exactly the knowledge perception could re-learn — accumulation strategies
  compound straight through the generalization probe.
- **Champion metrics select accumulators** (child_count×6, age, energy,
  raw capacity), i.e. they extract the blind-priority shape we are trying to
  leave behind. Only learner_champion is near-axis, and its fit counter is
  longevity-confounded.
- **A second ungated spawn-timing channel**: the combine-intent window (6-19
  ticks, passive resolution) + cheap combine reserve + the 2.5% exploration
  floor lets a blind policy reproduce via combine without ever ranking
  coordinate.
- The crafting stack is also where passes 3, 4, and 10 of the vocabulary
  neutralization all concentrated — it is the maintenance and surface burden
  of the repo, on top of being dilution.

## 2. Decision: era 2

Authorized by Alex (full freedom to reshape; cuts are true removals, not
config gates). Reversibility = git: the last era-1 commit is tagged
**`era1-full-env`**; era-1 checkpoints stay probe-able (probes read sizes
from the checkpoint itself).

### 2.1 CUT (remove from code)

| Cut | Approx. weight | Grounds |
|---|---|---|
| Actions `pickup`, `craft`, `build`, `use_tool`, `mark`, `observe` | ~1,400 lines of handlers + targeting | gate-blind tax; verb/argument split; D/X verdicts in ACTION_MAP §2 |
| Artifacts / materials / inventory (MATERIALS table, Artifact, wear, inventory fields) | ~565 lines | survival feedback real but invisible to the policy by construction (zero obs dims) |
| Structures (Structure, capability derivation, decay model, world-side processing) | ~475 lines | public-good free-rider economics; decay modulates unperceivable durability |
| Marks chain (write/read/lessons/portable/author feedback/erosion) | ~525 lines + 8 counters | the single largest dead surface; payoff rounds to zero |
| Causal challenges (generation, advancement, unlock release) | ~190 lines | flagship-in-principle, but state has zero obs dims and is consumed only by handler scripts; its role passes to the cue contract (§2.3) |
| tool_skill / SKILL_TRANSFER / skill-breadth obs dim | ~535 lines | non-neural, non-heritable adaptation channel competing with the controller for credit |
| Collaboration/helpers (candidacy, support, expedition relocation, collective materials) | ~180 lines | unperceivable success variance; recruits controllers without their choice |
| `planning` amplifier (`_interaction_control`) | ~15 lines + call sites | converts prediction fit into scripted success bonuses; consumers all cut |
| place_memory + memory-greedy move targeting + movement motive telemetry | ~250 lines | harness-side perception→action loop; write-only telemetry |
| demonstrations, first_tool bucket, success_profile trimmed to survivors | ~100 lines | consumers removed; novelty bookkeeping |

### 2.2 KEEP (the lean selection core)

eat, absorb_solar (clean perception-or-die energy loops, observable
resources); rest (+replay — controller-internal substrate); move (leave/stay
decision; destination becomes an honest uniform-random neighbor — no scripted
scorer; relocation economics kept, telemetry cut); forage (simplified:
resource seeding only, converters need it); drain (the density contest —
target becomes **random co-located** instead of weakest, removing the last
targeting oracle; observable-crowding precondition is the perception demand);
signal (the one social channel that reaches the observation vector);
coordinate/clone_perturb + Optimizer (the selection currency; both
free-state channels remain gated); collector/converter scripted kinds (the
food base, load-bearing for boot); physics/terrain/resources/sealed reserve;
world refresh (**now also clears signals on refresh**; structures/marks gone
with their subsystems, so the accumulation leak closes itself); patch
recovery (parked, unchanged); the whole controller substrate
(TinyController, modular blocks, plasticity, prediction heads, episodic
replay — this is the deliverable's body and is now load-bearing for the
pocketknife bar); checkpointing pipeline (re-aimed, §2.4); aggregates,
structure_events genealogy, coupling probe, run digest.

### 2.3 ADD: `tap` and the cue contract (the winnable demand)

Cutting the gate-blind stack removes the *unwinnable* tax but leaves only
thin gates (adult, mobility). The winnable demand is installed in the same
move — this is one design decision, not two:

- **`tap` (new action, era 2.0).** Gate-free. Attempts to release the
  place's sealed reserve. Payoff keyed to an observable cue channel: when
  the cue is high, release `min(sealed, 2 + 8·cue)` — actor takes a share,
  the rest lands in place resources; when the cue is low, the tap misfires:
  energy cost plus a small health hit. Both inputs (sealed reserve, cue) are
  existing observation dims. A constant-tap blind policy bleeds; a policy
  that reads two channels prospers. This replaces the use_tool/causal-unlock
  payoff topology (its S part) at ~1/20th the machinery.
- **Cue contract drift (era 2.1, the pocketknife pressure).** The identity
  of the cue channel is drawn **per world refresh** from a small physics set
  (e.g. residue_activity / current_exposure / wet_dry_cycle). The contract
  is discoverable only through tap outcomes and prediction errors — so
  selection rewards lines whose controllers *re-map quickly within a
  lifetime* (plasticity, prediction, replay), not lines that hardcode one
  channel weight. Fixed-cue 2.0 boots and validates the mechanism; drifted
  2.1 is what aims at general competence. Config: `tap_cue_drift` (off =
  2.0 behavior).

### 2.4 Adjustments to keep selection and extraction honest

- **Champion scoring re-aim**: `learner_champion` keys on prediction-fit
  *rate* (fit per tick alive, not the longevity-confounded accumulator);
  `tool_champion` → `tap_champion` (successful discriminating taps);
  `overall_champion` re-weighted toward fit rate + tap discrimination, with
  child_count/age/capacity terms demoted. spawn_champion and line_founder
  stay (genealogy instruments).
- **`_partner_score`**: drop the skill-breadth and child_count terms
  (accumulation leaks); keep health/energy/mobility/manipulator + param
  distance + noise.
- **New knobs (legacy-default)**: `combine_intent_window_scale` (1.0 =
  legacy 6-19 tick window; the ungated spawn-timing twin becomes gateable),
  `exploration_floor` (0.025 = legacy; annealable later).
- **Observation vector (era 2)**: remove the inventory and skill-breadth
  dims → 40 base + 8 trace + 6 prediction + 8 event memory + 8 signal =
  **70 dims**. The `tool` labels in trace/event-memory/prediction-heads
  become `tap`. No other channel changes — the cue rides existing physics
  dims by design.
- **Probe compat**: relax `probe_worlds.py`'s hard OBSERVATION_SIZE assert
  to read sizes from the checkpoint; parametrize the two hard-coded 72/15
  constants in the Catch notebook's control arms.

### 2.5 Deferred (pre-registered as later rungs, not this surgery)

Drive-injection annealing (Stage 1.5); exploration-floor annealing; patch
recovery recalibration (parked); a controller-owned move-destination head
and typed interface contracts (R2/R3); partner choice as a perception
problem; any second observation-schema change.

## 3. Pre-registered predictions

- **P-E1 (tax becomes signal).** Era-2 k=1 at h1.6: the gate-blind share of
  infeasible commits collapses from ~83% to <5%; remaining tax concentrates
  on coordinate/clone_perturb (observable gates — legitimate pressure);
  per-neural-choice tax declines vs Stage 0.5.
- **P-E2 (boot non-inferiority).** Boot at h1.6 in ≥2/3 of seeds
  {341, 44, 45} at campaign config, k=1/drive 1.0. Directional: s341 boots
  (its failure tracked the unwinnable tax).
- **P-E3 (tap discrimination, era 2.0).** In booted runs, corr(tap rate,
  cue level) > 0 emerges within the run; mis-tap fraction declines over
  cycles. Instrument: per-action tap/mis-tap counters in aggregates.
- **P-E4 (the Stage-1 read, unchanged).** Champion coupling ratio rises
  above the #2867 baseline (0.155) across the 6-h tier; capacity-erosion
  slope shallows (P5 unification).
- **P-E5 (re-mapping signature, era 2.1).** After each refresh, mis-tap rate
  spikes then re-declines *within* lifetimes for high-coupling lines; the
  re-decline is absent/slower in low-coupling lines. This is the
  pocketknife's in-world fingerprint.
- **No-go line (transfer bar).** A champion that clears Catch but fails a
  qualitatively different second probe is a no-go, not a partial win. The
  ladder gets that second game before any win claim.

## 4. Risks

- **Boot dynamics shift.** The Stage-0 pairing boom rode k=1 no-op
  economics; removing six actions changes juvenile fall-through costs.
  Mitigation: minute-scale local shakedown, then a 45-min Kaggle validation
  at h1.6 (the Stage 0.5 protocol) before committing the 6-h tier.
- **Converter food base.** forage loses material seeding; verify resource
  seeding is preserved so converters still stock the residue economy.
- **Champion-metric churn.** Re-aimed scores change which checkpoints get
  saved; genealogy instruments (line_founder, spawn) are kept unchanged as
  the cross-era control.
