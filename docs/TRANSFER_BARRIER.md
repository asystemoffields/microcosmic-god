# Why it didn't transfer to Catch — diagnosis & program (2026-06-12)

The original cross-task test (Catch) showed no advantage for developed
controllers over random. We jumped straight to it; the adapter problem
dominated and the result was uninformative (see git log / TRANSFER_RUNWAY).
Yesterday's probe-worlds + gauntlet evidence lets us replace "it didn't
transfer" with specific, mostly-testable causes.

## Diagnosis (evidence in hand)

1. **The competence is indexical — tied to the input schema.** Gauntlet:
   permuted (same weight *values*, shuffled) → 0.02 alive vs trained 0.64
   across 12 paired worlds. The merit lives in weight *arrangement* against a
   fixed observation layout. Catch arrived through an ad-hoc obs/action
   adapter — from the controller's view, an input permutation. We now have a
   direct knob for exactly this (the `remapped` arm).
2. **No fast re-mapping machinery.** Gauntlet: frozen (learning off) ≈ trained,
   even across 3 physics rewrites/probe. Lifetime learning contributes ~0 to
   robustness. So when Catch demanded "re-map new inputs to old computation,"
   the substrate had no mechanism for it — its learning ops are tuned for slow
   within-world value drift, not interface remapping.
3. **Selection strips the very machinery transfer needs.** At h1.6 the
   pools develop plasticity scales DOWN (→0.90–0.96) and capacity erodes
   (docs/LONG6H). Nothing at home rewards re-mapping or spare capacity, so
   neither is maintained. Transfer is out-of-distribution by construction: a
   *demand type* the home world never makes.
4. **Adapter dominance (documented proximate cause).** Both trained and random
   had to learn the Catch interface from scratch; that learning swamped any
   core-competence gap.

## The interface-distance ladder (the instrument)

Localize where transfer dies instead of asserting a binary failure. Each rung
adds interface distance; the same probe-world battery scores all of them:

- R0 native — trained baseline (have it).
- **R1 within-group input remap** — permute each typed encoder's columns
  inside its group span; types preserved, wiring scrambled. *(running now —
  `remapped` arm; the controlled Catch question)*
- R2 cross-group remap — permute whole observation groups (resource span lands
  where the obstacle span was). Tests whether typing itself is load-bearing.
- R3 typed Catch — Catch observations sorted into the existing type ontology,
  native action schema preserved.
- R4 raw Catch — the original failure, for calibration.

Prediction if the diagnosis holds: a cliff at R1 (indexical), partial rescue
at R3 vs R4 (typing carries some competence), and the gap that remains is what
new machinery must close.

## What to invent / equip (ranked by leverage)

1. **Typed interface contracts.** Tie competence to *types*, not channel
   indices; express a new task by typing its raw observations into the
   existing ontology. The encoders already do per-type encoding — push it so a
   new task is a typing judgment, not a learned remap. R2 vs R1 says how real
   the typing is today; R3 says how much it buys.
2. **Schema drift as home pressure.** Per-line channel remaps within groups,
   applied at instantiation, so selection pays for mapping robustness — the
   perceptual cousin of the contract-drift world feature. This is the lever
   that would make controllers *non*-indexical in the first place. Default-off
   flag, then rerun the campaign and read R1 transfer as the dependent
   variable.
3. **Fast-weights mapping layer.** High-rate, low-capacity plasticity on the
   encoders only; core stays slow. A dedicated re-mapping mechanism instead of
   overloading persistence-tuned learning. Tests against R1/R3 directly.

## 2026-06-12 — R1 measured, and the diagnosis went deeper than the ladder

**R1 result (n=29/36 at writing, 12 paired worlds, same battery as the
gauntlet): remapped alive 0.634 vs trained 0.635.** No cliff. The registered
prediction (death between R0 and R1) is falsified — but not for the hopeful
reason (typing as the portable boundary). Trace analysis found the real
mechanism:

- On its own live observation stream (1600 forward calls in a probe world),
  #2867 emits **2 distinct action rankings**, differing by one adjacent swap.
  Zero-observation input spawns the behavioral head exactly:
  `drain > coordinate > build > move ≈ use_tool > clone_perturb`.
- Per-action output std over time is 0.002-0.004; the fixed gaps between
  actions are 0.05-0.5. The observation pathway is functionally disconnected
  — a two-orders-of-magnitude-too-small perturbation on a constant program.
- The program is attractor-encoded, not a bias trick: bias_o alone gives a
  different head (`drain > eat > pickup`); the settled recurrent state
  contributes 2× the output spread (std 0.144 vs 0.070) and produces the
  realized ordering. The competence genuinely lives in weight *arrangement* —
  which is why `permuted` collapses to 0.02 — but none of it is perceptual.
- Realized behavior in-world: ~84% coordinate (cheap idle), ~11% drain
  (whenever feasible), everything else at the exploration floor. A blind
  energy-drainer: it claims energy from whatever individual becomes adjacent,
  without reading which one or whether it is the right move.

**The world's action-resolution mechanism is doing the perceiving.** Action
choice walks the controller's ranked list until something is *feasible*; the
feasibility gates (target adjacent? materials in hand? adult + energy?) are
computed by the world from exactly the state the controller would otherwise
have to observe. A fixed priority list + feasibility fallback IS a reactive,
context-sensitive policy — with the context sensitivity supplied free by the
harness. Selection found this channel and used it instead of perception,
because wiring real perception through perturbation is expensive and the
leak is free. This explains every arm in one stroke: frozen ≈ trained
(learning never mattered), remapped ≈ trained (inputs never mattered),
permuted dead (the ordering is destroyed), random poor (wrong ordering).

**Confirmation arms — LANDED as predicted (full battery, 12 paired worlds):**
`blind` (encoder weights zeroed) alive **0.667 vs trained 0.635**, paired
worlds 4 wins / 4 losses / 4 ties — dead even; the observation pathway can be
removed outright without cost. `outswapped` (action identities permuted on
the output side only) alive **0.101 vs permuted 0.021** (trained 0.635) —
collapse toward permuted, 7/12 paired wins over it, a small residue
presumably from preserved core dynamics. The obs-side ladder (R1-R4)
collapses to a single point for this champion; the effector-side ordering is
the entire competence.

**Second free-state channel (found while building the lever):** the realized
energy-conditional behavior (84% coordinate when rich, 11% drain when poor)
is not the feasibility walk alone — the chooser itself injects a boost into
coordinate/clone_perturb outputs exactly when an individual is adult and
energy-rich (`drive_injection_scale`, now config-gated). The champion's
context sensitivity was harness-supplied twice over: the walk filtered by
feasibility, and the injection timed spawning. Both are now removable
knobs; the full subsidy ledger and growth program live in
docs/PERCEPTION_PROGRAM.md.

## What this does to the program

1. The inventions list above is mooted *in this order*: schema drift can't
   bite (nothing reads the schema), fast-weights binding has nothing to bind,
   typed contracts protect an interface that carries no information. They
   become relevant only after perception pays.
2. **The actual lever: close the feasibility leak.** Make blindness
   unprofitable, then re-measure. Options, cheapest first: (a)
   feasibility-blind resolution — an infeasible chosen action wastes the tick
   (and a little energy) instead of falling through to the next ranked
   feasible one; (b) state that matters but isn't gate-visible — e.g. a
   resource variant that harms unless a cue channel distinguishes it; (c) the
   already-planned cue-reliability world feature. (a) is a one-knob change to
   `_choose_action_from_outputs` and converts the gates from oracle to cost.
   *(Built 2026-06-12: `action_search_depth` (k=1 = full pressure) plus
   `drive_injection_scale` (0 = no harness-timed spawning), with an
   `infeasible_commits` counter in every aggregate and
   `analysis/coupling_probe.py` as the dependent variable. Necessity vs
   sufficiency, pre-registered predictions, and staging:
   docs/PERCEPTION_PROGRAM.md.)*
3. Pre-registered predictions for feasibility-blind resolution: blind arm
   drops below trained; obs-output coupling (output std on a fixed trace)
   becomes selectable and rises across cycles; only then does the ladder
   measure anything, and only then is Catch worth re-asking.
4. This unifies with the capacity-erosion stream: capacity erodes and
   perception is absent for the same reason — the world asks no question that
   only observation can answer. The transfer barrier and capacity erosion are
   one phenomenon seen from two sides.

## Why this may matter beyond the sandbox

"What makes a learned competence portable across interfaces rather than tied
to the one it grew up on" is the micro-scale form of a central question in
transfer/continual learning. The ladder turns it into a measurement, and the
substrate lets us *develop* the answer rather than hand-design it. Keep scope
honest: claims are about this sandbox; the value is the method (typed
contracts + drift pressure + a distance ladder), not a leaderboard number.
