# Why it didn't transfer to Catch — diagnosis & program (2026-06-12)

The original cross-task test (Catch) showed no advantage for evolved
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
   populations evolve plasticity scales DOWN (→0.90–0.96) and capacity erodes
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
2. **Schema drift as home pressure.** Per-lineage channel remaps within groups,
   applied at instantiation, so selection pays for mapping robustness — the
   perceptual cousin of the contract-drift world feature. This is the lever
   that would make controllers *non*-indexical in the first place. Default-off
   flag, then rerun the campaign and read R1 transfer as the dependent
   variable.
3. **Fast-weights mapping layer.** High-rate, low-capacity plasticity on the
   encoders only; core stays slow. A dedicated re-mapping mechanism instead of
   overloading survival-tuned learning. Tests against R1/R3 directly.

## Why this may matter beyond the sandbox

"What makes a learned competence portable across interfaces rather than tied
to the one it grew up on" is the micro-scale form of a central question in
transfer/continual learning. The ladder turns it into a measurement, and the
substrate lets us *evolve* the answer rather than hand-design it. Keep scope
honest: claims are about this sandbox; the value is the method (typed
contracts + drift pressure + a distance ladder), not a leaderboard number.
