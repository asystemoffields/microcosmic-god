# The World Battery — named probe worlds and the road past one-number transfer

*Proposed 2026-06-11, while the first probe-world sweeps were still running. This extends
`docs/TRANSFER_RUNWAY.md` ("Probe Worlds Before Video Games") from resampled seeds to a
designed battery. Nothing here modifies the in-flight experiment.*

## Why named worlds

The current sweep measures transfer to *generic* held-out worlds — same generator, new
seeds, one harshness shift. That answers "does developed structure help at all?" but
compresses everything into a few numbers. A controller is a bundle of competencies
(energy budgeting, spatial adaptation, causal unlocking, recovery from shocks), and a
generic world exercises them in an uncontrolled mix.

A **battery of named worlds**, each built to stress one competency, turns the transfer
question into a *profile*: this champion carries its tool competence anywhere but exhausts
under scarcity; that one persists everything and attempts nothing. Profiles are
falsifiable, comparable across champions, and tell us what the substrate actually
selects for.

(They are also simply more fun to build and reason about, and the geographic/meteorological
naming register keeps the project's vocabulary effortlessly neutral.)

## Prerequisite: behavioral fingerprinting (do this first)

Before any new world, the probe harness should log per-founder behavior, not just
outcomes: action-distribution entropy, move rate, observe rate, tool *attempts* (not just
successes), places visited. Two open questions from the first sweep need exactly this:

- **Strategy or stasis?** The learner champion persists on high energy with near-zero
  tool use. Smart conservation and catatonia look identical in outcome metrics.
- **Skill or spam?** The tool champion posts 350–400 tool successes. Without the attempt
  denominator we can't distinguish precision from volume.

Implementation: a `--fingerprint` flag on `transfer/probe_worlds.py` adding per-founder
counters to the JSONL records. Cheap (counters already mostly exist on the individual's
success/action profiles).

## The battery

Worlds runnable **today** with existing config knobs:

| World | Config | Competency under test |
|---|---|---|
| **Lean Season** | `initial_collectors`/`initial_converters` at ~40% of standard, harshness 1.6 | Energy budgeting under scarcity. The learner champion's presumptive home turf. |
| **Quiet Eden** | Abundant resources, harshness 0.8 | Control world. Does trained structure *cost* anything when pressure is off? Detects overfit-to-harshness. |
| **Shifting Ground** | `world_refresh_every` ≈ 150 (physics resampled ~3× per 500-tick life) | Adaptation to distribution shift *within* a lifetime. The v2 multiworld champion was selected for exactly this; the battery tests whether that's real. |
| **Season of Storms** | garden mode + a fixed intervention script: `disaster` at tick 150, `climate_shift` at 300 | Shock recovery. The same script applies to every condition, so the comparison stays paired. |

Worlds needing **small world-gen parameter exposure** (each is a few new `RunConfig`
fields plumbed into `World.generate`):

| World | New knobs | Competency under test |
|---|---|---|
| **The Torrent** | current/slope magnitude ranges | Spatial/motor adaptation when movement has strong costs and asymmetries. |
| **Locked Larder** | causal-challenge density; fraction of energy gated behind challenges | Causal competence as persistence necessity, not opportunity. Away game for everyone except (maybe) the tool champion. |

Battery protocol: same paired design as the current sweep (trained / random / permuted,
shared params, frozen spawning, identical world seeds across conditions), ~8 seeds
per named world. A champion's result is its **transfer profile** across the battery —
report the table, resist averaging it.

## Mechanism probes (after the battery says where to look)

1. **Plasticity ablation (2×2).** {trained, random} × {lifetime learning on, off}.
   Decomposes the transfer advantage into *inherited weights* vs *learning machinery*.
   The learning rule is part of what evolution tuned — this measures how much.
2. **Params swap.** Trained controller weights paired with a different champion's params
   (learning rates, valences, mobility). Is the advantage in the network or in the
   hyperparameters it was developed alongside?
3. **Experienced transplant.** The current harness resets hidden state and episodic
   memory on entry (correct for measuring weights). The complement: transplant a
   *active* founder mid-life, memories intact. Does experience transfer, or anchor the
   controller to the world it came from?

## Scale-aware discipline (how to apply the gate here)

The vine-style gate (incremental + cross-draw-stable + held-out) transfers to this project,
but with one substrate-specific amendment: **in mg, some effects need time and scale to
show.** Lifetime learning accumulates over hundreds of ticks; tool chains and causal
unlocks are rare events; developed structure reflects how long and how hard selection ran.
A null at 500 ticks × cohort 16 × a 30-minute selection run is often "not visible at this
scale," not "false."

So the discipline is two-tier:

- **KILL** is reserved for claims falsified *at a scale pre-registered as adequate* —
  e.g. a paired contrast that stays at zero while its detection scale is raised 4×, or a
  control (permuted/random) that matches trained at every rung.
- **PARK** is the verdict for nulls at current scale. A parked claim must state the rung
  at which it should show (more ticks, bigger cohort, more worlds, longer/harder
  selection runs) — that makes "needs more scale" a testable prediction instead of an
  excuse.
- Every battery claim pre-registers both: the kill criterion *and* the expected
  scale-to-signal. Escalation ladder, cheapest first: ticks (500→2000) → cohort
  (16→48) → world seeds (8→24) → upstream selection compute (the expensive rung —
  champions from longer, multi-world selection runs).

Corollary for the current sweep: weak positive or null contrasts for the *learner*
champions do not indict the substrate — those champions came from short selection runs,
and lifetime-learning effects (pred_err_delta) may need 4× the ticks to separate from
noise. The tool champion's large effects, if they hold, are the floor of what the
substrate produces, not the ceiling.

## Further out (kept honest by the battery results)

- **Harshness ladder.** Transfer as a function of world distance: harshness 1.35 → 1.65
  → 1.95 → 2.25 with fixed seeds. The current indist/ood pair gives two points; a curve
  says whether developed structure degrades gracefully or falls off a cliff.
- **Develop-in-battery.** If profiles are spiky (every champion a specialist), the next
  selection run should *rotate worlds during evolution* (multi-world already exists) with
  the battery as held-out test — selecting for the profile, not a niche.
- **Kaggle scale-out.** Each (champion × world × condition) cell is an independent
  process; the battery is embarrassingly parallel across free CPU kernels (~30GB RAM,
  no quota) if the local box stays contended.
