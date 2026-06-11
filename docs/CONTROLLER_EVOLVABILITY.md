# Controller evolvability — can anything remarkable ever evolve here?

*2026-06-11. Prompted by Alex: "if they don't have — genuinely — the ability to evolve,
dramatically, then the idea that we'll ever recover anything remarkable from one is
unlikely." This audits what the current substrate lets evolution change, and proposes
the minimal redesign that opens the ceiling.*

## What can evolve today (audit of brain.py / params.py / optimization.py)

| Dimension | Evolvable? | Mechanism / bound |
|---|---|---|
| Hidden size | **Yes (capacity only)** | `neural_budget` genome field, legal range 0–512, init 4–13; `resize_hidden` is function-preserving; upkeep cost scales with it |
| Episodic memory capacity | Yes | genome field, 0–32 slots |
| Plasticity *rates* | Yes | scalar genome fields (learning rate, trace decay, etc.) |
| All weights | Yes | uniform gaussian on every entry at reproduction |
| Topology (depth, modules, wiring) | **No** | exactly one dense recurrent layer, forever |
| Attention head | **No gain path** | inherited if present; a lineage without one can never acquire it; form fixed (H×72, all-or-nothing) |
| Learning-rule *form* | No | fixed family; only its scalars evolve |
| Prediction heads | No | fixed set |
| Mutation granularity | No | one global scale for all weights; template recombination picks ONE parent's network whole |

## The verdict

Alex's worry is correct, in a specific and fixable way. The substrate can scale
**capacity** across two orders of magnitude, but it cannot change **shape**: every
controller that will ever exist is a point in "single dense recurrent layer + fixed
plasticity family." Dramatic evolution — new modules, qualitatively new computation,
architectural innovation — is structurally impossible, not just unlikely.

And there's an empirical tell that even the capacity axis is dead: **every champion we
have sits at hidden 8–14, inside the 4–13 initialization range.** Across all runs,
evolution has never meaningfully grown a controller. The likely mechanism is economic:
upkeep cost is proportional to `neural_budget` and paid *immediately*, while the benefit
of extra capacity arrives only after lifetime learning fills it. That is a fitness
valley at every rung of growth — selection prunes the investment before it pays. (Same
shape as the project's scale-discipline principle: effects need time to show; the
substrate's own economics currently refuse to wait.)

## Proposed program

### Phase 0 — diagnose (cheap, CPU-now)
Census `neural_budget` trajectories across existing run archives: distribution over
ticks, max ever reached by any lineage, survival curves conditioned on budget. If no
lineage ever crosses ~16, growth is being strangled, and the economics fix is justified
by data rather than theory.

### Phase 1 — fix the growth economics (small change, big unlock)

**RESULT (2026-06-11, capacity demography, 3 seeds/arm, 3000 ticks, ~19k
small-capacity individuals per arm as matched baseline):** the developmental
subsidy (grace 150, floor 0.35) **establishes the 17-48 capacity corridor** -
2.65x more mid-capacity individuals (100 -> 265), reach-age-150 up 51% -> 61%,
and reproduction up 0.62 -> 1.18 offspring (+90%), while leaving the small-
capacity baseline undistorted (70/75% reach-150, ~1.95 offspring, both arms).
Large (49-128) survives under subsidy (67% vs 0% reach-150) but did not yet
reproduce; random-init giants (>128) never establish in either arm - the
subsidy delays their death (median 32 -> 54 ticks), nothing more.

Interpretation: the valley is a *staircase*, not one cliff, and the subsidy
opens exactly the corridor that Phase 2's structural growth climbs through -
duplicate-and-diverge adds ~8-unit blocks on top of already-fit behavior, so
capacity ascends stepwise through the now-open mid range rather than leaping
to a random giant. The "giant leap" path (rare budget resets to 100+) is dead
on arrival and was never the modular mechanism anyway.
Pick one (or A/B them):
- **Developmental subsidy:** young individuals pay a ramp (say 30%→100% of size-upkeep
  over the first ~150 ticks) — gives lifetime learning time to make capacity pay.
- **Activity-priced capacity:** upkeep scales with *used* capacity (mean hidden
  activation mass), not allocated capacity — idle reserve is cheap, exploited capacity
  costs. (Pairs naturally with making `observe` cost energy, the parked attention-economy
  item.)

Gate (scale-aware): after the fix, lineages must *explore* budget >20 within a standard
run. They don't have to keep it — we're testing that the valley is crossable, not that
big is better.

### Phase 2 — typed modular genome (the real unlock; aligns with the v3 controller)
Replace "one dense layer" with a **composition of typed blocks** behind uniform
interfaces:

```
typed observation tokens (self / fields / objects / agents / consequences)
  -> shared token encoder (index-agnostic — the vine/V2H lesson)
  -> K recurrent core blocks, genome-owned wiring mask between them
  -> heads (action, prediction, optional per-block attention / episodic)
```

Structural genes: K, per-block module flags, wiring mask, per-block plasticity scalars.
Structural mutation operators, in order of importance:

1. **Duplicate-and-diverge:** copy a block (weights included), let the copy drift.
   Biology's proven route to innovation, and trivially dimension-safe with typed
   interfaces.
2. **Add-block, neutrally:** new block enters with its output gate near zero —
   function-preserving, so structural mutation is *survivable* (Net2Net-style). The
   single biggest historical blocker for dramatic neuroevolution is that structural
   mutations are lethal; neutrality removes the lethality.
3. **Prune-block / rewire-mask:** the cheap directions.

Why typed interfaces matter doubly: they make these operators dimension-compatible by
construction, AND they fix the slot-identity overfitting that the Catch postmortem
identified — the same redesign serves transfer and evolvability at once.

### Phase 3 — evolvable plasticity
Per-block learning rates and trace decays (genome vector, not one scalar), plus a
**neuromodulatory gate**: one controller output that scales plasticity each tick.
Context-dependent learning ("learn now, don't learn now") is itself then under
selection. The learning-rule *form* stays fixed for now — rule evolution is a later,
separate swing.

## Discipline

Per the scale-aware gate (`WORLD_BATTERY.md`): structural innovation has long
time-to-signal *by construction* (a duplicated block is neutral on day one). Phase 2
claims must be evaluated on long runs with the battery as held-out test, and nulls at
short scale get PARKED with a stated escalation rung, not killed. Pre-registered kill
example: if, after the economics fix and 4× standard run length, no structurally-mutated
lineage ever out-survives the fixed-shape pool, the modular genome is not earning its
complexity at this world-difficulty and we revisit world pressure before blaming the
architecture.

## Serialization note

All of Phase 2 fits the existing checkpoint philosophy: blocks serialize as a list of
typed dicts; `_LEGACY_KEYS`-style shims keep old single-block checkpoints loading as
K=1 configurations. Old champions remain valid citizens of the new space — they're just
the simplest expressible body plan.
