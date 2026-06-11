# Pre-flight readout — 2026-06-11 (E1/E2/E3 + first cloud run)

Inputs for the open decision in `docs/RESUME_2026-06-11.md`: pick the configuration
for the ~6-hour structural-accumulation run. All local pre-flights ran their full
45-minute walls (one seed per arm unless noted); census numbers from
`analysis/architecture_trajectory.py`.

## E2 — structural-rate sweep (h 1.35, no refresh, all-modular, seed 301)

| rate | outcome | end blk_mean | blk_max | end cap_mean |
|---|---|---|---|---|
| 0.06 | viable, pool ~900 | 1.18 | 4 | 11.3 |
| 0.15 | **collapsed to 7** by t~2100, limped to wall | 1.14 | 2 | 12.0 |
| 0.30 | viable, pool ~740 | **2.89** | **7** | **21.9** |

Viability is non-monotonic in rate, so at n=1/arm the 0.15 collapse is not
attributable to rate — world-trajectory stochasticity dominates. What E2 *does*
establish: the feared failure mode (high rate churns lineages to extinction)
did **not** appear even at 0.30, and 0.30 produced by far the most structure.
Per the pre-registered rule (highest rate that does not churn to extinction):
**take 0.30**.

## E1 — head-start competition (mixed founders: legacy / 1-block / ≤3-block)

- **seed 311**: modular founders sweep to fixation — legacy 21→0 by t1600;
  modular pool ~1000 at wall, blk_mean ~2.0, blk_max 5–6.
- **seed 312**: depressed world; modular dwindles to 1 while a small legacy
  population (10–50) lingers.

Read: in a viable world, structured founders win the direct competition
decisively; in a marginal world the cheaper architecture outlasts them.
n=2 and seed-dominated — treat as directional only.

## E3 — pressure ladder (mixed founders, seed 321)

| cell | pool | census movement |
|---|---|---|
| r0 / h1.35 | legacy-dominated (~500–800) | flat (~1.6 blk among few modulars) |
| r0 / h1.6 | **near-total collapse** (<10 total) | none |
| r1500 / h1.35 | thin, legacy-dominated | flat |
| r1500 / h1.6 | large legacy pool (800–1600), refresh waves visible | **strongest anywhere: blk_mean 2.6–2.7, cap_mean 28–29, rising at wall; modulars rebounding 7→39** |

Read: refresh×harshness is the only cell that moves the architecture census —
but h1.6 is only survivable when legacy founders carry the ecology.

## Cloud run s41 (all-modular, rate 0.15, r1500, h1.6, grace 150, seed 41)

**Neural extinction at tick 10,113** (~50 min in). Pool collapsed 67→11 by
t900 and limped at 1–3 individuals for ~9000 ticks before dying. Confirms E3:
**all-modular founders cannot establish at harshness 1.6**, grace 150
notwithstanding. s42 (same config, seed 42) still running as of 13:00; if it
also dies, the conclusion is solid.

## Provisional 6-hour configuration (final call held until s42 lands)

The tension: the only world that *demands* structure (r1500/h1.6) is one that
all-modular founders can't boot in. Two coherent resolutions:

1. **Mixed-boot (recommended)**: `--modular-fraction 0.5`, r1500, h1.6,
   structural-rate 0.30, max-blocks 3, grace 150/0.35. Legacy founders carry
   the ecology through establishment; modulars fight the E1 competition inside
   the E3 cell that actually rewards structure. The architecture census +
   modular/legacy share over time *is* the experiment's readout.
2. **All-modular at gentler pressure**: modular-fraction 1.0, r1500, h1.35,
   rate 0.30, max-blocks 3. Boots reliably (E2), but E3 says h1.35 cells
   showed a flat census — risk of a 6-hour null.

Decision rule when s42 lands: if s42 also went extinct → option 1. If s42
survived to wall with a moving census → option 1 anyway (mixed boot
dominates: it answers the competition question and the accumulation question
in one run), unless its census is decisively richer than E3-r1500/h1.6's, in
which case all-modular h1.6 with seed-luck retries becomes defensible.

Hygiene for the 6-hour run: launch under
`systemd-run --user --scope -p MemoryHigh=NG`, keep under the 12 h Kaggle
bound if cloud, and note runs are not bit-reproducible (PYTHONHASHSEED
unpinned).
