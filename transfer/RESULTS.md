# Probe-world transfer results — 2026-06-11

**Question** (docs/TRANSFER_RUNWAY.md): does a controller developed in this sandbox carry
competence into held-out worlds it never saw, against architecture-matched controls?

**Answer: yes — large, structural, and champion-shaped.**

## Design

`probe_worlds.py`. Per champion: trained vs 3× random-init vs 3× permuted (same per-array
weight distribution, structure destroyed), all sharing the champion's params and
architecture, lifetime learning ON for everyone. Cohorts of 16 injected as founders into
12 held-out worlds (places=36), spawning frozen, 500 ticks, paired by world seed.
In-dist = harshness 1.35, seeds 70000+; OOD = harshness 1.95, seeds 80000+.
t = paired mean/SE over 12 world seeds; w = fraction of worlds where trained wins.

## Headline contrasts (trained − random)

| Sweep | alive | lifespan | energy | tools | notes |
|---|---|---|---|---|---|
| learner, in-dist | **+0.52** (t 21, w 100%) | +91 (t 31) | +38.0 (t 15) | −27 (t −10) | persister-specialist; zero tool use |
| learner, OOD | **+0.33** (t 9, w 100%) | +85 (t 18) | +31.5 (t 11) | −24 (t −8) | transfer persists the harshness shift |
| overall, in-dist | **+0.43** (t 12, w 100%) | +99 (t 16) | +33.8 (t 18) | **+211** (t 17) | the generalist: persistence AND tools |
| tool, in-dist | −0.05 (t −1.2, ns) | −24 (t −4) | +25.3 (t 8) | **+369** (t 61) | pure specialist; no persistence edge |
| v2 learner, in-dist | **+0.58** (t 16, w 100%) | +56 (t 5) | **+64.1** (t 70) | +54 (t 8) | multi-world selection story holds |

All five sweeps complete (84/84 runs each); full numbers in `results/*.json`, per-run
records in `results/*.runs.jsonl`.

## Findings

1. **Within-substrate transfer is real.** Every champion except the tool specialist
   carries a persistence advantage into novel worlds — winning all 12 paired worlds in
   every persistence sweep. This is the result the Catch test could not deliver: with the
   native observation/action schema preserved, developed structure beats random
   decisively.
2. **Structure, not conditioning.** Trained beats *permuted* nearly everywhere
   (e.g. learner OOD lifespan +96, t 20). The permuted control also revealed a clean
   decomposition: weight *statistics* buy energetic efficiency (permuted recovers most
   of the energy advantage), but *structure* buys staying alive and all behavior.
3. **Transfer is champion-shaped.** The learner transfers persistence and suppresses
   exploration; the tool champion transfers prolific tool use (369 vs 0.6) at a small
   persistence cost; the overall champion transfers both; the v2 (episodic-memory,
   multi-world-selected) champion posts the largest energy dominance in the table
   (t 70). Single-number transfer scores would have hidden all of this — profiles are
   the right unit (see docs/WORLD_BATTERY.md).
4. **The tool champion is the only one whose prediction error improves over its
   lifetime in novel worlds** (pred_err_delta −0.046 vs random, t −28) — the strongest
   lifetime-learning signature observed. The learner champions' flat deltas are PARKED,
   not killed: 500 ticks may be below detection threshold for their learning effects
   (see scale-aware discipline in WORLD_BATTERY.md).

## Open questions

- **Strategy or stasis?** The learner persists with near-zero tool/causal activity. Needs
  behavioral fingerprinting (action entropy, attempts vs successes, coverage) before we
  call it strategic conservation.
- **OOD coverage** ran only for the learner champion; the other three champions' OOD
  sweeps are queued battery work.
- Champions developed at hidden 5–14 — capacity exploration is absent in the substrate's
  history (docs/CONTROLLER_ADAPTABILITY.md); what richer-bodied evolution transfers is
  the program now in motion.

## Provenance

Sweeps of 2026-06-10 21:05 died at 21:41 (machine OOM cascade; nothing saved — results
then only wrote at sweep end). Re-run 2026-06-11 with per-run JSONL checkpointing +
resume in probe_worlds.py; a second OOM at 08:59 cost one in-flight run per sweep;
final relaunch ran in an isolated systemd scope (MemoryHigh=3G) to completion.
PYTHONHASHSEED was unpinned for early runs (pinned to 0 from the final relaunch);
runs are not bit-repeatable across invocations, which the paired multi-seed design
absorbs. Permuted/random instance seeds are fixed (1000+i / 2000+i).
