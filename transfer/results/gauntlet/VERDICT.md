# The #2867 solitude gauntlet — verdict (2026-06-11)

**Subject:** seed341's bottleneck survivor, t3000 checkpoint (3×13 blocks,
cap 39), taken mid-solitude. **Battery:** 12 held-out paired worlds, h 1.6,
physics rewritten every 600 ticks (3 rewrites per 2000-tick probe), cohort 8,
shared genome, identical placements. Arms: trained / frozen (trained weights,
lifetime learning off) / permuted ×3 / random ×3. 96 runs.

| metric | trained | frozen | permuted | random |
|---|---|---|---|---|
| alive @2000 | **0.635** | **0.688** | 0.021 | 0.229 |
| mean lifespan | 1503 | 1570 | 250 | 687 |
| mean energy | 70.8 | 71.2 | 17.7 | 32.2 |
| tool successes | 2.2 | 1.9 | 64.0 (!) | 1.0 |
| places visited | 3.2 | 3.8 | 2.0 | 2.1 |

Paired contrasts (per-seed means, n=12):
- **trained − permuted**: alive +0.61 (t+9.4, 12/12 worlds), lifespan +1254
  (t+15.5), energy +53 (t+22.4). Same weight *values*, shuffled → near-total
  death. The merit is in the *arrangement*.
- **trained − random**: alive +0.41 (t+8.3, 12/12), lifespan +816 (t+14.3).
  A fresh controller of its exact architecture does not survive these worlds.
- **trained − frozen**: alive −0.05 (t−1.0, n.s.), all metrics ~equal.
  **Lifetime learning from the snapshot onward contributes nothing
  measurable — through three world-rewrites per probe.**

## Reading

1. **Merit established.** Luck (12 paired worlds), position (identical
   placement), genome (shared), weight statistics (permuted), and architecture
   (random) are all controlled. The survivor's competence is real, large, and
   lives in the organized structure of its weights.
2. **The merit is baked in, not re-learned.** The frozen arm matches trained
   everywhere, including across physics rewrites. Whatever #2867 needed to
   survive regime shifts was already written into its weights by t3000 (i.e.
   genome + selection + its first ~2,080 ticks of lived learning). Robustness
   here is an innate-policy property, not an adaptation-machinery property.
   - Consistent with the population signal: 341/342's pools evolved plasticity
     *down* (→0.90–0.96) under h1.6. The world currently pays for robust
     policy, not for learning. Same shape as the capacity finding in
     `docs/LONG6H_2026-06-11.md` — the world demands neither blocks nor
     plasticity yet.
3. **The fingerprint metrics caught a spam phenotype.** Permuted controllers
   rack up 64 tool successes while dying at 250 ticks — scrambled nets hammer
   tool actions. Success-counts without survival are noise; profiles, not
   single numbers (again).
4. **Caveat (stated honestly):** the subject was chosen *because* it survived
   — this battery proves the surviving controller is special, not that
   survival was predictable ex ante. The pre-registered version selects
   champions by rule from fresh runs and gauntlets them blind. The harness now
   does this for any checkpoint in one command.

Raw: `shard{0-3}.json` / `.runs.jsonl` in this directory.
