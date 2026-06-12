# Microcosmic God Handoff

Last updated: 2026-05-07

This is the quick-start context for a fresh Codex instance taking over Microcosmic God.

## Repo State

- Repo: `C:\Users\power\Documents\Codex\2026-05-01\i-have-an-exciting-and-fun\microcosmic-god`
- Remote: `https://github.com/asystemoffields/microcosmic-god`
- Active branch: **`main`** (sessions are now landing directly on main; `codex/colab-a100-run-notebook` was kept in sync but main is canonical)
- Latest pushed commit: `3c825b0 Align Catch transfer harness with v2 controller architecture (A+B+C+D)`
- Recent commits (newest → oldest, last few sessions):
  - `3c825b0` Align Catch transfer harness with v2 controller architecture (multi-ball, persistent hidden, replay-on-stay, adaptive mode)
  - `37392d8` Add v2 controller (episodic memory + replay) and fix multi-world refresh (preserve resources)
  - `aa95bbf` Add multi-world selection: regenerate world every N ticks
  - `bc16cff` Add skeptic controls; the single-seed +0.38 was noise
  - `2c3bb5b` Add Catch transfer harness as Colab notebook
  - `e2e3c7c` Rebalance energetic cost and attention update for cognitive payoff
  - `36ec8a0` Add physics-conditional prep steps to causal challenges
  - `c634607` Add neuroplastic attention head with bounded fidelity budget
  - `10b93d4` Allow controllers to grow and shrink across spawning
  - `e10a91c` Move TinyBrain core to numpy (~11x controller ops speedup)
  - `eb5dd82` Add environment harshness + situation-aware affordance + line tracking + counterdrain + exposure-pressure
  - `2eee68e` Add arc report (`analysis/scripts/arc_report.py`)
  - `696fc43` Drop senescence as a death cause
  - `7d22c19` Specialist-trap penalty in arc scoring
- Pre-session commits still relevant:
  - `0b75780` Add torch controller simulation backend
  - `2a9e4f0` Use CUDA for checkpoint SAE analysis
  - `34ddb4e` Add Colab A100 run notebook
  - `5680f86` Lock movement energy costs

77/77 tests pass at HEAD.

Before editing, run:

```powershell
git status --short
```

The expected clean state after this handoff is either clean or only this file changed if it has not been committed yet.

## Project Aim

Microcosmic God is a headless artificial-life sandbox for developing agents in a consistent causal universe. The north star is not to hand-code intelligence, cooperation, language, or technology, but to create general world laws where those capacities can become useful through action.

The user cares deeply about:

- ANN agents whose weights can be checkpointed and later transferred/adapted to other worlds or simple RL/game tasks.
- Tools and structures derived from material properties, not recipes.
- Information transfer through temporary signals and durable-but-decaying marks.
- General causal rules over explicit achievements.
- Compute efficiency locally, with a path to larger cloud/GPU runs.
- Rich, dangerous, rewarding environments where smart action matters.

Useful docs:

- `README.md`
- `docs/GENERALITY_AND_SCALING.md`
- `docs/LEARNING_ARCHITECTURE.md`
- `docs/EVENT_MONITORING.md`
- `docs/TRANSFER_RUNWAY.md`
- `docs/PHYSICS_KERNEL.md`
- `docs/LANGUAGE_RUNWAY.md`
- `docs/MACHINE_PROFILE.md`

## Design Commitments

- Sealed runs stay sealed. Garden interventions are allowed only when explicitly requested and logged.
- Do not add direct hidden rewards for "being smart", "cooperating", "durable symbol encoding", or "using tools".
- Intelligence should pay off only because actions become better: less wasted movement, better tool outcomes, remembered places, causal unlocks, useful marks, persistence, spawning.
- Specialists are legitimate. A narrow master should not masquerade as a universal engineer, but genuine repeated mastery should count.
- Inactive controllers disappear unless checkpoint policy saved them.
- Competitive interaction currently exists through agent `drain` behavior. Do not add a separate drainer type yet; the user explicitly became unsure that dedicated drainer individuals are the right lever.
- Movement must cost energy. This is now locked by tests for easy success, failure, and helper-assisted expeditions.

## Current Mechanics Snapshot

World:

- Sparse graph world, not a grid.
- Environment archetypes include pelagic, reef, trench, hydrothermal vent, tidal marsh, high ridge, mineral scree, forest edge, desert glass, and cavern.
- Places carry resources, materials, obstacles, local-condition fields, physics fields, causal challenges, marks, signals, and structures.
- Recent enrichment makes hostile places also valuable: trenches/vents/mineral zones can hold high-value energy/material opportunities.

Movement:

- Movement cost includes base cost, load, distance, slope/current/barrier pressure, relocation shock, and failure penalty.
- Relocation shock comes from physical deltas: temperature, fluid level, pressure, humidity, salinity, elevation, oxygen, and hazard pressure.
- Planning, destination memory, protection, traversal, insulation, containment, and helper support can mitigate but not erase the cost.
- Movement telemetry is in summaries under `movement`: attempts, success/failure, energy/health cost, barriers, support, relocation shock, motives, routes.

Tools and structures:

- Tools derive capabilities from material properties.
- Crafting chooses a target affordance and components under noisy planning/skill constraints.
- Skill transfer is local in affordance-space. Example: `bind` can help related craft/build/support/carry/record channels, but not unrelated conduction or heat concentration.
- Structures are place-level artifacts with scale, durability, decay, and field effects such as support, shelter, channeling, filtering, gradient harvest, reaction surfaces.

Collaboration:

- Active helpers can support expeditions, tool use, build attempts, and causal challenge steps/unlocks.
- Help is indirect and costly. Helpers pay energy and gain only action-mediated feedback.
- Collaboration telemetry is in summaries under `collaboration_events`.

Communication and memory:

- `signal` is temporary local communication with no built-in semantics.
- `mark` is a durable-but-decaying physical trace.
- Intentional lesson traces can encode recent tool/craft/problem experience when the agent has relevant experience, attention, materials, and `inscribe` skill.
- Reading useful traces improves `interpret_mark` and relevant action skills. Self-reading counts as external memory, not information transfer.
- Portable record-capable artifacts can carry inscriptions.

Evolution and checkpoints:

- Spawning is routed through `EvolutionEngine`.
- In-world operators are `clone_perturb` and `coordinate` leading to `combine`.
- Controller checkpoints are bucketed: first tools, interval/final champions, spawn/tool/causal/learner champions, line founders, notable removals.
- Seed 63 standout controllers were archived in `archives/controllers/seed63_run204018`.

## Recent Empirical Notes

Seed 1 30-minute v2 controller + multi-world run (`runs/cpu_30m_seed1_v2_multiworld/20260507_153023_seed1_minute/`):

- Multi-world selection enabled (`world_refresh_every=1200`). World refreshes preserve resources, structures, marks; only physics + obstacles + causal_challenge are regenerated.
- v2 controller features active: episodic memory bank (params-adaptable capacity), replay-during-rest, attention head with budget=0.95.
- 5,440-tick budget but only reached tick 3,113 — pool pressure + episodic computation slowed per-tick. **1,641 neural agents at end** (3,622 total) — the largest neural pool observed.
- 60% combination rate (was 33% in single-world).
- **Line 460 = 909 active agents at run-end**, max gen 19, 7,539 cumulative successors. Largest dominant pool cluster observed in the substrate.
- **Notable narrative arc**: individual 5506 was active across ticks 993-1658 across the world refresh at tick 1200. At place 33 it solved `cut > bind` for residue_store in World A; after the refresh, place 33 became different physics, and 5506 switched to `crack > lever > contain` for essence and unlocked it 4 more times. Concrete in-substrate evidence of cognitive flexibility under distribution shift — the behavior the substrate was designed to select for.
- Controller capacity transient peak: 218 hidden units at tick 2,300 (selected against, but the substrate is exploring large controllers).

**Catch transfer test results (10 seeds, 4 conditions, frozen + adaptive modes, multi-ball harness):**

The harness was aligned to the controller's architecture (A+B+C+D from `3c825b0`): multi-ball Catch episodes with paddle persistence, controller hidden state persists across balls within an episode, Catch's "stay" action triggers `controller.replay_episode()`, and an adaptive mode that fires controller plasticity from per-ball reward.

Results in multi-ball units (max possible reward per episode = +30):
- Direct linear policy (no controller): **+16.66 ± 2.11** — best.
- Random-init controller + adapter: +13.37 ± 2.66 — beats trained controller.
- v2 trained controller + adapter: +5.27 ± 5.22 — third.
- Permuted v2 controller + adapter: +0.29 ± 9.29 — worst.

**Decisive findings:**
- `trained − permuted = +4.98` — permutation test passes; the substrate IS producing structured cognition (not just well-conditioned weights).
- `trained − random_controller = -8.10` — random controllers beat trained controllers. The mg-trained representations are actively misaligned with Catch.
- `trained − direct = -11.39` — the controller hinders compared to no controller.
- Adaptive mode catastrophically broke all controller conditions (collapse to ~-13 reward). The naive Hebbian rule trains the wrong mg-action's weights — there's no alignment between the controller's intended mg-action argmax and the projected Catch action chosen. Adaptive transfer needs a redesign before it's testable.

**Read:** Catch is the wrong test target. Even with full harness alignment, mg-trained controllers underperform random controllers. The substrate produces structure (permutation test confirms) but that structure doesn't generalize to a 4-feature linearly-solvable game. Either the substrate isn't producing transferable cognition yet (true negative for the project's "transferable minds" claim), OR Catch can't probe what's actually being produced. Probably both.

Seed 1 30-minute payoff-rebalance run (`runs/cpu_30m_seed1_payoff_v2/20260507_124346_seed1_minute/`):

- 5,429 ticks, 546 neural agents at end (peaked at 861 active at tick 4,900).
- **Controller capacity grew across the run for the first time**: mean 7.7 → 10.8 (+40%), max 13 → 26 (2x). Earlier runs had controllers stuck at ~8 throughout.
- **Three competing line strategies** (vs v1's single dominant): line 490 collaborate-heavy (241 active, 3,060 successors), line 430 balanced (195 active, 3,990 tools), line 489 tool-master (105 active, 9,882 tools). Genuinely differentiated cognitive specializations in the same world.
- Environment denser and more competitive: energy depletion 1,240 (was 701 in v1), counterdrain 246 (was 53).
- Attention concentration moved from 0.01 → 0.02 — slight movement after the raw-values rule fix, but still well below the convergence we'd want. Probably needs longer runs.

Seed 1 30-minute full-pipeline run (`runs/cpu_30m_seed1_full_pipeline/20260507_113805_seed1_minute/`):

- 5,440 ticks, 397 neural agents at end (out of 2,482 total pool).
- **Line 489 reached generation 38** with 369 active agents, 6,459 cumulative successors, 54,067 tool successes, collaboration profile = 351,218. By far the deepest selection-driven dominant pool cluster observed in this substrate.
- Combination shifted to 33% of spawnings (was 5%). Cross-line params mixing is now a dominant spawning mode at scale.
- Tool repertoire genuinely diversified: bind 14,982, lever 13,970, crack 9,869, kindle 9,742, contain 8,237. No single dominant affordance.
- **Individual 416 was active 2,044 ticks** (38% of the run) at place 12 and built a single `structure_support_anchor_gradient_harvest` from scale 6 → 372 across 104 build/extend events, working solo. Removed by competitive interaction with 1 child. Long-horizon coherence in one controller's lifetime.
- Attention concentration stayed flat at 0.01-0.02 across the whole run — the neuroplastic update rule is calibrated too gently to converge in 5,000 ticks.
- Controller capacity mean stayed at ~7.9 throughout. Controllers aren't growing despite the mechanism being enabled.

Seed 1 5-minute textured-harshness run (`runs/cpu_5m_harsh_env_textured/20260506_181846_seed1_minute/`):

- Final tick 1016, final pool 2284 (575 neural).
- Tool repertoire shifted dramatically vs the pre-textured seed-1 run:
  - `kindle` 7 → 1198 (cold-place prep rule biting)
  - `bind` 321 → 1530 (abrasion-prep)
  - `filter` 2 → 217
  - `lever` 880 → 2121, `crack` 572 → 1264
- Neural pool went from 232 → 575 (+148%) — neural agents outcompete non-neurals more strongly when puzzles demand cognitive work.
- **Line 474** is the new dominant pool cluster. 41 spawnings, max generation 7. Solves four physics-regime puzzles across four places: `crack>lever>contain` at place 15, `cut>bind` at place 14, `bind>contain>filter` at place 16, `kindle>conduct` at place 23. Three different prep-step types in one line = controller template generalizing the physics-conditional rule.
- Run was ~30% slower per tick (1016 vs 1456 ticks in same wall budget).

Seed 1 5-minute pre-textured run (`runs/cpu_5m_harsh_env/20260502_070301_seed1_minute/`) — kept for comparison:

- Final tick 1456, final pool 2123 (232 neural).
- Lever-dominated tool monoculture (lever 880, crack 572, all others <250). One 3-step unlock in the entire run.
- Codex-flagged narrative arcs: 424 (early crack specialist removed by a drainer), 422 (founder of dominant line), 1551 (clean crack→lever causal arc), 3692 (best team-problem-solving), 3427 (builder-then-solver across places 7+8). The arc tool also surfaces 2025 as the run's biggest specialist trap (tool_use=330 from 328 lever-only successes at place 8, removed by energy depletion, no children).

Seed 63 10-minute run:

- Run dir: `runs\20260501_204018_seed63_minute`
- Final tick: 1689 due wall limit.
- Final pool: total 2613, neural 1634.
- Standout: individual 1958.
- 1958 was a bind specialist, not a true universal tool master. The old `bind` path incorrectly incremented every tool skill. That loophole has been fixed.
- 1958 likely moved between places 28 and 6 because those places were directly connected and place 6 became better for its strategy, but ordinary movement/pickup was not logged at enough detail then.

Recent smoke after hostile biomes/collaboration:

- Run dir: `runs\20260501_223526_seed72_smoke`
- Seed 72, 160 ticks.
- Produced structures, tool diversity, collaboration events, and movement telemetry.
- Movement average energy cost was about `0.313`.
- Average relocation shock was about `0.399`.

Recent smoke after movement-cost test lock:

- Run dir: `runs\20260501_223958_seed73_smoke`
- Seed 73, 100 ticks.
- Movement average energy cost was about `0.235`.
- Existing competitive interaction was visible as `deaths: {'depletion': 15}` from agent drains.

## Verification Commands

Use these before and after meaningful edits:

```powershell
python -m compileall microcosmic_god tests analysis\scripts
python -m unittest discover -s tests -v
python -m microcosmic_god run --profile smoke --seed 73 --ticks 100 --wall-seconds 15 --quiet-events
```

Summarize a run:

```powershell
python analysis\scripts\summarize_run.py runs\<run_dir>
```

Machine specs:

```powershell
python -m microcosmic_god specs
```

## Current Test Coverage Highlights

Important tests are in `tests/test_causal_contracts.py`.

The 2026-05-06 session added a `TexturedHarshnessTests` class covering the prep-step rule:

- `test_temperate_dry_place_skips_prep`
- `test_cold_place_prepends_kindle`
- `test_flooded_place_prepends_contain`
- `test_high_pressure_place_prepends_contain`
- `test_unstable_place_prepends_bind`
- `test_cold_and_flooded_stacks_two_prep_steps`
- `test_prep_step_not_duplicated_when_base_sequence_already_contains_it`

Earlier important tests still in place:

- `test_bind_practice_transfers_only_to_related_skills`
- `test_specialists_keep_cognitive_credit_from_repeated_practice`
- `test_active_helpers_can_supply_build_materials`
- `test_collective_support_and_relocation_shock_are_tracked_for_moves`
- `test_successful_movement_spends_energy_even_when_easy`
- `test_failed_movement_spends_energy`
- `test_environment_generation_has_hostile_treasure_biomes`

As of commit `36ec8a0`, full suite passes:

```text
Ran 54 tests
OK
```

## Likely Next Actions

The 2026-05-07 session pinned the open thread to a clean question: *the substrate produces structured cognition (permutation test passes), but that cognition doesn't transfer to Catch.* Three ranked threads to pull next:

- **Within-substrate transfer test (highest priority)**: take a trained controller from one mg seed, drop it into a fresh mg seed (different physics + puzzles, same observation/action space), measure persistence/tool-use/causal-unlock vs random-init baseline. No action-space mismatch, no input-distribution shift to fight, exercises exactly the cognition the controller has. This is the cleanest probe for whether the substrate produces general cognition. **Build this next.**

- **Smarter adaptive transfer**: the current adaptive mode (`adaptive_lr > 0` in the Catch harness) is broken because it trains the controller's mg-action policy on Catch reward — but the controller's argmax mg-action and the projected Catch action are decoupled. Fix: representation-level plasticity from prediction errors only (the controller's native learning signal), no policy updates. Lets the controller rapidly form representations of new environments without the action-space mismatch destroying it.

- **v3 controller architecture (typed inputs / cross-attention over input tokens)**: addresses the "input-distribution-specific priors" root cause that makes mg-trained controllers hurt vs. random controllers on out-of-distribution tasks. Bigger lift but the deepest fix. Worth investigating after within-substrate transfer is validated.

Earlier 2026-05-06 threads still relevant:

- **Information cost**: make `observe` cost energy proportional to detail extracted. Currently `observe` averaged -0.026 energy in seed-1 (nearly free), so attention has no economy and marks/signals have no compressive value. Tighten this and `mark`/`signal`/`mark_lesson_*` channels gain real economic weight.
- **Push diversity-aware scoring into `simulation.py`'s `_checkpoint_score`**: arc_report's `diversity_factor` correctly demotes specialist-trap controllers in *analysis*, but the simulator still archives them via the same accumulator-style score. Fixing this means the controllers saved for transfer will reflect the agentic intelligence the project actually wants, not rote memorizers.
- **Controller-checkpoint trajectory metrics**: action diversity over a window, novelty of place-action pairs. Currently checkpoints rank on cumulative counts; trajectory metrics would catch individuals whose intelligence is in *adaptation*, not volume.
- **Decompose `terrain_mismatch`** into the underlying physical pressures it conflates. The pressures already exist; the label is redundant and obscures cause-of-removal analysis.

Earlier carry-over ideas (still good):

- Calibrate relocation teeth: enough failures to matter, not so much that lines randomly collapse every time.
- Improve event story tooling around movement: identify costly relocations, repeated routes, local-condition traps, successful expeditions, and agents that learned to avoid bad moves. Note: `arc_report.py` partially addresses this for individual-centric arcs; place-centric and movement-centric arcs are not yet covered.
- Expand environmental resource coupling in general ways: flow gradients, hydro-like structures, pressure/thermal/electrical reservoirs, sea treasures with consistent risks.
- Add richer object/structure attention to observations so ANNs can notice local affordance causes more directly.

Avoid for now:

- Do not add a separate drainer type unless the user reaffirms it. Existing competitive interaction via agent drain is enough pressure to inspect first.
- Do not make cooperation mandatory.
- Do not add recipe-like tools such as "axe cuts wood" as a special case. Note: textured-harshness prep steps are *not* recipes — the rule is global (e.g., "cold places need warming first") and physics varies per place.
- Do not reward marks/durable symbol encoding directly; only changed action consequences should matter.
- Do not start a long run before giving the user key specs: profile, seed, ticks/wall limit, places, initial pools, max pool, checkpoint cadence.
- Do not collapse fixed `kind` (agent/converter/collector/neural) into emergent kinds yet. The user explicitly held off on this in the 2026-05-06 session.

## User Preferences

- The user likes bold, open-ended mechanisms but is sensitive to overprogramming.
- They prefer general physical/causal relationships over hand-authored goals.
- They are comfortable with bigger pools, but local machine constraints matter.
- They want short progress updates and concrete implementation momentum.
- Before launching a run, provide key run specs.
- If a run is clearly not going to answer the question, stop it and pivot.

## One-Sentence North Star

Build a cheap, scalable universe where persistence and spawning favor agents that learn causal structure, manipulate materials, manage energy, communicate useful traces, and sometimes become transferable minds, without the simulator secretly telling them to do any of that.
