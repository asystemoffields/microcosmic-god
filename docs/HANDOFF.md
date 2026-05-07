# Microcosmic God Handoff

Last updated: 2026-05-06

This is the quick-start context for a fresh Codex instance taking over Microcosmic God.

## Repo State

- Repo: `C:\Users\power\Documents\Codex\2026-05-01\i-have-an-exciting-and-fun\microcosmic-god`
- Remote: `https://github.com/asystemoffields/microcosmic-god`
- Active branch: `codex/colab-a100-run-notebook` (8 commits ahead of `main`, pushed to origin)
- Latest pushed commit: `36ec8a0 Add physics-conditional prep steps to causal challenges`
- Recent commits (this session, oldest → newest):
  - `eb5dd82` Add environment harshness + situation-aware affordance choice + lineage tracking + counterattack + exposure-pressure hazard
  - `2eee68e` Add arc report (`analysis/scripts/arc_report.py`) — narrative-arc curator over `story_events.jsonl`
  - `696fc43` Drop senescence as a death cause
  - `7d22c19` Penalize specialist-trap signatures in arc scoring + surface them in a dedicated section
  - `36ec8a0` Add physics-conditional prep steps to causal challenges (textured harshness)
- Pre-session commits still relevant:
  - `0b75780` Add torch brain simulation backend
  - `2a9e4f0` Use CUDA for checkpoint SAE analysis
  - `34ddb4e` Add Colab A100 run notebook
  - `5680f86` Lock movement energy costs

54/54 tests pass at HEAD.

Before editing, run:

```powershell
git status --short
```

The expected clean state after this handoff is either clean or only this file changed if it has not been committed yet.

## Project Aim

Microcosmic God is a headless artificial-life sandbox for evolving agents in a consistent causal universe. The north star is not to hand-code intelligence, cooperation, language, or technology, but to create general world laws where those capacities can become useful through action.

The user cares deeply about:

- ANN agents whose weights can be checkpointed and later transferred/adapted to other worlds or simple RL/game tasks.
- Tools and structures derived from material properties, not recipes.
- Knowledge transmission through temporary signals and durable-but-decaying marks.
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
- Do not add direct hidden rewards for "being smart", "cooperating", "writing", or "using tools".
- Intelligence should pay off only because actions become better: less wasted movement, better tool outcomes, remembered places, causal unlocks, useful marks, survival, reproduction.
- Specialists are legitimate. A narrow master should not masquerade as a universal engineer, but genuine repeated mastery should count.
- Dead brains disappear unless checkpoint policy saved them.
- Predation currently exists through agent `attack` behavior. Do not add a separate predator species yet; the user explicitly became unsure that predatory animals are the right lever.
- Movement must cost energy. This is now locked by tests for easy success, failure, and helper-assisted expeditions.

## Current Mechanics Snapshot

World:

- Sparse graph world, not a grid.
- Environment archetypes include pelagic, reef, trench, hydrothermal vent, tidal marsh, high ridge, mineral scree, forest edge, desert glass, and cavern.
- Places carry resources, materials, obstacles, habitat fields, physics fields, causal challenges, marks, signals, and structures.
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
- Reading useful traces improves `interpret_mark` and relevant action skills. Self-reading counts as external memory, not knowledge transmission.
- Portable record-capable artifacts can carry inscriptions.

Evolution and checkpoints:

- Reproduction is routed through `EvolutionEngine`.
- In-world operators are `clone_mutate` and `coordinate` leading to `recombine`.
- Brain checkpoints are bucketed: first tools, interval/final champions, reproductive/tool/causal/learner champions, lineage founders, notable deaths.
- Seed 63 standout brains were archived in `archives/brains/seed63_run204018`.

## Recent Empirical Notes

Seed 1 30-minute payoff-rebalance run (`runs/cpu_30m_seed1_payoff_v2/20260507_124346_seed1_minute/`):

- 5,429 ticks, 546 neural agents at end (peaked at 861 alive at tick 4,900).
- **Brain capacity grew across the run for the first time**: mean 7.7 → 10.8 (+40%), max 13 → 26 (2x). Earlier runs had brains stuck at ~8 throughout.
- **Three competing lineage strategies** (vs v1's single dominant): lineage 490 collaborate-heavy (241 living, 3,060 offspring), lineage 430 balanced (195 living, 3,990 tools), lineage 489 tool-master (105 living, 9,882 tools). Genuinely differentiated cognitive niches in the same world.
- Ecology denser and more competitive: starvation 1,240 (was 701 in v1), counterattack 246 (was 53).
- Attention concentration moved from 0.01 → 0.02 — slight movement after the raw-values rule fix, but still well below the convergence we'd want. Probably needs longer runs.

Seed 1 30-minute full-pipeline run (`runs/cpu_30m_seed1_full_pipeline/20260507_113805_seed1_minute/`):

- 5,440 ticks, 397 neural agents at end (out of 2,482 total population).
- **Lineage 489 reached generation 38** with 369 living agents, 6,459 cumulative offspring, 54,067 tool successes, collaboration profile = 351,218. By far the deepest selection-driven civilization observed in this substrate.
- Recombination shifted to 33% of births (was 5%). Cross-lineage genome mixing is now a dominant reproduction mode at scale.
- Tool repertoire genuinely diversified: bind 14,982, lever 13,970, crack 9,869, concentrate_heat 9,742, contain 8,237. No single dominant affordance.
- **Organism 416 lived 2,044 ticks** (38% of the run) at place 12 and built a single `structure_support_anchor_gradient_harvest` from scale 6 → 372 across 104 build/extend events, working solo. Died of predation with 1 offspring. Long-horizon coherence in one brain's lifetime.
- Attention concentration stayed flat at 0.01-0.02 across the whole run — the neuroplastic update rule is calibrated too gently to converge in 5,000 ticks.
- Brain capacity mean stayed at ~7.9 throughout. Brains aren't growing despite the mechanism being enabled.

Seed 1 5-minute textured-harshness run (`runs/cpu_5m_harsh_env_textured/20260506_181846_seed1_minute/`):

- Final tick 1016, final population 2284 (575 neural).
- Tool repertoire shifted dramatically vs the pre-textured seed-1 run:
  - `concentrate_heat` 7 → 1198 (cold-place prep rule biting)
  - `bind` 321 → 1530 (abrasion-prep)
  - `filter` 2 → 217
  - `lever` 880 → 2121, `crack` 572 → 1264
- Neural population went from 232 → 575 (+148%) — neural agents outcompete non-neurals more strongly when puzzles demand cognitive work.
- **Lineage 474** is the new dominant civilization. 41 births, max generation 7. Solves four physics-regime puzzles across four places: `crack>lever>contain` at place 15, `cut>bind` at place 14, `bind>contain>filter` at place 16, `concentrate_heat>conduct` at place 23. Three different prep-step types in one lineage = brain template generalizing the physics-conditional rule.
- Run was ~30% slower per tick (1016 vs 1456 ticks in same wall budget).

Seed 1 5-minute pre-textured run (`runs/cpu_5m_harsh_env/20260502_070301_seed1_minute/`) — kept for comparison:

- Final tick 1456, final population 2123 (232 neural).
- Lever-dominated tool monoculture (lever 880, crack 572, all others <250). One 3-step unlock in the entire run.
- Codex-flagged narrative arcs: 424 (early crack specialist eaten), 422 (founder of dominant lineage), 1551 (clean crack→lever causal arc), 3692 (best team-problem-solving), 3427 (builder-then-solver across places 7+8). The arc tool also surfaces 2025 as the run's biggest specialist trap (tool_use=330 from 328 lever-only successes at place 8, died of starvation, no offspring).

Seed 63 10-minute run:

- Run dir: `runs\20260501_204018_seed63_minute`
- Final tick: 1689 due wall limit.
- Final population: total 2613, neural 1634.
- Standout: organism 1958.
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
- Existing predation was visible as `deaths: {'predation': 15}` from agent attacks.

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
- `test_cold_place_prepends_concentrate_heat`
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

The 2026-05-06 session left several open design threads worth pulling next, ranked roughly by leverage:

- **Information cost**: make `observe` cost energy proportional to detail extracted. Currently `observe` averaged -0.026 energy in seed-1 (nearly free), so attention has no economy and marks/signals have no compressive value. Tighten this and `mark`/`signal`/`mark_lesson_*` channels gain real economic weight.
- **Push diversity-aware scoring into `simulation.py`'s `_checkpoint_score`**: arc_report's `diversity_factor` correctly demotes specialist-trap brains in *analysis*, but the simulator still archives them via the same accumulator-style score. Fixing this means the brains saved for transfer will reflect the agentic intelligence the project actually wants, not rote memorizers.
- **Brain-checkpoint trajectory metrics**: action diversity over a window, novelty of place-action pairs. Currently checkpoints rank on cumulative counts; trajectory metrics would catch organisms whose intelligence is in *adaptation*, not volume.
- **Decompose `habitat_mismatch`** into the underlying physical pressures it conflates. The pressures already exist; the label is redundant and obscures cause-of-death analysis.

Earlier carry-over ideas (still good):

- Calibrate relocation teeth: enough failures to matter, not so much that lineages randomly collapse every time.
- Improve event story tooling around movement: identify costly relocations, repeated routes, habitat traps, successful expeditions, and agents that learned to avoid bad moves. Note: `arc_report.py` partially addresses this for organism-centric arcs; place-centric and movement-centric arcs are not yet covered.
- Expand environmental resource coupling in general ways: flow gradients, hydro-like structures, pressure/thermal/electrical reservoirs, sea treasures with consistent risks.
- Add richer object/structure attention to observations so ANNs can notice local affordance causes more directly.

Avoid for now:

- Do not add a separate predator species unless the user reaffirms it. Existing predation via agent attack is enough pressure to inspect first.
- Do not make cooperation mandatory.
- Do not add recipe-like tools such as "axe cuts wood" as a special case. Note: textured-harshness prep steps are *not* recipes — the rule is global (e.g., "cold places need warming first") and physics varies per place.
- Do not reward marks/writing directly; only changed action consequences should matter.
- Do not start a long run before giving the user key specs: profile, seed, ticks/wall limit, places, initial populations, max population, checkpoint cadence.
- Do not collapse fixed `kind` (agent/fungus/plant/neural) into emergent kinds yet. The user explicitly held off on this in the 2026-05-06 session.

## User Preferences

- The user likes bold, open-ended mechanisms but is sensitive to overprogramming.
- They prefer general physical/causal relationships over hand-authored goals.
- They are comfortable with bigger populations, but local machine constraints matter.
- They want short progress updates and concrete implementation momentum.
- Before launching a run, provide key run specs.
- If a run is clearly not going to answer the question, stop it and pivot.

## One-Sentence North Star

Build a cheap, scalable universe where survival and reproduction favor agents that learn causal structure, manipulate materials, manage energy, communicate useful traces, and sometimes become transferable minds, without the simulator secretly telling them to do any of that.
