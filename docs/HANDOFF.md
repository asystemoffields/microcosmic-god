# Microcosmic God Handoff

Last updated: 2026-05-01

This is the quick-start context for a fresh Codex instance taking over Microcosmic God.

## Repo State

- Repo: `C:\Users\power\Documents\Codex\2026-05-01\i-have-an-exciting-and-fun\microcosmic-god`
- Remote: `https://github.com/asystemoffields/microcosmic-god`
- Branch: `main`
- Latest pushed commit: `5680f86 Lock movement energy costs`
- Recent commits:
  - `5680f86` lock movement energy-cost tests
  - `4654066` add collaboration, movement telemetry, relocation shock, hostile biomes
  - `42fc5a9` archive seed 63 standout brains
  - `2629c28` add portable memory and gear capacities
  - `5f223c4` add literacy value feedback for marks

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

Recent additions cover:

- `test_bind_practice_transfers_only_to_related_skills`
- `test_specialists_keep_cognitive_credit_from_repeated_practice`
- `test_active_helpers_can_supply_build_materials`
- `test_collective_support_and_relocation_shock_are_tracked_for_moves`
- `test_successful_movement_spends_energy_even_when_easy`
- `test_failed_movement_spends_energy`
- `test_environment_generation_has_hostile_treasure_biomes`

As of commit `5680f86`, full suite passed:

```text
Ran 35 tests
OK
```

## Likely Next Actions

Good next work:

- Run a 5-10 minute sealed run and inspect movement/collaboration/tool/mark stories.
- Calibrate relocation teeth: enough failures to matter, not so much that lineages randomly collapse every time.
- Improve event story tooling around movement: identify costly relocations, repeated routes, habitat traps, successful expeditions, and agents that learned to avoid bad moves.
- Expand environmental resource coupling in general ways: flow gradients, hydro-like structures, pressure/thermal/electrical reservoirs, sea treasures with consistent risks.
- Add richer object/structure attention to observations so ANNs can notice local affordance causes more directly.
- Improve checkpoint ranking so it catches specialists, collaborators, causal unlockers, tool makers, literacy users, and strong movers separately.

Avoid for now:

- Do not add a separate predator species unless the user reaffirms it. Existing predation via agent attack is enough pressure to inspect first.
- Do not make cooperation mandatory.
- Do not add recipe-like tools such as "axe cuts wood" as a special case.
- Do not reward marks/writing directly; only changed action consequences should matter.
- Do not start a long run before giving the user key specs: profile, seed, ticks/wall limit, places, initial populations, max population, checkpoint cadence.

## User Preferences

- The user likes bold, open-ended mechanisms but is sensitive to overprogramming.
- They prefer general physical/causal relationships over hand-authored goals.
- They are comfortable with bigger populations, but local machine constraints matter.
- They want short progress updates and concrete implementation momentum.
- Before launching a run, provide key run specs.
- If a run is clearly not going to answer the question, stop it and pivot.

## One-Sentence North Star

Build a cheap, scalable universe where survival and reproduction favor agents that learn causal structure, manipulate materials, manage energy, communicate useful traces, and sometimes become transferable minds, without the simulator secretly telling them to do any of that.
