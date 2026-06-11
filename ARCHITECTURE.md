# Microcosmic God Architecture

## Purpose

Microcosmic God is a headless artificial life research sandbox. Its first job is not to solve a task, but to host a small, consistent, evolving universe where intelligence, tool use, communication, prediction, reproduction, and social behavior can become useful if the environment makes them useful.

Prototype 0 should run in minutes on CPU. Longer runs, larger populations, and GPU-scale epochs should be possible later without changing the conceptual model.

## Design Commitments

- Encode world laws, not desired behaviors.
- Let survival, reproduction, energy capture, and causal consequences drive selection.
- Make intelligence expensive in upkeep from the beginning.
- Do not directly reward tool use, cooperation, communication, curiosity, teaching, family behavior, trade, culture, or science.
- Keep genomes readable enough to debug early collapses.
- Infer species after runs; do not assign fixed species labels.
- Allow extinction and produce useful debriefs when it happens.
- Keep a logged intervention hatch for garden runs while preserving sealed runs.

## Current Stack

Prototype 0 currently uses a dependency-light Python 3.12 core with analysis-friendly outputs. This was chosen because Rust is not installed on the local machine, and getting a working experimental loop matters more than waiting on a toolchain.

- Python simulation core:
  - deterministic stepping
  - portable run folders
  - simple debugging
  - compact enough for local minute-scale runs
  - easy migration path to Rust, NumPy, JAX, Modal, or Colab
- Structured output:
  - JSONL for early logs
  - binary snapshots later when volume grows
- Python analysis layer:
  - plotting
  - extinction debriefs
  - species/lineage clustering
  - saved-controller transfer experiments
  - notebooks or scripts

The first version does not need a visual UI.

Rust remains a strong candidate for the future hot-loop simulator once the model stabilizes.

## World Model

Use a sparse environment graph rather than a grid.

```text
World
  places: environment patches
  routes: connections between places
  entities: individuals, materials, tools, structures
  fields: local energy and environmental conditions
  laws: causal rules for matter, energy, action, damage, decay
```

Places represent meaningful local contexts: pond edge, fungal mat, sunlit stone, mineral vent, burrow, canopy, dry basin, river crossing. Routes represent possible movement, not just geometry. Barriers and distance can be represented through route cost, danger, required affordances, or environmental exposure.

This preserves locality and exploration while avoiding the cost of simulating empty space.

## Tick Loop

Each simulation tick should be deterministic given seed and configuration.

```text
1. Update environmental fields.
2. Update non-neural individuals.
3. Build observations for neural individuals.
4. Run neural policies and local learning.
5. Resolve actions through world laws.
6. Apply upkeep, damage, repair, growth, and decay.
7. Resolve reproduction attempts.
8. Remove inactive individuals and decay abandoned state.
9. Record compact logs and optional checkpoints.
10. Stop or debrief if extinction or time limit occurs.
```

Use short runs first. A useful Prototype 0 run should complete in minutes, even if it ends in extinction.

## Energy And Matter

Energy must not collapse into a single resource score. Represent typed energy and conversion routes.

```text
EnergyKind
  solar
  essence
  organic_store
  thermal
  mechanical
  electrical
  dense_node
```

Objects and individuals can store, convert, waste, or exploit these forms depending on body modules, tools, structures, and learned skill.

Example continuity:

```text
sunlight
  simple use: solar-energy capture
  system-level use: seasons, drying, warming
  tool use: concentrating heat
  advanced use: photovoltaic-like conversion
  deep use: stellar/nuclear-inspired high-density power chains
```

Matter should expose stable properties:

```text
mass
hardness
sharpness
flexibility
brittleness
durability
conductivity
combustibility
toxicity
nutritional_value
thermal_capacity
absorption
phase
```

World laws operate over these properties rather than over hard-coded object recipes.

## Individuals

Not every individual needs a neural network.

```text
Individual
  id
  genome
  body
  upkeep
  location
  age
  energy stores
  health/damage state
  optional controller
  optional memory
  lineage metadata
```

Early individual categories:

- Non-neural individuals: producers, consumers, microbial analogs, simple environmental individuals.
- Primitive neural individuals: mobile agents with small expensive controllers.
- Higher-cost neural individuals: rare agents with memory, prediction, tool manipulation, or richer learning.

These are implementation categories, not permanent species labels.

## Genome And Development

Use structured, debuggable genomes at first.

```text
Genome
  upkeep parameters
  body module parameters
  sensor parameters
  effector parameters
  neural capacity parameters
  learning/plasticity parameters
  valence wiring parameters
  communication parameters
  reproduction parameters
  mutation/recombination parameters
  developmental budget parameters
```

Mutation should eventually affect every attribute with a real-world analog. Early implementation can expose a small set of numeric parameters and expand from there.

Inheritance follows standard parameter inheritance by default:

- Children inherit genome/development parameters.
- Lifetime-learned neural weights are not directly inherited.
- Teaching, imitation, parental investment, and cultural transfer can emerge behaviorally.

## Controllers

Controller capacity should have explicit cost:

```text
neural_tick_cost = base_cost + neuron_cost + memory_cost + prediction_cost + plasticity_cost
```

Prototype 0 controller:

```text
ControllerCore
  small recurrent policy
  compact hidden state
  optional prediction head
  evolved plasticity parameters
```

The controller receives local observations, body state, memory summaries, available action affordances, and evolved valence signals. It outputs action choices, signal emissions, attention/use priorities, and possibly learning gates.

## Learning

Learning should be real but not task-rewarded.

Inputs to learning:

- evolved pain/pleasure/valence signals from body state
- prediction errors about local consequences
- success/failure of actions under physics
- observation of other individuals' actions
- memory retrieval

Avoid direct novelty rewards. Novelty matters only when predictive improvements, resource discovery, survival, reproduction, or cultural transfer make it useful.

## Tools And Skill

Tools are affordance bundles, not just inventory labels.

```text
tool usefulness = object affordances + body compatibility + learned skill + context prediction
```

Possession is not competence. An agent may hold a tool and still fail, waste energy, break it, injure itself, or use it in the wrong context.

Represent skill through a combination of:

- neural control competence
- learned action sequencing
- predictive models of consequences
- body/tool compatibility
- prior practice or observation

## Reproduction

Support both asexual and sexual reproduction.

Asexual reproduction:

- one parent
- lower coordination burden
- lower developmental complexity ceiling
- mutated copy of genome/development parameters

Sexual reproduction:

- two parents
- requires behavioral coordination
- requires compatibility
- recombines genomes
- can unlock higher developmental complexity budgets
- allows pairing selection to evolve from perception and behavior

Do not expose a direct pairing fitness score. Individuals may perceive health, energy, age, behavior, territory, signals, morphology, tool competence, or past outcomes if their sensors and memory support it.

## Communication

Start with a cheap but limited signal channel.

```text
Signal
  emitter
  local target or broadcast radius
  token or low-dimensional vector
  intensity
  energy cost
  medium constraints
```

Signals begin without fixed semantic meaning. Meaning emerges if agents learn or evolve useful associations.

## Environmental Variation

World laws should remain stable, but conditions should drift.

Initial variation candidates:

- seasons
- local resource depletion
- regrowth and succession
- weather-like field variation
- migration pressure
- rare disasters
- climate drift

Variation should create system-level pressure without becoming a hidden curriculum.

## Interventions

Support two run modes.

```text
sealed run:
  no interventions after initialization

garden run:
  researchers may add/remove/change entities, fields, or events
  every intervention is logged with tick, author, and reason
```

Interventions are allowed for exploration, but debriefs must separate natural dynamics from touched dynamics.

## Controller Checkpointing

Active controllers exist in memory. Inactive agents' learned weights disappear by default unless selected for archival.

Checkpoint candidates:

- manual selection
- lineage champions
- novelty outliers
- first use of a new tool chain
- long-lived individuals
- reproductively successful individuals
- rare system-level strategies
- random population samples

Saved controller package:

```text
ControllerCheckpoint
  controller weights
  architecture metadata
  genome
  body configuration
  adapter/schema version
  memory summary, optional
  lineage
  run configuration
  environment context
  reason saved
```

This enables later transfer experiments into new worlds, games, or embodied tasks.

## Logging And Debriefing

Logs should be compact but scientifically useful.

Core logs:

- run configuration and seed
- population counts
- creations, removals, and causes
- energy availability by type and place
- lineage events
- reproduction events
- mutation summaries
- major environmental changes
- intervention events
- checkpoint events
- extinction state, if reached

Extinction debrief should summarize:

- final population timeline
- last surviving lineages
- likely bottlenecks
- removal cause distribution
- resource and energy collapse patterns
- mutation load signs
- antagonistic interaction or competition pressure
- environmental shifts near collapse
- whether collapse was sudden or gradual

## Scaling Path

Prototype 0:

- single-process CPU
- small environment graph
- hundreds to low thousands of individuals
- minute-scale runs
- JSONL logs

Prototype 1:

- data-oriented hot loops
- parallel place updates where safe
- binary snapshots
- larger worlds and longer runs
- Python debrief tooling

Prototype 2:

- vectorized neural inference
- batched environments
- GPU-backed controller evaluation where useful
- large epoch orchestration
- transfer-learning experiments from saved controllers

## Proposed Repository Shape

```text
microcosmic-god/
  ARCHITECTURE.md
  PROJECT_BRIEF.md
  README.md
  pyproject.toml
  microcosmic_god/
    brain.py
    checkpoints.py
    cli.py
    config.py
    debrief.py
    energy.py
    genome.py
    interventions.py
    organisms.py
    runlog.py
    simulation.py
    world.py
  crates/
    sim/
      src/
        world/
        organisms/
        genome/
        brain/
        energy/
        logging/
        experiments/
  analysis/
    notebooks/
    scripts/
  examples/
  runs/
    .gitkeep
  docs/
```

The first code milestone is now implemented as a Python package. Future work can replace hot loops with Rust or vectorized backends without changing the run-folder contract.
