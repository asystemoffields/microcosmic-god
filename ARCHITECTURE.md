# Microcosmic God Architecture

## Purpose

Microcosmic God is a headless artificial life research sandbox. Its first job is not to solve a task, but to host a small, consistent, developing universe where intelligence, tool use, communication, prediction, spawning, and social behavior can become useful if the environment makes them useful.

Prototype 0 should run in minutes on CPU. Longer runs, larger pools, and GPU-scale epochs should be possible later without changing the conceptual model.

## Design Commitments

- Encode world laws, not desired behaviors.
- Let persistence, spawning, energy capture, and causal consequences drive selection.
- Make intelligence expensive in upkeep from the beginning.
- Do not directly reward tool use, cooperation, communication, curiosity, teaching, family behavior, trade, culture, or science.
- Keep param_vectors readable enough to debug early collapses.
- Infer type after runs; do not assign fixed type labels.
- Allow washout and produce useful debriefs when it happens.
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
  - washout debriefs
  - type/line clustering
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
7. Resolve spawning attempts.
8. Remove inactive individuals and decay abandoned state.
9. Record compact logs and optional checkpoints.
10. Stop or debrief if washout or time limit occurs.
```

Use short runs first. A useful Prototype 0 run should complete in minutes, even if it ends in washout.

## Energy And Matter

Energy must not collapse into a single resource score. Represent typed energy and conversion routes.

```text
EnergyKind
  solar
  essence
  residue_store
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
  params
  body
  upkeep
  location
  age
  energy stores
  health/damage state
  optional controller
  optional memory
  line metadata
```

Early individual categories:

- Non-neural individuals: producers, consumers, microbial analogs, simple environmental individuals.
- Primitive neural individuals: mobile agents with small expensive controllers.
- Higher-cost neural individuals: rare agents with memory, prediction, tool manipulation, or richer learning.

These are implementation categories, not permanent type labels.

## Params And Development

Use structured, debuggable param_vectors at first.

```text
Params
  upkeep parameters
  body module parameters
  sensor parameters
  effector parameters
  neural capacity parameters
  learning/plasticity parameters
  valence wiring parameters
  communication parameters
  spawning parameters
  perturbation/combination parameters
  developmental budget parameters
```

Perturbation should eventually affect every attribute with a real-world analog. Early implementation can expose a small set of numeric parameters and expand from there.

Inheritance follows standard parameter inheritance by default:

- Children inherit params/development parameters.
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
  developed plasticity parameters
```

The controller receives local observations, body state, memory summaries, available action affordances, and developed valence signals. It outputs action choices, signal emissions, attention/use priorities, and possibly learning gates.

## Learning

Learning should be real but not task-rewarded.

Inputs to learning:

- developed pain/pleasure/valence signals from body state
- prediction errors about local consequences
- success/failure of actions under physics
- observation of other individuals' actions
- memory retrieval

Avoid direct novelty rewards. Novelty matters only when predictive improvements, resource discovery, persistence, spawning, or cultural transfer make it useful.

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

## Spawning

Support both solo and paired spawning.

Solo spawning:

- one parent
- lower coordination burden
- lower developmental complexity ceiling
- perturbed copy of params/development parameters

Paired spawning:

- two parents
- requires behavioral coordination
- requires compatibility
- combines param_vectors
- can unlock higher developmental complexity budgets
- allows pairing selection to develop from perception and behavior

Do not expose a direct pairing quality score. Individuals may perceive health, energy, age, behavior, territory, signals, morphology, tool competence, or past outcomes if their sensors and memory support it.

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

Signals begin without fixed semantic meaning. Meaning emerges if agents learn or develop useful associations.

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
- line champions
- novelty outliers
- first use of a new tool chain
- long-lived individuals
- via_spawning successful individuals
- rare system-level strategies
- random pool samples

Saved controller package:

```text
ControllerCheckpoint
  controller weights
  architecture metadata
  params
  body configuration
  adapter/schema version
  memory summary, optional
  line
  run configuration
  environment context
  reason saved
```

This enables later transfer experiments into new worlds, games, or embodied tasks.

## Logging And Debriefing

Logs should be compact but scientifically useful.

Core logs:

- run configuration and seed
- pool counts
- creations, removals, and causes
- energy availability by type and place
- line events
- spawning events
- perturbation summaries
- major environmental changes
- intervention events
- checkpoint events
- washout state, if reached

Washout debrief should summarize:

- final pool timeline
- last persisting lines
- likely bottlenecks
- removal cause distribution
- resource and energy collapse patterns
- perturbation load signs
- inter-individual interaction or competition pressure
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
    controller.py
    checkpoints.py
    cli.py
    config.py
    debrief.py
    energy.py
    params.py
    interventions.py
    individuals.py
    runlog.py
    simulation.py
    world.py
  crates/
    sim/
      src/
        world/
        individuals/
        params/
        controller/
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
