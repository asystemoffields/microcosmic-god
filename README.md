# Microcosmic God

Microcosmic God is a headless artificial-life sandbox for evolving ecology, intelligence, tool use, communication, and reproduction inside a cheap but consistent causal universe.

The current implementation is Prototype 0: a runnable Python simulation designed for short local experiments first, with portable run folders that can later move to Modal, Colab, or a faster systems core.

## What Exists Now

- Sparse ecological graph world, not a grid.
- Multiple energy gradients from day one: radiant, chemical, biological storage, thermal, mechanical, electrical, and high-density latent energy.
- Evolving non-neural organisms: plant and fungus analogs.
- Evolving neural agents with recurrent ANNs, eligibility traces, predictive learning, and representational plasticity.
- Darwinian inheritance of genome/development parameters and innate brain templates.
- Lifetime learning through evolved valence wiring and plasticity.
- Evolution operators are split from world physics through an `EvolutionEngine`.
- `clone_mutate` and `recombine` are the first two in-world operators; they replace biology-first action labels.
- Recombine requires behavioral coordination through short-lived local intent states.
- Short action-result traces and event-memory slots let agents perceive recent consequences of their own behavior.
- Multiple prediction heads learn energy, damage, reproduction, social, tool, and movement-hazard outcomes.
- Observer-only success profiles track energy capture, prediction fit, tool making/use, structures, causal unlocks, social learning, and reproduction for checkpointing and debriefs without becoming hidden reward.
- Tool affordances derived from material properties.
- Tool use requires object affordance, body compatibility, learned skill, and context.
- Crafting now has proto-reasoning: agents infer a local target affordance, choose components under noisy planning/skill constraints, and only get better artifacts when the materials actually fit the problem.
- Place-level causal challenges can require short affordance sequences, such as containment then filtering, to unlock finite local energy payoffs through consistent physical interaction.
- The payoff curve is intentionally steep: learned sequencing, useful tools, and structures can open much larger energy reservoirs than surface foraging.
- Composite artifacts with derived capabilities, durability, and tiered resistance against harder materials/obstacles.
- Persistent material structures built from gathered components, including place-level boundaries, channels, supports, filters, and gradient harvesters.
- Structure decay through general material-environment coupling: mechanical wear, corrosion, biological decay, thermal damage, solubility, radiation, and fatigue.
- Graph-field physics for temperature, fluid level, pressure/depth, humidity, salinity, elevation, current exposure, edge slope, edge current, permeability, and conductance.
- Environmental degradation fields such as oxygen-like exposure, acidity, biological activity, abrasion, and wet/dry cycling are visible to agents.
- Physics-driven effects such as heat/pressure/current stress, chemical advection, signal advection, mark erosion, current-assisted movement, and gravity falls.
- Material-coupled artifact capabilities including filtering, floating, anchoring, traversal, insulation, conductivity, containment, cracking, cutting, levering, and heat concentration.
- General artifact capacities for carrying, protection, and record surfaces, so backpack-like, protective, and portable-writing objects can emerge from material properties.
- Inside/outside is modeled as boundary physics: `enclose`, `permeable`, and `shelter` are separate capabilities, so not every inside is protective.
- Diversified habitats: aquatic/terrestrial balance, depth, salinity, humidity, heat, height, water, and thorn barriers.
- Hostile-but-valuable environment archetypes such as pelagic water, reefs, trenches, vents, ridges, scree, desert glass, forest edges, marshes, and caverns. These bias fields and resources without creating quests.
- Evolving habitat tolerances, so some organisms can drown, desiccate, or specialize for aquatic/dry niches.
- Movement has real teeth: distance, load, slope, current, pressure, barriers, and relocation shock impose energy/health costs, while tools, memory, planning, and collaboration can mitigate them.
- Collaboration can indirectly matter through active helper support for expeditions, tool use, causal unlocks, and pooled-material structures, without making cooperation a required objective.
- Local signals with no fixed semantics.
- Durable-but-decaying place marks, a primitive external memory channel analogous to writing.
- Intentional lesson inscriptions are distinct from plain marks. Agents must have recent tool/problem experience and discover/use `inscribe`; readers improve through `interpret_mark`.
- Predation and ecological collapse can happen.
- Extinction or run-limit debriefs are written automatically.
- Selective brain checkpoints are saved for notable neural agents.
- Brain checkpoints include cognitive context: recent trace, event memory, lesson memory, signal associations, and place memory.
- Garden mode supports logged interventions.
- Reproduction failure telemetry, per-action energy accounting, and deaths split by organism kind.
- Movement telemetry summarizes attempts, cost, relocation shock, support, motives, and routes.
- Regression tests for locality, signal observability, crafting failure costs, and reproduction capacity contracts.

## Quick Start

```powershell
python -m microcosmic_god specs
python -m microcosmic_god run --profile smoke --seed 7 --dry-run
python -m microcosmic_god run --profile smoke --seed 7
```

Run outputs are written under `runs/`.

Each run folder contains:

- `config.json`
- `events.jsonl`
- `story_events.jsonl`
- `summary.json`
- `world_final.json`
- `checkpoints/` for saved ANN snapshots

## Profiles

| Profile | Intended Use | Defaults |
| --- | --- | --- |
| `smoke` | Validate the engine in seconds | 12 places, 118 initial organisms, 300 ticks, 20s wall cap |
| `minute` | Local exploratory run | 36 places, 490 initial organisms, 8,000 ticks, 300s wall cap |
| `long` | Overnight or day-scale local run | 96 places, 3,000 initial organisms, 1,000,000 ticks, 24h wall cap |
| `modal` | Cloud/remote scaling target | 256 places, 5,800 initial organisms, 10,000,000 ticks, 3d wall cap |

Override any key size directly:

```powershell
python -m microcosmic_god run --profile minute --seed 42 --ticks 20000 --wall-seconds 900 --agents 80 --max-population 2500
```

Disable the wall limit for an intentionally long local run:

```powershell
python -m microcosmic_god run --profile long --seed 42 --wall-seconds 0
```

## Machine Fit

The first target machine has:

- AMD Ryzen 5 7530U
- 6 physical cores, 12 logical threads
- 7.28 GiB usable RAM
- integrated AMD Radeon graphics
- about 306 GiB free disk at setup time

Recommended local starting points:

- Use `smoke` for correctness checks.
- Use `minute` for iteration.
- Use `long` only after `minute` runs stop collapsing immediately and RAM is not under pressure.
- Use `--quiet-events` for longer runs to reduce log size.

## Garden Runs

Sealed runs are the default. Garden runs allow interventions, and every intervention is logged.

```powershell
python -m microcosmic_god run --profile smoke --garden --interventions examples/interventions.example.json
```

## ANN Checkpoints

Dead agents lose their live ANN by default. The simulator only saves brains when selected by checkpoint policy:

- first successful use of a new tool affordance
- interval champion
- notable death after reproduction or tool success

Checkpoint files include:

- live brain weights
- innate brain template
- genome
- organism state
- tool skills
- observer success profile
- signal associations
- recent action-result trace and event memory
- recent lesson memory
- place memory
- ecological context
- reason saved

Checkpoint slots are bucketed so one dramatic failure mode cannot consume the whole archive. First tool successes, interval/final living champions, reproductive champions, tool champions, causal champions, learner champions, lineage founders, and notable deaths each get their own quota inside the global checkpoint limit.

These are the transfer candidates for future experiments in other worlds or games.

Prototype 0 brains are still intentionally compact, but they now carry hidden state, input/hidden eligibility traces, learned action preferences, prediction weights, and valence-modulated input-to-hidden plasticity. Genome neural budgets can mutate far beyond the local starting sizes, with metabolic cost deciding whether larger brains survive.

Run summaries include an `evolution_policy` block describing the active operators. The current sealed policy is still triggered by in-world action and interaction, while the code is now shaped to support future farm-mode policies like "these were effective operators and learners, make more like that."

See [docs/TRANSFER_RUNWAY.md](docs/TRANSFER_RUNWAY.md) for the plan to separate reusable brain cores from world-specific adapters and test saved agents in held-out worlds, simple games, and eventually richer RL environments.

See [docs/LEARNING_ARCHITECTURE.md](docs/LEARNING_ARCHITECTURE.md) for the north-star contract: evolve compact causal learners that may transfer through new encoders/action heads, not just policies that memorize this sandbox.

## SAE Inspection

Train a small sparse autoencoder on brain checkpoints after a run:

```powershell
python analysis\scripts\train_checkpoint_sae.py runs\<run_dir>\checkpoints archives\brains --latent 16 --steps 1500 --out analysis\sae_models\run_sae.npz
```

Inspect a checkpoint through the trained SAE:

```powershell
python analysis\scripts\inspect_sae.py analysis\sae_models\run_sae.npz runs\<run_dir>\checkpoints\<brain_file>.json
```

This is an analysis microscope only. SAE features never feed back into agent reward, perception, reproduction, or world physics.

## Knowledge Transmission

Agents have two low-level channels, neither with built-in meaning:

- `signal`: temporary local emission, like vocalization or gesture. It decays within a few ticks.
- `mark`: costly local inscription, like a primitive durable sign. It persists for many ticks but decays through age, volatility, and water exposure.

Agents can learn associations between observed tokens and later internal valence. A token only becomes useful if ecology makes it useful.

Marks are plain tokens by default. Some marks can intentionally preserve a fuzzy trace of the maker's recent tool/craft/problem experience: action, affordance, rough success, method quality, components, and a coarse local problem frame when the inscription is clear enough. Writing that packet is not free or guaranteed; it requires useful recent experience, body/material capacity, attention, and the learnable `inscribe` skill. Observing it can slightly improve relevant skill only when the reader spends an observe action and has enough sensor/memory/attention plus `interpret_mark` ability to extract the trace. This is not language yet, but it gives durable writing-like behavior a physical channel to matter across time, and agents can theoretically copy useful traces elsewhere.

The world now treats literacy as an action-mediated advantage, not a hidden score. A mark's `writing_quality` emerges from inscription clarity, lesson coherence, and the value of the underlying tool/problem experience. Good writing is rewarded only if another agent can use it: useful reads increase the reader's `interpret_mark` ability, record mark `reads` and `value_transmitted`, and can feed local feedback to the living, co-present author through `knowledge_transmitted`. Bad, irrelevant, or unread writing remains mostly cost and noise.

Record-capable artifacts can now hold portable lesson traces. A self-read can serve as external memory and improve later action without counting as knowledge transmission; another agent reading that carried trace can still create the ordinary teaching feedback if the author is present. Carry-capable artifacts expand material/tool capacity, while protect-capable artifacts reduce environmental, accident, and predation damage through the same material-derived artifact system.

Mark creation is summarized in aggregate/debrief counters rather than logged as one event per mark, so agents are free to mark obsessively if that behavior evolves. Intentional lesson writes and successful reads can be promoted to `story_events.jsonl`.

See [docs/LANGUAGE_RUNWAY.md](docs/LANGUAGE_RUNWAY.md) for the longer path from meaningless signals to possible future language-transfer experiments.

## Generality And Scaling

Every new mechanic should be governed by reusable laws rather than special-case objectives. Tool and habitat mechanics should compose through properties like hardness, resistance, containment, traversal, insulation, conductivity, and learned skill.

See [docs/GENERALITY_AND_SCALING.md](docs/GENERALITY_AND_SCALING.md) for the project contract that keeps the simulator general enough for open-ended evolution and structured enough to move from local CPU runs to many-world GPU epochs.

See [docs/PHYSICS_KERNEL.md](docs/PHYSICS_KERNEL.md) for the plan to add graph-field physics: thermodynamics, currents, gravity/elevation, pressure/depth, and material coupling as cheap local laws.

See [docs/EVENT_MONITORING.md](docs/EVENT_MONITORING.md) for the logging strategy: cheap counters for routine substrate dynamics, bounded rolling context, and durable story records only when events become important.

See [side_projects/](side_projects/) for speculative branches, including Universal Genesis: an unseeded big-bang-to-life thought experiment kept separate from the practical simulator.

## Design Rule

Do not directly reward intelligence, tool use, communication, curiosity, family, teaching, trade, or culture.

Make those behaviors possible, make them costly, and let ecology decide whether they survive.
