# Microcosmic God

Microcosmic God is a headless artificial-life sandbox for evolving ecology, intelligence, tool use, communication, and reproduction inside a cheap but consistent causal universe.

The current implementation is Prototype 0: a runnable Python simulation designed for short local experiments first, with portable run folders that can later move to Modal, Colab, or a faster systems core.

## What Exists Now

- Sparse ecological graph world, not a grid.
- Multiple energy gradients from day one: radiant, chemical, biological storage, thermal, mechanical, electrical, and high-density latent energy.
- Evolving non-neural organisms: plant and fungus analogs.
- Evolving neural agents with small recurrent ANNs.
- Darwinian inheritance of genome/development parameters and innate brain templates.
- Lifetime learning through evolved valence wiring and plasticity.
- Asexual and sexual reproduction.
- Sexual reproduction requires behavioral coordination through short-lived courtship/receptivity states.
- Tool affordances derived from material properties.
- Tool use requires object affordance, body compatibility, learned skill, and context.
- Composite artifacts with derived capabilities, durability, and tiered resistance against harder materials/obstacles.
- Graph-field physics for temperature, fluid level, pressure/depth, humidity, salinity, elevation, current exposure, edge slope, edge current, permeability, and conductance.
- Physics-driven effects such as heat/pressure/current stress, chemical advection, signal advection, mark erosion, current-assisted movement, and gravity falls.
- Material-coupled artifact capabilities including filtering, floating, anchoring, traversal, insulation, conductivity, containment, cracking, cutting, levering, and heat concentration.
- Diversified habitats: aquatic/terrestrial balance, depth, salinity, humidity, heat, height, water, and thorn barriers.
- Evolving habitat tolerances, so some organisms can drown, desiccate, or specialize for aquatic/dry niches.
- Local signals with no fixed semantics.
- Durable-but-decaying place marks, a primitive external memory channel analogous to writing.
- Predation and ecological collapse can happen.
- Extinction or run-limit debriefs are written automatically.
- Selective brain checkpoints are saved for notable neural agents.
- Garden mode supports logged interventions.
- Reproduction failure telemetry, per-action energy accounting, and deaths split by organism kind.
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
- signal associations
- ecological context
- reason saved

These are the transfer candidates for future experiments in other worlds or games.

Prototype 0 brains are small leaky reactive policies: hidden state carries forward with fixed decay, while lifetime learning updates output/action preferences and prediction weights. Rich learned recurrence is a future brain-core upgrade.

See [docs/TRANSFER_RUNWAY.md](docs/TRANSFER_RUNWAY.md) for the plan to separate reusable brain cores from world-specific adapters and test saved agents in held-out worlds, simple games, and eventually richer RL environments.

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

Mark creation is summarized in aggregate/debrief counters rather than logged as one event per mark, so agents are free to mark obsessively if that behavior evolves.

See [docs/LANGUAGE_RUNWAY.md](docs/LANGUAGE_RUNWAY.md) for the longer path from meaningless signals to possible future language-transfer experiments.

## Generality And Scaling

Every new mechanic should be governed by reusable laws rather than special-case objectives. Tool and habitat mechanics should compose through properties like hardness, resistance, containment, traversal, insulation, conductivity, and learned skill.

See [docs/GENERALITY_AND_SCALING.md](docs/GENERALITY_AND_SCALING.md) for the project contract that keeps the simulator general enough for open-ended evolution and structured enough to move from local CPU runs to many-world GPU epochs.

See [docs/PHYSICS_KERNEL.md](docs/PHYSICS_KERNEL.md) for the plan to add graph-field physics: thermodynamics, currents, gravity/elevation, pressure/depth, and material coupling as cheap local laws.

See [side_projects/](side_projects/) for speculative branches, including Universal Genesis: an unseeded big-bang-to-life thought experiment kept separate from the practical simulator.

## Design Rule

Do not directly reward intelligence, tool use, communication, curiosity, family, teaching, trade, or culture.

Make those behaviors possible, make them costly, and let ecology decide whether they survive.
