# Brain Transfer Runway

This document describes how Microcosmic God should be shaped so saved agent brains can eventually seed learning in different worlds, simple games, or other RL environments.

The near-term target is modest but important:

```text
Does a brain evolved in Microcosmic God learn a new environment faster, more robustly, or with better exploration than a random brain of the same size?
```

Directly wiring one saved policy into an unrelated video game is unlikely to work if the input and action spaces are unrelated. The plausible transfer target is the evolved internal machinery: recurrent state, prediction habits, learned control priors, memory dynamics, and sensorimotor abstractions.

## Core Idea

Separate the brain into reusable parts:

```text
world-specific observation adapter
  -> reusable recurrent/predictive core
  -> world-specific action adapter
  -> optional value/valence/prediction heads
```

The adapter changes when moving from Microcosmic God to another task. The core is the part we hope evolution and lifetime learning make interesting.

## Current Prototype Status

Prototype 0 uses `TinyBrain`:

```text
inputs -> recurrent hidden state -> action logits
                         -> energy prediction head
```

Checkpoint files already save:

- brain weights and live hidden state
- innate brain template
- genome
- body and organism state
- tool skill
- signal associations
- ecological context
- checkpoint reason

This is enough for archival and early inspection. It is not yet enough for clean cross-environment transfer because input and output matrices are tied to the Microcosmic observation/action schema.

## Transfer Package Contract

Future saved brains should export a `BrainPackage` with explicit schema metadata:

```text
BrainPackage
  package_version
  brain_architecture
  reusable_core_weights
  observation_adapter_weights
  action_adapter_weights
  prediction_heads
  hidden_state_optional
  genome
  body_metadata
  training_history_summary
  observation_schema
  action_schema
  world_schema
  checkpoint_reason
```

The package should say which weights are expected to transfer and which are sandbox-specific.

## Observation Design For Transfer

Microcosmic observations should avoid becoming arbitrary feature soup. They should be organized around reusable physical concepts:

- self state: energy, damage, age, motion, internal memory
- local fields: heat, water, salinity, light, current, slope, pressure
- objects: material properties, affordances, relative availability
- agents: motion, proximity, emitted signals, observed action traces
- consequences: recent action result, energy delta, prediction error
- communication: temporary signal tokens and durable mark tokens

This gives a future game adapter something to map into. A simple game like catch can expose ball position, velocity, body state, and action result. A richer game can use a learned visual encoder to produce similar tokens from pixels.

## Action Design For Transfer

Reusable action structure matters as much as observation structure.

Microcosmic actions should keep pointing toward general verbs:

- move
- attend/observe
- manipulate/use
- collect/pickup
- combine/craft
- emit signal
- make mark
- wait/rest
- pursue/attack
- court/mate

For another RL environment, the action adapter maps these internal action tendencies to task-specific controls. In a catch game, `move` maps to left/right/up/down. In Atari, a small action head maps recurrent-core output to joystick/button logits.

## Transfer Protocol

For a new environment:

1. Load a saved `BrainPackage`.
2. Keep the reusable recurrent/predictive core.
3. Replace or reinitialize the observation adapter for the new environment.
4. Replace or reinitialize the action adapter for the new environment.
5. Train adapters first while the core is frozen.
6. Fine-tune the core slowly if adapter-only training plateaus.
7. Compare against:
   - random brain with same architecture
   - randomly initialized core plus trained adapters
   - scratch-trained baseline
   - shuffled or damaged saved core

The transfer claim only means something if saved brains beat these controls.

## Selection Without Hidden Objectives

The simulator should not evolve agents for transfer. Transfer candidates should be selected after the run by observer heuristics.

Good checkpoint signals:

- survived across multiple habitat regimes
- used multiple tool affordances successfully
- improved prediction error during life
- reproduced in more than one ecological context
- carried useful artifacts or moved through barriers
- communicated or marked before later adaptive behavior
- recovered from scarcity, predation pressure, or environmental drift
- performed well in held-out Microcosmic probe worlds

These are analysis filters, not rewards.

## Probe Worlds Before Video Games

Before attempting an unrelated game, test saved brains in held-out Microcosmic variants:

- changed resource distribution
- changed currents, heat, salinity, or terrain barriers
- different material availability
- new locked resources using known affordance laws
- altered ecology and predation pressure
- different communication decay rates

If a brain cannot adapt to nearby worlds, it is unlikely to help in a video game.

## Video Game Path

Simple vector game first:

```text
Catch / dodge / collect game
  observation adapter: object positions, velocities, hazards, reward-like body signal
  reusable core: saved Microcosmic recurrent/predictive weights
  action adapter: movement buttons
```

Then small pixel game:

```text
Pixel encoder
  -> object/field latent tokens
  -> saved recurrent/predictive core
  -> action adapter
```

Atari-like transfer is a later target. It probably requires a visual encoder trained separately, with the Microcosmic brain acting as a compact decision/memory/prediction core rather than as a raw pixel policy.

## Architecture Requirements

To keep this path open:

- Keep brain serialization explicit and versioned.
- Keep observation and action schemas named and stable.
- Add a modular brain architecture before serious transfer claims.
- Preserve prediction heads; predictive machinery is likely one of the most transferable pieces.
- Save both live learned weights and innate template weights.
- Save enough ecological context to understand why a brain was interesting.
- Build adapter-training scripts for simple external environments.
- Run ablations, especially random-core and shuffled-core controls.

## Near-Term Implementation Steps

1. Add schema names to every checkpoint: observation features, action names, and brain segment labels.
2. Add a `brain_package` exporter that can convert a checkpoint into arrays plus metadata.
3. Split `TinyBrain` successor into `encoder`, `core`, and `heads`.
4. Add a tiny external transfer test environment, starting with vector catch.
5. Add a transfer evaluation script that compares saved cores to random controls.
6. Add held-out Microcosmic probe worlds before claiming any cross-domain generality.

The dream version is not magic weights that instantly play anything. It is evolved machinery that brings useful priors: memory, causal prediction, exploration under scarcity, tool-like action sequencing, and adaptation under unfamiliar physics.
