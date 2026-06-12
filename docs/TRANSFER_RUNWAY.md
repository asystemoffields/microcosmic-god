# Controller Transfer Runway

This document describes how the simulation should be shaped so saved agent controllers can eventually seed learning in different worlds, simple games, or other RL environments.

The near-term target is modest but important:

```text
Does a controller developed in the simulation learn a new environment faster, more robustly, or with better exploration than a random controller of the same size?
```

Directly wiring one saved policy into an unrelated video game is unlikely to work if the input and action spaces are unrelated. The plausible transfer target is the developed internal machinery: recurrent state, prediction habits, learned control priors, memory dynamics, and sensorimotor abstractions.

## Core Idea

Separate the controller into reusable parts:

```text
world-specific observation adapter
  -> reusable recurrent/predictive core
  -> world-specific action adapter
  -> optional value/valence/prediction heads
```

The adapter changes when moving from Microcosmic God to another task. The core is the part we hope evolution and lifetime learning make interesting.

## Current Prototype Status

Prototype 0 uses `TinyController`:

```text
inputs -> recurrent hidden state + eligibility traces -> action logits
                                                   -> energy prediction head
```

Checkpoint files already save:

- controller weights and live hidden state
- input and hidden eligibility traces
- innate controller template
- params
- body and individual state
- tool skill
- signal associations
- environment context
- checkpoint reason

Lifetime learning can now update output preferences, prediction weights, and input-to-hidden representations. It is still intentionally cheap, but it gives agents a path toward learning which environmental factors predict later consequences rather than only learning which action was recently rewarding.

This is enough for archival and early inspection. It is not yet enough for clean cross-environment transfer because input and output matrices are tied to the Microcosmic observation/action schema.

## Transfer Package Contract

Future saved controllers should export a `ControllerPackage` with explicit schema metadata:

```text
ControllerPackage
  package_version
  controller_architecture
  reusable_core_weights
  observation_adapter_weights
  action_adapter_weights
  prediction_heads
  hidden_state_optional
  params
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
- local fields: heat, water, salinity, light, current, slope, pressure, oxygen-like exposure, acidity, residue activity, abrasion, and wet/dry cycling
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
- pursue/drain
- court/pair

For another RL environment, the action adapter maps these internal action tendencies to task-specific controls. In a catch game, `move` maps to left/right/up/down. In Atari, a small action head maps recurrent-core output to joystick/button logits.

## Transfer Protocol

For a new environment:

1. Load a saved `ControllerPackage`.
2. Keep the reusable recurrent/predictive core.
3. Replace or reinitialize the observation adapter for the new environment.
4. Replace or reinitialize the action adapter for the new environment.
5. Train adapters first while the core is frozen.
6. Fine-tune the core slowly if adapter-only training plateaus.
7. Compare against:
   - random controller with same architecture
   - randomly initialized core plus trained adapters
   - scratch-trained baseline
   - shuffled or damaged saved core

The transfer claim only means something if saved controllers beat these controls.

## Selection Without Hidden Objectives

The simulator should not develop agents for transfer. Transfer candidates should be selected after the run by observer heuristics.

Good checkpoint signals:

- persisted across multiple environment regimes
- used multiple tool affordances successfully
- improved prediction error during its lifetime
- spawned in more than one environment context
- carried useful artifacts or moved through barriers
- communicated or marked before later adaptive behavior
- recovered from scarcity, competitive-interaction pressure, or environmental drift
- performed well in held-out Microcosmic probe worlds

These are analysis filters, not rewards.

## Probe Worlds Before Video Games

Before attempting an unrelated game, test saved controllers in held-out Microcosmic variants:

- changed resource distribution
- changed currents, heat, salinity, or terrain barriers
- different material availability
- new locked resources using known affordance laws
- altered environment and competitive-interaction pressure
- different communication decay rates

If a controller cannot adapt to nearby worlds, it is unlikely to help in a video game.

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

Atari-like transfer is a later target. It probably requires a visual encoder trained separately, with the Microcosmic controller acting as a compact decision/memory/prediction core rather than as a raw pixel policy.

## Architecture Requirements

To keep this path open:

- Keep controller serialization explicit and versioned.
- Keep observation and action schemas named and stable.
- Add a modular controller architecture before serious transfer claims.
- Preserve prediction heads; predictive machinery is likely one of the most transferable pieces.
- Save both live learned weights and innate template weights.
- Save enough environment context to understand why a controller was interesting.
- Build adapter-training scripts for simple external environments.
- Run ablations, especially random-core and shuffled-core controls.

## Near-Term Implementation Steps

1. Add schema names to every checkpoint: observation features, action names, and controller segment labels.
2. Add a `controller_package` exporter that can convert a checkpoint into arrays plus metadata.
3. Split `TinyController` successor into `encoder`, `core`, and `heads`.
4. Add a tiny external transfer test environment, starting with vector catch.
5. Add a transfer evaluation script that compares saved cores to random controls.
6. Add held-out Microcosmic probe worlds before claiming any cross-domain generality.

The dream version is not magic weights that instantly play anything. It is developed machinery that brings useful priors: memory, causal prediction, exploration under scarcity, tool-like action sequencing, and adaptation under unfamiliar physics.
