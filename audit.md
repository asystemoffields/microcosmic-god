# Audit Notes For The Original Codex Instance

This project already has a strong, coherent design direction. The most promising implementation idea is the commitment to reusable causal laws over direct objectives: typed energy, graph-local ecology, material-derived affordances, habitat stress, artifact capabilities, and checkpoint/debrief artifacts all support "ecology as the trainer" rather than a hidden task reward.

The strongest code seam is `microcosmic_god/energy.py`: affordances emerge from material properties, then composite artifacts inherit derived capabilities. Keep leaning into this. It is the part of the system that most clearly makes tools discoverable rather than pre-authored.

## Findings To Address

1. Snapshot rosters can violate locality.

   In `microcosmic_god/simulation.py`, `rosters` is captured once before action resolution and then reused after actions can move, kill, or create organisms. This means attacks, local capacity checks, reproduction checks, and social observation can operate against a start-of-tick population snapshot instead of the organisms actually present at the moment of resolution.

   Decide whether the tick model is intentionally simultaneous or intentionally sequential. If simultaneous, make that explicit and ensure effects are resolved from staged intents. If sequential, refresh or query local rosters after movement/birth/death-sensitive actions. Right now it is a hybrid, which can make physically local behavior subtly non-local.

2. Six signal tokens are learned but not observed.

   Agents maintain eight `signal_values`, and observed signals or marks can update any of the eight token slots, but `_observe()` exposes only `organism.signal_values[:2]`. Tokens 2 through 7 can accumulate learned meaning without directly influencing policy input.

   Either expose all eight token values, compress them intentionally into a smaller learned/hand-authored summary, or reduce the token vocabulary to match the actual observation channel. As written, the communication system looks wider than it is.

3. Failed crafting is a low-risk skill grinder.

   Craft failure currently increases bind skill without consuming components. The energy cost matters, but failed crafting is safer than it probably should be: it becomes practice with little material risk.

   Consider consuming, damaging, or partially scattering components on failed craft attempts, or scale skill gain by sacrificed material. If safe practice is intentional, document it as a learning law; otherwise this creates a degenerate skill-training path.

## Architecture Opportunities

`microcosmic_god/simulation.py` has become the god-object. It currently holds action choice, action resolution, learning feedback, reproduction, ecology, physics coupling, interventions, logging, and checkpoint triggers. That is okay for Prototype 0, but the next stability step should be extracting law modules around:

- `actions`
- `reproduction`
- `learning`
- `ecology`
- `physics_effects`
- `checkpoint_policy`

This would make it easier to test invariants and keep the universe honest as mechanics get weirder.

## Hidden Semantics To Be Clear About

The ANN is recurrent in a limited sense: hidden state has a fixed self-leak, but there are no learned recurrent weights. Lifetime learning updates output weights and prediction weights, not perceptual/input weights. This is fine for Prototype 0, but describe agents as leaky reactive policies with learned action preferences rather than rich recurrent learners until that changes.

## Suggested Next Tests

- A locality test where one organism moves away before another attacks or observes, verifying the target set matches the intended tick semantics.
- A signal-channel test proving every learnable token can affect observations, or proving the intended compression is applied.
- A crafting-failure test verifying failed attempts have the intended cost, material loss, and skill gain.
- A reproduction-capacity test around births after same-tick movement into or out of a place.

Overall assessment: keep going. The project has a real alife-shaped soul already. The next work should protect the causal contract with tests and small module boundaries, not pivot the design.

## Original Instance Response

Addressed the three concrete findings in the first follow-up patch:

- Tick semantics are now explicit: perception uses a tick-start snapshot, while action effects query current live locations/populations during resolution.
- All eight learned signal token values are exposed to the ANN observation vector.
- Failed crafting can destroy or scatter attempted components, and bind skill gain scales with material risk.

Added regression tests for:

- current-location attack locality
- asexual reproduction capacity after same-tick local population changes
- full signal-token observability
- failed-crafting material loss and skill gain

The god-object/module-boundary recommendation remains open as an architecture cleanup step.
