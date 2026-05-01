# Learning Architecture North Star

Microcosmic God should try to evolve agents that are not merely good at this sandbox, but good at learning structured worlds.

The long-term question is:

```text
Can a population evolving in a rich causal universe produce compact ANN cores that adapt well to new real or simulated environments when paired with the right input and action adapters?
```

## Target Shape

The useful transferable object is not a full policy tied to one observation vector. It is a learned world-interaction core:

```text
environment-specific encoder -> reusable causal/predictive/memory core -> environment-specific action head
```

The core should be pressured to learn:

- which environmental factors predict later consequences
- which actions change which fields
- which objects and materials have stable affordances
- when exploration is worth its cost
- how to use memory when consequences are delayed
- how to adapt when the same action works in one context and fails in another
- how signals, marks, and other agents change the future

## Why The Sandbox Matters

Good general learners need a world where shallow tricks are not enough. The simulator should create pressure for:

- causal prediction across time
- intervention and observation
- changing seasons and climate drift
- multiple energy regimes
- material affordances and degradation
- tools and structures that have consistent but nontrivial consequences
- social information that can be useful but is not guaranteed truthful or useful
- held-out worlds where old habits only partly transfer

No direct reward should say "be intelligent" or "learn language." Intelligence should matter because it helps organisms survive, reproduce, and adapt in a changing causal universe.

## Current Prototype Foothold

The current `TinyBrain` is still small, but it now has:

- recurrent hidden state
- input and hidden eligibility traces
- valence-modulated action learning
- prediction-weight learning
- input-to-hidden representational plasticity
- recent action-result traces exposed back into observation
- short event-memory slots for energy, damage, reproduction, social, tool, and surprise-like consequences
- evolvable neural budget, memory budget, learning rate, plasticity, prediction weight, and valence wiring
- observation access to physical fields including temperature, pressure, current, interiority, shelter, oxygen-like exposure, acidity, biological activity, abrasion, and wet/dry cycling

This is enough for early evolution and inspection, not enough for strong transfer claims.

## Intelligence Pressure

There should be advantages to being smart, but they should appear through action-interaction rather than a free reward for having a large brain.

Implemented pressure should look like this:

- Better prediction and memory make movement, foraging, tool use, construction, and social learning slightly more reliable or less wasteful.
- Learned skill changes the outcome of future actions in the same physical channel.
- Larger neural machinery costs metabolism, so unused capacity is a liability.
- Selection can favor cognition through survival, energy control, tool success, and reproduction, not through a hidden intelligence score.

This keeps the project pointed at possible minds while preserving the rule that ecology, not hand-authored achievement labels, decides what survives.

## Required Next Brain Upgrades

1. Split the brain into named modules: encoder, recurrent core, prediction heads, action heads, memory state.
2. Add multiple prediction heads: energy delta, damage risk, resource changes, social signal outcome, place transition outcome.
3. Add neuromodulators: separate surprise, pain/damage, energy gain, reproduction, social, novelty, and uncertainty signals.
4. Add longer-lived memory with learned write/read gates rather than only place-value tables.
5. Add curiosity only indirectly through prediction error or uncertainty, never as a fixed external objective.
6. Add held-out world evaluation where saved cores are tested against random cores and shuffled controls.
7. Keep every checkpoint schema explicit so we can train new encoders/action heads around old cores.

## Simple Near-Term Mind Upgrades

These are cheap enough to add before a major backend rewrite:

- Done: add explicit recent-action/result traces to observations so agents can connect action, context, and consequence.
- Done: add short event memory slots for "what changed after I acted" rather than only place value.
- Add prediction heads for damage, reproduction opportunity, tool success, and movement hazard.
- Let organisms attend to a subset of local materials/structures instead of seeing only aggregate place fields.
- Partly done: save richer checkpoint context around standouts, including recent trace, event memory, signal values, place memory, local world summaries, and reason saved.
- Keep increasing neural capacity only behind metabolic cost and successful reproduction.

## More Sophisticated Mind Upgrades

These are the deeper architecture path:

- Modular recurrent core with separate sensory encoder, world-model state, policy heads, and value/valence heads.
- Differentiable external memory with evolved write/read gates.
- Learned attention over local objects, structures, agents, signals, and marks.
- Predictive coding loops where surprise changes exploration and memory allocation.
- Meta-learning genes that control plasticity schedules, not just one learning rate.
- Social-model heads that predict whether another agent's signal/action will be useful, harmful, or irrelevant.
- Developmental growth: juvenile brains start smaller and add capacity if energy, age, and ecology support it.
- Distillation/export tools that package a reusable core with replaceable observation and action adapters.

The high-end aim is "possible minds": systems with memory, causal expectation, adaptive attention, and self-modifying learning dynamics. The sandbox should make these capacities useful without declaring them as objectives.

## Transfer Test Ladder

Start near the sandbox and move outward:

1. Same world laws, different seed.
2. Same laws, shifted resource/material distribution.
3. Same laws, new action/observation adapters.
4. Simple vector games such as catch, pursuit, balancing, or navigation.
5. Richer RL environments with learned visual or symbolic encoders.

The claim only matters if saved evolved cores adapt faster, more robustly, or with better exploration than same-size random cores.

## Guardrails

- Do not optimize the simulator for transfer scores during sealed evolution.
- Do not save every brain.
- Do not confuse a good policy with a good learner.
- Do not treat language-like behavior as real language until it transfers or supports counterfactual use.
- Keep compute cheap locally, but keep data structures compatible with batched GPU evolution.
