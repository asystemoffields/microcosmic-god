# Microcosmic God

## Working Vision

Build a low-compute artificial life sandbox where embodied agents learn, remember, communicate, reproduce or persist, discover tools, and interact with an open world whose valuable resources often require tool use or multi-step capabilities.

The first milestone should run competently on CPU. The architecture should also admit larger training/evolution runs on GPU clusters without changing the conceptual model.

## Core Design Pillars

1. Agents have capturable neural controllers.
2. The world contains resources, obstacles, tools, and tool chains.
3. Agents learn from experience, not only from fixed evolutionary selection.
4. Agents can communicate, remember, and form useful relational knowledge.
5. Compute cost stays brutally low at first, with clear paths to scale.

## Current Preferences

- Primary flavor: biology/evolution first.
- Lifetimes: both individual learning and reproductive/generational selection matter.
- Communication: undecided; should probably begin minimal and become meaningful through use.
- Visualization: not important in the early stage.
- Tools: abstract tools are acceptable, but a consistent physics-like universe may encourage creative transfer.
- Social behaviors: family, trade, bonding, and culture should be possible emergent outcomes, not guaranteed scripts.
- Agent brains: saveable when interesting, but not retained forever by default. Dead agents' networks should normally disappear unless explicitly checkpointed, inherited, archived, or sampled.
- Guiding directive: avoid preprogramming goals, social roles, tool recipes, family behavior, trade, or culture wherever possible. Build a consistent world and let selection discover strategies.
- Reproduction: support both asexual and sexual reproduction. One agent can create a similar primitive offspring; two agents are required for more complex offspring.
- Inheritance: Darwinian by default. Offspring inherit genome/development parameters, not lifetime-learned weights. Teaching offspring can still emerge behaviorally.
- Ecology: mixed predatory and non-predatory life, closer to an ecosystem than a pure game.
- Learning: agents should be capable of both trial-and-error learning and imitation/observational learning.
- Reward: pain, pleasure, and reinforcement signals should themselves be evolution-friendly rather than fixed task rewards.
- Energy: energy must exist in multiple forms, not as a single food score. The same source should sometimes be usable directly by simple life and sometimes harnessable through increasingly complex tools and knowledge.
- Physical universe: define a consistent reality with learnable, predictable causal regularities. It should be rich enough to create many ecological niches while staying cheap to simulate.
- Organism substrate: not all organisms need neural networks. Plants, fungi, microbial analogs, and environmental life can use simpler rules.
- Non-neural evolution: plants, fungi, microbial analogs, and other background organisms should evolve too, not merely serve as fixed scenery.
- Mutation/development: mutation should eventually affect every trait with a real-world analog: brain, body, sensors, metabolism, reproduction, learning rules, communication, morphology, and tool-use capacities.
- Body evolution: open-ended in the long run.
- Prediction: agents should be able to evolve and learn science-like predictive models of the world. Better prediction should improve planning, tool use, and survival without granting direct "science rewards."
- Tool skill: advanced tools should require learned skill, not only possession. Agents may have the tool but fail, waste energy, break it, hurt themselves, or use it poorly without practice or teaching.
- Environmental variation: world rules should be stable, but environments should vary slowly through seasons, climate shifts, resource depletion, disasters, succession, and ecological feedback.
- Curiosity: curiosity/novelty should not be rewarded directly unless it provides an actual advantage through survival, reproduction, prediction, resource discovery, or cultural transfer.
- Knowledge transmission: include temporary signals analogous to gesture/vocalization and more durable but decaying external marks analogous to writing. Neither channel has fixed semantics.
- Language runway: do not program language directly, but preserve a path from meaningless tokens to associative communication, durable marks, proto-symbol systems, and later ANN transfer experiments.
- Intelligence cost: neural capacity, memory, prediction, and learning should carry real metabolic and developmental costs from the beginning.
- Extinction: full run extinction is allowed. The simulator should preserve a debrief explaining likely causes and final ecological state.
- Intervention hatch: researchers may intervene by changing conditions, adding entities, or triggering events, but interventions should be recorded separately from sealed evolutionary runs.
- Sexual reproduction: should require behavioral coordination and some mechanism of genuine mate/fitness selection, not automatic proximity cloning.
- Initial energy diversity: multiple energy gradients should exist from day one rather than sunlight alone.
- Genome readability: genomes should remain relatively debuggable, especially while early worlds may collapse frequently.
- Species: do not assign fixed species labels. Infer species after the fact from lineage, genetic distance, reproductive compatibility, morphology, behavior, and ecological niche.
- Engineering target: aim for both scientific observability and high performance. Logging, debriefing, and replay should be built in without making the simulator sluggish.
- Implementation preference: use a very fast but workable stack; exact language/framework is open.

## Early Architecture Hypothesis

- World: start with the cheapest abstraction that still supports locality, resource discovery, barriers, tool affordances, and causality. This may be a graph/world-state model rather than a 2D grid.
- Body: agents have position, energy, inventory, sensors, and action budget.
- Brain: small neural policy plus small memory module; saveable as structured weights and metadata.
- Learning: combine lifetime reinforcement learning with evolutionary selection across generations.
- Tools: define tools as composable affordances, not hard-coded objects only.
- Communication: start with limited symbolic tokens or learned discrete signals.
- Memory: begin with compact episodic traces and relationship scores.
- Scaling: keep the simulator vectorizable and headless from day one.

## Energy Model Direction

Avoid treating energy as one generic reward. Represent energy as typed gradients and conversions:

- radiant energy
- chemical energy
- mechanical energy
- thermal energy
- electrical energy
- stored biological energy
- nuclear-like high-density energy, likely unavailable until very advanced tool chains

Simple organisms may directly exploit only a few sources, while advanced agents can discover conversion chains through tools, structures, and learned causal models.

Example: sunlight can feed photosynthetic organisms directly, dry materials thermally, guide seasonal behavior, power primitive solar collectors, and eventually inspire or support advanced high-energy technologies.

## Anti-Shortcut Rules

- Do not directly reward tool use, communication, cooperation, curiosity, intelligence, teaching, family behavior, trade, or culture.
- Do make those behaviors possible when they produce real advantages under world physics and ecology.
- Keep possession separate from competence: an agent may hold a tool but fail to use it well without learned control, timing, sequencing, or contextual knowledge.
- Keep knowledge grounded in prediction: internal models are useful only when they help agents anticipate consequences and act better.
- Prefer universal mechanics over special-case labels. Predators, prey, families, teachers, traders, and cultures should be interpretations of behavior, not preassigned roles.

## Open Questions

- How should the first graph/world-state model represent physical distance, barriers, and local causality without becoming a grid?
- Should communication start as predefined symbols, learned signals, or both?
- What is the smallest viable evolvable genome that can express metabolism, body modules, neural capacity, learning rules, and reproduction strategy?
- How should skill be represented for tool use: neural control competence, learned action sequences, predictive models, or all three?
- How should we measure interestingness for selective brain checkpointing without turning it into a hidden objective?
- What kinds of environmental variation should appear first: seasons, resource depletion, climate drift, disasters, migration, or ecological succession?
- Should the first mobile agents be seeded as primitive neural organisms, or should neural mobility itself have to evolve from simpler life?
- How inspectable should agent brains and memories be during a run?
- Which implementation stack should Prototype 0 use: Rust core with Python analysis, Python/NumPy first, Julia, C++ core, or another hybrid?
