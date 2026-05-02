# Language Runway

Microcosmic God should not start by programming language. It should start by making language-like behavior possible.

The current communication substrate has two channels:

- Temporary `signal` emissions, analogous to gesture or vocalization.
- Durable but decaying `mark` inscriptions, analogous to primitive writing or environmental signs.

Neither channel has fixed semantics. A token only matters if organisms learn or evolve useful associations between that token and later consequences.

Marks are plain tokens unless an agent intentionally inscribes a lesson trace. Intentional traces require useful recent tool/problem experience plus enough body/material capacity, attention, memory, and `inscribe` skill to encode anything beyond the token. This is not a built-in word. It is a costly, lossy residue that can be attended to, ignored, misread, copied, or made useful by later agents with enough `interpret_mark` skill.

## Current Literacy Contract

The simulator gives writing a causal channel without promising that language exists:

- Writers can create higher-quality lesson traces when inscription clarity, lesson coherence, and underlying tool/problem value line up.
- Readers gain more from marks when their sensors, memory, attention, and `interpret_mark` skill let them extract the trace with high fidelity.
- Marks remember cheap use history through `reads`, `last_read_tick`, and `value_transmitted`, so reused inscriptions become visible without saving every dull mark event forever.
- Authors are not globally rewarded for writing. A living author only gets local feedback when a reader successfully uses the author's mark in the same place, making "teaching by writing" something that must be discovered and situated.
- The reader's benefit flows through ordinary capabilities such as relevant tool skill, signal association, and future action outcomes. A clearer mark matters because it helps an action work, not because the simulator says literacy is intrinsically good.

This keeps the door open for literacy-like behavior while preserving the core rule: symbols matter only when organisms make them useful in the world.

## Possible Developmental Stages

1. Bare signaling:
   - agents emit tokens
   - tokens are locally observed
   - most signals are useless or costly

2. Associative signaling:
   - agents learn that some tokens predict danger, food, tools, mates, or movement
   - no grammar exists
   - meaning is local and lineage-specific

3. Socially useful signaling:
   - tokens alter behavior in ways that improve survival or reproduction
   - deception, alarm, recruitment, or mating signals may appear

4. External marks:
   - agents leave decaying place-local marks
   - marks can outlive the author
   - marks may support route memory, resource warnings, or tool-site cues
   - plain marks do not automatically carry lessons
   - intentional inscription can encode fuzzy problem/solution traces when that behavior is discovered
   - observation can extract fuzzy tool traces when the reader has enough attention, memory, and interpretation skill

5. Proto-symbol systems:
   - combinations of signals and marks become useful
   - tokens may become tied to action sequences or tool affordances
   - imitation and observation make communication more valuable
   - useful marks can theoretically be copied across places, allowing tool knowledge to move across space as well as time

6. Transfer experiments:
   - checkpoint a neural agent with interesting communication behavior
   - attach it to a new environment adapter
   - test whether its learned associations, memory dynamics, or predictive machinery can be trained toward human language tasks

## ANN Extraction Implication

Brain checkpoints should preserve:

- neural weights
- innate brain template
- genome and body configuration
- signal association values
- memory summary
- tool skills
- ecological context
- observed communication/marking history when available
- recent lesson-memory traces when available

The first transfer experiments will probably fail. That is acceptable. The goal is to build the archive format and agent interface so a later successful communicator is not trapped inside its birth world.
