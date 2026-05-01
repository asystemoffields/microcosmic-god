# Generality And Scaling Contract

This project should grow by adding general world laws, not by adding hand-authored achievements for agents to discover.

## Mechanic Admission Test

Every new mechanic should pass these checks before it enters the simulator:

- It is expressed as reusable properties, fields, costs, conversions, or constraints.
- It creates consequences, not objectives.
- It can apply to many places, organisms, artifacts, and future worlds.
- It does not directly reward intelligence, tool use, cooperation, communication, curiosity, teaching, family, trade, language, or culture.
- It is observable enough for debriefs, but observer heuristics do not feed back into fitness.
- Its expensive calculations can be batched, cached, approximated, or moved behind a faster backend later.

## World Law Pattern

Prefer laws shaped like this:

```text
outcome = actor capacity + artifact capability + local field + material resistance + learned skill + stochastic margin
```

Avoid laws shaped like this:

```text
if agent has axe and target is tree:
  give wood
```

The first form can support axes, chisels, shells, pumps, boats, heat lenses, conductors, underwater tools, future environments, and unknown combinations. The second form only supports the scene we imagined in advance.

## Open-Ended Interaction

Casual structures and relationships should arise from shared causal channels rather than special social or puzzle labels.

Examples:

- A current can transport organisms, nutrients, toxins, offspring, signals, marks, heat, tools, and debris.
- A durable mark can be memory, warning, territory, lure, courtship trace, path marker, or useless scratch depending on later consequences.
- A tool can be useful as weapon, lever, filter, float, anchor, insulator, heat store, conductor, container, bridge, or component because its materials allow those interactions.
- An inside/outside relation can be shelter, trap, nest, kiln, pipe, dam, filter, storage pocket, or useless enclosure depending on exchange with surrounding fields.
- A repeated association can become navigation, habit, social preference, avoidance, proto-teaching, or proto-trade without a dedicated "relationship type" being assigned first.

When possible, add fields and material laws that multiple behaviors can use. Avoid creating one-purpose relationship slots unless the physics and memory layers cannot express the relationship at all.

## Tool And Environment Mastery

Tools should be capability transformers. A composite artifact can matter because it changes what physical regimes are reachable:

- `cut`: opens fibrous/thorn/soft biological barriers.
- `crack`: opens brittle shells, stones, and mineral seams up to a resistance tier.
- `lever`: moves heavy barriers or exposes mechanically locked resources.
- `contain`: carries fluid, buffers wet habitats, or enables chemical concentration.
- `filter`: harvests suspended resources from flows, fluids, or porous media.
- `float`: changes how currents, water barriers, pressure, and movement hazards apply.
- `anchor`: resists current, slope, washout, and some traversal hazards.
- `traverse`: reduces movement penalties through water, height, unstable ground, or future terrain.
- `insulate`: buffers heat, cold, conductivity, or chemical exposure.
- `conduct`: routes electrical/thermal/electrochemical gradients.
- `concentrate_heat`: converts radiant or thermal gradients into higher local intensity.

Hardness and resistance tiers matter. A weak cutting artifact can work on fiber and fail or break against diamond-like material. The same rule should handle wood, shell, stone, crystal, deep vents, ocean edges, and future exotic environments.

## Persistent Structures

Place-level structures are artifacts left in the world. They are built from material properties, have durability and scale, and affect local fields without being assigned a special purpose.

- `enclose`: creates an inside/outside boundary that limits exchange.
- `permeable`: lets selected flows through the boundary.
- `shelter`: buffers stress only when enclosure combines with support, insulation, anchoring, and scale.
- `channel`: redirects flows and can make gradients more exploitable.
- `support`: changes footing, traversal, and resistance to collapse or washout.
- `gradient_harvest`: converts flow/slope/current gradients into mechanical or electrical energy when materials support it.
- `reaction_surface`: makes chemical/thermal reactions easier by exposing area, porosity, or conductivity.

This is the groundwork for dams, bridges, dens, rafts, traps, kilns, hydro-like structures, aquaculture filters, or future machinery without hardcoding any of those as recipes.

Structures also decay through general wear channels. The same material-environment law should explain corrosion near salty water, biological rot in warm wet habitats, abrasion in high-current places, thermal cracking, soluble materials leaching away, and durable materials lasting longer.

## Neural Growth And Plasticity

Agents need enough learning capacity to connect environmental causes with later consequences. Keep this governed by evolvable costs:

- Neural budget and memory budget can grow through mutation and recombination.
- Larger brains cost more metabolism, so size only persists if ecology pays for it.
- Plasticity, learning rate, prediction weight, and valence wiring are inherited traits.
- Lifetime learning updates action preferences, prediction weights, and internal representations through eligibility traces.
- Observation features should expose reusable causal fields, not hidden score labels.

The goal is not to preinstall intelligence. The goal is to make enough adaptive machinery available that evolution can discover richer cognition when the world rewards it.

Smartness should never be paid directly as a global scalar. It should matter because action-interaction is easier for agents that have built useful internal structure:

- remembered places improve later route choices
- lower prediction error reduces wasted/risky movement and manipulation
- learned tool skill changes future tool outcomes
- event memory gives the next action access to recent causal consequences
- social observation only helps if a demonstrated action had useful downstream consequences

This is still an explicit selection pressure for cognition, but the pressure is mediated by physical and social interactions rather than by a hidden "intelligence reward."

## Evolution Operators

The current code exposes reproduction as `asexual_reproduce` and `mate`, but those should be understood as provisional evolutionary operators rather than sacred biology.

- Clone/mutate reproduction preserves useful local solutions and explores nearby variants.
- Recombine reproduction mixes genomes and brain templates, and currently requires behavioral coordination so coordination can itself matter.
- Future intelligence-farming modes may add non-biological operators such as tournament selection, novelty archives, cross-world trials, curriculum transfer, or task-adoption trials.
- Any non-biological operator must be separated from sealed ecological runs or logged as garden/search machinery.

The aim is not to mimic animal drives unless those drives are useful. The aim is to create robust selection toward agents that learn, adapt, communicate, manipulate environments, and transfer.

## Scaling Strategy

The conceptual model should remain stable while the implementation backend changes.

Local CPU runs:

- Small sparse worlds.
- Hundreds to a few thousand organisms.
- Small neural controllers.
- Compact JSONL summaries.
- Minutes per experiment.

Workstation or cloud CPU runs:

- Many sealed seeds in parallel.
- Larger sparse worlds.
- Heavier lineage and checkpoint sampling.
- Longer ecological timescales.

GPU runs on A100/H100-class hardware:

- Batch neural inference across organisms and worlds.
- Run many independent worlds at once for evolutionary diversity.
- Keep world-law resolution data-oriented so hot loops can move to NumPy, JAX, Rust, CUDA, or another accelerator-backed core.
- Use checkpoint policies to save rare brains and stories without retaining every dead agent.
- Treat the GPU as epoch throughput, not as permission to make each organism individually bloated.

The expected path for a five-hour A100-class run is not one giant hand-built world. It is a large set of sealed worlds, batched brain evaluation, compact observer heuristics, and selective archival of rare lineages, tools, habitats, and communication patterns.

## Non-Negotiables

- Sealed runs stay sealed.
- Garden interventions are logged and separated from natural dynamics.
- Species are inferred after runs.
- Dead brains disappear unless checkpoint policy saved them.
- Analysis tools can identify interesting organisms, but the simulator must not optimize for the analysis score.
- If a behavior looks impressive, first ask whether it was caused by a general rule or by a shortcut we accidentally installed.
