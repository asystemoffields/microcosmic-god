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

## Tool And Environment Mastery

Tools should be capability transformers. A composite artifact can matter because it changes what physical regimes are reachable:

- `cut`: opens fibrous/thorn/soft biological barriers.
- `crack`: opens brittle shells, stones, and mineral seams up to a resistance tier.
- `lever`: moves heavy barriers or exposes mechanically locked resources.
- `contain`: carries fluid, buffers wet habitats, or enables chemical concentration.
- `traverse`: reduces movement penalties through water, height, unstable ground, or future terrain.
- `insulate`: buffers heat, cold, conductivity, or chemical exposure.
- `conduct`: routes electrical/thermal/electrochemical gradients.
- `concentrate_heat`: converts radiant or thermal gradients into higher local intensity.

Hardness and resistance tiers matter. A weak cutting artifact can work on fiber and fail or break against diamond-like material. The same rule should handle wood, shell, stone, crystal, deep vents, ocean edges, and future exotic environments.

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
