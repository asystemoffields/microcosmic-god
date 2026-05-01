# Physics Kernel Runway

Microcosmic God should not become a full rigid-body or fluid simulator. It should gain cheap physical fields whose local rules create learnable, exploitable consequences.

The target is:

```text
simple local laws + coupled fields + material properties -> rich ecological pressure
```

## Graph Field Physics

The world remains a sparse ecological graph. Physics lives on places and edges.

Place fields:

- temperature
- pressure/depth
- humidity
- salinity
- fluid level
- light/radiant exposure
- chemical concentration
- elevation
- thermal mass
- wind/current exposure

Edge fields:

- distance/cost
- slope
- permeability
- current direction and strength
- heat conductance
- fluid conductance
- danger/exposure
- required traversal capability

## Local Law Pattern

Each tick runs cheap field updates:

```text
heat diffuses along conductive edges
fluid moves downhill or with current-biased edges
salinity and chemicals advect with fluid
humidity rises near water and falls in heat
radiance warms exposed places
organisms and reactions produce waste heat
pressure increases with depth/fluid column
materials absorb and release heat based on thermal mass
```

No global solver is required at first. Approximate local updates are enough if they are consistent and stable.

## Emergent Interaction Targets

Physics should move and transform things agents care about:

- organisms
- offspring/larvae/spores
- nutrients
- toxins
- signals
- durable marks
- heat
- water
- tools and debris
- dead bodies and stored energy

This allows currents, gravity, and heat to shape ecology before any agent "understands" them.

## Gravity And Elevation

Elevation should create:

- climbing costs
- falling or washout risks
- downhill water flow
- basins, cliffs, ledges, caves, and sheltered pockets
- mechanical-energy opportunities
- route asymmetry

Tools and body traits can interact with this through `traverse`, `anchor`, `grip`, `lever`, `float`, and `contain`.

## Ocean And Fluid Currents

Aquatic places should have flow. Currents can:

- transport organisms and resources
- make movement cheaper with the flow and harder against it
- disperse signals and erase marks
- move toxins or nutrients
- strand terrestrial organisms
- create pressure, salinity, and depth gradients
- make anchoring, floating, filtering, and channeling useful

This gives ocean life real physical pressures without simulating every unit of water.

## Thermodynamics

Thermal fields should matter because they interact with life, materials, and tools:

- heat stress and cold stress
- evaporation and desiccation
- insulation and heat storage
- chemical reaction rates
- phase-like thresholds for water, ice, vapor, or future materials
- radiant concentration
- thermal gradients as exploitable energy
- conductive and insulating artifacts

The same heat rule should make sunlight, vents, desert basins, ocean depth, fire-like reactions, and heat tools intelligible.

## Material Coupling

Materials should expose properties that physics can use:

- hardness
- brittleness
- sharpness
- flexibility
- porosity
- buoyancy
- density/heaviness
- conductivity
- thermal capacity
- combustibility/reactivity
- solubility
- toxicity
- absorbency
- containment quality

Artifacts should gain capabilities from these properties rather than from named recipes.

## Agent-Relevant Consequences

The physics kernel should create opportunities for:

- sheltering from heat, water, salt, or predators
- riding currents or resisting them
- filtering nutrients from flow
- storing heat or carrying water
- using gravity for transport or mechanical work
- opening locked resources through pressure, heat, chemistry, or force
- making habitats accessible through tools or evolved body traits

None of these should be directly rewarded. They matter only if they change survival, reproduction, prediction, or energy capture.

## Scaling Rules

The physics kernel must stay data-oriented:

- fixed-size numeric field arrays per place
- fixed-size numeric field arrays per edge
- local neighbor updates
- bounded iterations per tick
- no per-agent path search unless requested by action
- aggregate logs instead of per-field event spam
- optional lower-frequency physics ticks for slow fields

This keeps the layer compatible with CPU runs now and vectorized/GPU backends later.

## Near-Term Implementation Steps

1. Add place fields for temperature, elevation, fluid level, pressure/depth, current exposure, humidity, and salinity.
2. Add edge fields for slope, current, permeability, and conductance.
3. Add a deterministic physics update before organism actions.
4. Let movement, marks, signals, and loose resources be affected by currents and slope.
5. Let artifacts interact with field gradients through `contain`, `traverse`, `insulate`, `conduct`, `concentrate_heat`, and future `anchor`/`float` capabilities.
6. Log field-driven ecological stories compactly: washouts, heat bottlenecks, current-assisted spread, depth specialization, and barrier crossings.

The goal is not photorealistic physics. The goal is a consistent universe where causal structure is rich enough for evolution to exploit.
