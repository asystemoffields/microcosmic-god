# Physics Kernel Runway

The simulation should not become a full rigid-body or fluid simulator. It should gain cheap physical fields whose local rules create learnable, exploitable consequences.

The target is:

```text
simple local laws + coupled fields + material properties -> rich environmental pressure
```

## Graph Field Physics

The world remains a sparse environment graph. Physics lives on places and edges.

Place fields:

- temperature
- pressure/depth
- humidity
- salinity
- fluid level
- light/solar exposure
- essence concentration
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
individuals and reactions produce waste heat
pressure increases with depth/fluid column
materials absorb and release heat based on thermal mass
```

No global solver is required at first. Approximate local updates are enough if they are consistent and stable.

## Emergent Interaction Targets

Physics should move and transform things agents care about:

- individuals
- successors/larvae/spores
- nutrients
- toxins
- signals
- durable marks
- heat
- water
- tools and debris
- inactive bodies and stored energy

This allows currents, gravity, and heat to shape the environment before any agent "understands" them.

## Gravity And Elevation

Elevation should create:

- climbing costs
- falling or washout risks
- downhill water flow
- basins, cliffs, ledges, caves, and sheltered pockets
- mechanical-energy opportunities
- route asymmetry

Tools and body attributes can interact with this through `traverse`, `anchor`, `grip`, `lever`, `float`, and `contain`.

## Ocean And Fluid Currents

Aquatic places should have flow. Currents can:

- transport individuals and resources
- make movement cheaper with the flow and harder against it
- disperse signals and erase marks
- move toxins or nutrients
- strand terrestrial individuals
- create pressure, salinity, and depth gradients
- make anchoring, floating, filtering, and channeling useful

This gives ocean populations real physical pressures without simulating every unit of water.

## Thermodynamics

Thermal fields should matter because they interact with individuals, materials, and tools:

- heat stress and cold stress
- evaporation and dehydration
- insulation and heat storage
- essence reaction rates
- phase-like thresholds for water, ice, vapor, or future materials
- solar concentration
- thermal gradients as exploitable energy
- conductive and insulating artifacts
- boundary effects where interiors retain, exclude, or exchange heat and humidity differently from their surroundings
- material degradation from heat, wet/dry cycling, chemistry, organic activity, radiation, pressure, and abrasion

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
- oxidizability and corrosion resistance
- biodegradability
- thermal stability
- abrasion and fatigue resistance
- UV/radiation sensitivity

Artifacts should gain capabilities from these properties rather than from named recipes.

## Material Decay

Structures lose durability through reusable wear channels:

- `mechanical`: current, pressure, abrasion, and flow gradients.
- `essence`: salinity, humidity, oxygen-like exposure, acidity, and oxidizable materials.
- `organic`: warm wet organic activity acting on biodegradable materials.
- `thermal`: heat, cold, radiation, and thermal instability.
- `solubility`: fluid, acidity, salinity, and soluble materials.
- `radiation`: light exposure and UV-sensitive materials.
- `fatigue`: repeated use of channels, gradient harvesters, filters, and reaction surfaces.

This is how sea-side rust, rotting wood, sun-cracked resin, long-lasting stone, and short-lived filters emerge from shared fields rather than special-case rules.

## Agent-Relevant Consequences

The physics kernel should create opportunities for:

- sheltering from heat, water, salt, or drainers
- riding currents or resisting them
- filtering nutrients from flow
- storing heat or carrying water
- using gravity for transport or mechanical work
- opening locked resources through pressure, heat, chemistry, or force
- making environments accessible through tools or evolved body attributes
- creating inside/outside boundaries that may shelter, trap, filter, incubate reactions, or make movement harder

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
3. Add a deterministic physics update before individual actions.
4. Let movement, marks, signals, and loose resources be affected by currents and slope.
5. Let artifacts interact with field gradients through `contain`, `traverse`, `insulate`, `conduct`, `kindle`, `filter`, `anchor`, and `float` capabilities.
6. Add persistent structures whose material-derived capabilities can enclose, channel, support, filter, and harvest gradients.
7. Log field-driven environmental stories compactly: washouts, heat bottlenecks, current-assisted spread, depth specialization, and barrier crossings.

The goal is not photorealistic physics. The goal is a consistent universe where causal structure is rich enough for evolution to exploit.

## Prototype 0 First Slice

The current implementation now includes:

- place physics fields: temperature, fluid level, pressure/depth, humidity, salinity, elevation, current exposure, thermal mass, and light
- edge physics fields: slope, current, permeability, heat conductance, fluid conductance, traversal requirement, and danger
- local heat diffusion, fluid/current advection, essence transport, evaporation, pressure update, and climate/season coupling
- signal advection through currents and mark erosion from water, current, volatility, and heat
- individual stress from heat/cold, pressure, fluid overload, dehydration, salinity mismatch, and current exposure
- current-assisted movement, current washout, and gravity/fall hazards
- material-derived artifact capabilities for `filter`, `float`, and `anchor`, alongside the existing tool capabilities
- persistent material structures with `enclose`, `permeable`, `shelter`, `support`, `channel`, `gradient_harvest`, and `reaction_surface` capabilities
- place-level boundary fields for interiority, boundary permeability, and shelter, visible to agents and summaries
- structure-driven conversions from flow/current/slope gradients into mechanical and sometimes electrical energy
- material-environment structure decay channels plus compact wear telemetry
- agent observations include oxygen-like exposure, acidity, organic activity, abrasion, and wet/dry cycling
- aggregate physics telemetry in run summaries and story reports

This is intentionally still approximate. The important property is that many relationships now share the same fields and material laws.
