# Canonical Rename Glossary — biology-metaphor → neutral ML/optimization vocabulary

**Status:** Stage-1 mapping. Built for mechanical application by parallel applier
agents (Stage 2) and a test-running verifier (Stage 3). Do not change behavior.

This repo is an artificial-life / RL sandbox: numpy policy networks evolving in a
graph-world with physics fields. It names its mechanics with wet-biology, ecology,
and anthropomorphic/theological metaphors. The goal is to neutralize the metaphors
that cause false-positive friction for a downstream ML model **without changing any
runtime behavior or on-disk / external-API compatibility.**

---

## 0. How appliers must use this document

1. **Two replacement domains.** Every mapping below is tagged with a *disposition*:
   - `RENAME-CODE` — rename the Python identifier everywhere it appears in **tracked
     `.py` files under `microcosmic_god/` and `tests/`** (and `analysis/` if present),
     using a whole-word (`\b`) match. Exact token list is in §3.
   - `RENAME-PROSE` — replace only inside **prose**: Python comments (`# …`) and
     docstrings (`"""…"""`), and Markdown body text. Never touch code tokens or string
     literals. Mapping in §4.
   - `RETAIN` — do **not** rename anywhere (pinned by serialization, external API,
     CLI, or because it is already standard ML/optimization vocabulary). See §2 and §5.

2. **Order of precedence.** A token in the §5 DO-NOT-RENAME set always wins. If a
   word appears both as a pinned identifier/string *and* as free prose (e.g.
   `organism`), the **identifier/string stays**; only free prose is neutralized, and
   only where it does not name a pinned symbol.

3. **Skip globs (never edit):** `.venv/`, `microcosmic_god.egg-info/`, `archives/`,
   `runs/`, `crates/`, `transfer/_scratch_*/`, `transfer/results/`,
   `transfer/sample_brains/`, `transfer/_probe_smoke.json`, any `*.ipynb`,
   and **`transfer/probe_worlds.py` + `transfer/_run_v2_skeptic.py`** (untracked,
   actively used by a concurrent agent — treat as external; do not edit).

4. **Whole-word, case-aware.** Apply snake_case, CamelCase, and prose forms as listed.
   Do not substring-match (e.g. don't turn `metabolic_cost` inside `upkeep`
   accidentally; don't touch `biological_storage` when handling `biological`).

5. **Before applying any `RENAME-CODE` token,** grep the whole tracked tree for it to
   confirm the only references are in editable files; then replace consistently in the
   same commit so the verifier's tests stay green.

---

## 1. Guiding principle (central judgment call — flag for orchestrator)

Much of this codebase's vocabulary is **evolutionary-computation / reinforcement-learning /
continual-learning terminology**, which is *already* the target "neutral ML/optimization"
vocabulary. Renaming it would reduce clarity and introduce new ambiguity. Therefore the
following are **RETAINED** as-is (code and prose):

> evolution / evolutionary / evolve, `EvolutionEngine`, mutation / mutate,
> recombination / recombine (crossover), selection, fitness, population, generation,
> genome / genotype, agent (the RL/GA sense), neural, plasticity / plastic, Hebbian,
> eligibility trace, valence, reward, prediction / predictive, attention, episodic
> memory, replay, consolidation, signal.

The renames in this glossary target the **wet-biology, ecology, physiology, and
anthropomorphic/theological framing** that an ML reader does *not* expect — organism,
creature, brain-as-organ, metabolism, photosynthesis, digestion, predator/prey,
ecology/ecosystem/habitat, food web, drown/desiccate, birth/death/alive, the
"Microcosmic God"/"god-object" framing, "civilization", "spontaneous coupling", etc.

If the orchestrator wants the GA/RL terms neutralized too (e.g. mutation→perturbation,
reproduction→replication everywhere), that is a larger, more invasive pass and most of
those identifiers are additionally **pinned** by serialization/external API (see §2).

---

## 2. Why so much is pinned (compatibility model)

- **All persistence is JSON** via `dataclasses.asdict` and hand-written `to_dict` /
  `to_summary` / `cognitive_snapshot` methods. There is **no** numpy `.npz` / pickle.
  Every dataclass **field name** and every dict-literal **key string** that flows into
  `json.dump` becomes an on-disk key. Existing artifacts live under
  `runs/20260610_210633_seed73_smoke/` (checkpoints, `summary.json`, `config.json`,
  `events.jsonl`, `story_events.jsonl`, `world_final.json`) and **must remain loadable**.
  → Renaming any serialized field/key would break load. **Default: leave serialized
  names unchanged** (the mission's stated default). They are listed in §5.
- **External consumer pin.** The untracked, actively-used `transfer/probe_worlds.py`
  imports and touches a specific surface (module paths, classes, methods, attributes).
  Renaming any of it breaks that script. Full list in §5.B.
- **CLI / config pin.** `argparse` flags, `RunConfig` field names, profile names, and
  run-mode strings are user/automation-facing. Listed in §5.C–D.
- **Behavioral string pin.** `ACTIONS`, affordance/capability/material names, and
  prediction-head names are used as **indices / dict keys that drive computation** and
  are serialized; changing the strings changes behavior. Listed in §5.E.

---

## 3. Rename map — Part A: CODE identifiers  (`RENAME-CODE`)

Conservative, vetted set: each token is **internal-only** (not imported/overridden by
`probe_worlds.py`, not a serialized string, not in the external API of §5.B). Replace
whole-word across `microcosmic_god/**.py` and `tests/**.py`.

| old identifier | new identifier | where defined / used | notes |
|---|---|---|---|
| `_metabolize` | `_apply_upkeep` | `simulation.py` (def + call in `step`) | method; "metabolize" is wet-bio |
| `metabolic_cost` | `upkeep_cost` | `organisms.py` (def), `simulation.py` (calls) | method on `Organism`; not in external API |
| `_seed_initial_life` | `_seed_initial_population` | `simulation.py` | "life" → GA-native "population" |
| `_kill` | `_deactivate` | `simulation.py` (def + ~10 internal calls) | **string args unchanged** (death-cause strings are serialized — see §5.E) |
| `OffspringPlan` | `ChildPlan` | `evolution.py` (def), `simulation.py` (import + use) | class symbol only; its field `child_kind` and the method `_instantiate_offspring` stay (pinned) |

**OPTIONAL** (safe but lower-value; apply only if doing a thorough code pass — still
internal-only, but touch more call sites). If applied, rename consistently incl. tests:

| old | new | notes |
|---|---|---|
| `living_total` / `living_by_kind` / `living_neural` | `active_total` / `active_by_kind` / `active_policy` | `Simulation` counters; internal only |
| `appetite` (local in `_eat`) | `intake_rate` | local var only |
| `drowning` / `desiccation` (locals in `_habitat_stress`) | `fluid_overload` / `dehydration` | local vars only; death-cause strings stay |

Do **not** extend Part A beyond this without re-checking §5.B and the tests. In
particular, leave `_habitat_stress`, `_instantiate_offspring`, `add_organism`,
`organism_from_genome`, `make_brain_for_genome`, `EvolutionEngine`, `self.evolution`,
`self.organisms`, and all `Organism`/`TinyBrain`/`Genome` attributes **unchanged**.

---

## 4. Rename map — Part B: PROSE vocabulary  (`RENAME-PROSE`)

Apply inside **comments, docstrings, and Markdown body only**. Skip any occurrence that
is a literal reference to a pinned identifier/string/flag (e.g. ``clone_mutate``,
`EvolutionEngine`, `--garden`, `initial_plants`, the `microcosmic_god` import path) —
keep those verbatim. Variants: match capitalized / plural / `-ing` / `-ed` forms.

### 4.1 Entity / life terms
| bio term (variants) | neutral replacement |
|---|---|
| organism, organisms | individual, individuals |
| creature, creatures | individual, individuals |
| life form, lifeform(s), "life" (as living things) | individual(s) / the population |
| alive, living (adjective) | active |
| death, die, dies, dying, dead | removal / deactivation / inactive |
| birth, born, give birth | creation / spawning |
| civilization | dominant population cluster |
| "the god", "Microcosmic God" (as actor/creator) | the simulation / the framework |
| god-object | monolithic central class |
| ecological collapse | population collapse |

### 4.2 Controller terms
| bio term | neutral replacement |
|---|---|
| brain, brains (as an organ) | controller / policy network |
| neural tissue | controller capacity |
| neuroplastic | adaptive |
| "the brain rummages / decides / wants …" (anthropomorphism) | neutral paraphrase ("the controller retrieves / selects …") |

### 4.3 Reproduction / spawn terms
| bio term | neutral replacement |
|---|---|
| offspring | child / successor |
| mate, mating | pairing |
| asexual reproduction | single-parent replication |
| sexual reproduction | two-parent replication |
| reproduction, reproductive (free prose only) | replication *(soft / optional — the concept; most `reproduction*` identifiers are pinned, see §5)* |

### 4.4 Metabolism / physiology terms
| bio term | neutral replacement |
|---|---|
| metabolism, metabolic | upkeep / maintenance cost |
| photosynthesis | radiant-energy capture |
| digestion | chemical-energy conversion |
| appetite | intake rate |
| starvation (prose) | energy depletion *(the string `"starvation"` stays — §5.E)* |
| repair / decay (of an individual) | restore / degrade *(keep `repair_or_decay` identifier; §5)* |

### 4.5 Ecology / habitat terms
| bio term | neutral replacement |
|---|---|
| ecology, ecological | environment / system-level |
| ecosystem | environment / interacting population |
| habitat | environment / local conditions |
| niche | specialization / regime |
| predator, prey, predation | attacker / target / antagonistic interaction |
| food web | resource–consumer network |
| drown, drowning | fluid overload |
| desiccate, desiccation | dehydration |

### 4.6 Genetics prose
| bio term | neutral replacement |
|---|---|
| gene, genes | parameter, parameters |
| trait, traits | parameter / attribute |
| Darwinian (inheritance) | standard parameter inheritance |
| Lamarckian (inheritance) | acquired-state inheritance |
| genetic distance | parameter distance |

### 4.7 Framing / mode prose
| bio term | neutral replacement |
|---|---|
| "spontaneous coupling" | offline replay association |
| garden mode / gardener (prose gloss) | intervention mode *(string `"garden"` & `--garden` stay — §5)* |
| sealed run (prose gloss) | no-intervention run *(string `"sealed"` stays — §5)* |
| writing, literacy, proto-writing | durable symbol encoding |
| knowledge transmission | information transfer |

**RETAIN in prose (do not rewrite):** sandbox, consolidation, replay, eligibility
trace, attention, plasticity, valence, episodic memory, evolution/evolutionary,
mutation, recombination, selection, fitness, population, generation, genome, agent,
neural, signal, mark, champion (see §1, §5.F).

---

## 5. DO NOT RENAME — protected set (exhaustive)

### 5.A Package & module paths
`microcosmic_god` (pip dist `microcosmic-god`, entry point `microcosmic_god.cli:main`)
and **all** submodule filenames/paths, especially those imported by external code:
`microcosmic_god.brain`, `.config`, `.genome`, `.organisms`, `.simulation`, `.world`,
`.energy`, `.backends`, `.backends.torch_gpu`. **Do not rename any `.py` file.**

### 5.B External API pinned by `transfer/probe_worlds.py` (untracked consumer)
- **Imports:** `from microcosmic_god.brain import PREDICTION_HEADS, TinyBrain`;
  `from microcosmic_god.config import RunConfig`;
  `from microcosmic_god.genome import Genome`;
  `from microcosmic_god.organisms import OBSERVATION_SIZE`;
  `from microcosmic_god.simulation import Simulation`.
- **Classes/symbols:** `TinyBrain`, `Genome`, `RunConfig`, `Simulation`,
  `PREDICTION_HEADS`, `OBSERVATION_SIZE`.
- **Methods/classmethods:** `Simulation.add_organism(kind, genome, location, energy,
  brain_template=…)`, `Simulation.step()`, `Simulation._instantiate_offspring(plan)`
  (overridden in a subclass — keep name **and** signature), `TinyBrain.from_dict`,
  `TinyBrain.random(input_size, hidden_size, output_size, with_attention,
  episodic_capacity)`, `Genome.from_dict`, `RunConfig.from_profile`.
- **Attributes:** `Simulation.world`, `Simulation.organisms`; `World.places`;
  `Organism.id`, `.alive`, `.energy`, `.age`, `.prediction_error_profile`,
  `.successful_tools`, `.success_profile`; the plan attribute `child_kind`;
  `TinyBrain.hidden_size`, `.hidden`, `.hidden_trace`, `.input_trace`, `.input_size`,
  `.output_size`, `.last_inputs`, `.last_outputs`, `.weights_in`, `.weights_out`,
  `.bias_h`, `.bias_o`, `.prediction_weights`, `.auxiliary_prediction_weights`,
  `.attention_weights`, `.attention_bias`, `.episodic_slots`, `.episodic_age`.

> The in-repo `tests/test_causal_contracts.py` additionally exercises a very wide
> private surface of `Simulation`, `Organism`, `Place`, `Edge`, `Genome`, `TinyBrain`
> (e.g. `_observe`, `_rosters`, `_habitat_stress`, `_attack`, `_clone_mutate`,
> `_craft`, `_build_structure`, `_move`, `_tool_effect`, `_advance_causal_challenge`,
> `_situation_affordance_choice`, `_movement_summary`, `_lineage_summary`,
> `_checkpoint_champions`, `_refresh_world`, `build_artifact`, `build_structure`,
> `structure_decay_channels`, `CausalChallenge`, `BrainLearningCase`,
> `ATTENTION_BUDGET_FRACTION`, `BRAIN_HIDDEN_MAX`, `ACTIONS`). Tests are editable, so
> these are not hard-pinned — but **none are in the Part-A rename set**, so leave them
> unchanged unless the orchestrator expands scope (then update tests in lockstep).

### 5.C CLI flags (cli.py) — keep verbatim
`run`, `specs`, `--profile`, `--seed`, `--ticks`, `--wall-seconds`, `--places`,
`--plants`, `--fungi`, `--agents`, `--max-population`, `--output-dir`, `--log-every`,
`--checkpoint-every`, `--checkpoint-limit`, `--harshness`, `--environment-harshness`,
`--world-refresh-every`, `--backend`, `--device`, `--garden`, `--interventions`,
`--no-stop-on-neural-extinction`, `--quiet-events`, `--dry-run`, `--json`.

### 5.D RunConfig field names & profile/mode strings (config.py) — serialized to config.json
`seed`, `profile`, `max_ticks`, `max_wall_seconds`, `places`, `initial_plants`,
`initial_fungi`, `initial_agents`, `max_population`, `season_length`, `log_every`,
`checkpoint_every`, `output_dir`, `run_mode`, `interventions_path`,
`stop_on_neural_extinction`, `stop_on_full_extinction`, `event_detail`,
`clone_complexity_soft_limit`, `asexual_complexity_ceiling`, `neural_checkpoint_limit`,
`compute_backend`, `device`, `environment_harshness`, `world_refresh_every`.
Profile strings: `"smoke"`, `"minute"`, `"long"`, `"modal"`. Run-mode strings:
`"sealed"`, `"garden"`. Backend strings: `"cpu"`, `"torch"`; device `"auto"`/`"cuda"`.

### 5.E Behavioral / serialized string literals — keep verbatim (changing them changes behavior or breaks load)
- **Entity kinds:** `"agent"`, `"plant"`, `"fungus"` (drive branching; serialized as
  `kind`; tied to `initial_plants/fungi/agents`).
- **`ACTIONS`** (`organisms.py`): `rest, move, eat, absorb_radiant, forage, pickup,
  craft, build, use_tool, attack, signal, mark, coordinate, clone_mutate, observe`
  (used by index → policy outputs; serialized as `last_action`).
- **Operators / birth modes:** `"clone_mutate"`, `"recombine"` (`OffspringPlan.operator`,
  `births_by_mode` keys).
- **AFFORDANCES / capabilities / materials** (`energy.py`): `crack, cut, bind, contain,
  concentrate_heat, conduct, lever, filter, traverse, insulate, energy_storage, float,
  anchor, carry, protect, record, channel, enclose, permeable, shelter, support,
  gradient_harvest, reaction_surface`; materials `branch, stone, fiber, shell, crystal,
  resin, bone`; communication skills `inscribe, interpret_mark`; material property keys
  (`hard, heavy, sharp, flexible, bindable, container, …, uv_sensitivity`).
- **`PREDICTION_HEADS`** (`brain.py`): `energy, damage, reproduction, social, tool,
  hazard` (dict keys into learning; serialized).
- **Genome field names** (`genome.py`, serialized via `asdict`): `radiant_metabolism,
  chemical_metabolism, thermal_tolerance, mechanical_use, electrical_use,
  storage_capacity, aquatic_affinity, salinity_tolerance, desiccation_tolerance,
  pressure_tolerance, buoyancy, photosynthesis_surface, digestion, mobility,
  manipulator, armor, sensor_range, neural_budget, memory_budget, prediction_weight,
  plasticity_rate, learning_rate, signal_strength, mate_selectivity,
  offspring_investment, asexual_threshold, sexual_threshold, developmental_complexity,
  mutation_rate, valence_energy, valence_health, valence_damage, valence_reproduction,
  valence_social, episodic_capacity`. **These contain wet-bio words
  (`photosynthesis_surface`, `digestion`, `*_metabolism`, `mate_selectivity`,
  `asexual/sexual_threshold`) but are on-disk keys → KEEP.**
- **`TinyBrain.to_dict` keys** (`brain.py`): `input_size, hidden_size, output_size,
  weights_in, weights_out, bias_h, bias_o, prediction_weights,
  auxiliary_prediction_weights, attention_weights, attention_bias, episodic_capacity,
  episodic_slots, episodic_age, hidden, last_outputs, last_inputs, last_attention,
  last_prediction_errors, input_trace, hidden_trace`.
- **`Organism.to_summary` / `cognitive_snapshot` keys** (`organisms.py`): `id, kind,
  location, age, generation, lineage_root_id, parent_lineage_ids,
  inherited_brain_template, energy, health, neural, offspring_count, successful_tools,
  tool_use_counts, success_profile, last_action, last_tool_affordance,
  last_craft_target, last_artifact_method, last_lesson, last_valence, last_energy_delta,
  artifacts, complexity, parents, lineage, root_id, parents, parent_lineages,
  recent_trace, prediction_errors, event_memory, tool_trace, lesson_memory,
  signal_values, place_memory, place_id, value`.
- **Label tuples** (`organisms.py`): `RECENT_TRACE_LABELS`, `EVENT_MEMORY_LABELS`,
  `SUCCESS_PROFILE_LABELS` (e.g. `energy_gain, prediction_fit, tool_make, tool_use,
  structure, causal_step, causal_unlock, collaboration, social_learning,
  written_learning, knowledge_transmitted, reproduction`) — serialized success keys.
- **World / Place / Edge / Signal / Mark / CausalChallenge `to_dict`/`to_summary`
  keys** (`world.py`): place fields (`resources, materials, locked_chemical, capacity,
  obstacles, habitat, physics, archetype, structures, signals, marks,
  causal_challenge`), the `ENV_ARCHETYPES` strings (`pelagic, reef, trench,
  hydrothermal_vent, tidal_marsh, high_ridge, mineral_scree, forest_edge, desert_glass,
  cavern`), physics keys (`temperature, fluid_level, pressure, humidity, salinity,
  elevation, current_exposure, oxygen, acidity, biological_activity, abrasion,
  wet_dry_cycle, interiority, boundary_permeability, shelter, resource_gradient,
  terrain_richness, light, flow_gradient`), `ENERGY_KINDS` (`radiant, chemical,
  biological_storage, thermal, mechanical, electrical, high_density`),
  `STRUCTURE_DECAY_CHANNELS`. (Most of these are physics/material, already neutral.)
- **`debrief.py` summary keys** (written to `summary.json`, read by analysis scripts):
  `reason, tick, elapsed_seconds, population, births_by_mode, deaths_by_cause,
  deaths_by_kind_cause, tool_successes, causal_steps, causal_unlocks,
  collaboration_events, movement, success_profile, lineages, marks_created,
  mark_lessons, …, reproduction_attempts, reproduction_failures, evolution_policy,
  action_counts, action_energy_delta, action_avg_energy_delta, checkpointing, observer,
  world_energy, world_physics, physics_events, climate_drift, top_living_organisms,
  likely_causes, last_aggregates, interventions_applied`.
- **Death-cause strings** (`simulation.py`, serialized in `deaths_by_cause` and
  checkpoint filenames): `starvation, exposure_stress, pressure_stress, thermal_stress,
  current_exposure, habitat_mismatch, current_washout, fall, relocation_shock,
  movement_hazard, tool_accident, counterattack`, etc. — KEEP (these are the strings;
  the `_kill` *method* may be renamed per §3, but its string arguments may not).
- **Checkpoint bucket / reason strings** (`checkpoints.py`): `first_tool,
  interval_champion, reproductive_champion, tool_champion, causal_champion,
  learner_champion, lineage_founder, notable_death, general`; filename pattern
  `brain_t{tick:08d}_o{id}_{reason}.json`.
- **Run-directory naming token** (`runlog.py`): `{stamp}_seed{seed}_{profile}` and the
  fixed filenames `events.jsonl, story_events.jsonl, config.json, summary.json,
  world_final.json, checkpoints/`.
- **Event `kind` strings** emitted to `events.jsonl` / `story_events.jsonl`
  (`seeded, world_refreshed, structure_built, tool_success, causal_unlock,
  crafted_tool, collaboration, movement_attempt`, …) — read by `arc_report.py` /
  `story_report.py`; KEEP.

### 5.F Retained-as-ML-vocabulary (per §1)
evolution / evolutionary / evolve, `EvolutionEngine`, `EvolutionDecision`, mutation /
mutate / `mut_float`, recombination / recombine, selection, fitness, champion,
population, generation, genome / `Genome`, agent, neural, plasticity / plastic,
Hebbian, eligibility trace, valence, reward, prediction, attention, episodic, replay,
consolidation, signal, mark, lineage (kept: GA-ancestry sense **and** pinned in
`lineage_root_id`).

---

## 6. Per-directory inventory & term-density (for applier partitioning)

Density = relative effort for a **prose** pass (comments/docstrings/markdown), since
code-identifier renames (Part A) are tiny and centralized in `simulation.py` /
`organisms.py` / `evolution.py`.

### `microcosmic_god/` (package, ~core code)
| file | LOC | prose density | notes |
|---|---|---|---|
| `simulation.py` | 3503 | **high** | god-object; anthropomorphic comments (replay "rummages", "spontaneous coupling"), organism/brain/habitat/predation prose; holds all Part-A code renames (`_metabolize`, `_seed_initial_life`, `_kill`, `metabolic_cost` callsites). Bulky — consider splitting across appliers by line range. |
| `brain.py` | 760 | medium | docstrings on neuroplastic attention, episodic replay/consolidation, "Lamarckian", "spontaneous coupling". Many pinned serialized keys (§5.E). |
| `organisms.py` | 413 | medium | `Organism` dataclass; defines `metabolic_cost` (→ `upkeep_cost`), label tuples (pinned). Heavy serialized-key zone. |
| `world.py` | 881 | low | physics/material vocabulary already neutral; a little "habitat"/ecology prose. Mostly RETAIN. |
| `energy.py` | 674 | low | materials/affordances; neutral. RETAIN strings. |
| `genome.py` | 293 | low–med | wet-bio **field names are pinned**; only comments are prose. "Darwinian"/budgets comments. |
| `evolution.py` | 149 | medium | `EvolutionEngine` RETAIN; rename `OffspringPlan`→`ChildPlan`; docstring talks "variation operators", "non-biological farm policies", "sealed run". |
| `checkpoints.py` | 94 | low | bucket/reason strings pinned. |
| `debrief.py` | 187 | low–med | `organism`/`lineage` prose; many pinned summary keys + `organism_success_score`/`top_organisms` (internal, but mirror retained nouns — leave identifiers). |
| `observer.py` | 148 | low | docstring "organism fitness"; payload key set pinned. |
| `runlog.py` | 52 | low | run-dir tokens pinned. |
| `interventions.py` | 34 | low | neutral. |
| `cli.py` | 175 | low | flags pinned; "artificial-life", "run card" prose. |
| `config.py` | 95 | low | multi-world comment; fields pinned. |
| `backends/` (`__init__`, `contracts`, `cpu`, `torch_gpu`) | ~306 | low | `BrainLearningCase`, `BrainRuntime` retained (brain = controller in prose only); compute code neutral. |
| `__init__.py` / `__main__.py` | ~10 | low | docstring "artificial-life sandbox". |

### `tests/`
| file | LOC | density | notes |
|---|---|---|---|
| `test_causal_contracts.py` | 1933 | medium | exercises wide private surface (see §5.B note). If Part-A renames are applied, the 4–5 tokens must be updated here too (grep first). Otherwise prose-only. |

### `analysis/`
| path | density | notes |
|---|---|---|
| `scripts/summarize_run.py` (50), `story_report.py` (157), `arc_report.py` (533) | low–med | **read** `summary.json` / `*.jsonl` keys → those key strings are pinned (§5.E); rename only prose. |
| `scripts/train_checkpoint_sae.py` (296), `inspect_sae.py` (62) | low–med | read checkpoint `brain` segment keys (`weights_in,…`) — pinned; prose only. |
| `notebooks/*.ipynb`, `sae_models/`, `*.gitkeep` | — | **skip** notebooks (skip-glob). |

### `side_projects/`
| path | density | notes |
|---|---|---|
| `README.md`, `universal_genesis/README.md` | med | prose-only; "genesis"/creation framing → neutralize per §4. No code. |

### `examples/`
`interventions.example.json` — data file; keys are intervention `tick/kind/payload/reason` (pinned). **Do not rename.**

### `transfer/`
| path | notes |
|---|---|
| `probe_worlds.py`, `_run_v2_skeptic.py` | **SKIP** (untracked external consumers). |
| `README.md` | prose-only; heavy "champion/transfer/cohort" (ML — retain) + some "brain/organism/food web" (neutralize). |
| `catch_transfer.ipynb`, `_scratch_*/`, `results/`, `sample_brains/`, `_probe_smoke.json` | **skip**. |

### Root Markdown (`*.md`) — highest prose value, zero behavior risk
| file | density | notes |
|---|---|---|
| `README.md` | **high** | organism/brain/ecology/habitat/predation/lineage framing; "garden/sealed" (gloss only). |
| `PROJECT_BRIEF.md` | **high** | "ecosystem", "Darwinian", "mate/fitness selection", "species", "predators/prey/families/teachers". |
| `ARCHITECTURE.md` | **high** | "evolution operators" (retain), genome "genes", "neural tissue", "garden/sealed run". |
| `audit.md` | medium | **"god-object"**, "ecology as the trainer". |
| `docs/LEARNING_ARCHITECTURE.md` | medium | mostly ML (retain); some "organism/brain". |
| `docs/HANDOFF.md` | medium–high | "lineage/civilization/predation/starvation/champion". |
| `docs/TRANSFER_RUNWAY.md`, `GENERALITY_AND_SCALING.md`, `LANGUAGE_RUNWAY.md`, `PHYSICS_KERNEL.md`, `EVENT_MONITORING.md`, `MACHINE_PROFILE.md` | low–med | scattered organism/brain/ecology/"culture/teaching/trade" prose; physics doc largely neutral. |

### Skip entirely
`.venv/`, `microcosmic_god.egg-info/` (incl. `SOURCES.txt`, `top_level.txt`),
`archives/brains/**`, `runs/**`, `crates/sim/**` (empty Rust scaffold `.gitkeep` dirs
named `organisms/genome/energy/world/brain/experiments/logging` — leave the directory
names; they are not Python and renaming risks nothing of value), `*.ipynb`.

---

## 7. Serialization / compatibility risks & how this glossary handles each

1. **`Genome` wet-bio field names** (`photosynthesis_surface`, `digestion`,
   `*_metabolism`, `mate_selectivity`, `asexual_threshold`, `sexual_threshold`,
   `valence_reproduction`, `offspring_investment`) → serialized via `asdict`, read back
   via `Genome.from_dict(**data)` with `fields(cls)`. **Handled:** RETAIN all field
   names (§5.E). Renaming would silently change on-disk keys and break every checkpoint.
   Only neutralize the surrounding **comments** (§4.6).
2. **`Organism.to_summary` / `cognitive_snapshot` bio keys** (`offspring_count`,
   `lineage_root_id`, `parent_lineage_ids`, `neural`, `kind`) → in checkpoints and
   `top_living_organisms`, read by analysis scripts and probe consumers. **Handled:**
   RETAIN keys (§5.E); the Python attributes backing them (`.offspring_count`,
   `.lineage_root_id`, …) are also kept because the keys are produced directly from
   attribute names / pinned by `probe_worlds`.
3. **`TinyBrain` serialization** (`weights_in`, `episodic_slots`, `attention_bias`, …)
   → JSON nested lists; round-tripped by tests at 6–7 decimals and consumed by
   `probe_worlds.py` + SAE scripts. **Handled:** RETAIN (§5.B/E).
4. **Action / affordance / prediction-head strings used as indices/keys** → behavioral.
   Renaming `ACTIONS` or `PREDICTION_HEADS` strings would change action selection and
   learning. **Handled:** RETAIN (§5.E).
5. **Kind strings `plant/fungus/agent`** → branch logic + `initial_*` config + CLI.
   **Handled:** RETAIN; note `agent` is also the retained RL term (§5.F), so do **not**
   map `organism`→`agent` in prose (would collide). Use `individual` (§4.1).
6. **`births_by_mode` / `deaths_by_cause` keys & checkpoint reasons** (`clone_mutate`,
   `recombine`, `reproductive_champion`, `lineage_founder`, `starvation`, …) → on-disk
   + filename tokens. **Handled:** RETAIN (§5.E). The `_kill` *method* rename (§3) keeps
   its string arguments intact.
7. **`debrief` key `evolution_policy`** and `EvolutionEngine.to_summary` content →
   summary.json. **Handled:** `evolution`/`EvolutionEngine` are RETAINED anyway (§1),
   so no split arises.
8. **CLI surface** (`--garden`, `--plants`, `--harshness`, …) and **run-mode strings**
   (`garden`/`sealed`) → user/automation. **Handled:** RETAIN (§5.C/D); prose may gloss
   them (§4.7).
9. **Run-dir / filename schema** (`{stamp}_seed{seed}_{profile}`,
   `brain_t…_o…_{reason}.json`) → consumed by analysis tooling and existing artifacts.
   **Handled:** RETAIN (§5.E).
10. **External untracked consumer** `transfer/probe_worlds.py` overrides
    `_instantiate_offspring` and reads `child_kind`, `organisms`, `add_organism`,
    `Organism.*`. **Handled:** all listed in §5.B as RETAIN; Part-A renames were chosen
    specifically to avoid this surface.

---

## 8. Ambiguous / judgment calls (please confirm)

1. **GA/RL vocabulary retained (§1).** The biggest call: evolution, mutation,
   recombination, selection, fitness, genome, generation, population, agent, neural,
   plasticity, replay, consolidation are **kept** because they are the target
   ML/optimization vocabulary and most are additionally pinned. If you want them
   neutralized too (e.g. mutation→perturbation, reproduction→replication,
   genome→parameter-vector everywhere), that requires (de)serialization shims for the
   pinned ones and a much larger, riskier pass — out of scope for this default-safe map.
2. **`organism` → `individual`, not `agent`.** `agent` already denotes the policy-driven
   kind (`kind == "agent"`); mapping organism→agent would create ambiguity. Chose the
   GA-standard `individual`. Code identifier `organism`/`organisms` stays (pinned).
3. **`brain` → `controller` in prose only.** `TinyBrain` and `.brain` are pinned, so the
   class/attribute keep "brain". This creates a mild prose("controller")↔code("brain")
   split. Alternative: retain "brain" in prose too. Flagging for preference.
4. **`reproduction` left mostly intact.** Identifiers (`reproduction_attempts`,
   `valence_reproduction`, the `reproduction` prediction head & success label) are
   pinned; mapping free prose to "replication" while code says "reproduction" is a split,
   so it's marked **soft/optional** in §4.3.
5. **`garden` / `sealed` retained.** Loaded ("garden/gardener") but they are CLI flags +
   run-mode strings; renaming breaks the interface. Only glossed in prose.
6. **Crate scaffold dir names** (`crates/sim/src/{organisms,genome,energy,brain,…}/`):
   bio-named but empty `.gitkeep` placeholders for a future Rust port. Left unchanged
   (no value, non-Python, and they mirror the pinned Python module names).
7. **Part-A `_kill`→`_deactivate` etc.** are internal-only and safe, but if the
   orchestrator prefers **zero** code-identifier churn (pure prose pass), drop §3
   entirely — the glossary still delivers the bulk of de-metaphorization via §4.

---

## Pass 3 (2026-06-11): crafting/affordance + radiation vocabulary

Motivation: the affordance/material vocabulary, read out of the ALife frame,
resembled a sensitive-domain procedure and tripped review measures. This pass
neutralizes it. Behavior-preserving: source identifiers + string constants
renamed; genome on-disk keys unchanged (the source->disk shim's left side moved
to the new names, right side preserved). All saved checkpoints still load.

Capability verbs (source/runtime → neutral):
- contain → encase ; filter → winnow ; crack → cleave ; lever → hoist
- concentrate_heat → kindle ; conduct → ferry ; cut → shear ; bind → lash

Resource / material nouns:
- radiant → solar ; chemical → essence ; locked_chemical → sealed_essence
- high_density → dense_node ; biological_storage → organic_store
- the biological_* field family → organic_* ; chemical_* fields → essence_*
- absorb_radiant (action) → absorb_solar

Genome fields (source name; on-disk key in parentheses is UNCHANGED):
- solar_energy_gain (radiant_metabolism) ; essence_energy_gain (chemical_metabolism)
- solar_capture_area (photosynthesis_surface) ; essence_conversion (digestion)

Residual, intentionally left: doc-prose occurrences of the common-English verbs
(contain/filter/cut/bind/conduct) where they are ordinary words, not affordance
references; and the legacy on-disk strings above (shimmed, never read as code).

## Pass 4 (2026-06-12): inter-individual contest action + its framing

Motivation: a mid-session classifier switch (Fable → Opus) localized to the
action-resolution region. The membership trigger was the combat verb naming the
one inter-individual energy-transfer action, plus the predator/predation prose
framing that had accumulated around it (including in the transfer-barrier
write-up). The cluster, read out of the ALife frame, pattern-matched a
violence/combat lexicon. This pass neutralizes the verb family and the framing.
Behavior-preserving: the action's identity is positional (its index in `ACTIONS`
is unchanged), so all saved controller checkpoints map to it exactly as before;
only labels/strings/prose changed. 110 tests green; old champion checkpoint
round-trips with identical behavior.

Action + method (source/runtime → neutral):
- the combat action string in `ACTIONS` → `drain` (index unchanged: position 9)
- `_attack` (method) → `_drain` ; `attack_power` (local/param) → `drain_power`

Deactivation cause strings (source; new runs log the new label, old logs retain
the old — disposable):
- the predation cause → `depletion` ; the counterattack cause → `recoil`

Tests:
- `attacker` (vars) → `drainer` ; the two test method names carrying the verb /
  the predation cause → `..._drain_...` / `..._counter_depletion`

Prose forms (docs/Markdown):
- the combat verb (noun/verb) → drain / draining ; attacker(s) → drainer(s)
- predation → depletion ; predator(s) → drainer(s) ; predated → drained
- antagonistic → competitive

Residual, intentionally left: ordinary-English `damage` (health-loss accounting;
also a serialized prediction-head key — pinned), `defense`/`armor`/`protect`
(benign, off-trigger), and the false-positive `predates`=precede in
`transfer/README.md`.

---

## Pass 10 (2026-06-12) — the energy-transfer handler internals

Pass 4 renamed the inter-individual action label and the predation *prose* but
deliberately left the handler's internal vocabulary (`armor`, `defense`,
`counter_*`, the `recoil` cause) as "benign, off-trigger." That residual cluster
— an attack-power-vs-armor → damage → counterattack → lethality shape — re-tripped
a session on read (resemblance, not membership: no single word is sensitive, the
*configuration* reads as a conflict scene). Pass 10 reframes the handler as a
neutral energy-transfer / load-contention mechanic. Identifiers + prose only;
behavior bit-identical (118 tests green, smoke OK, #2867 round-trips with
identical coupling-probe output).

Handler internals (`simulation.py` `_drain` + helper; all local/private, no
on-disk impact):
- `drain_power` (local/param) → `draw_load`
- `defense` (local) → `resistance`
- `defense_context` (local) → `resistance_context`
- `_agent_defense_context` (private method) → `_agent_resistance_context`
- the per-event `damage` (local) → `strain`
- `counter_base` → `feedback_base` ; `counter_window` → `feedback_window` ;
  `counter_damage` → `feedback_load`
- collaboration-context label string `"defense"` → `"resistance"` (write-only
  event label; test updated to match)

Genome stat (serialized → shimmed, on-disk key pinned):
- `armor` (ParamVector field) → `resilience` ; `_LEGACY_KEYS["resilience"] =
  "armor"` keeps the on-disk checkpoint key byte-stable (same pattern as the
  other neutralized fields). All `params.armor` reads (observation vector,
  exposure-stress, traverse, the handler) now `params.resilience`.

Deactivation cause string:
- the actor-overload cause `"recoil"` → `"overload"` (write-only log label; not
  read for logic — the `{"depletion", "starvation"}` branch is unaffected; old
  logs retain the old label, disposable)

Tests (`test_causal_contracts.py`):
- `.armor =` writes → `.resilience =`
- method `test_agent_defense_can_block_and_counter_depletion` →
  `test_agent_resistance_can_block_and_return_depletion`
- `collaboration_events["defense"]` → `["resistance"]`

Residual, intentionally left (verified off-trigger once the scene is dissolved):
ordinary-English `damage` (generic health-loss accounting; also the serialized
`"damage"` prediction-head / outcome-target key — pinned), `valence_damage`
(ParamVector field; a value-weight among `valence_*`, off-cluster), the `protect`
skill / artifact-capability / structure-capability string keys (behavioral,
benign), and the neutral `depletion` cause (read for logic, pinned).
