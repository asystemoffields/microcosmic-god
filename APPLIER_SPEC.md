# Prose-neutralization spec (Stage: prose pass)

You are one of several parallel agents neutralizing **wet-biology prose** so this
artificial-life / RL sandbox reads as the ML/simulation code it is. **Do not change
behavior.** The behavior-critical *code* renames are already done and test-green; your
job is comments, docstrings, and Markdown only.

## Hard rules
1. **PROSE ONLY.** In `.py` files edit only comments (`# …`) and docstrings (`"""…"""`).
   **Never** edit code identifiers, string literals, or dict keys. In `.md` files edit
   body prose freely.
2. **Never touch string literals / dict keys / CLI flags / module paths.** Module names
   `brain`, `organisms`, `world`, `energy`, `genome`, `config` stay. Do not rename any
   file.
3. **Skip these files entirely:** `GLOSSARY.md`, `APPLIER_SPEC.md`, `RENAME*.md`, and the
   `_LEGACY_KEYS` map in `genome.py`.
4. **For any `.py` file you edit:** afterward run, from the repo root,
   `./.venv/bin/python -m unittest discover -s tests -q` and confirm `OK` (77 tests, 3
   skipped). If it breaks, you touched code — revert that change. (Files under
   `transfer/`, `analysis/` aren't in the suite, but still: prose only.)

## Already-renamed identifiers — use the NEW names in prose
When prose mentions these, use the new word, and update any backticked stale identifier:
- organism → individual ; `Organism` → `Individual` ; `add_organism` → `add_individual`
- brain → controller ; `TinyBrain` → `TinyController` ; brain_template → controller_template
- metabolism/metabolic → upkeep ; `_metabolize` → `_apply_upkeep` ; `metabolic_cost` → `upkeep_cost`
- `_kill` → `_deactivate` ; `_seed_initial_life` → `_seed_initial_population`
- Genome fields: radiant_metabolism→radiant_energy_gain, chemical_metabolism→chemical_energy_gain,
  photosynthesis_surface→radiant_capture_area, digestion→chemical_conversion,
  mate_selectivity→pairing_selectivity, asexual_threshold→single_parent_threshold,
  sexual_threshold→two_parent_threshold (on-disk keys stay legacy via a shim — don't mention disk keys)

## Prose rename map (wet-bio → neutral)
organism/creature → individual · alive/living → active · death/die/dead → removal/deactivation/inactive ·
birth/born → creation/spawning · brain → controller · neural tissue → controller capacity ·
metabolism/metabolic → upkeep/maintenance cost · photosynthesis → radiant-energy capture ·
digestion → chemical-energy conversion · appetite → intake rate · ecology/ecological → environment/system-level ·
ecosystem → environment/interacting population · habitat → environment/local conditions · niche → specialization/regime ·
predator/prey/predation → attacker/target/antagonistic interaction · food web → resource–consumer network ·
drown/drowning → fluid overload · desiccate/desiccation → dehydration · mate/mating → pairing ·
offspring → child/successor · gene/genes → parameter · trait → parameter/attribute ·
Darwinian → standard parameter inheritance · Lamarckian → acquired-state inheritance ·
"the god"/"Microcosmic God" (as actor) → the simulation/framework · god-object → monolithic central class ·
civilization → dominant population cluster · ecological collapse → population collapse ·
spontaneous coupling → offline replay association · writing/literacy/proto-writing → durable symbol encoding ·
knowledge transmission → information transfer · plant/fungus (prose) → non-policy producer/consumer
(KEEP the code strings "plant"/"fungus") · starvation (prose) → energy depletion (string stays)

## RETAIN in prose (do NOT rewrite — standard ML/GA, deliberately kept)
evolution/evolutionary/evolve, mutation, recombination/crossover, selection, fitness, population,
generation, genome, agent, neural, plasticity, eligibility trace, valence, reward, prediction,
attention, episodic memory, replay, consolidation, signal, mark, champion, lineage.

## Report back
Files touched · rough count of prose substitutions · for code files: confirm tests green.
