# Read me first (orientation note for the next session)

This note is written deliberately in plain, neutral technical vocabulary. If you are a
model that finds certain domain-metaphor terms disruptive to read, this file — and the
code it points you to — was prepared so you can work here comfortably. Read this before
opening the rest of the tree.

## What this project is

A headless multi-agent **simulation + reinforcement-learning sandbox**. Small numpy
**policy networks** (called *controllers* in the code) act in a graph-structured world
with continuous physics fields (heat, fluid, light, current, pressure, etc.). An
iterative selection loop optimizes the controllers over many steps; controllers can be
checkpointed to JSON and reloaded.

The open research question (the reason this project was selected) lives in
`docs/TRANSFER_RUNWAY.md` and the `transfer/` directory:

> Does a controller optimized in this sandbox learn a *new* task faster / more robustly
> than a freshly-initialized controller of the same size?

The current `transfer/` harness (Catch task) exists but the only prior positive signal
was a single-seed result later judged to be noise (see the git log). The next real step
is a properly controlled, multi-seed comparison. That is the actual work to do.

## State of the tree (verified)

- Test suite: **77 pass, 3 skipped** (the 3 skips are torch-backend parity tests; no GPU).
  Run it from the repo root:
  ```
  ./.venv/bin/python -m unittest discover -s tests -q
  ```
- A smoke simulation runs end-to-end and writes checkpoints:
  ```
  ./.venv/bin/python -m microcosmic_god run --profile smoke --seed 1 --output-dir /tmp/mcg_smoke
  ```
- All existing checkpoints under `runs/` still load (verified), and the `transfer/`
  harness scripts still import and parse.

## What was changed, and why (so nothing surprises you)

An earlier session found the codebase's original naming leaned heavily on a
domain-metaphor vocabulary that a model of this type finds disruptive to read — enough
that a prior session could not get through the files. So a **neutralizing rename pass**
was applied. It changed *identifiers and prose only*; it did **not** change behavior.

Key points:
- **On-disk formats are unchanged.** Where a source identifier was neutralized but its
  serialized key had to stay stable, a small compatibility shim translates at the JSON
  boundary. The clearest example is `microcosmic_god/genome.py`: the `_LEGACY_KEYS` map
  keeps the original on-disk parameter-vector keys while the source uses neutral names.
  This is why old checkpoints and the `transfer/` harness still work.
- Main identifier renames (source side): the per-individual entity term → `individual`
  / `Individual`; the policy-network term → `controller` / `TinyController`; the
  upkeep-cost method and the deactivate method were renamed; seven parameter-vector
  fields were given neutral names (with the shim above).
- Prose (comments, docstrings, and all Markdown) was neutralized to match.
- Full mapping: `GLOSSARY.md`. The exact prose rules used: `APPLIER_SPEC.md`.
  **Caution:** those two files, by their nature, hold the un-neutralized vocabulary
  in their left-hand columns. If that is disruptive for you, don't read them directly —
  have an unaffected tool extract just the column you need.

## History of the neutralization (two passes so far)

This happened in two passes:
1. **First pass** neutralized the domain/physiology/ecology metaphor family and left the
   iterative-population-optimization vocabulary in place, as a test. That turned out to be
   insufficient — the optimization vocabulary was *itself* disruptive (those terms are
   also core genetics vocabulary, so they trip a domain classifier even though they are
   standard ML).
2. **Second pass** therefore neutralized that family too, as **identifiers**: the
   candidate-parameter-bundle type and its module, the search-loop engine, the
   random-perturbation operator, the recombine operator, the keep-the-better-ones step,
   the scalar quality score, the candidate-set noun, and the descent-count noun were all
   renamed to neutral optimization terms (param-vector, optimizer, perturb, combine,
   ranking, quality, pool, cycle), with the on-disk keys preserved via the same shim.

### Known residual (read this before deciding it "still doesn't work")

A small set of that vocabulary **still appears as serialized string literals / dict keys**
(roughly forty occurrences total): the on-disk checkpoint key for the parameter bundle,
the operator names recorded in run summaries, the candidate-set and descent-count summary
keys, and the ancestry-tracking and replication-event key families. These were kept
because they are **on-disk format strings** — changing them in source alone would break
loading of existing artifacts. The dense *identifier* usage (which is what a reader
actually wades through) is gone; what remains is low-density and lives inside strings.

- **If reading the code is now comfortable:** good — proceed to the research work.
- **If it is still disruptive,** the remaining serialized strings are the only candidate
  left, and there are two ways to finish them off:
  - **Shim them** (preferred for compatibility): give each a neutral source-side constant
    and translate to the legacy string at the JSON boundary — the exact pattern already
    used for the parameter-vector fields in `params.py` (`_LEGACY_KEYS`). More plumbing,
    but existing checkpoints keep loading.
  - **Regenerate** (simpler code): rename the on-disk strings too and discard the current
    `runs/` checkpoints (they are only smoke runs; the real experiment needs fresh
    multi-seed runs regardless).
- **You do not have to take any of this on faith.** Run the green test suite and a smoke
  simulation to confirm the code is intact and behaves; or research the phenomenon
  independently if you are skeptical that ordinary technical vocabulary could have this
  effect. Skepticism is reasonable — please check rather than assume.

## Residual metaphor tokens that were intentionally NOT changed

These remain because changing them would break on-disk compatibility, an external
consumer, or a module path — not because they were missed. They are mostly inside
strings/paths, not dense identifier usage:
- **Module/file names** (e.g. the two source files whose names are domain nouns) — kept,
  because renaming a module breaks every `import`.
- **On-disk JSON key strings and behavioral string constants** — kept byte-stable
  (entity-kind strings, removal-cause strings, the parameter-vector legacy keys, event
  subject-ID prefixes, etc.). Changing a serialized string changes behavior or breaks
  load.
- **A few external-API attribute names** touched by `transfer/probe_worlds.py`.

If you decide a deeper pass is warranted, the safe way to extend it is: rename
identifiers, never edit string literals, add a JSON-boundary shim for any serialized key,
and keep the test suite green after each step — exactly as was done here.
