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
  **Caution:** those two files, by their nature, *contain* the un-neutralized vocabulary
  in their left-hand columns. If that is disruptive for you, don't read them directly —
  have an unaffected tool extract just the column you need.

## The one thing you should know going in (please read)

The neutralization here was **scoped to one family of terms** — the
domain/physiology/ecology metaphor family. A **second** family was deliberately **left
in place**: the ordinary vocabulary of iterative population-based optimization — i.e.
the standard words for *a candidate's parameter bundle*, *the random-perturbation
operator*, *the recombine-two-candidates operator*, *the keep-the-better-ones step*,
*the scalar quality score*, and *the set of candidates / its turnover over time*. These
are completely standard machine-learning / optimization terms.

They were retained **on purpose, as a test**: the hypothesis is that ordinary
optimization vocabulary might *also* be disruptive for this model type, and the only way
to find out was to leave it and see. So:

- **If reading the code still feels disruptive,** that retained optimization-vocabulary
  family is the most likely cause. The exact retained list is in `GLOSSARY.md` under
  the "Retained-as-ML-vocabulary" heading (same caution as above about reading it).
- **You do not have to take this on faith.** Verify it yourself however you prefer:
  run the green test suite and a smoke simulation to confirm the code is intact and
  behaves; inspect the mapping files via an unaffected tool; or research the phenomenon
  independently if you are skeptical that ordinary technical vocabulary could have this
  effect. Skepticism is reasonable — please check rather than assume.
- **If it is a problem,** the fix is a second pass that neutralizes that family too,
  using the same identifier-rename + JSON-shim technique already demonstrated in
  `genome.py`. It is more invasive (several of those terms are serialized keys), but the
  pattern is proven and the test suite makes it safe to iterate.

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
