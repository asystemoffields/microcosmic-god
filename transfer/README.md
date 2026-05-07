# Transfer harness

A test rig for evaluating whether brains evolved in microcosmic-god have transferable representations — does the cognition that emerged in the alife substrate generalize to a different task it has never seen?

## What's here

- `catch_transfer.ipynb` — Colab-ready notebook. Implements a simple Catch environment, loads a microcosmic-god brain checkpoint, trains a small linear adapter around it (brain frozen), and compares against a control with random-init brain weights. Open in Colab and run all cells.
- `sample_brains/` — committed brain checkpoints from the 30-min seed-1 run for quick experimentation:
  - `learner_champion_hidden14.json` — the default. A long-lived learner champion (hidden_size=14, attention head present).
  - `final_overall_champion.json` — the run's overall champion.
  - `final_tool_champion.json` — a tool-master brain with a different selection profile.

## Run it on Colab

Click "Open in Colab" or upload the notebook. The first cell clones this repo so the sample checkpoints are accessible. Then run all cells. About 5-15 minutes for the full experiment depending on Colab's CPU allocation.

## What it tells you

- If `trained > control`: the brain's hidden representations are doing useful work in Catch. Transfer worked. The substrate is producing genuinely transferable minds.
- If `trained ≈ control`: the linear adapter alone is solving Catch through whatever nonlinearity the brain provides. No real transfer signal.
- Either way: try multiple sample brains. Different selection pressures (tool-master vs causal-champion vs overall-champion) may produce different transferability profiles.

## Try your own brain

Drop any `brain_*.json` from `runs/<your-run>/checkpoints/` into `sample_brains/` and update `SAMPLE_BRAIN` in the experiment cell. Brain shape (input/hidden/output) is read from the checkpoint, so the adapter sizes itself to fit.

## Architecture

The brain is **frozen** during the test — only the input projection (`W_in`: 4 → input_size) and output projection (`W_out`: output_size → 3) train. The brain's `forward()` and attention head run in pure numpy, mirroring `microcosmic_god/brain.py`. Attention noise is set to zero for deterministic evaluation.

Training uses Evolution Strategies (no autograd). This keeps the brain genuinely frozen — no gradient flow through it, no library dependency beyond numpy.
