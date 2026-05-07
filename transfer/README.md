# Transfer harness

A test rig for evaluating whether brains evolved in microcosmic-god have transferable representations — does the cognition that emerged in the alife substrate generalize to a different task it has never seen?

## What's here

- `catch_transfer.ipynb` — Colab-ready notebook. Implements a simple Catch environment, loads a microcosmic-god brain checkpoint, trains a small linear adapter around it (brain frozen), and compares against four conditions including skeptic controls.
- `sample_brains/` — committed brain checkpoints from the 30-min seed-1 run for quick experimentation:
  - `learner_champion_hidden14.json` — the default. A long-lived learner champion.
  - `final_overall_champion.json` — the run's overall champion.
  - `final_tool_champion.json` — a tool-master brain.

## Run it on Colab

Open the notebook in Colab. The first cell clones this repo to access the sample checkpoints. Then run all cells. About 10-20 minutes for the full experiment depending on Colab's CPU allocation.

## What we found (10 seeds, 150 ES generations, learner_champion_hidden14)

```
Direct (no brain)     : +0.084 ± 0.252
Random-init brain     : +0.004 ± 0.400
Permuted trained brain: -0.227 ± 0.197
Trained brain         : -0.020 ± 0.360

trained − permuted     = +0.207   structure matters
trained − random_brain = -0.024   no transfer beyond random
trained − direct       = -0.104   brain hinders slightly
```

**The permutation test passes.** Shuffling the trained brain's weights (preserving distribution, destroying structure) makes it +0.21 worse than the unshuffled version. The substrate is producing *structured cognition*, not just well-conditioned random functions.

**But the structure doesn't help on Catch.** Trained and random-init brains transfer equally well. A direct linear policy slightly beats both. Catch is too simple a probe — solvable by a 12-parameter linear policy that doesn't need rich representations of causal/temporal structure.

A genuine transfer test should target what mg's substrate was selected for: temporal reasoning, causal sequencing, partial observability. Candidates: memory-based maze, multi-step puzzle box, sequential prediction.

## Try your own brain

Drop any `brain_*.json` from `runs/<your-run>/checkpoints/` into `sample_brains/` and update `SAMPLE_BRAIN` in the experiment cell. The adapter sizes itself to the brain's dimensions.

## Architecture

The brain is **frozen** during the test — only the input projection (`W_in`: 4 → input_size) and output projection (`W_out`: output_size → 3) train. The brain's `forward()` and attention head run in pure numpy, mirroring `microcosmic_god/brain.py`. Attention noise is set to zero for deterministic evaluation.

Training uses Evolution Strategies (no autograd). This keeps the brain genuinely frozen — no gradient flow through it, no library dependency beyond numpy.

## Honesty note

An early single-seed run showed a +0.38 trained-vs-control advantage that I initially read as "transfer worked." The 10-seed multi-condition controls above showed that result was noise. The robust signal is **trained beats permuted by +0.21** — the substrate is producing structure — but that structure doesn't transfer to Catch specifically. The single-seed claim was wrong; the structured-cognition claim survives the controls.
