# Patch recovery — feature note (2026-06-11)

First of the ranked world features (see `docs/RESUME_2026-06-11.md`): make
efficient foraging require spatio-temporal memory, so recurrence has something
to earn its keep on. Opportunity-shaped by design — it adds an exploitable
regularity without raising the survival floor (the floor only moves if the
knobs are set aggressively; see calibration below).

## Mechanic

A substantial feeding event (extraction > 0.5 energy in one act) at a place
starts a **recovery window**: for `patch_recovery_ticks` ticks, that place's
staple regeneration is multiplied by `patch_recovery_floor` (default 0 — fully
suppressed). Repeat feeds extend the window (max, never shrink). Current
resource *levels* stay fully observable; what is not observable is whether a
patch you left has recovered yet — that is the memory problem.

- **`patch_recovery_jitter` = 0**: the window is exactly N ticks. Remembering
  where/when you fed predicts when a return visit pays → route rotation is
  learnable.
- **`patch_recovery_jitter` = 1**: window drawn per event from a same-mean
  exponential. Mean energetics identical, predictability severed. This is the
  **scrambled control**: any census/fitness gap between fixed and scrambled
  arms is attributable to *exploiting the regularity*, not to the energy
  change the feature introduces.

Defaults off (`patch_recovery_ticks = 0`); behavior is bit-identical to the
pre-feature code when disabled. World refresh carries recovery windows over
(consistent with refresh preserving resource depletion). Implementation:
`world.note_patch_depletion`, the regen branch in `world.update_environment`,
the trigger in `simulation._eat`. Tests: `tests/test_patch_recovery.py` (8).

## Observability for analysis

- `summary.json: patch_recovery_triggers` — count of window starts/extensions.
- `physics_events: patch_recovering` — place-ticks spent suppressed.
- `movement` summary (`movement_routes`) — the route-rotation readout.

## Calibration note

The 0.5 trigger threshold is below a typical meal (appetite ≥ 2.0 → extraction
≥ 1.1 when food is present), so effectively **every successful meal triggers**.
In a 300-tick smoke at default density with a 60-tick window, ~93% of
place-ticks were suppressed — that is starvation pressure, not memory pressure.
Window length (and floor) set the regime; start near 120 ticks / floor 0 at
moderate density and tune by the `patch_recovering` share (a useful target to
explore: 30–60% of place-ticks suppressed).

## E5 — first behavioral A/B (in flight)

`runs/e5_patch/{off,fixed,scrambled}`: seed 351, h 1.35, no refresh,
all-modular (max-blocks 3, rate 0.30, grace 150/0.35), 25-min walls; window
120, floor 0. Predictions if the feature works as intended:
1. fixed and scrambled both depress raw energy intake vs off (same magnitude —
   energetics match);
2. movement/route diversity rises in both feature arms vs off;
3. any *advantage* of fixed over scrambled (survival, energy, census growth)
   is the memory-exploitation signal. At 25 minutes a census difference is
   unlikely (scale discipline: PARK, don't KILL — this scale only validates
   mechanics and calibration, the real test rides along a long run).
