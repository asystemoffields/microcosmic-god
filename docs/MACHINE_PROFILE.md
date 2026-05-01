# Local Machine Profile

Captured during project setup on 2026-05-01.

## Hardware

- CPU: AMD Ryzen 5 7530U with Radeon Graphics
- Physical cores: 6
- Logical processors: 12
- RAM visible to Windows: about 7.28 GiB
- Available RAM during setup: about 1.2 GiB
- GPU: AMD Radeon integrated graphics, reported adapter RAM 512 MiB
- Free disk on C: about 306 GiB

## Implications

This machine is good for short and medium CPU-bound headless runs. It is not ideal for very large neural populations or heavy GPU workloads.

Default local strategy:

- `smoke` profile for correctness.
- `minute` profile for real iteration.
- `long` profile only after the ecology is stable.
- keep `--quiet-events` on for long runs if log volume grows.

## Cloud Scaling Notes

Modal and Google Colab are suitable next steps once local runs produce interesting dynamics. The run folders are portable: each one contains its config, event log, summary, final world state, and brain checkpoints.

The expected workflow is:

1. Tune configs locally with `smoke` and `minute`.
2. Commit a stable experiment config.
3. Launch larger sealed runs on Modal or Colab.
4. Bring back `summary.json`, `events.jsonl`, and selected `checkpoints/`.

