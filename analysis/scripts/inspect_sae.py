from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_checkpoint_sae import SEGMENTS, load_checkpoint, vectorize


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def segment_spans(sizes: dict[str, int]) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for segment in SEGMENTS:
        size = sizes[segment]
        spans[segment] = (cursor, cursor + size)
        cursor += size
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a brain checkpoint with a trained checkpoint SAE.")
    parser.add_argument("model", help="SAE .npz file")
    parser.add_argument("checkpoint", help="brain checkpoint JSON")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    data = np.load(args.model, allow_pickle=False)
    metadata: dict[str, Any] = json.loads(str(data["metadata"]))
    sizes = {key: int(value) for key, value in metadata["segment_sizes"].items()}
    checkpoint = load_checkpoint(Path(args.checkpoint))
    x_raw = vectorize(checkpoint, sizes)[None, :]
    x = (x_raw - data["mean"]) / data["std"]
    h = relu(x @ data["w_enc"] + data["b_enc"])[0]
    top_units = np.argsort(h)[-args.top :][::-1]
    spans = segment_spans(sizes)

    print(f"checkpoint: {args.checkpoint}")
    print(f"organism: {checkpoint['organism']}")
    print("top SAE units")
    for unit in top_units:
        activation = float(h[unit])
        if activation <= 0.0:
            continue
        decoder = np.asarray(data["w_dec"][unit])
        segment_energy = []
        for segment, (start, end) in spans.items():
            segment_energy.append((float(np.sum(np.abs(decoder[start:end]))), segment))
        segment_energy.sort(reverse=True)
        top_segments = ", ".join(f"{name}:{value:.3f}" for value, name in segment_energy[:4])
        print(f"  unit {int(unit):>3}: activation={activation:.5f} decoder_segments=[{top_segments}]")


if __name__ == "__main__":
    main()

