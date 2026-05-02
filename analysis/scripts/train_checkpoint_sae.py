from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SEGMENTS = ("weights_in", "weights_out", "bias_h", "bias_o", "prediction_weights", "hidden", "last_outputs")


def checkpoint_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix == ".json":
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("brain_*.json")))
    return sorted(set(paths))


def load_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_sizes(checkpoints: list[dict[str, Any]]) -> dict[str, int]:
    sizes = {segment: 0 for segment in SEGMENTS}
    for checkpoint in checkpoints:
        brain = checkpoint["brain"]
        for segment in SEGMENTS:
            sizes[segment] = max(sizes[segment], len(brain.get(segment, [])))
    return sizes


def vectorize(checkpoint: dict[str, Any], sizes: dict[str, int]) -> np.ndarray:
    brain = checkpoint["brain"]
    parts: list[np.ndarray] = []
    for segment in SEGMENTS:
        values = np.asarray(brain.get(segment, []), dtype=np.float32)
        target = sizes[segment]
        if len(values) < target:
            values = np.pad(values, (0, target - len(values)))
        elif len(values) > target:
            values = values[:target]
        parts.append(values)
    return np.concatenate(parts).astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def train_sae_numpy(x_raw: np.ndarray, latent: int, steps: int, lr: float, l1: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mean = x_raw.mean(axis=0, keepdims=True)
    std = x_raw.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x_raw - mean) / std
    n, dim = x.shape
    w_enc = rng.normal(0.0, 0.03, size=(dim, latent)).astype(np.float32)
    b_enc = np.zeros((latent,), dtype=np.float32)
    w_dec = rng.normal(0.0, 0.03, size=(latent, dim)).astype(np.float32)
    b_dec = np.zeros((dim,), dtype=np.float32)
    losses: list[float] = []

    for step in range(steps):
        z = x @ w_enc + b_enc
        h = relu(z)
        recon = h @ w_dec + b_dec
        err = recon - x
        mse = float(np.mean(err * err))
        sparsity = float(np.mean(np.abs(h)))
        losses.append(mse + l1 * sparsity)

        grad_recon = (2.0 / max(1, n * dim)) * err
        grad_w_dec = h.T @ grad_recon
        grad_b_dec = grad_recon.sum(axis=0)
        grad_h = grad_recon @ w_dec.T + (l1 / max(1, n * latent))
        grad_z = grad_h * (z > 0.0)
        grad_w_enc = x.T @ grad_z
        grad_b_enc = grad_z.sum(axis=0)

        w_dec -= lr * grad_w_dec
        b_dec -= lr * grad_b_dec
        w_enc -= lr * grad_w_enc
        b_enc -= lr * grad_b_enc

        if step > 10 and not np.isfinite(losses[-1]):
            raise RuntimeError("SAE loss became non-finite; lower --lr")

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "w_enc": w_enc,
        "b_enc": b_enc,
        "w_dec": w_dec,
        "b_dec": b_dec,
        "losses": losses,
        "backend": "numpy",
        "device": "cpu",
    }


def train_sae_torch(
    x_raw: np.ndarray,
    latent: int,
    steps: int,
    lr: float,
    l1: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for --backend torch") from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_raw_t = torch.as_tensor(x_raw, dtype=torch.float32, device=device)
    mean = x_raw_t.mean(dim=0, keepdim=True)
    std = x_raw_t.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    x = (x_raw_t - mean) / std
    n, dim = x.shape

    w_enc = (torch.randn((dim, latent), dtype=torch.float32, device=device) * 0.03).requires_grad_(True)
    b_enc = torch.zeros((latent,), dtype=torch.float32, device=device, requires_grad=True)
    w_dec = (torch.randn((latent, dim), dtype=torch.float32, device=device) * 0.03).requires_grad_(True)
    b_dec = torch.zeros((dim,), dtype=torch.float32, device=device, requires_grad=True)
    params = (w_enc, b_enc, w_dec, b_dec)
    losses: list[float] = []

    for step in range(steps):
        h = torch.relu(x @ w_enc + b_enc)
        recon = h @ w_dec + b_dec
        err = recon - x
        loss = torch.mean(err * err) + l1 * torch.mean(torch.abs(h))
        losses.append(float(loss.detach().cpu()))
        if step > 10 and not torch.isfinite(loss):
            raise RuntimeError("SAE loss became non-finite; lower --lr")

        loss.backward()
        with torch.no_grad():
            for param in params:
                assert param.grad is not None
                param -= lr * param.grad
                param.grad.zero_()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return {
        "mean": mean.detach().cpu().numpy().astype(np.float32),
        "std": std.detach().cpu().numpy().astype(np.float32),
        "w_enc": w_enc.detach().cpu().numpy().astype(np.float32),
        "b_enc": b_enc.detach().cpu().numpy().astype(np.float32),
        "w_dec": w_dec.detach().cpu().numpy().astype(np.float32),
        "b_dec": b_dec.detach().cpu().numpy().astype(np.float32),
        "losses": losses,
        "backend": "torch",
        "device": device,
    }


def train_sae(x_raw: np.ndarray, latent: int, steps: int, lr: float, l1: float, seed: int) -> dict[str, Any]:
    return train_sae_numpy(x_raw, latent, steps, lr, l1, seed)


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_backend(backend: str, device: str) -> tuple[str, str]:
    if backend == "auto":
        if device not in {"auto", "cpu"} or torch_cuda_available():
            return "torch", device
        return "numpy", "cpu"
    if backend == "numpy":
        return "numpy", "cpu"
    return "torch", device


def build_report(model: dict[str, Any], x_raw: np.ndarray, files: list[Path], metadata: dict[str, Any]) -> dict[str, Any]:
    mean = model["mean"]
    std = model["std"]
    w_enc = model["w_enc"]
    b_enc = model["b_enc"]
    assert isinstance(mean, np.ndarray)
    assert isinstance(std, np.ndarray)
    assert isinstance(w_enc, np.ndarray)
    assert isinstance(b_enc, np.ndarray)
    h = relu(((x_raw - mean) / std) @ w_enc + b_enc)
    top_checkpoints: list[dict[str, Any]] = []
    for row, path in zip(h, files):
        top_units = np.argsort(row)[-5:][::-1]
        top_checkpoints.append(
            {
                "file": str(path),
                "top_units": [{"unit": int(i), "activation": float(row[i])} for i in top_units if row[i] > 0.0],
                "active_units": int(np.sum(row > 1e-6)),
            }
        )
    active_rate = np.mean(h > 1e-6, axis=0)
    unit_strength = np.mean(h, axis=0)
    return {
        "metadata": metadata,
        "loss_start": float(model["losses"][0]) if model["losses"] else None,
        "loss_end": float(model["losses"][-1]) if model["losses"] else None,
        "unit_active_rate": [float(v) for v in active_rate],
        "unit_mean_activation": [float(v) for v in unit_strength],
        "checkpoint_activations": top_checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small sparse autoencoder on saved Microcosmic God brain checkpoints.")
    parser.add_argument("inputs", nargs="+", help="checkpoint JSON files, checkpoint dirs, or run dirs")
    parser.add_argument("--latent", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1_500)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l1", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto", help="torch device for --backend torch, such as cuda, cuda:0, cpu, or auto")
    parser.add_argument("--out", default="analysis/sae_models/latest_sae.npz")
    args = parser.parse_args()

    files = checkpoint_paths(args.inputs)
    if not files:
        raise SystemExit("No checkpoint files found.")
    checkpoints = [load_checkpoint(path) for path in files]
    sizes = segment_sizes(checkpoints)
    x_raw = np.stack([vectorize(checkpoint, sizes) for checkpoint in checkpoints])
    if x_raw.shape[0] < 4:
        print("Warning: very few checkpoints; SAE will be a microscope scaffold, not a reliable feature model.")

    backend, device = resolve_backend(args.backend, args.device)
    if backend == "torch":
        model = train_sae_torch(x_raw, args.latent, args.steps, args.lr, args.l1, args.seed, device)
    else:
        model = train_sae_numpy(x_raw, args.latent, args.steps, args.lr, args.l1, args.seed)
    metadata = {
        "files": [str(path) for path in files],
        "segments": SEGMENTS,
        "segment_sizes": sizes,
        "input_dim": int(x_raw.shape[1]),
        "checkpoint_count": int(x_raw.shape[0]),
        "latent": args.latent,
        "steps": args.steps,
        "lr": args.lr,
        "l1": args.l1,
        "seed": args.seed,
        "backend": model["backend"],
        "device": model["device"],
    }
    report = build_report(model, x_raw, files, metadata)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        mean=model["mean"],
        std=model["std"],
        w_enc=model["w_enc"],
        b_enc=model["b_enc"],
        w_dec=model["w_dec"],
        b_dec=model["b_dec"],
        metadata=json.dumps(metadata),
    )
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"trained SAE on {x_raw.shape[0]} checkpoints, input_dim={x_raw.shape[1]}, "
        f"latent={args.latent}, backend={model['backend']}, device={model['device']}"
    )
    print(f"model: {out}")
    print(f"report: {report_path}")
    print(f"loss_start={report['loss_start']:.6f} loss_end={report['loss_end']:.6f}")


if __name__ == "__main__":
    main()
