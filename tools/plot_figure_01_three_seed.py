#!/usr/bin/env python3
"""Create the three-seed convergence plot for manuscript Figure 1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(text: str) -> list[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise argparse.ArgumentTypeError("Seeds must be a non-empty unique list.")
    return seeds


def load_histories(root: Path) -> dict[int, list[dict]]:
    histories: dict[int, list[dict]] = {}
    for path in sorted(root.rglob("history.json")):
        meta_path = path.parent / "run_metadata.json"
        status_path = path.parent / "run_status.json"
        if not meta_path.is_file() or not status_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("return_code") != 0:
            continue
        if meta.get("target") == "figure_01" and meta.get("method_preset") == "dt1d":
            seed = int(meta["independent_seed"])
            if seed in histories:
                raise SystemExit(f"Duplicate Figure-1 history for seed {seed}.")
            histories[seed] = json.loads(path.read_text(encoding="utf-8"))
    return histories


def stack(histories: list[list[dict]], key: str) -> np.ndarray:
    length = min(len(history) for history in histories)
    values = []
    for history in histories:
        values.append([float(history[i].get(key, np.nan)) for i in range(length)])
    return np.asarray(values, dtype=float)


def maybe_percent(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmax(np.abs(finite))) <= 1.5:
        return values * 100.0
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "cnn_paper_three_seed")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cnn_paper_three_seed" / "figures" / "figure_01_three_seed.png")
    parser.add_argument("--require-seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument("--require-count", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    by_seed = load_histories(root)
    required = set(args.require_seeds)
    if args.require_count is not None and args.require_count != len(required):
        raise SystemExit("--require-count conflicts with --require-seeds.")
    missing = sorted(required - set(by_seed))
    extra = sorted(set(by_seed) - required)
    if missing or extra:
        raise SystemExit(f"Figure-1 seed mismatch: missing={missing}, unexpected={extra}")
    histories = [by_seed[seed] for seed in args.require_seeds]

    train_acc = maybe_percent(stack(histories, "train_class_acc"))
    val_acc = maybe_percent(stack(histories, "val_acc1"))
    train_loss = stack(histories, "train_loss")
    val_loss = stack(histories, "val_loss")
    epochs = np.arange(1, train_acc.shape[1] + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for data, label in ((train_acc, "Train accuracy"), (val_acc, "Validation accuracy")):
        mean, std = np.nanmean(data, axis=0), np.nanstd(data, axis=0, ddof=1)
        axes[0].plot(epochs, mean, label=label)
        axes[0].fill_between(epochs, mean - std, mean + std, alpha=0.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy behavior (mean ± std, seeds 0/1/2)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    for data, label in ((train_loss, "Train loss"), (val_loss, "Validation loss")):
        mean, std = np.nanmean(data, axis=0), np.nanstd(data, axis=0, ddof=1)
        axes[1].plot(epochs, mean, label=label)
        axes[1].fill_between(epochs, mean - std, mean + std, alpha=0.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss behavior (mean ± std, seeds 0/1/2)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Caltech101 + ResNet-18 + DT1D-Adapter")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
