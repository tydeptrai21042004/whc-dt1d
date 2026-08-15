#!/usr/bin/env python3
"""Create the three-seed accuracy/parameter trade-off for manuscript Figure 4."""
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "cnn_paper_three_seed")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "cnn_paper_three_seed" / "figures" / "figure_04_three_seed.png")
    parser.add_argument("--require-seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument("--require-count", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    required = set(args.require_seeds)
    if args.require_count is not None and args.require_count != len(required):
        raise SystemExit("--require-count conflicts with --require-seeds.")

    grouped: dict[str, dict[int, tuple[float, float]]] = {}
    for meta_path in sorted((root / "figure_04").rglob("run_metadata.json")):
        run_dir = meta_path.parent
        meta = read_json(meta_path)
        status = read_json(run_dir / "run_status.json")
        test = read_json(run_dir / "test_summary.json")
        conv = read_json(run_dir / "convergence_summary.json")
        if status.get("return_code") != 0 or "acc1" not in test:
            continue
        label = str(meta.get("method_label", meta.get("method_preset")))
        seed = int(meta["independent_seed"])
        params = conv.get("n_trainable_parameters")
        if params is None:
            raise SystemExit(f"Missing trainable-parameter count for {label}, seed {seed}.")
        if seed in grouped.setdefault(label, {}):
            raise SystemExit(f"Duplicate Figure-4 run for {label}, seed {seed}.")
        grouped[label][seed] = (float(test["acc1"]), float(params))

    if not grouped:
        raise SystemExit(f"No successful Figure-4 runs found under {root}.")
    incomplete = {
        label: {"missing": sorted(required - set(values)), "unexpected": sorted(set(values) - required)}
        for label, values in grouped.items()
        if set(values) != required
    }
    if incomplete:
        raise SystemExit(f"Incomplete Figure-4 seed groups: {incomplete}")

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for label, by_seed in sorted(grouped.items()):
        acc = np.asarray([by_seed[seed][0] for seed in args.require_seeds], dtype=float)
        param_values = np.asarray([by_seed[seed][1] for seed in args.require_seeds], dtype=float)
        if not np.all(param_values == param_values[0]):
            raise SystemExit(f"Trainable parameters changed across seeds for {label}: {param_values.tolist()}")
        params = float(param_values[0])
        x = params
        mean, std = float(np.mean(acc)), float(np.std(acc, ddof=1))
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=4, linestyle="none")
        ax.annotate(label, (x, mean), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test top-1 accuracy (%)")
    ax.set_title("Accuracy–parameter trade-off (mean ± std, seeds 0/1/2)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
