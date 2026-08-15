#!/usr/bin/env python3
"""Generate a deterministic two-way or three-way index manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--test-ratio", default=0.0, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("Require val_ratio > 0 and val_ratio + test_ratio < 1.")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(args.length, generator=generator).tolist()

    if args.test_ratio > 0:
        holdout_ratio = args.val_ratio + args.test_ratio
        n_holdout = max(2, int(round(holdout_ratio * args.length)))
        n_val = max(1, int(round(args.val_ratio / holdout_ratio * n_holdout)))
        n_test = n_holdout - n_val
        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_holdout]
        train_indices = indices[n_holdout:]
        manifest = {
            "schema_version": 3,
            "dataset": args.dataset,
            "dataset_length": args.length,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "algorithm": "torch.randperm(seed); disjoint test, validation, then training partitions",
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "n_test": len(test_indices),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        }
    else:
        n_val = max(1, int(round(args.val_ratio * args.length)))
        manifest = {
            "schema_version": 2,
            "dataset": args.dataset,
            "dataset_length": args.length,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "algorithm": "torch.randperm(seed); first n_val validation, remaining training",
            "n_train": args.length - n_val,
            "n_val": n_val,
            "train_indices": indices[n_val:],
            "val_indices": indices[:n_val],
        }

    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["content_sha256_without_this_field"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
