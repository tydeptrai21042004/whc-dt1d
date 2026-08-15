from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten(path: Path, payload: dict) -> dict:
    row = {
        "path": str(path),
        "target": path.parents[2].name,
        "method_dir": path.parents[1].name,
        "seed_dir": path.parent.name,
        "task": payload["task"],
        "pipeline": payload["pipeline"],
        "dataset": payload["dataset"],
        "method": payload["method"],
        "seed": payload["seed"],
        "best_epoch": payload["best_epoch"],
        "best_validation_metric": payload["best_validation_metric"],
    }
    row.update({f"test_{key}": value for key, value in payload.get("test_metrics", {}).items()})
    row.update({f"eff_{key}": value for key, value in payload.get("efficiency", {}).items()})
    return row


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/dense_prediction"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dense_prediction/summary"))
    parser.add_argument("--require-seeds", default="0,1,2")
    args = parser.parse_args(argv)
    rows = [flatten(path, json.loads(path.read_text())) for path in args.input_root.rglob("summary.json") if args.output_dir not in path.parents]
    if not rows:
        raise SystemExit("No dense summary.json files found")
    frame = pd.DataFrame(rows)
    required = {int(seed) for seed in args.require_seeds.split(",") if seed.strip()}
    incomplete = []
    for (target, method), group in frame.groupby(["target", "method"]):
        found = set(int(seed) for seed in group["seed"])
        if found != required:
            incomplete.append({"target": target, "method": method, "found": sorted(found), "required": sorted(required)})
    if incomplete:
        raise SystemExit("Incomplete seed groups:\n" + json.dumps(incomplete, indent=2))
    numeric = [column for column in frame.columns if column.startswith(("test_", "eff_")) and pd.api.types.is_numeric_dtype(frame[column])]
    grouped = frame.groupby(["target", "task", "pipeline", "method"])[numeric].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(filter(None, map(str, column))).rstrip("_") if isinstance(column, tuple) else column for column in grouped.columns]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "dense_per_seed.csv", index=False)
    grouped.to_csv(args.output_dir / "dense_mean_std.csv", index=False)
    print(grouped.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
