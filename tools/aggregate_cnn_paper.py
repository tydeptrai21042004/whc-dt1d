#!/usr/bin/env python3
"""Aggregate CNN paper runs into raw, mean±std, JSON, and LaTeX tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items() if not isinstance(value, (dict, list))}


def discover(root: Path, target: str | None) -> list[Path]:
    base = root / target if target else root
    return sorted({p.parent for p in base.rglob("run_metadata.json")})


def display_metric(mean: Any, std: Any, count: Any, digits: int = 3) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std) or int(count or 0) <= 1:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "cnn_paper_three_seed")
    parser.add_argument("--target", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-seeds", default="0,1,2")
    parser.add_argument("--allow-incomplete", action="store_true", help="Write partial summaries without failing.")
    ns = parser.parse_args()

    root = ns.root if ns.root.is_absolute() else ROOT / ns.root
    output_dir = ns.output_dir or (root / "aggregated" / (ns.target or "all"))
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_seeds = {int(v.strip()) for v in ns.require_seeds.split(",") if v.strip()}

    rows: list[dict[str, Any]] = []
    for run_dir in discover(root, ns.target):
        meta = load_json(run_dir / "run_metadata.json")
        args = load_json(run_dir / "args.json")
        status = load_json(run_dir / "run_status.json")
        test = load_json(run_dir / "test_summary.json")
        val = load_json(run_dir / "eval_summary.json")
        conv = load_json(run_dir / "convergence_summary.json")
        eff = load_json(run_dir / "efficiency_profile.json")
        if not args:
            resolved = load_json(run_dir / "resolved_config.json")
            args = resolved.get("args", {})
        row: dict[str, Any] = {
            "run_dir": str(run_dir),
            "target": meta.get("target"),
            "kind": meta.get("kind"),
            "method": meta.get("method_preset"),
            "method_label": meta.get("method_label"),
            "variant": meta.get("variant"),
            "seed": int(meta.get("independent_seed", args.get("seed", -1))),
            "dataset": args.get("dataset"),
            "backbone": args.get("backbone"),
            "epochs": args.get("epochs"),
            "batch_size": args.get("batch_size"),
            "return_code": status.get("return_code"),
            "successful": status.get("return_code") == 0 and bool(test),
        }
        row.update(flatten("test", test))
        row.update(flatten("eval", val))
        row.update(flatten("conv", conv))
        row.update(flatten("eff", eff))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No runs found under {root}.")

    raw = pd.DataFrame(rows).sort_values(["target", "method_label", "variant", "seed"], na_position="last")
    raw_path = output_dir / "raw_runs.csv"
    raw.to_csv(raw_path, index=False)

    identity = ["target", "kind", "dataset", "backbone", "method", "method_label", "variant", "epochs", "batch_size"]
    successful_raw = raw[raw["successful"] == True].copy()  # noqa: E712
    if successful_raw.empty:
        raise SystemExit("No successful runs with test_summary.json were found.")

    metric_cols: list[str] = []
    for column in successful_raw.columns:
        if column in identity + ["run_dir", "seed", "return_code", "successful"]:
            continue
        numeric = pd.to_numeric(successful_raw[column], errors="coerce")
        if numeric.notna().any():
            successful_raw[column] = numeric
            metric_cols.append(column)

    grouped = successful_raw.groupby(identity, dropna=False)
    stats = grouped[metric_cols].agg(["mean", "std", "count"]).reset_index()
    stats.columns = ["__".join([str(v) for v in col if str(v)]) if isinstance(col, tuple) else str(col) for col in stats.columns]
    stats_path = output_dir / "mean_std_numeric.csv"
    stats.to_csv(stats_path, index=False)

    pretty = stats[identity].copy()
    for metric in metric_cols:
        mean_col = f"{metric}__mean"
        std_col = f"{metric}__std"
        count_col = f"{metric}__count"
        if mean_col in stats:
            pretty[metric] = [
                display_metric(m, s, c, digits=3)
                for m, s, c in zip(stats[mean_col], stats.get(std_col), stats.get(count_col))
            ]
    pretty_path = output_dir / "mean_std_pretty.csv"
    pretty.to_csv(pretty_path, index=False)

    completeness: list[dict[str, Any]] = []
    for keys, group in grouped:
        seeds = {int(v) for v in group["seed"].tolist()}
        completeness.append({
            **dict(zip(identity, keys if isinstance(keys, tuple) else (keys,))),
            "observed_seeds": sorted(seeds),
            "expected_seeds": sorted(expected_seeds),
            "complete": seeds == expected_seeds,
            "missing_seeds": sorted(expected_seeds - seeds),
        })
    (output_dir / "seed_completeness.json").write_text(json.dumps(completeness, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n", encoding="utf-8")

    summary = {
        "root": str(root),
        "target": ns.target,
        "expected_seeds": sorted(expected_seeds),
        "discovered_run_count": len(raw),
        "successful_run_count": len(successful_raw),
        "failed_or_incomplete_run_count": len(raw) - len(successful_raw),
        "group_count": len(stats),
        "all_groups_complete": all(item["complete"] for item in completeness),
        "files": {
            "raw": str(raw_path),
            "numeric": str(stats_path),
            "pretty": str(pretty_path),
        },
    }
    (output_dir / "aggregation_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n", encoding="utf-8")

    # A compact manuscript-oriented table with the most common classification metrics.
    compact_metrics = [
        metric for metric in (
            "conv_n_trainable_parameters", "conv_n_total_parameters",
            "test_acc1", "test_acc5", "test_loss", "conv_best_val_acc1",
            "conv_best_epoch", "eff_flops_g", "eff_latency_ms_per_image",
            "eff_fps", "conv_total_train_time_sec", "conv_mean_epoch_time_sec",
            "conv_peak_train_memory_mb", "eff_peak_inference_memory_mb",
        ) if metric in pretty.columns
    ]
    compact = pretty[identity + compact_metrics]
    compact.to_csv(output_dir / "manuscript_compact.csv", index=False)
    try:
        latex = compact.to_latex(index=False, escape=True, longtable=True)
        (output_dir / "manuscript_compact.tex").write_text(latex, encoding="utf-8")
    except Exception as exc:
        (output_dir / "latex_export_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if not summary["all_groups_complete"]:
        print("[error] Some method/variant groups do not contain all required successful seeds.")
        if not ns.allow_incomplete:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
