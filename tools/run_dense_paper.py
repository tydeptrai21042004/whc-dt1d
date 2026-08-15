#!/usr/bin/env python3
"""Run one or more self-contained dense-prediction experiment configs."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_DIR = ROOT / "configs" / "dense" / "experiments"
INDEX = ROOT / "configs" / "dense" / "index.yaml"


def parse_csv_ints(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Seeds must be unique independent runs")
    return values


def parse_csv_strings(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def load_index() -> dict:
    payload = yaml.safe_load(INDEX.read_text(encoding="utf8"))
    if payload.get("schema_version") != 2 or not payload.get("experiments"):
        raise SystemExit(f"Invalid dense index: {INDEX}")
    return payload


def load_config(name: str) -> dict:
    from tools.run_dense_from_config import load_config as _load
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"Unknown dense experiment {name!r}: {path}")
    return _load(path)


def select_targets(requested: str, index: dict) -> list[str]:
    names = list(index["experiments"]) if requested == "all" else parse_csv_strings(requested)
    unknown = [name for name in names if name not in index["experiments"]]
    if unknown:
        raise SystemExit(f"Unknown dense experiment(s): {unknown}")
    return names


def select_methods(config: dict, requested: str) -> list[str]:
    available = list(config["method_order"])
    if not requested or requested == "target":
        return available
    names = parse_csv_strings(requested)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"Method(s) {unknown} are not enabled for {config['experiment_id']}; available={available}")
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="all", help="all or comma-separated dense experiment IDs")
    parser.add_argument("--methods", default="target", help="target or comma-separated methods")
    parser.add_argument("--seeds", type=parse_csv_ints, default=None)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "dense_prediction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--profile-latency", choices=["true", "false"], default=None)
    parser.add_argument("--skip-if-complete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    options = parser.parse_args(argv)

    index = load_index()
    target_names = select_targets(options.target, index)
    output_root = options.output_root if options.output_root.is_absolute() else ROOT / options.output_root
    data_root = options.data_root if options.data_root.is_absolute() else ROOT / options.data_root

    runs = []
    for target_name in target_names:
        config = load_config(target_name)
        seeds = options.seeds if options.seeds is not None else [int(seed) for seed in config["seeds"]]
        if any(seed not in config["seeds"] for seed in seeds):
            raise SystemExit(f"Requested seed not declared by {target_name}")
        for method in select_methods(config, options.methods):
            for seed in seeds:
                output = output_root / target_name / method / f"seed_{seed}"
                data_path = data_root / config.get("data_subdir", "")
                runs.append((target_name, method, seed, output, data_path))
    if options.max_runs:
        runs = runs[: options.max_runs]
    if not runs:
        raise SystemExit("No dense runs selected")

    plan = {
        "config_index": str(INDEX),
        "targets": target_names,
        "method_request": options.methods,
        "run_count": len(runs),
        "smoke": options.smoke,
        "generated_yaml_configs": 0,
        "runs": [
            {
                "config": str(CONFIG_DIR / f"{target}.yaml"),
                "output_dir": str(output),
                "target": target,
                "method": method,
                "seed": seed,
            }
            for target, method, seed, output, _ in runs
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "execution_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    print(json.dumps(plan, indent=2))
    if options.plan_only:
        return 0

    failures = []
    for run_index, (target, method, seed, output, data_path) in enumerate(runs, start=1):
        command = [
            sys.executable,
            str(ROOT / "tools" / "run_dense_from_config.py"),
            str(CONFIG_DIR / f"{target}.yaml"),
            "--method", method,
            "--seed", str(seed),
            "--data-path", str(data_path.resolve()),
            "--output-dir", str(output.resolve()),
            "--device", "cpu" if options.smoke else options.device,
        ]
        if options.num_workers is not None:
            command += ["--num-workers", str(options.num_workers)]
        if options.download:
            command.append("--download")
        if options.no_pretrained:
            command.append("--no-pretrained")
        if options.profile_latency is not None:
            command += ["--profile-latency", options.profile_latency]
        if options.smoke:
            command.append("--smoke")
        if options.dry_run:
            command.append("--dry-run")
        if options.skip_if_complete:
            command.append("--skip-if-complete")
        print(f"[{run_index}/{len(runs)}] {shlex.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            failures.append({"target": target, "method": method, "seed": seed, "returncode": result.returncode})
            if not options.continue_on_error:
                break

    report = {"runs_selected": len(runs), "failures": failures}
    (output_root / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
