#!/usr/bin/env python3
"""Run one method/seed from one self-contained dense experiment YAML."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DENSE_LOCKED_METHOD_KEYS = {
    "task", "pipeline", "dataset", "data_path", "num_classes", "input_size",
    "seed", "epochs", "batch_size", "num_workers", "pretrained", "download",
    "lr", "weight_decay", "use_amp", "device", "deterministic", "pin_mem",
    "profile_latency", "latency_warmup", "latency_iterations",
    "max_train_samples", "max_val_samples", "max_test_samples",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf8"))
    required = {
        "schema_version", "experiment_id", "seeds", "task", "pipeline", "dataset",
        "num_classes", "input_size", "epochs", "batch_size", "fairness",
        "common_args", "methods", "method_order",
    }
    if not isinstance(payload, dict) or not required <= set(payload):
        missing = sorted(required - set(payload or {}))
        raise SystemExit(f"Invalid dense config {path}; missing={missing}")
    if payload["schema_version"] != 2:
        raise SystemExit(f"Unsupported dense config schema: {payload['schema_version']}")
    if payload["method_order"] != list(payload["methods"]):
        raise SystemExit(f"{path}: method_order must match method declaration order")
    if payload["seeds"] != [0, 1, 2]:
        raise SystemExit(f"{path}: publication dense configs require seeds 0,1,2")
    fairness = payload["fairness"]
    if fairness.get("test_used_for_selection") is not False or fairness.get("evaluate_test_once") is not True:
        raise SystemExit(f"{path}: dense test-selection contract is invalid")
    for method, spec in payload["methods"].items():
        locked = DENSE_LOCKED_METHOD_KEYS & set(spec.get("args", {}))
        if locked:
            raise SystemExit(f"{path}:{method} overrides locked dense-budget keys: {sorted(locked)}")
    return payload


def build_args(
    payload: dict[str, Any],
    method: str,
    seed: int,
    *,
    smoke: bool = False,
    download: bool | None = None,
    pretrained: bool | None = None,
    num_workers: int | None = None,
    profile_latency: bool | None = None,
) -> dict[str, Any]:
    if method not in payload["methods"]:
        raise SystemExit(f"Unknown method {method!r}; available={list(payload['methods'])}")
    if int(seed) not in [int(x) for x in payload["seeds"]]:
        raise SystemExit(f"Seed {seed} is not declared by {payload['experiment_id']}")

    args = copy.deepcopy(payload["common_args"])
    args.update({
        "task": payload["task"],
        "pipeline": payload["pipeline"],
        "dataset": payload["dataset"],
        "num_classes": int(payload["num_classes"]),
        "input_size": int(payload["input_size"]),
        "epochs": int(payload["epochs"]),
        "batch_size": int(payload["batch_size"]),
        "seed": int(seed),
    })
    args.update(copy.deepcopy(payload["methods"][method].get("args", {})))

    # Operational overrides apply globally to the selected run and never come
    # from a method row, so they cannot create a hidden baseline advantage.
    if download is not None:
        args["download"] = bool(download)
    if pretrained is not None:
        args["pretrained"] = bool(pretrained)
    if num_workers is not None:
        args["num_workers"] = int(num_workers)
    if profile_latency is not None:
        args["profile_latency"] = bool(profile_latency)

    if smoke:
        args.update({
            "dataset": {
                "binary_segmentation": "fake_binary",
                "semantic_segmentation": "fake_semantic",
                "detection": "fake_detection",
            }[payload["task"]],
            "pretrained": False,
            "epochs": 1,
            "input_size": 64,
            "batch_size": 1 if payload["task"] == "detection" else 2,
            "fake_train_size": 2 if payload["task"] == "detection" else 4,
            "fake_val_size": 1 if payload["task"] == "detection" else 2,
            "fake_test_size": 1 if payload["task"] == "detection" else 2,
            "max_train_samples": 0,
            "max_val_samples": 0,
            "max_test_samples": 0,
            "profile_latency": False,
            "save_checkpoint": False,
            "use_amp": False,
            "pin_mem": False,
            "num_workers": 0,
            "vit_variant": "tiny" if payload["pipeline"] == "vit_b16_dense" else "vit_b16",
            "detector_variant": "tiny" if payload["task"] == "detection" else "mobilenet_v3_fpn",
        })
    return args


def build_command(args: dict[str, Any], *, data_path: Path, output_dir: Path, device: str) -> list[str]:
    merged = dict(args)
    merged.update({"data_path": str(data_path), "output_dir": str(output_dir), "device": device})
    command = [sys.executable, str(ROOT / "dense_main.py")]
    for key, value in merged.items():
        command.extend([f"--{key}", cli_value(value)])
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--profile-latency", choices=["true", "false"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-if-complete", action="store_true")
    ns = parser.parse_args(argv)

    config_path = ns.config if ns.config.is_absolute() else ROOT / ns.config
    payload = load_config(config_path)
    output_dir = ns.output_dir if ns.output_dir.is_absolute() else ROOT / ns.output_dir
    data_path = ns.data_path if ns.data_path.is_absolute() else ROOT / ns.data_path
    summary = output_dir / "summary.json"
    if ns.skip_if_complete and summary.is_file():
        print(json.dumps({"status": "skipped_complete", "summary": str(summary)}, indent=2))
        return 0

    profile_latency = None if ns.profile_latency is None else ns.profile_latency == "true"
    args = build_args(
        payload,
        ns.method,
        ns.seed,
        smoke=ns.smoke,
        download=True if ns.download else None,
        pretrained=False if ns.no_pretrained else None,
        num_workers=ns.num_workers,
        profile_latency=profile_latency,
    )
    command = build_command(args, data_path=data_path.resolve(), output_dir=output_dir.resolve(), device="cpu" if ns.smoke else ns.device)

    from dense_main import build_parser
    build_parser().parse_args(command[2:])
    print(shlex.join(command), flush=True)
    if ns.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps({
        "schema_version": 2,
        "source_config": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        "source_config_sha256": sha256(config_path),
        "experiment_id": payload["experiment_id"],
        "method": ns.method,
        "seed": int(ns.seed),
        "proposal": ns.method == "dt1d",
        "fairness": payload["fairness"],
        "args": args,
    }, indent=2) + "\n", encoding="utf8")
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
