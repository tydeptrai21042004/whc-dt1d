#!/usr/bin/env python3
"""Run one self-contained DT1D paper experiment fairly.

Each committed experiment is one YAML file. The runner never generates
seed-specific YAML files. For every method it trains the same declared LR grid
for every requested seed, selects **one shared LR for that method** using the
mean best-validation accuracy across those seeds, and only then evaluates each
seed's selected checkpoint on the test split exactly once.

This avoids both test-set hyperparameter selection and per-seed LR cherry
picking. The best checkpoint epoch is still validation-selected independently
inside each seed, which is standard checkpoint selection.

Output layout
-------------
<output-root>/<experiment_id>/<method>/
    lr_selection_summary.json
    seed_<seed>/
        run_metadata.json
        resolved_config.json
        selection_summary.json
        convergence_summary.json
        efficiency_profile.json
        test_summary.json
        run_status.json
        args.json
        selection/lr_<value>/

The selection subdirectories do not contain ``run_metadata.json`` so the
standard aggregator sees only final, test-evaluated seed directories.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from proposal_contract import runtime_metadata

# Method entries may describe model parameterization only. These training/data
# controls are locked by the experiment-level config so a baseline cannot obtain
# a hidden budget or preprocessing advantage.
FAIR_LOCKED_METHOD_KEYS = {
    "dataset", "backbone", "epochs", "batch_size", "lr", "min_lr",
    "warmup_epochs", "warmup_steps", "weight_decay", "weight_decay_end",
    "weight_decay_dt1d", "update_freq", "weights", "pretrained",
    "input_size", "imagenet_norm", "imagenet_default_mean_and_std",
    "val_ratio", "test_ratio", "use_amp", "deterministic",
    "num_workers", "pin_mem", "final_test",
}


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid experiment config: {path}")
    required = (
        "experiment_id", "seeds", "dataset", "backbone", "epochs",
        "batch_size", "fairness", "common_args", "methods", "method_order",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(f"{path}: missing required keys: {missing}")
    if set(data["method_order"]) != set(data["methods"]) or len(data["method_order"]) != len(data["methods"]):
        raise SystemExit(f"{path}: method_order must contain every method exactly once")
    fairness = data["fairness"]
    if not isinstance(fairness.get("lr_candidates"), list) or not fairness["lr_candidates"]:
        raise SystemExit(f"{path}: fairness.lr_candidates must be a non-empty list")
    if fairness.get("selection_metric", "best_val_acc1") != "best_val_acc1":
        raise SystemExit("Current runner supports selection_metric=best_val_acc1 only")
    if fairness.get("lr_selection_scope", "method_across_seeds") != "method_across_seeds":
        raise SystemExit("Paper protocol requires lr_selection_scope=method_across_seeds")
    if fairness.get("lr_aggregation", "mean_best_val_acc1") != "mean_best_val_acc1":
        raise SystemExit("Paper protocol requires lr_aggregation=mean_best_val_acc1")
    if not bool(fairness.get("evaluate_test_once", True)):
        raise SystemExit("Paper protocol requires evaluate_test_once=true")
    if bool(fairness.get("test_used_for_selection", False)):
        raise SystemExit("Paper protocol requires test_used_for_selection=false")
    if fairness.get("selection_mode", "max") != "max":
        raise SystemExit("Paper protocol requires selection_mode=max")
    if fairness.get("tie_break", "lower_lr") != "lower_lr":
        raise SystemExit("Paper protocol requires tie_break=lower_lr")
    if fairness.get("same_lr_grid_for_all_methods") is not True:
        raise SystemExit("Paper protocol requires same_lr_grid_for_all_methods=true")
    if fairness.get("same_epoch_budget") is not True:
        raise SystemExit("Paper protocol requires same_epoch_budget=true")
    if fairness.get("same_batch_size_within_experiment") is not True:
        raise SystemExit("Paper protocol requires same_batch_size_within_experiment=true")
    if fairness.get("same_seed_and_split_within_seed") is not True:
        raise SystemExit("Paper protocol requires same_seed_and_split_within_seed=true")
    if fairness.get("same_preprocessing") is not True:
        raise SystemExit("Paper protocol requires same_preprocessing=true")
    for method_key, method in data["methods"].items():
        overridden = FAIR_LOCKED_METHOD_KEYS & set(method.get("args", {}))
        if overridden:
            raise SystemExit(
                f"{path}:{method_key} overrides locked fair-protocol keys: {sorted(overridden)}"
            )
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deep_merge(*maps: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in maps:
        for key, value in mapping.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def csv_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def csv_ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != len(set(values)):
        raise SystemExit("Seeds must be unique")
    return values


def lr_slug(value: float) -> str:
    return f"{float(value):.8g}".replace("+", "").replace("-", "m").replace(".", "p")


def normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_command(args_map: dict[str, Any], output_dir: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "main.py")]
    merged = dict(args_map)
    merged["output_dir"] = str(output_dir)
    for key, value in merged.items():
        if value is None:
            continue
        if isinstance(value, list):
            command.append(f"--{key}")
            command.extend(normalize_value(v) for v in value)
        else:
            command.extend([f"--{key}", normalize_value(value)])
    return command


def validate_args(args_map: dict[str, Any], output_dir: Path) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from main import get_args_parser

    parser = get_args_parser()
    parser.parse_args(build_command(args_map, output_dir)[2:])


def split_for(dataset: str, seed: int) -> str | None:
    candidate = ROOT / "splits" / dataset / f"seed{seed}_holdout20.json"
    return str(candidate.resolve()) if candidate.is_file() else None


def base_args(config: dict[str, Any], method_key: str, seed: int, data_path: Path, device: str) -> dict[str, Any]:
    method = config["methods"][method_key]
    args = deep_merge(
        config["common_args"],
        {
            "dataset": config["dataset"],
            "backbone": config["backbone"],
            "epochs": int(config["epochs"]),
            "batch_size": int(config["batch_size"]),
        },
        config.get("args", {}),
        method.get("args", {}),
    )
    args["seed"] = int(seed)
    args["data_path"] = str(data_path.resolve())
    args["device"] = device
    split = split_for(str(config["dataset"]), int(seed))
    if split:
        args["split_file"] = split
    return args


def smoke_args(args: dict[str, Any]) -> dict[str, Any]:
    out = dict(args)
    out.pop("split_file", None)
    out.update({
        "dataset": "fake",
        "weights": "none",
        "pretrained": False,
        "device": "cpu",
        "input_size": 64,
        "epochs": 1,
        "batch_size": 4,
        "fake_train_size": 16,
        "fake_val_size": 8,
        "fake_test_size": 8,
        "fake_num_classes": 5,
        "num_workers": 0,
        "use_amp": False,
        "pin_mem": False,
        "profile_efficiency": False,
        "measure_eval_latency": False,
        "deterministic": True,
    })
    return out


def run_process(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(command), flush=True)
    if dry_run:
        log_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def select_shared_lr(
    candidate_rows: list[dict[str, Any]],
    *,
    seeds: list[int],
    lr_candidates: list[float],
) -> dict[str, Any]:
    """Select one LR for a method by mean validation accuracy across seeds."""
    expected = {(int(seed), float(lr)) for seed in seeds for lr in lr_candidates}
    observed = {(int(row["seed"]), float(row["lr"])) for row in candidate_rows}
    if observed != expected or len(candidate_rows) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            "Fair LR search is incomplete or duplicated; "
            f"missing={missing}, extra={extra}, rows={len(candidate_rows)}, expected={len(expected)}"
        )

    invalid = [
        row for row in candidate_rows
        if int(row.get("return_code", 1)) != 0
        or not math.isfinite(float(row.get("best_val_acc1", float("nan"))))
    ]
    if invalid:
        raise RuntimeError(
            "Fair LR search is incomplete; every method/seed/LR candidate must "
            f"finish successfully before test evaluation. Failed candidates: {invalid}"
        )

    aggregates: list[dict[str, Any]] = []
    for lr in lr_candidates:
        rows = sorted(
            [row for row in candidate_rows if float(row["lr"]) == float(lr)],
            key=lambda row: int(row["seed"]),
        )
        values = [float(row["best_val_acc1"]) for row in rows]
        aggregates.append({
            "lr": float(lr),
            "mean_best_val_acc1": float(statistics.mean(values)),
            "std_best_val_acc1": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "per_seed_best_val_acc1": {str(int(row["seed"])): float(row["best_val_acc1"]) for row in rows},
            "per_seed_best_epoch": {str(int(row["seed"])): int(row["best_epoch"]) for row in rows},
        })

    selected = max(
        aggregates,
        key=lambda row: (float(row["mean_best_val_acc1"]), -float(row["lr"])),
    )
    return {
        "selection_scope": "method_across_seeds",
        "aggregation": "mean_best_val_acc1",
        "tie_break": "lower_lr",
        "seeds": [int(s) for s in seeds],
        "lr_candidates": [float(lr) for lr in lr_candidates],
        "candidates": aggregates,
        "selected_lr": float(selected["lr"]),
        "selected_mean_best_val_acc1": float(selected["mean_best_val_acc1"]),
    }


def candidate_for(candidate_rows: list[dict[str, Any]], seed: int, lr: float) -> dict[str, Any]:
    matches = [
        row for row in candidate_rows
        if int(row["seed"]) == int(seed) and float(row["lr"]) == float(lr)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one candidate for seed={seed}, lr={lr}; got {len(matches)}")
    return matches[0]


def materialize_final_metadata(
    *,
    config: dict[str, Any],
    config_path: Path,
    method_key: str,
    seed: int,
    seed_dir: Path,
    selected: dict[str, Any],
    seed_candidates: list[dict[str, Any]],
    method_selection: dict[str, Any],
    final_args: dict[str, Any],
    return_code: int,
    elapsed_seconds: float,
) -> None:
    method = config["methods"][method_key]
    run_metadata = {
        "schema_version": 4,
        "target": config["experiment_id"],
        "kind": config.get("kind", "comparison"),
        "manuscript_tables": config.get("manuscript_tables", []),
        "manuscript_figures": config.get("manuscript_figures", []),
        "method_preset": method_key,
        "method_label": method.get("label", method_key),
        "variant": method.get("variant"),
        "independent_seed": int(seed),
        "proposal": method_key == "dt1d",
        "reviewer_control": bool(method.get("reviewer_control", False)),
        **runtime_metadata(ROOT),
    }
    (seed_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

    resolved = {
        "schema_version": 4,
        "source_config": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        "source_config_sha256": sha256(config_path),
        "experiment_id": config["experiment_id"],
        "method": method_key,
        "seed": int(seed),
        "fairness": config["fairness"],
        "selected_lr": float(method_selection["selected_lr"]),
        "selected_checkpoint": selected["checkpoint"],
        "lr_selection_scope": "method_across_seeds",
        "args": final_args,
        **runtime_metadata(ROOT),
    }
    (seed_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (seed_dir / "selection_summary.json").write_text(json.dumps({
        "selection_metric": "best_val_acc1",
        "test_used_for_selection": False,
        "lr_selection_scope": "method_across_seeds",
        "lr_aggregation": "mean_best_val_acc1",
        "method_selection": method_selection,
        "seed_candidates": seed_candidates,
        "selected_seed_checkpoint": selected,
    }, indent=2) + "\n", encoding="utf-8")
    (seed_dir / "run_status.json").write_text(json.dumps({
        "return_code": int(return_code),
        "elapsed_seconds": float(elapsed_seconds),
    }, indent=2) + "\n", encoding="utf-8")


def cleanup_checkpoints(candidate_rows: list[dict[str, Any]], selected_lr: float, keep_all: bool) -> None:
    if keep_all:
        return
    for row in candidate_rows:
        path = Path(row["output_dir"]).resolve()
        keep_selected_best = float(row["lr"]) == float(selected_lr)
        for ckpt in path.glob("checkpoint-*.pth"):
            if keep_selected_best and ckpt.name == "checkpoint-best.pth":
                continue
            try:
                ckpt.unlink()
            except OSError:
                pass


def run_method(
    *,
    config: dict[str, Any],
    config_path: Path,
    method_key: str,
    seeds: list[int],
    output_root: Path,
    data_path: Path,
    device: str,
    smoke: bool,
    dry_run: bool,
    skip_if_complete: bool,
) -> list[dict[str, Any]]:
    method_root = output_root / config["experiment_id"] / method_key
    method_selection_path = method_root / "lr_selection_summary.json"

    if skip_if_complete and method_selection_path.is_file():
        complete = all((method_root / f"seed_{seed}" / "test_summary.json").is_file() for seed in seeds)
        if complete:
            selection = load_json(method_selection_path)
            selected_on = {int(x) for x in selection.get("seeds", [])}
            if set(seeds) <= selected_on:
                print(f"[reuse method] {method_root}")
                rows: list[dict[str, Any]] = []
                for seed in seeds:
                    test = load_json(method_root / f"seed_{seed}" / "test_summary.json")
                    rows.append({
                        "method": method_key,
                        "seed": int(seed),
                        "selected_lr": float(test.get("selected_lr", selection["selected_lr"])),
                        "best_val_acc1": float(test.get("best_val_acc1", float("nan"))),
                        "test_acc1": float(test.get("test_acc1_at_best_val", float("nan"))),
                        "return_code": 0,
                        "status": "reused",
                    })
                return rows

    # Partial/stale output cannot safely participate in a shared-LR selection.
    shutil.rmtree(method_root, ignore_errors=True)
    method_root.mkdir(parents=True, exist_ok=True)

    lr_candidates = [float(x) for x in config["fairness"]["lr_candidates"]]
    if smoke:
        lr_candidates = [lr_candidates[0]]

    candidate_rows: list[dict[str, Any]] = []
    base_by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        base = base_args(config, method_key, seed, data_path, device)
        if smoke:
            base = smoke_args(base)
        base_by_seed[int(seed)] = base
        selection_root = method_root / f"seed_{seed}" / "selection"
        selection_root.mkdir(parents=True, exist_ok=True)

        for lr in lr_candidates:
            candidate_dir = selection_root / f"lr_{lr_slug(lr)}"
            shutil.rmtree(candidate_dir, ignore_errors=True)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            args = dict(base)
            args.update({
                "lr": lr,
                "final_test": False,
                "eval": False,
                "save_ckpt": True,
                "save_history": True,
                "profile_efficiency": False,
                "measure_eval_latency": False,
            })
            validate_args(args, candidate_dir)
            command = build_command(args, candidate_dir)
            started = time.time()
            rc = run_process(command, ROOT, candidate_dir / "stdout.log", dry_run=dry_run)
            elapsed = time.time() - started
            if dry_run:
                best_val = 0.0
                best_epoch = 0
                checkpoint = str(candidate_dir / "checkpoint-best.pth")
            else:
                conv = load_json(candidate_dir / "convergence_summary.json")
                best_val = conv.get("best_val_acc1", float("nan"))
                best_epoch = conv.get("best_epoch", -1)
                checkpoint_path = candidate_dir / "checkpoint-best.pth"
                checkpoint = str(checkpoint_path)
                if rc == 0 and not checkpoint_path.is_file():
                    rc = 97
            candidate_rows.append({
                "seed": int(seed),
                "lr": float(lr),
                "best_val_acc1": best_val,
                "best_epoch": best_epoch,
                "return_code": int(rc),
                "elapsed_seconds": float(elapsed),
                "output_dir": str(candidate_dir),
                "checkpoint": checkpoint,
                "test_evaluated": False,
            })

    # Critical barrier: no test process starts until *all* validation candidates
    # for this method and all requested seeds have finished successfully.
    method_selection = select_shared_lr(candidate_rows, seeds=seeds, lr_candidates=lr_candidates)
    method_selection.update({
        "schema_version": 4,
        "experiment_id": config["experiment_id"],
        "method": method_key,
        "test_used_for_selection": False,
        "test_policy": "one test evaluation per seed after shared method-level LR selection",
    })
    method_selection_path.write_text(json.dumps(method_selection, indent=2) + "\n", encoding="utf-8")
    selected_lr = float(method_selection["selected_lr"])

    result_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = method_root / f"seed_{seed}"
        selected = candidate_for(candidate_rows, seed, selected_lr)
        selected_dir = Path(selected["output_dir"])
        base = base_by_seed[int(seed)]
        final_args = dict(base)
        final_args.update({
            "lr": selected_lr,
            "eval": True,
            "final_test": True,
            "finetune": str(selected["checkpoint"]),
            "save_ckpt": False,
            "profile_efficiency": bool(config["common_args"].get("profile_efficiency", True)) and not smoke,
            "measure_eval_latency": bool(config["common_args"].get("measure_eval_latency", True)) and not smoke,
        })
        validate_args(final_args, seed_dir)
        command = build_command(final_args, seed_dir)
        started = time.time()
        final_rc = run_process(command, ROOT, seed_dir / "final_stdout.log", dry_run=dry_run)
        elapsed = time.time() - started

        if dry_run:
            eval_summary = {"acc1": 0.0, "loss": 0.0}
            convergence = {
                "best_val_acc1": selected["best_val_acc1"],
                "best_epoch": selected["best_epoch"],
            }
        else:
            eval_summary = load_json(seed_dir / "eval_summary.json")
            convergence = load_json(selected_dir / "convergence_summary.json")
            if final_rc == 0 and not eval_summary:
                final_rc = 98

        (seed_dir / "convergence_summary.json").write_text(
            json.dumps(convergence, indent=2, default=str) + "\n", encoding="utf-8"
        )
        test_summary = dict(eval_summary)
        test_summary.update({
            "test_acc1_at_best_val": float(eval_summary.get("acc1", float("nan"))),
            "best_val_acc1": float(selected["best_val_acc1"]),
            "selected_checkpoint_epoch": int(selected["best_epoch"]),
            "selected_lr": selected_lr,
            "selected_lr_mean_val_acc1": float(method_selection["selected_mean_best_val_acc1"]),
            "lr_selection_scope": "method_across_seeds",
            "selection_rule": (
                "one LR selected per method by mean best-validation accuracy across seeds; "
                "each seed then evaluated once on test using its best-validation checkpoint at that LR"
            ),
            "test_used_for_hyperparameter_selection": False,
        })
        (seed_dir / "test_summary.json").write_text(
            json.dumps(test_summary, indent=2, default=str) + "\n", encoding="utf-8"
        )

        seed_candidates = [row for row in candidate_rows if int(row["seed"]) == int(seed)]
        materialize_final_metadata(
            config=config,
            config_path=config_path,
            method_key=method_key,
            seed=seed,
            seed_dir=seed_dir,
            selected=selected,
            seed_candidates=seed_candidates,
            method_selection=method_selection,
            final_args=final_args,
            return_code=final_rc,
            elapsed_seconds=elapsed,
        )
        result_rows.append({
            "method": method_key,
            "seed": int(seed),
            "selected_lr": selected_lr,
            "best_val_acc1": float(selected["best_val_acc1"]),
            "test_acc1": float(test_summary["test_acc1_at_best_val"]),
            "return_code": int(final_rc),
        })
        if final_rc != 0:
            break

    cleanup_checkpoints(
        candidate_rows,
        selected_lr,
        keep_all=bool(config["fairness"].get("keep_unselected_checkpoints", False)),
    )
    return result_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "experiments")
    ap.add_argument("--data-path", type=Path, default=ROOT / "data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default=None, help="Comma-separated subset of method keys")
    ap.add_argument("--seeds", default=None, help="Comma-separated subset of seeds")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--skip-if-complete", action="store_true")
    ns = ap.parse_args()

    config_path = ns.config if ns.config.is_absolute() else ROOT / ns.config
    config = load_yaml(config_path)
    output_root = ns.output_root if ns.output_root.is_absolute() else ROOT / ns.output_root
    data_path = ns.data_path if ns.data_path.is_absolute() else ROOT / ns.data_path

    methods = csv_list(ns.methods) or list(config["method_order"])
    unknown = [m for m in methods if m not in config["methods"]]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    requested_seeds = csv_ints(ns.seeds)
    seeds = requested_seeds or [int(x) for x in config["seeds"]]
    declared_seeds = [int(x) for x in config["seeds"]]
    if any(seed not in declared_seeds for seed in seeds):
        raise SystemExit("Requested seed not declared by experiment config")

    lr_candidates = [float(x) for x in config["fairness"]["lr_candidates"]]
    effective_lr_count = 1 if ns.smoke else len(lr_candidates)
    plan = {
        "config": str(config_path),
        "experiment_id": config["experiment_id"],
        "methods": methods,
        "seeds": seeds,
        "lr_candidates": lr_candidates,
        "lr_selection_scope": "method_across_seeds",
        "lr_aggregation": "mean_best_val_acc1",
        "validation_runs": len(methods) * len(seeds) * effective_lr_count,
        "test_runs": len(methods) * len(seeds),
        "test_policy": "once per seed after shared method-level validation-only LR selection",
        "generated_yaml_configs": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_root = output_root / config["experiment_id"]
    experiment_root.mkdir(parents=True, exist_ok=True)
    (experiment_root / "execution_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(config_path, experiment_root / "experiment_config.yaml")
    print(json.dumps(plan, indent=2))
    if ns.plan_only:
        return 0

    rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows = run_method(
            config=config,
            config_path=config_path,
            method_key=method,
            seeds=seeds,
            output_root=output_root,
            data_path=data_path,
            device="cpu" if ns.smoke else ns.device,
            smoke=ns.smoke,
            dry_run=ns.dry_run,
            skip_if_complete=ns.skip_if_complete,
        )
        rows.extend(method_rows)
        if any(int(row.get("return_code", 0)) != 0 for row in method_rows):
            (experiment_root / "session_summary.json").write_text(
                json.dumps(rows, indent=2) + "\n", encoding="utf-8"
            )
            return next(int(row["return_code"]) for row in method_rows if int(row.get("return_code", 0)) != 0)

    (experiment_root / "session_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
