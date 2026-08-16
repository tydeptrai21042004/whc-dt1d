#!/usr/bin/env python3
"""Validate every committed experiment config for the single-proposal release."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
INDEX = CONFIG_DIR / "index.yaml"


def load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"{path}: expected mapping")
    return data


def assert_fair(cfg: dict[str, Any], path: Path) -> None:
    fairness = cfg["fairness"]
    assert cfg["seeds"] == [0, 1, 2], f"{path}: paper/reviewer experiments require seeds 0,1,2"
    assert fairness["selection_metric"] == "best_val_acc1", path
    assert fairness["selection_mode"] == "max", path
    assert fairness["tie_break"] == "lower_lr", path
    assert fairness["lr_selection_scope"] == "method_across_seeds", path
    assert fairness["lr_aggregation"] == "mean_best_val_acc1", path
    assert fairness["evaluate_test_once"] is True, path
    assert fairness["test_used_for_selection"] is False, path
    assert fairness["same_lr_grid_for_all_methods"] is True, path
    assert fairness["same_epoch_budget"] is True, path
    assert fairness["same_batch_size_within_experiment"] is True, path
    assert fairness["same_seed_and_split_within_seed"] is True, path
    assert fairness["same_preprocessing"] is True, path
    assert fairness["same_optimizer"] == "AdamW", path
    assert fairness["same_scheduler"] == "cosine", path
    assert len(fairness["lr_candidates"]) >= 1, path
    assert cfg["method_order"] == list(cfg["methods"]), f"{path}: methods must be declared in run order"


def validate_parser(cfg: dict[str, Any], path: Path) -> int:
    sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
    from tools.run_experiment import base_args, validate_args
    checked = 0
    for method in cfg["method_order"]:
        for seed in cfg["seeds"]:
            args = base_args(cfg, method, int(seed), ROOT / "data", "cpu")
            args["lr"] = float(cfg["fairness"]["lr_candidates"][0])
            validate_args(args, ROOT / "outputs" / "_config_validation")
            checked += 1
    return checked


def main() -> int:
    index = load(INDEX)
    assert index["schema_version"] == 4
    assert index["proposal"] == {
        "method_key": "dt1d",
        "name": "DT1D-Adapter",
        "architecture": "R124-P2-G16-Axis-LearnedGate",
    }

    names = list(index["main_experiments"]) + list(index["reviewer_ablations"])
    assert len(names) == len(set(names)), "duplicate experiment IDs"
    checked = 0
    main_rows = 0
    ablation_rows = 0
    ablation_pairs: list[tuple[str, str]] = []
    proposal_keys: set[str] = set()

    required_coverage = {
        "direct_vs_shifted",
        "shared_vs_unshared",
        "single_axis_vs_two_axis",
        "fixed_vs_learned_routing",
        "one_vs_multiple_dilations",
        "gate_on_vs_off",
        "pointwise_off_vs_on",
        "weighted_shift_core",
        "stability_projection_core",
        "previous_version_side_by_side",
    }

    for name in names:
        path = CONFIG_DIR / f"{name}.yaml"
        assert path.is_file(), path
        cfg = load(path)
        assert cfg["schema_version"] == 4, path
        assert cfg["experiment_id"] == name, path
        assert_fair(cfg, path)
        assert "dt1d" in cfg["methods"], f"{path}: canonical proposal missing"
        assert cfg["methods"]["dt1d"]["args"]["tuning_method"] == "dt1d", path
        assert not cfg["methods"]["dt1d"].get("reviewer_control", False), path
        proposal_keys.add(cfg["methods"]["dt1d"]["args"]["tuning_method"])

        if name in index["main_experiments"]:
            main_rows += len(cfg["methods"])
            forbidden = {
                k for k, v in cfg["methods"].items()
                if v.get("reviewer_control", False)
                or v.get("args", {}).get("tuning_method") in {"dt1d_ablation", "reviewer_routing"}
            }
            assert not forbidden, f"{path}: reviewer controls leaked into main comparison: {sorted(forbidden)}"
        else:
            ablation_rows += len(cfg["methods"])
            ablation_pairs.append((str(cfg["dataset"]), str(cfg["backbone"])))
            coverage = set(cfg.get("reviewer_coverage", {}))
            assert required_coverage <= coverage, f"{path}: missing coverage {sorted(required_coverage - coverage)}"
            for key, spec in cfg["methods"].items():
                if key != "dt1d":
                    assert spec.get("reviewer_control") is True, f"{path}:{key} must be reviewer_control"

        checked += validate_parser(cfg, path)

    assert proposal_keys == {"dt1d"}, proposal_keys
    assert len(set(ablation_pairs)) >= 2, "reviewer ablation must cover at least two dataset/backbone settings"
    assert len({b for _, b in ablation_pairs}) >= 2, "reviewer ablation must cover at least two backbones"

    dense_index = load(ROOT / "configs" / "dense" / "index.yaml")
    assert dense_index["schema_version"] == 2
    assert dense_index["proposal"]["method_key"] == "dt1d"
    from tools.run_dense_from_config import load_config as load_dense_config, build_args as build_dense_args, build_command as build_dense_command
    from dense_main import build_parser as build_dense_parser
    dense_checked = 0
    dense_dir = ROOT / "configs" / "dense" / "experiments"
    for dense_name in dense_index["experiments"]:
        dense_path = dense_dir / f"{dense_name}.yaml"
        dense_cfg = load_dense_config(dense_path)
        assert dense_cfg["experiment_id"] == dense_name
        assert "dt1d" in dense_cfg["methods"]
        assert dense_cfg["methods"]["dt1d"]["args"]["tuning_method"] == "dt1d"
        for method in dense_cfg["method_order"]:
            for seed in dense_cfg["seeds"]:
                dense_args = build_dense_args(dense_cfg, method, int(seed), smoke=False, num_workers=0, profile_latency=False)
                command = build_dense_command(dense_args, data_path=ROOT / "data", output_dir=ROOT / "outputs" / "_dense_config_validation", device="cpu")
                build_dense_parser().parse_args(command[2:])
                dense_checked += 1

    report = {
        "status": "PASS",
        "proposal": index["proposal"],
        "main_experiments": len(index["main_experiments"]),
        "reviewer_ablation_experiments": len(index["reviewer_ablations"]),
        "main_method_rows": main_rows,
        "ablation_method_rows": ablation_rows,
        "parser_validated_method_seed_configs": checked,
        "dense_experiments": len(dense_index["experiments"]),
        "dense_parser_validated_method_seed_configs": dense_checked,
        "ablation_dataset_backbone_pairs": ablation_pairs,
        "protocol": "one shared LR per method selected by mean validation across seeds; test once per seed",
    }
    out = ROOT / "reproducibility" / "config_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
