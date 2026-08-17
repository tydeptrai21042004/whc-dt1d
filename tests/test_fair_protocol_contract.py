from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tools.run_experiment import FAIR_LOCKED_METHOD_KEYS, load_yaml, select_shared_lr

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"


def _rows(values: dict[float, list[float]]):
    rows = []
    for lr, per_seed in values.items():
        for seed, val in enumerate(per_seed):
            rows.append({
                "seed": seed,
                "lr": lr,
                "best_val_acc1": val,
                "best_epoch": seed + 1,
                "return_code": 0,
                "checkpoint": f"/tmp/{lr}/{seed}/checkpoint-best.pth",
                "output_dir": f"/tmp/{lr}/{seed}",
            })
    return rows


def test_shared_lr_is_selected_by_mean_validation_not_best_single_seed():
    rows = _rows({
        1e-3: [90.0, 90.0, 90.0],
        5e-3: [99.0, 80.0, 80.0],
    })
    selected = select_shared_lr(rows, seeds=[0, 1, 2], lr_candidates=[1e-3, 5e-3])
    assert selected["selected_lr"] == pytest.approx(1e-3)
    assert selected["selection_scope"] == "method_across_seeds"


def test_shared_lr_tie_break_prefers_lower_lr():
    rows = _rows({1e-3: [80.0, 81.0, 82.0], 5e-3: [81.0, 81.0, 81.0]})
    selected = select_shared_lr(rows, seeds=[0, 1, 2], lr_candidates=[1e-3, 5e-3])
    assert selected["selected_lr"] == pytest.approx(1e-3)


def test_shared_lr_rejects_missing_seed_lr_candidate():
    rows = _rows({1e-3: [80.0, 81.0, 82.0], 5e-3: [83.0, 84.0, 85.0]})[:-1]
    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        select_shared_lr(rows, seeds=[0, 1, 2], lr_candidates=[1e-3, 5e-3])


def test_shared_lr_rejects_failed_candidate_before_any_test_selection():
    rows = _rows({1e-3: [80.0, 81.0, 82.0], 5e-3: [83.0, 84.0, 85.0]})
    rows[-1]["return_code"] = 17
    with pytest.raises(RuntimeError, match="every method/seed/LR candidate"):
        select_shared_lr(rows, seeds=[0, 1, 2], lr_candidates=[1e-3, 5e-3])


def test_all_configs_lock_seed_policy_and_validation_only_selection():
    index = yaml.safe_load((CONFIG_DIR / "index.yaml").read_text())
    for name in index["main_experiments"] + index["reviewer_ablations"]:
        cfg = load_yaml(CONFIG_DIR / f"{name}.yaml")
        if cfg.get("seed_policy") == "single_seed_figure":
            assert len(cfg["seeds"]) == 1
        else:
            assert len(cfg["seeds"]) >= 3
        assert cfg["fairness"]["lr_selection_scope"] == "method_across_seeds"
        assert cfg["fairness"]["lr_aggregation"] == "mean_best_val_acc1"
        assert cfg["fairness"]["test_used_for_selection"] is False
        assert cfg["fairness"]["evaluate_test_once"] is True


def test_method_configs_cannot_override_locked_training_budget():
    index = yaml.safe_load((CONFIG_DIR / "index.yaml").read_text())
    for name in index["main_experiments"] + index["reviewer_ablations"]:
        cfg = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
        for method, spec in cfg["methods"].items():
            assert not (FAIR_LOCKED_METHOD_KEYS & set(spec.get("args", {}))), (name, method)


def test_configs_are_self_contained_and_do_not_reference_fragment_inheritance():
    forbidden_keys = {"include", "includes", "extends", "inherit", "inherits", "base_config", "preset_file"}
    index = yaml.safe_load((CONFIG_DIR / "index.yaml").read_text())
    for name in index["main_experiments"] + index["reviewer_ablations"]:
        cfg = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
        assert not (forbidden_keys & set(cfg)), name
        for required in ("seeds", "dataset", "backbone", "epochs", "batch_size", "fairness", "common_args", "methods", "method_order"):
            assert required in cfg, (name, required)


def test_main_experiments_have_exactly_one_proposal_key():
    index = yaml.safe_load((CONFIG_DIR / "index.yaml").read_text())
    for name in index["main_experiments"]:
        cfg = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
        proposal_rows = [
            key for key, spec in cfg["methods"].items()
            if key == "dt1d" or spec.get("proposal") is True
        ]
        assert proposal_rows == ["dt1d"], name
        assert not any(spec.get("reviewer_control") for spec in cfg["methods"].values())


def test_plan_declares_shared_method_lr_selection(tmp_path):
    out = tmp_path / "plan"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_experiment.py"),
            str(CONFIG_DIR / "table_05.yaml"),
            "--methods", "dt1d,linear",
            "--seeds", "0,1",
            "--plan-only",
            "--output-root", str(out),
            "--device", "cpu",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    plan = json.loads((out / "table_05" / "execution_plan.json").read_text())
    assert plan["lr_selection_scope"] == "method_across_seeds"
    assert plan["lr_aggregation"] == "mean_best_val_acc1"
    assert plan["test_runs"] == 4
    assert plan["validation_runs"] == 4 * len(plan["lr_candidates"])


def test_load_yaml_rejects_hidden_method_lr_override(tmp_path):
    cfg = yaml.safe_load((CONFIG_DIR / "table_05.yaml").read_text())
    cfg["methods"]["dt1d"]["args"]["lr"] = 0.123
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(SystemExit, match="locked fair-protocol keys"):
        load_yaml(path)


def test_two_seed_smoke_uses_one_shared_lr_and_test_once(tmp_path):
    out = tmp_path / "smoke"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_experiment.py"),
            str(CONFIG_DIR / "table_05.yaml"),
            "--methods", "dt1d",
            "--seeds", "0,1",
            "--smoke",
            "--output-root", str(out),
            "--data-path", str(tmp_path / "data"),
            "--device", "cpu",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout[-10000:]
    method_root = out / "table_05" / "dt1d"
    selection = json.loads((method_root / "lr_selection_summary.json").read_text())
    assert selection["seeds"] == [0, 1]
    assert selection["selection_scope"] == "method_across_seeds"
    shared_lr = float(selection["selected_lr"])
    for seed in (0, 1):
        test = json.loads((method_root / f"seed_{seed}" / "test_summary.json").read_text())
        per_seed_selection = json.loads((method_root / f"seed_{seed}" / "selection_summary.json").read_text())
        assert float(test["selected_lr"]) == pytest.approx(shared_lr)
        assert test["lr_selection_scope"] == "method_across_seeds"
        assert test["test_used_for_hyperparameter_selection"] is False
        assert per_seed_selection["test_used_for_selection"] is False
