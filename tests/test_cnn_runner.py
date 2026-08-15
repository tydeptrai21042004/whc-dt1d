from __future__ import annotations

import json
from pathlib import Path
import yaml

from tools.run_cnn_paper import parse_names
from tools.run_experiment import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_runner_uses_one_committed_config_per_experiment_and_no_generated_yaml(tmp_path):
    names = parse_names("all")
    index = yaml.safe_load((ROOT / "configs/experiments/index.yaml").read_text())
    assert names == index["main_experiments"]
    for name in names:
        cfg = load_yaml(ROOT / "configs/experiments" / f"{name}.yaml")
        assert cfg["experiment_id"] == name
        assert cfg["seeds"] == [0, 1, 2]
    assert not list((ROOT / "configs").rglob("generated"))


def test_plan_only_reports_validation_grid_and_zero_generated_yaml(tmp_path):
    import subprocess, sys
    out = tmp_path / "out"
    command = [
        sys.executable, str(ROOT / "tools/run_experiment.py"),
        str(ROOT / "configs/experiments/table_05.yaml"),
        "--methods", "dt1d,linear", "--seeds", "0", "--plan-only",
        "--output-root", str(out), "--device", "cpu",
    ]
    done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    plan = json.loads((out / "table_05/execution_plan.json").read_text())
    assert plan["generated_yaml_configs"] == 0
    assert plan["test_policy"] == "once per seed after shared method-level validation-only LR selection"
    assert plan["lr_selection_scope"] == "method_across_seeds"
    assert plan["lr_aggregation"] == "mean_best_val_acc1"
    assert plan["validation_runs"] == 2 * len(plan["lr_candidates"])
    assert plan["test_runs"] == 2
