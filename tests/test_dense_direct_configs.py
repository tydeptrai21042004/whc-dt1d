from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from dense_main import build_parser
from tools.run_dense_from_config import build_args, build_command, load_config

ROOT = Path(__file__).resolve().parents[1]
DENSE_DIR = ROOT / "configs" / "dense" / "experiments"


def test_dense_experiments_are_committed_self_contained_and_three_seed_complete(tmp_path):
    index = yaml.safe_load((ROOT / "configs/dense/index.yaml").read_text())
    groups = defaultdict(set)
    run_count = 0
    for target_name in index["experiments"]:
        path = DENSE_DIR / f"{target_name}.yaml"
        cfg = load_config(path)
        assert cfg["experiment_id"] == target_name
        assert cfg["seeds"] == [0, 1, 2]
        assert cfg["method_order"] == list(cfg["methods"])
        for method in cfg["method_order"]:
            for seed in cfg["seeds"]:
                args = build_args(cfg, method, seed, smoke=False, num_workers=0, profile_latency=False)
                command = build_command(args, data_path=ROOT / "data", output_dir=tmp_path / "out", device="cpu")
                parsed = build_parser().parse_args(command[2:])
                assert parsed.seed == seed
                assert parsed.tuning_method == cfg["methods"][method]["args"]["tuning_method"]
                groups[(target_name, method)].add(seed)
                run_count += 1
    assert run_count == 108
    assert len(groups) == 36
    assert all(seeds == {0, 1, 2} for seeds in groups.values())
    assert not list((ROOT / "configs/dense").rglob("generated"))


def test_dense_configs_do_not_use_fragment_inheritance():
    forbidden = {"include", "includes", "extends", "inherit", "inherits", "base_config", "preset_file"}
    index = yaml.safe_load((ROOT / "configs/dense/index.yaml").read_text())
    for target_name in index["experiments"]:
        cfg = yaml.safe_load((DENSE_DIR / f"{target_name}.yaml").read_text())
        assert not (forbidden & set(cfg)), target_name
        assert cfg["fairness"]["test_used_for_selection"] is False
        assert cfg["fairness"]["evaluate_test_once"] is True


def test_dense_orchestrator_smoke_runs_committed_config_without_generated_yaml(tmp_path):
    import json
    import subprocess
    import sys

    output_root = tmp_path / "dense_orchestrator"
    command = [
        sys.executable,
        str(ROOT / "tools" / "run_dense_paper.py"),
        "--target", "binary_vit_drive",
        "--methods", "dt1d",
        "--seeds", "0",
        "--smoke",
        "--max-runs", "1",
        "--output-root", str(output_root),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr

    plan = json.loads((output_root / "execution_plan.json").read_text())
    assert plan["generated_yaml_configs"] == 0
    assert plan["run_count"] == 1
    assert plan["runs"][0]["config"].endswith("configs/dense/experiments/binary_vit_drive.yaml")

    run_dir = output_root / "binary_vit_drive" / "dt1d" / "seed_0"
    assert (run_dir / "summary.json").is_file()
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    assert resolved["source_config"] == "configs/dense/experiments/binary_vit_drive.yaml"
    assert resolved["proposal"] is True
    assert resolved["fairness"]["test_used_for_selection"] is False
