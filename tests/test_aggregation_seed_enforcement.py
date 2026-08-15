from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_run(root: Path, seed: int) -> None:
    run = root / "table_x" / "fake" / "resnet18" / "dt1d" / f"seed_{seed}"
    run.mkdir(parents=True)
    (run / "run_metadata.json").write_text(json.dumps({
        "target": "table_x",
        "kind": "comparison",
        "method_preset": "dt1d",
        "method_label": "DT1D-Adapter",
        "variant": None,
        "independent_seed": seed,
    }))
    (run / "args.json").write_text(json.dumps({
        "dataset": "fake", "backbone": "resnet18", "epochs": 1, "batch_size": 4,
    }))
    (run / "run_status.json").write_text(json.dumps({"return_code": 0}))
    (run / "test_summary.json").write_text(json.dumps({"acc1": 50.0 + seed, "loss": 1.0}))
    (run / "convergence_summary.json").write_text(json.dumps({
        "n_trainable_parameters": 100, "n_total_parameters": 1000,
    }))


def call(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable,
        str(ROOT / "tools/aggregate_cnn_paper.py"),
        "--root", str(root),
        "--target", "table_x",
        "--require-seeds", "0,1,2",
        *extra,
    ], text=True, capture_output=True)


def test_aggregator_fails_on_missing_seed_and_passes_when_complete(tmp_path):
    write_run(tmp_path, 0)
    incomplete = call(tmp_path)
    assert incomplete.returncode == 2
    partial = call(tmp_path, "--allow-incomplete")
    assert partial.returncode == 0
    write_run(tmp_path, 1)
    write_run(tmp_path, 2)
    complete = call(tmp_path)
    assert complete.returncode == 0, complete.stderr + complete.stdout
    summary = json.loads((tmp_path / "aggregated/table_x/aggregation_summary.json").read_text())
    assert summary["all_groups_complete"] is True
    assert summary["successful_run_count"] == 3
