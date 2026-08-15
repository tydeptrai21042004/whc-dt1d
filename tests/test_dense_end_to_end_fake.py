from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "task,pipeline,dataset,extra",
    [
        ("binary_segmentation", "deeplab_mobilenet_v3", "fake_binary", []),
        ("binary_segmentation", "vit_b16_dense", "fake_binary", ["--vit_variant", "tiny"]),
        ("detection", "fasterrcnn_mobilenet_v3_fpn", "fake_detection", ["--detector_variant", "tiny"]),
    ],
)
def test_dense_pipeline_completes_train_val_test(tmp_path, task, pipeline, dataset, extra):
    output = tmp_path / pipeline
    command = [
        sys.executable,
        str(ROOT / "dense_main.py"),
        "--task", task,
        "--pipeline", pipeline,
        "--dataset", dataset,
        "--tuning_method", "dt1d",
        "--pretrained", "false",
        "--input_size", "64",
        "--epochs", "1",
        "--batch_size", "1" if task == "detection" else "2",
        "--fake_train_size", "2" if task == "detection" else "4",
        "--fake_val_size", "1" if task == "detection" else "2",
        "--fake_test_size", "1" if task == "detection" else "2",
        "--num_workers", "0",
        "--pin_mem", "false",
        "--device", "cpu",
        "--use_amp", "false",
        "--profile_latency", "false",
        "--save_checkpoint", "false",
        "--output_dir", str(output),
        *extra,
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-10000:]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["best_epoch"] == 0
    assert summary["checkpoint_selection_rule"].startswith("highest validation")
    assert summary["dataset_sizes"]["train"] > 0
    assert summary["test_metrics"]
