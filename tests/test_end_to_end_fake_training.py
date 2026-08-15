from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "method_args",
    [
        ["--tuning_method", "dt1d"],
        ["--tuning_method", "plain_axial", "--axial_kernel_mode", "unrestricted"],
    ],
)
def test_fake_data_training_validation_and_test_complete(tmp_path, method_args):
    output = tmp_path / method_args[1]
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--dataset", "fake",
        "--backbone", "resnet18",
        "--weights", "none",
        "--pretrained", "False",
        "--device", "cpu",
        "--use_amp", "False",
        "--input_size", "64",
        "--epochs", "1",
        "--batch_size", "4",
        "--fake_train_size", "8",
        "--fake_val_size", "4",
        "--fake_test_size", "4",
        "--fake_num_classes", "5",
        "--num_workers", "0",
        "--pin_mem", "False",
        "--deterministic", "True",
        "--save_ckpt", "False",
        "--save_history", "True",
        "--final_test", "True",
        "--profile_efficiency", "False",
        "--measure_eval_latency", "False",
        "--output_dir", str(output),
        *method_args,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout[-8000:]
    convergence = json.loads((output / "convergence_summary.json").read_text())
    test = json.loads((output / "test_summary.json").read_text())
    assert convergence["best_epoch"] == 0
    assert "best_val_acc1" in convergence
    assert "test_acc1_at_best_val" in test
