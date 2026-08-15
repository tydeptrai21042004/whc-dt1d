from __future__ import annotations

from pathlib import Path
import yaml

from tools import benchmark_reviewer_efficiency as benchmark

ROOT = Path(__file__).resolve().parents[1]

# Snapshot of the baseline rows that were present before legacy-proposal cleanup.
# DT1D itself is included so the exact method order of every manuscript config is
# protected against accidental cleanup edits.
EXPECTED_METHODS = {
    "table_03": ["linear", "bitfit", "ssf", "dt1d", "residual", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_04": ["linear", "bitfit", "ssf", "dt1d", "residual", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_05": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_06": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_07": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_08": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_09": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_10": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_11": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_12": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_13": ["linear", "bitfit", "ssf", "dt1d", "conv_r8", "conv_r6", "conv_r4", "conv_r2", "full"],
    "table_14_15": ["linear", "bitfit", "ssf", "dt1d", "bam", "lora_conv", "residual", "sidetune", "conv_r4", "full"],
    "table_18_19": ["linear", "dt1d", "bam", "conv_r4", "full"],
    "figure_01": ["dt1d"],
    "figure_04": ["linear", "bitfit", "ssf", "dt1d", "bam", "lora_conv", "residual", "sidetune", "conv_r4", "full"],
}


def test_each_main_experiment_keeps_exact_baseline_inventory_and_order():
    index = yaml.safe_load((ROOT / "configs/experiments/index.yaml").read_text())
    assert set(index["main_experiments"]) == set(EXPECTED_METHODS)
    for name, expected in EXPECTED_METHODS.items():
        cfg = yaml.safe_load((ROOT / "configs/experiments" / f"{name}.yaml").read_text())
        assert list(cfg["methods"]) == expected
        assert cfg["method_order"] == expected


def test_efficiency_registry_preserves_core_baselines():
    expected = {"linear", "full", "conv", "residual", "bam", "ssf", "bitfit", "lora_conv"}
    assert expected <= set(benchmark.METHOD_ARGS)


def test_baseline_module_files_still_exist():
    required = [
        "models/tuning_modules/bam_adapter.py",
        "models/tuning_modules/conv_adapter.py",
        "models/tuning_modules/lora_conv.py",
        "models/tuning_modules/residual_adapter.py",
        "models/tuning_modules/side_tuning.py",
        "models/tuning_modules/ssf.py",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative
