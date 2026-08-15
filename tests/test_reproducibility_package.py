from __future__ import annotations

import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_required_single_proposal_release_files_exist():
    required = [
        "requirements.txt", "environment.yml", "CITATION.cff", "README.md",
        "REPRODUCIBILITY.md", "models/dt1d_adapter.py",
        "models/dt1d_ablation_adapter.py",
        "tools/run_experiment.py", "tools/run_cnn_paper.py",
        "tools/validate_dt1d.py", "tools/validate_all_configs.py",
        "configs/experiments/index.yaml",
        "configs/experiments/ablation_dtd_resnet18.yaml",
        "configs/experiments/ablation_flowers102_resnet50.yaml",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative


def test_every_main_experiment_is_one_self_contained_yaml():
    index = yaml.safe_load((ROOT / "configs/experiments/index.yaml").read_text())
    assert index["proposal"]["method_key"] == "dt1d"
    assert index["proposal"]["name"] == "DT1D-Adapter"
    for name in index["main_experiments"]:
        path = ROOT / "configs/experiments" / f"{name}.yaml"
        data = yaml.safe_load(path.read_text())
        for key in ["seeds", "dataset", "backbone", "epochs", "batch_size", "fairness", "common_args", "methods", "method_order"]:
            assert key in data, (name, key)
        assert "dt1d" in data["methods"]
        assert data["methods"]["dt1d"]["args"]["tuning_method"] == "dt1d"


def test_caltech_splits_are_exact_disjoint_partitions():
    for seed in (0, 1, 2):
        split = json.loads((ROOT / f"splits/caltech101/seed{seed}_holdout20.json").read_text())
        train, val, test = split["train_indices"], split["val_indices"], split["test_indices"]
        assert len(train) == 6942 and len(val) == 868 and len(test) == 867
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)
        assert len(set(train + val + test)) == split["dataset_length"] == 8677


def test_release_metadata_versions_match():
    version = (ROOT / "VERSION").read_text().strip()
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    codemeta = json.loads((ROOT / "codemeta.json").read_text())
    zenodo = json.loads((ROOT / ".zenodo.json").read_text())
    environment_name = yaml.safe_load((ROOT / "environment.yml").read_text())["name"]
    assert citation["version"] == version
    assert codemeta["version"] == version
    assert zenodo["version"] == version
    assert environment_name.endswith(version)


def test_legacy_proposal_artifacts_are_absent():
    forbidden = [
        "models/legacy_dt1d_adapter.py",
        "models/whc_compact_dt1d_adapter.py",
        "models/whc_final_dt1d_adapter.py",
        "models/whc_tau3_dt1d_adapter.py",
        "tools/run_experiments_fastest_first.py",
    ]
    for relative in forbidden:
        assert not (ROOT / relative).exists(), relative
