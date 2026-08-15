from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DENSE_DIR = ROOT / "configs" / "dense" / "experiments"


def test_dense_index_contains_all_manuscript_pipelines_and_datasets():
    index = yaml.safe_load((ROOT / "configs/dense/index.yaml").read_text())
    assert index["schema_version"] == 2
    assert index["proposal"]["method_key"] == "dt1d"
    assert set(index["experiments"]) == {
        "binary_deeplab_pennfudan",
        "binary_deeplab_drive",
        "binary_vit_pennfudan",
        "binary_vit_drive",
        "semantic_pet_deeplab",
        "detection_pet_fasterrcnn",
    }
    targets = {
        name: yaml.safe_load((DENSE_DIR / f"{name}.yaml").read_text())
        for name in index["experiments"]
    }
    assert targets["binary_deeplab_pennfudan"]["dataset"] == "pennfudan"
    assert targets["binary_deeplab_drive"]["dataset"] == "drive"
    assert targets["semantic_pet_deeplab"]["dataset"] == "oxford_pet_segmentation"
    assert targets["detection_pet_fasterrcnn"]["dataset"] == "oxford_pet_detection"
    assert targets["detection_pet_fasterrcnn"]["common_args"]["max_test_samples"] == 80
    for target in targets.values():
        assert "dt1d" in target["methods"]
        proposal_rows = [key for key in target["methods"] if key == "dt1d"]
        assert proposal_rows == ["dt1d"]
