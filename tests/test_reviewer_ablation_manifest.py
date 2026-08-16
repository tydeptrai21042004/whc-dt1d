from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "configs/experiments/ablation_dtd_resnet18.yaml",
    ROOT / "configs/experiments/ablation_flowers102_resnet50.yaml",
]
REQUIRED = {
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


def test_reviewer_ablation_is_two_dataset_two_backbone_three_seed():
    pairs = []
    for path in FILES:
        data = yaml.safe_load(path.read_text())
        assert data["kind"] == "ablation"
        assert data["seeds"] == [0, 1, 2]
        assert data["methods"]["dt1d"]["args"]["tuning_method"] == "dt1d"
        assert REQUIRED <= set(data["reviewer_coverage"])
        assert len(data["methods"]) == 12
        for name, spec in data["methods"].items():
            if name != "dt1d":
                assert spec.get("reviewer_control") is True
        pairs.append((data["dataset"], data["backbone"]))
    assert len(set(pairs)) == 2
    assert len({backbone for _, backbone in pairs}) == 2


def test_ablation_fairness_is_matched_within_each_experiment():
    for path in FILES:
        data = yaml.safe_load(path.read_text())
        fair = data["fairness"]
        assert fair["same_lr_grid_for_all_methods"] is True
        assert fair["same_epoch_budget"] is True
        assert fair["same_batch_size_within_experiment"] is True
        assert fair["same_seed_and_split_within_seed"] is True
        assert fair["same_preprocessing"] is True
        assert fair["evaluate_test_once"] is True
        assert fair["test_used_for_selection"] is False
