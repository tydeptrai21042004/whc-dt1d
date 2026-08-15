from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_index_matches_committed_self_contained_configs():
    root = ROOT / "configs/experiments"
    index = yaml.safe_load((root / "index.yaml").read_text())
    expected = set(index["main_experiments"] + index["reviewer_ablations"])
    actual = {p.stem for p in root.glob("*.yaml") if p.name != "index.yaml"}
    assert actual == expected
    assert not list(root.rglob("generated"))


def test_only_dt1d_is_marked_as_proposal_in_main_configs():
    index = yaml.safe_load((ROOT / "configs/experiments/index.yaml").read_text())
    assert index["proposal"] == {
        "method_key": "dt1d", "name": "DT1D-Adapter", "architecture": "R124-P2-G16-Axis-LearnedGate"
    }
    for name in index["main_experiments"]:
        cfg = yaml.safe_load((ROOT / "configs/experiments" / f"{name}.yaml").read_text())
        assert cfg["methods"]["dt1d"]["args"]["tuning_method"] == "dt1d"
        for method, spec in cfg["methods"].items():
            assert spec.get("reviewer_control", False) is False
            assert spec["args"].get("tuning_method") not in {"dt1d_ablation", "reviewer_routing"}
