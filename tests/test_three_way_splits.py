from __future__ import annotations

import json
from types import SimpleNamespace

from datasets.build import _split_three_way_subset


def test_generated_three_way_split_is_disjoint_complete_and_repeatable(tmp_path):
    base = list(range(101))
    args = SimpleNamespace(
        seed=2,
        val_ratio=0.1,
        test_ratio=0.1,
        split_file=None,
        output_dir=str(tmp_path),
        dataset="dummy",
    )
    train = _split_three_way_subset(base, "train", args=args)
    val = _split_three_way_subset(base, "val", args=args)
    test = _split_three_way_subset(base, "test", args=args)
    train_ids, val_ids, test_ids = set(train.indices), set(val.indices), set(test.indices)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(range(101))
    manifest = json.loads((tmp_path / "split_manifest_used.json").read_text())
    assert manifest["schema_version"] == 3
    assert manifest["test_indices"] == list(test.indices)


def test_committed_three_way_split_is_authoritative(tmp_path):
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(json.dumps({
        "dataset_length": 10,
        "train_indices": [4, 5, 6, 7, 8, 9],
        "val_indices": [2, 3],
        "test_indices": [0, 1],
    }))
    args = SimpleNamespace(
        seed=0,
        val_ratio=0.2,
        test_ratio=0.2,
        split_file=str(manifest_path),
        output_dir=None,
        dataset="dummy",
    )
    assert _split_three_way_subset(list(range(10)), "train", args=args).indices == [4, 5, 6, 7, 8, 9]
    assert _split_three_way_subset(list(range(10)), "val", args=args).indices == [2, 3]
    assert _split_three_way_subset(list(range(10)), "test", args=args).indices == [0, 1]
