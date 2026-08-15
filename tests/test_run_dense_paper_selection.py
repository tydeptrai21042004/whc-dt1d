from __future__ import annotations

import pytest

from tools.run_dense_paper import load_config, load_index, parse_csv_ints, select_methods, select_targets


def test_dense_runner_defaults_and_csv_selection():
    index = load_index()
    assert parse_csv_ints("0,1,2") == [0, 1, 2]
    assert select_targets("binary_deeplab_pennfudan,detection_pet_fasterrcnn", index) == [
        "binary_deeplab_pennfudan", "detection_pet_fasterrcnn"
    ]
    target = load_config("binary_vit_drive")
    assert select_methods(target, "dt1d,linear") == ["dt1d", "linear"]


def test_dense_runner_rejects_duplicate_seeds_and_invalid_method():
    with pytest.raises(Exception):
        parse_csv_ints("0,0,1")
    target = load_config("binary_vit_drive")
    with pytest.raises(SystemExit):
        select_methods(target, "bam")
