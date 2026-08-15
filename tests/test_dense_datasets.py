from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dense_prediction.datasets import (
    DriveBinaryDataset,
    OxfordPetDetectionDataset,
    PennFudanBinaryDataset,
)
from dense_prediction.transforms import SegmentationTransform


def save_rgb(path: Path, size=(24, 20)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), 127, dtype=np.uint8)).save(path)


def save_mask(path: Path, size=(24, 20)):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[4:15, 5:18] = 255
    Image.fromarray(mask).save(path)


def test_pennfudan_binary_dataset_pairs_and_binarizes(tmp_path):
    root = tmp_path / "PennFudanPed"
    save_rgb(root / "PNGImages" / "FudanPed00001.png")
    save_mask(root / "PedMasks" / "FudanPed00001.png")
    dataset = PennFudanBinaryDataset(root, SegmentationTransform(size=32, train=False))
    image, mask = dataset[0]
    assert image.shape == (3, 32, 32)
    assert mask.shape == (32, 32)
    assert set(mask.unique().tolist()) <= {0, 1}


def test_drive_dataset_pairs_numeric_prefix(tmp_path):
    root = tmp_path / "DRIVE"
    save_rgb(root / "training" / "images" / "21_training.tif")
    save_mask(root / "training" / "1st_manual" / "21_manual1.gif")
    dataset = DriveBinaryDataset(root, "training", SegmentationTransform(size=32, train=False))
    image, mask = dataset[0]
    assert image.shape == (3, 32, 32)
    assert mask.sum() > 0


def test_oxford_pet_detection_target_from_trimap():
    trimap = torch.full((20, 24), 2, dtype=torch.long)
    trimap[3:18, 4:20] = 1
    trimap[2:19, 3] = 3
    target = OxfordPetDetectionDataset.target_from_trimap(trimap, 7)
    assert target["labels"].tolist() == [1]
    assert target["image_id"].tolist() == [7]
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    assert (x1, y1, x2, y2) == (3.0, 2.0, 20.0, 19.0)
    assert target["area"].item() > 0


def test_dense_sample_limit_is_deterministic():
    from types import SimpleNamespace
    from dense_prediction.datasets import build_dense_datasets

    args = SimpleNamespace(
        dataset="fake_detection", task="detection", input_size=32, seed=4,
        data_path=".", download=False, val_fraction=0.2, test_fraction=0.2,
        fake_train_size=10, fake_val_size=6, fake_test_size=9, num_classes=2,
        max_train_samples=3, max_val_samples=2, max_test_samples=4,
    )
    train1, val1, test1 = build_dense_datasets(args)
    train2, val2, test2 = build_dense_datasets(args)
    assert (len(train1), len(val1), len(test1)) == (3, 2, 4)
    assert train1.indices == train2.indices
    assert val1.indices == val2.indices
    assert test1.indices == test2.indices
