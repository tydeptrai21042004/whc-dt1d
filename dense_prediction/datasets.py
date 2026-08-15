from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import OxfordIIITPet

from .transforms import DetectionTransform, SegmentationTransform

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}


def _files(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def _find_child(root: Path, names: Sequence[str]) -> Path:
    root = root.expanduser().resolve()
    candidates = [root / name for name in names]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    lower = {name.lower() for name in names}
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() in lower:
            return candidate
    raise FileNotFoundError(f"Could not find any of {list(names)} below {root}")


def _stable_seed(seed: int, index: int) -> int:
    token = f"{int(seed)}:{int(index)}".encode("utf8")
    return int(hashlib.sha256(token).hexdigest()[:8], 16)


class IndexView(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class PennFudanBinaryDataset(Dataset):
    """Penn-Fudan pedestrians converted from instance masks to one binary mask."""

    def __init__(self, root: str | Path, transform: SegmentationTransform | None = None) -> None:
        root = Path(root)
        image_dir = _find_child(root, ("PNGImages", "images"))
        mask_dir = _find_child(root, ("PedMasks", "masks"))
        images = {p.stem: p for p in _files(image_dir)}
        masks = {p.stem: p for p in _files(mask_dir)}
        keys = sorted(set(images).intersection(masks))
        if not keys:
            raise RuntimeError(f"No matched PennFudan images/masks in {root}")
        self.samples = [(images[key], masks[key]) for key in keys]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        elif isinstance(mask, Image.Image):
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask = (mask > 0).long()
        return image, mask


class DriveBinaryDataset(Dataset):
    """DRIVE retinal-vessel dataset with robust official-folder discovery."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: SegmentationTransform | None = None,
    ) -> None:
        root = Path(root)
        split_aliases = ("training", "train") if split.lower() in {"training", "train"} else ("test", "testing")
        split_dir = _find_child(root, split_aliases)
        image_dir = _find_child(split_dir, ("images", "image"))
        mask_dir = _find_child(split_dir, ("1st_manual", "manual", "masks", "mask"))

        def key(path: Path) -> str:
            return path.stem.split("_")[0].lower()

        images = {key(p): p for p in _files(image_dir)}
        masks = {key(p): p for p in _files(mask_dir)}
        keys = sorted(set(images).intersection(masks))
        if not keys:
            raise RuntimeError(f"No matched DRIVE images/masks in {split_dir}")
        self.samples = [(images[k], masks[k]) for k in keys]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        elif isinstance(mask, Image.Image):
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, (mask > 0).long()


class OxfordPetSegmentationDataset(Dataset):
    """Oxford-IIIT Pet trimaps mapped from labels 1..3 to class indices 0..2."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: SegmentationTransform | None,
        download: bool = False,
    ) -> None:
        self.base = OxfordIIITPet(
            root=str(root), split=split, target_types="segmentation", download=bool(download)
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, trimap = self.base[index]
        if self.transform is not None:
            image, trimap = self.transform(image, trimap)
        else:
            trimap = torch.as_tensor(np.array(trimap), dtype=torch.long)
        target = (trimap.long() - 1).clamp_(0, 2)
        return image, target


class OxfordPetDetectionDataset(Dataset):
    """One-class pet detection targets derived deterministically from trimaps."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: DetectionTransform | None,
        download: bool = False,
    ) -> None:
        self.base = OxfordIIITPet(
            root=str(root), split=split, target_types="segmentation", download=bool(download)
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def target_from_trimap(trimap: Image.Image | torch.Tensor, image_id: int) -> Dict[str, torch.Tensor]:
        if isinstance(trimap, Image.Image):
            trimap = torch.as_tensor(np.array(trimap), dtype=torch.long)
        else:
            trimap = trimap.long()
        # Oxford trimap: 1=pet, 2=background, 3=border. Include border in the pet box.
        foreground = trimap != 2
        ys, xs = torch.where(foreground)
        if xs.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            x1, x2 = xs.min().float(), xs.max().float() + 1.0
            y1, y2 = ys.min().float(), ys.max().float() + 1.0
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            labels = torch.ones((1,), dtype=torch.int64)
        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
            "area": ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).to(torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

    def __getitem__(self, index: int):
        image, trimap = self.base[index]
        target = self.target_from_trimap(trimap, index)
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target


class FakeSegmentationDataset(Dataset):
    def __init__(self, size: int, image_size: int, num_classes: int, seed: int, binary: bool) -> None:
        self.size = int(size)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.binary = bool(binary)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(_stable_seed(self.seed, index))
        image = torch.rand((3, self.image_size, self.image_size), generator=generator)
        if self.binary:
            mask = torch.zeros((self.image_size, self.image_size), dtype=torch.long)
            lo = max(1, self.image_size // 4)
            hi = max(lo + 1, 3 * self.image_size // 4)
            mask[lo:hi, lo:hi] = 1
        else:
            mask = torch.randint(
                0, self.num_classes, (self.image_size, self.image_size), generator=generator
            )
        return image, mask


class FakeDetectionDataset(Dataset):
    def __init__(self, size: int, image_size: int, seed: int) -> None:
        self.size = int(size)
        self.image_size = int(image_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(_stable_seed(self.seed, index))
        image = torch.rand((3, self.image_size, self.image_size), generator=generator)
        q = self.image_size // 4
        boxes = torch.tensor([[q, q, self.image_size - q, self.image_size - q]], dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones((1,), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([(self.image_size - 2 * q) ** 2], dtype=torch.float32),
            "iscrowd": torch.zeros((1,), dtype=torch.int64),
        }
        return image, target


def _split_indices(length: int, seed: int, val_fraction: float, test_fraction: float = 0.0):
    if length < 3 and test_fraction > 0:
        raise ValueError("At least three samples are required for train/val/test splitting")
    indices = list(range(length))
    random.Random(int(seed)).shuffle(indices)
    n_test = int(round(length * test_fraction)) if test_fraction > 0 else 0
    n_val = max(1, int(round(length * val_fraction))) if length > 1 else 0
    if n_test > 0:
        n_test = max(1, n_test)
    while n_val + n_test >= length and n_val > 0:
        n_val -= 1
    test = indices[:n_test]
    val = indices[n_test : n_test + n_val]
    train = indices[n_test + n_val :]
    return train, val, test



def _limit_dataset(dataset: Dataset, maximum: int, seed: int) -> Dataset:
    maximum = int(maximum or 0)
    if maximum <= 0 or maximum >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(int(seed)).shuffle(indices)
    return IndexView(dataset, indices[:maximum])


def _build_dense_datasets_unlimited(args):
    dataset = str(args.dataset).lower()
    task = str(args.task).lower()
    size = int(args.input_size)
    train_seg = SegmentationTransform(size=size, train=True)
    eval_seg = SegmentationTransform(size=size, train=False)
    train_det = DetectionTransform(size=size, train=True)
    eval_det = DetectionTransform(size=size, train=False)
    root = Path(args.data_path)
    download = bool(getattr(args, "download", False))
    seed = int(args.seed)

    if dataset == "fake_binary":
        return (
            FakeSegmentationDataset(args.fake_train_size, size, 2, seed, True),
            FakeSegmentationDataset(args.fake_val_size, size, 2, seed + 1000, True),
            FakeSegmentationDataset(args.fake_test_size, size, 2, seed + 2000, True),
        )
    if dataset == "fake_semantic":
        return (
            FakeSegmentationDataset(args.fake_train_size, size, args.num_classes, seed, False),
            FakeSegmentationDataset(args.fake_val_size, size, args.num_classes, seed + 1000, False),
            FakeSegmentationDataset(args.fake_test_size, size, args.num_classes, seed + 2000, False),
        )
    if dataset == "fake_detection":
        return (
            FakeDetectionDataset(args.fake_train_size, size, seed),
            FakeDetectionDataset(args.fake_val_size, size, seed + 1000),
            FakeDetectionDataset(args.fake_test_size, size, seed + 2000),
        )

    if dataset == "pennfudan":
        base = PennFudanBinaryDataset(root, transform=None)
        train_idx, val_idx, test_idx = _split_indices(
            len(base), seed, args.val_fraction, args.test_fraction
        )
        train_base = PennFudanBinaryDataset(root, transform=train_seg)
        eval_base = PennFudanBinaryDataset(root, transform=eval_seg)
        return IndexView(train_base, train_idx), IndexView(eval_base, val_idx), IndexView(eval_base, test_idx)

    if dataset == "drive":
        train_raw = DriveBinaryDataset(root, "training", transform=None)
        train_idx, val_idx, _ = _split_indices(len(train_raw), seed, args.val_fraction)
        train_base = DriveBinaryDataset(root, "training", transform=train_seg)
        val_base = DriveBinaryDataset(root, "training", transform=eval_seg)
        test_base = DriveBinaryDataset(root, "test", transform=eval_seg)
        return IndexView(train_base, train_idx), IndexView(val_base, val_idx), test_base

    if dataset == "oxford_pet_segmentation":
        trainval_raw = OxfordPetSegmentationDataset(root, "trainval", None, download=download)
        train_idx, val_idx, _ = _split_indices(len(trainval_raw), seed, args.val_fraction)
        train_base = OxfordPetSegmentationDataset(root, "trainval", train_seg, download=False)
        val_base = OxfordPetSegmentationDataset(root, "trainval", eval_seg, download=False)
        test_base = OxfordPetSegmentationDataset(root, "test", eval_seg, download=download)
        return IndexView(train_base, train_idx), IndexView(val_base, val_idx), test_base

    if dataset == "oxford_pet_detection":
        trainval_raw = OxfordPetDetectionDataset(root, "trainval", None, download=download)
        train_idx, val_idx, _ = _split_indices(len(trainval_raw), seed, args.val_fraction)
        train_base = OxfordPetDetectionDataset(root, "trainval", train_det, download=False)
        val_base = OxfordPetDetectionDataset(root, "trainval", eval_det, download=False)
        test_base = OxfordPetDetectionDataset(root, "test", eval_det, download=download)
        return IndexView(train_base, train_idx), IndexView(val_base, val_idx), test_base

    raise ValueError(f"Unsupported dense dataset: {dataset}")



def build_dense_datasets(args):
    train, val, test = _build_dense_datasets_unlimited(args)
    train = _limit_dataset(train, getattr(args, "max_train_samples", 0), int(args.seed) + 11)
    val = _limit_dataset(val, getattr(args, "max_val_samples", 0), int(args.seed) + 23)
    test = _limit_dataset(test, getattr(args, "max_test_samples", 0), int(args.seed) + 37)
    return train, val, test

def detection_collate_fn(batch):
    return tuple(zip(*batch))


__all__ = [
    "PennFudanBinaryDataset",
    "DriveBinaryDataset",
    "OxfordPetSegmentationDataset",
    "OxfordPetDetectionDataset",
    "FakeSegmentationDataset",
    "FakeDetectionDataset",
    "build_dense_datasets",
    "detection_collate_fn",
]
