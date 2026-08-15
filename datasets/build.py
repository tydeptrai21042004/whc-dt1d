# datasets/build.py
"""
Dataset router for classification experiments.

Revision fixes:
1. Adds build_dataset_split(args, split) with explicit train/val/test semantics.
2. Adds ImageNet normalization for pretrained backbones.
3. Provides deterministic validation splits when torchvision has only train/test.
4. Keeps build_dataset(args, is_train) backward-compatible.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Tuple

import torch
import torchvision.transforms as T
from torchvision import datasets
from torchvision.datasets import CocoDetection


def _get_bool(args, name: str, default: bool) -> bool:
    return bool(getattr(args, name, default))


def _img_transforms(args, is_train: bool):
    size = int(getattr(args, "input_size", 224))
    train_aug = str(getattr(args, "train_aug", "standard")).lower()
    use_norm = _get_bool(args, "imagenet_norm", _get_bool(args, "imagenet_default_mean_and_std", True))

    interpolation = T.InterpolationMode.BICUBIC
    tfms = []
    if is_train and train_aug == "standard":
        tfms.extend([
            T.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=interpolation),
            T.RandomHorizontalFlip(),
        ])
    else:
        tfms.append(T.Resize((size, size), interpolation=interpolation))

    tfms.extend([
        T.ToTensor(),
        T.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
    ])

    if use_norm:
        tfms.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))

    return T.Compose(tfms)


def _write_generated_split_manifest(
    args,
    dataset_length: int,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int] | None = None,
) -> None:
    """Record the exact generated partition once per run."""
    if args is None:
        return
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return
    path = Path(output_dir) / "split_manifest_used.json"
    if path.exists():
        return
    payload = {
        "schema_version": 3 if test_idx is not None else 2,
        "dataset": str(getattr(args, "dataset", "unknown")),
        "dataset_length": int(dataset_length),
        "seed": int(getattr(args, "seed", 0)),
        "val_ratio": float(getattr(args, "val_ratio", 0.2)),
        "algorithm": "torch.randperm with a per-run torch.Generator seeded by --seed",
        "train_indices": [int(v) for v in train_idx],
        "val_indices": [int(v) for v in val_idx],
    }
    if test_idx is not None:
        payload["test_ratio"] = float(getattr(args, "test_ratio", 0.1))
        payload["test_indices"] = [int(v) for v in test_idx]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _split_subset(base, split: str, args=None, val_ratio: float = 0.2, seed: int = 42):
    """Return an exact or deterministically generated train/validation subset.

    When ``args.split_file`` is set, the JSON index manifest is authoritative.
    Otherwise the historical paper split is regenerated with ``torch.randperm``.
    """
    if split not in ("train", "val"):
        raise ValueError("_split_subset only supports train/val")

    if args is not None:
        seed = int(getattr(args, "seed", seed))
        val_ratio = float(getattr(args, "val_ratio", val_ratio))
        split_file = getattr(args, "split_file", None)
    else:
        split_file = None

    if split_file:
        path = Path(split_file).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_length = int(manifest["dataset_length"])
        if len(base) != expected_length:
            raise ValueError(
                f"Split file {path} expects dataset_length={expected_length}, "
                f"but the loaded dataset has {len(base)} items."
            )
        train_idx = [int(v) for v in manifest["train_indices"]]
        val_idx = [int(v) for v in manifest["val_indices"]]
        combined = train_idx + val_idx
        if len(combined) != len(base) or len(set(combined)) != len(base):
            raise ValueError(f"Split file {path} is not a disjoint full partition.")
        if min(combined, default=0) < 0 or max(combined, default=-1) >= len(base):
            raise ValueError(f"Split file {path} contains out-of-range indices.")
    else:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(base), generator=g).tolist()
        n_val = max(1, int(round(val_ratio * len(idx))))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        _write_generated_split_manifest(args, len(base), train_idx, val_idx)

    return torch.utils.data.Subset(base, train_idx if split == "train" else val_idx)


def _split_three_way_subset(base, split: str, args=None, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """Return disjoint train/validation/test subsets for datasets without official splits."""
    if split not in ("train", "val", "test"):
        raise ValueError("_split_three_way_subset supports train/val/test")

    if args is not None:
        seed = int(getattr(args, "seed", seed))
        val_ratio = float(getattr(args, "val_ratio", val_ratio))
        test_ratio = float(getattr(args, "test_ratio", test_ratio))
        split_file = getattr(args, "split_file", None)
    else:
        split_file = None

    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError(f"Invalid three-way ratios: val={val_ratio}, test={test_ratio}")

    if split_file:
        path = Path(split_file).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_length = int(manifest["dataset_length"])
        if len(base) != expected_length:
            raise ValueError(
                f"Split file {path} expects dataset_length={expected_length}, "
                f"but the loaded dataset has {len(base)} items."
            )
        train_idx = [int(v) for v in manifest["train_indices"]]
        val_idx = [int(v) for v in manifest["val_indices"]]
        if "test_indices" not in manifest:
            raise ValueError(f"Three-way split file {path} has no test_indices.")
        test_idx = [int(v) for v in manifest["test_indices"]]
    else:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(base), generator=generator).tolist()
        holdout_ratio = val_ratio + test_ratio
        n_holdout = max(2, int(round(holdout_ratio * len(indices))))
        n_val = max(1, int(round((val_ratio / holdout_ratio) * n_holdout)))
        n_test = n_holdout - n_val
        if n_test < 1 or n_holdout >= len(indices):
            raise ValueError("Validation and test partitions consume the entire dataset.")
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_holdout]
        train_idx = indices[n_holdout:]
        _write_generated_split_manifest(args, len(base), train_idx, val_idx, test_idx)

    combined = train_idx + val_idx + test_idx
    if len(combined) != len(base) or len(set(combined)) != len(base):
        raise ValueError("Three-way split is not a disjoint full partition.")
    if min(combined, default=0) < 0 or max(combined, default=-1) >= len(base):
        raise ValueError("Three-way split contains out-of-range indices.")

    selected = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    return torch.utils.data.Subset(base, selected)


# ------------------------------
# Dataset builders with split='train'|'val'|'test'
# ------------------------------
def _build_cifar10(args, split: str):
    if split in ("train", "val"):
        base = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=_img_transforms(args, split == "train"))
        return _split_subset(base, split, args=args), 10
    return datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=_img_transforms(args, False)), 10


def _build_cifar100(args, split: str):
    if split in ("train", "val"):
        base = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=_img_transforms(args, split == "train"))
        return _split_subset(base, split, args=args), 100
    return datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=_img_transforms(args, False)), 100


def _build_mnist_like(cls, args, split: str, num_classes: int, **kwargs):
    if split in ("train", "val"):
        base = cls(root=args.data_path, train=True, download=True, transform=_img_transforms(args, split == "train"), **kwargs)
        return _split_subset(base, split, args=args), num_classes
    ds = cls(root=args.data_path, train=False, download=True, transform=_img_transforms(args, False), **kwargs)
    return ds, num_classes


def _build_mnist(args, split: str):
    return _build_mnist_like(datasets.MNIST, args, split, 10)


def _build_fashion_mnist(args, split: str):
    return _build_mnist_like(datasets.FashionMNIST, args, split, 10)


def _build_emnist(args, split: str):
    return _build_mnist_like(datasets.EMNIST, args, split, 62, split="byclass")


def _build_kmnist(args, split: str):
    return _build_mnist_like(datasets.KMNIST, args, split, 10)


def _build_qmnist(args, split: str):
    tfm = _img_transforms(args, split == "train")
    if split in ("train", "val"):
        base = datasets.QMNIST(root=args.data_path, what="train", download=True, transform=tfm)
        return _split_subset(base, split, args=args), 10
    return datasets.QMNIST(root=args.data_path, what="test", download=True, transform=_img_transforms(args, False)), 10


def _build_usps(args, split: str):
    if split in ("train", "val"):
        base = datasets.USPS(root=args.data_path, train=True, download=True, transform=_img_transforms(args, split == "train"))
        return _split_subset(base, split, args=args), 10
    return datasets.USPS(root=args.data_path, train=False, download=True, transform=_img_transforms(args, False)), 10


def _build_svhn(args, split: str):
    if split in ("train", "val"):
        base = datasets.SVHN(root=args.data_path, split="train", download=True, transform=_img_transforms(args, split == "train"))
        return _split_subset(base, split, args=args), 10
    return datasets.SVHN(root=args.data_path, split="test", download=True, transform=_img_transforms(args, False)), 10


def _build_stl10(args, split: str):
    tv_split = "train" if split in ("train", "val") else "test"
    base = datasets.STL10(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        return _split_subset(base, split, args=args), 10
    return base, 10


def _build_food101(args, split: str):
    tv_split = "train" if split in ("train", "val") else "test"
    base = datasets.Food101(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        return _split_subset(base, split, args=args), 101
    return base, 101


def _build_pets(args, split: str):
    tv_split = "trainval" if split in ("train", "val") else "test"
    base = datasets.OxfordIIITPet(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        return _split_subset(base, split, args=args), 37
    return base, 37


def _build_flowers102(args, split: str):
    # Official splits exist: train / val / test.
    tv_split = {"train": "train", "val": "val", "test": "test"}[split]
    return datasets.Flowers102(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train")), 102


def _build_cars(args, split: str):
    tv_split = "train" if split in ("train", "val") else "test"
    base = datasets.StanfordCars(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        return _split_subset(base, split, args=args), 196
    return base, 196


def _build_caltech101(args, split: str):
    base = datasets.Caltech101(root=args.data_path, download=True, transform=_img_transforms(args, split == "train"))
    return _split_three_way_subset(base, split, args=args), 101


def _build_eurosat(args, split: str):
    base = datasets.EuroSAT(root=args.data_path, download=True, transform=_img_transforms(args, split == "train"))
    return _split_three_way_subset(base, split, args=args), 10


def _build_dtd(args, split: str):
    # Official DTD splits exist: train / val / test.
    return datasets.DTD(root=args.data_path, split=split, download=True, transform=_img_transforms(args, split == "train")), 47


def _build_fgvc_aircraft(args, split: str):
    tv_split = "trainval" if split in ("train", "val") else "test"
    base = datasets.FGVCAircraft(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        return _split_subset(base, split, args=args), 100
    return base, 100


def _build_sun397(args, split: str):
    # Some torchvision versions do not expose a split arg for SUN397.
    base = datasets.SUN397(root=args.data_path, download=True, transform=_img_transforms(args, split == "train"))
    if split == "test":
        split = "val"
    return _split_subset(base, split, args=args), 397


def _build_gtsrb(args, split: str):
    tv_split = "train" if split in ("train", "val") else "test"
    base = datasets.GTSRB(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        return _split_subset(base, split, args=args), 43
    return base, 43


def _build_fer2013(args, split: str):
    tv_split = "train" if split in ("train", "val") else "test"
    base = datasets.FER2013(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train"))
    if split in ("train", "val"):
        return _split_subset(base, split, args=args), 7
    return base, 7


def _build_pcam(args, split: str):
    tv_split = {"train": "train", "val": "val", "test": "test"}[split]
    return datasets.PCAM(root=args.data_path, split=tv_split, download=True, transform=_img_transforms(args, split == "train")), 2


class _CocoSingleLabel(torch.utils.data.Dataset):
    """Wrap CocoDetection and assign one label per image: the majority category."""

    def __init__(self, img_root: str, ann_file: str, transform):
        self.base = CocoDetection(img_root, ann_file, transform=None, target_transform=None)
        self.transform = transform
        cat_ids = sorted(self.base.coco.getCatIds())
        self.catid2idx = {cid: i for i, cid in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)
        keep, labels = [], []
        for i, img_id in enumerate(self.base.ids):
            ann_ids = self.base.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            if not ann_ids:
                continue
            anns = self.base.coco.loadAnns(ann_ids)
            counts = Counter(self.catid2idx[a["category_id"]] for a in anns if "category_id" in a)
            if counts:
                keep.append(i)
                labels.append(counts.most_common(1)[0][0])
        self.keep = keep
        self.labels = labels

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx: int):
        base_idx = self.keep[idx]
        img, _ = self.base[base_idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _build_coco(args, split: str):
    split_dir = "train2017" if split in ("train", "val") else "val2017"
    img_root = os.path.join(args.data_path, split_dir)
    ann_file = os.path.join(args.data_path, "annotations", f"instances_{split_dir}.json")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"[COCO] Missing images folder: {img_root}")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"[COCO] Missing annotations file: {ann_file}")
    base = _CocoSingleLabel(img_root, ann_file, _img_transforms(args, split == "train"))
    if split in ("train", "val"):
        return _split_subset(base, split, args=args), base.num_classes
    return base, base.num_classes


def _build_fake(args, split: str):
    """Small deterministic dataset used only by smoke tests and CI.

    It exercises the complete training/evaluation pipeline without downloads.
    Real manuscript scripts never select this dataset.
    """
    sizes = {
        "train": int(getattr(args, "fake_train_size", 24)),
        "val": int(getattr(args, "fake_val_size", 12)),
        "test": int(getattr(args, "fake_test_size", 12)),
    }
    num_classes = int(getattr(args, "fake_num_classes", 5))
    seed = int(getattr(args, "seed", 0)) + {"train": 0, "val": 10000, "test": 20000}[split]
    ds = datasets.FakeData(
        size=sizes[split],
        image_size=(3, int(getattr(args, "input_size", 64)), int(getattr(args, "input_size", 64))),
        num_classes=num_classes,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]),
        random_offset=seed,
    )
    return ds, num_classes


_BUILDERS = {
    ("fake", "fake_cnn", "smoke"): _build_fake,
    ("cifar10",): _build_cifar10,
    ("cifar100", "cifar_100"): _build_cifar100,
    ("mnist",): _build_mnist,
    ("fashion_mnist", "fashionmnist"): _build_fashion_mnist,
    ("emnist", "emnist_byclass"): _build_emnist,
    ("kmnist",): _build_kmnist,
    ("qmnist",): _build_qmnist,
    ("usps",): _build_usps,
    ("svhn",): _build_svhn,
    ("stl10",): _build_stl10,
    ("food101", "food_101"): _build_food101,
    ("oxfordiiitpet", "oxford_iiit_pet", "pets", "oxford_pets"): _build_pets,
    ("flowers102", "oxfordflowers102", "oxford_flowers102"): _build_flowers102,
    ("stanford_cars", "stanfordcars", "cars"): _build_cars,
    ("caltech101", "caltech_101"): _build_caltech101,
    ("dtd", "describable_textures", "textures"): _build_dtd,
    ("eurosat", "euro_sat"): _build_eurosat,
    ("fgvc_aircraft", "fgvca", "aircraft"): _build_fgvc_aircraft,
    ("sun397", "sun_397"): _build_sun397,
    ("gtsrb", "traffic_signs", "german_traffic_signs"): _build_gtsrb,
    ("fer2013", "fer_2013"): _build_fer2013,
    ("pcam", "patch_camelyon"): _build_pcam,
    ("coco", "coco2017", "mscoco", "mscoco2017"): _build_coco,
}


def _resolve_builder(name: str) -> Callable:
    name = (name or "").lower().replace("-", "_")
    for aliases, fn in _BUILDERS.items():
        if name in aliases:
            return fn
    raise NotImplementedError(f"Unsupported dataset '{name}'.")


def build_dataset_split(args, split: str) -> Tuple[torch.utils.data.Dataset, int]:
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")
    fn = _resolve_builder(args.dataset)
    return fn(args, split)


def build_dataset(args, is_train: bool) -> Tuple[torch.utils.data.Dataset, int]:
    """Backward-compatible API: train -> train, eval -> val."""
    return build_dataset_split(args, "train" if is_train else "val")
