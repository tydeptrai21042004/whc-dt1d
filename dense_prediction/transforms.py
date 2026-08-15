from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SegmentationTransform:
    size: int = 224
    train: bool = False
    normalize: bool = True

    def __call__(self, image: Image.Image, mask: Image.Image | torch.Tensor):
        image = TF.resize(image, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        if isinstance(mask, Image.Image):
            mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
            mask_tensor = torch.as_tensor(__import__("numpy").array(mask), dtype=torch.long)
        else:
            mask_tensor = mask.to(dtype=torch.long)
            mask_tensor = TF.resize(
                mask_tensor.unsqueeze(0), [self.size, self.size], interpolation=InterpolationMode.NEAREST
            ).squeeze(0)

        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask_tensor = torch.flip(mask_tensor, dims=(-1,))

        image_tensor = TF.pil_to_tensor(image).float().div_(255.0)
        if self.normalize:
            image_tensor = TF.normalize(image_tensor, _IMAGENET_MEAN, _IMAGENET_STD)
        return image_tensor, mask_tensor


@dataclass
class DetectionTransform:
    size: int = 320
    train: bool = False

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        old_w, old_h = image.size
        image = TF.resize(image, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        target = {key: value.clone() if torch.is_tensor(value) else value for key, value in target.items()}
        boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).to(torch.float32)
        if boxes.numel() > 0:
            scale_x = self.size / float(old_w)
            scale_y = self.size / float(old_h)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = self.size - x2
                boxes[:, 2] = self.size - x1
        target["boxes"] = boxes
        target["area"] = ((boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0))
        image_tensor = TF.pil_to_tensor(image).float().div_(255.0)
        return image_tensor, target


__all__ = ["SegmentationTransform", "DetectionTransform"]
