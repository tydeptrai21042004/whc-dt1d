from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn.functional as F

from .metrics import DetectionAP50Meter, SegmentationMeter
from .models import enforce_frozen_norm_eval, prepare_dt1d_inference_cache


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def segmentation_loss(outputs: dict, target: torch.Tensor, binary: bool) -> torch.Tensor:
    logits = outputs["out"]
    if binary:
        loss = F.binary_cross_entropy_with_logits(logits[:, 0], target.float())
    else:
        loss = F.cross_entropy(logits, target.long())
    if "aux" in outputs:
        aux = outputs["aux"]
        if binary:
            loss = loss + 0.4 * F.binary_cross_entropy_with_logits(aux[:, 0], target.float())
        else:
            loss = loss + 0.4 * F.cross_entropy(aux, target.long())
    return loss


def train_segmentation_epoch(model, loader, optimizer, device, binary: bool, use_amp: bool) -> Dict[str, float]:
    model.train()
    enforce_frozen_norm_eval(model)
    total_loss = 0.0
    samples = 0
    start = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and device.type == "cuda"))
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            outputs = model(images)
            loss = segmentation_loss(outputs, targets, binary)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch = images.shape[0]
        total_loss += float(loss.detach()) * batch
        samples += batch
    return {
        "loss": total_loss / max(1, samples),
        "epoch_time_sec": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate_segmentation(model, loader, device, num_classes: int, binary: bool, cache_dt1d: bool) -> Dict[str, float]:
    model.eval()
    if cache_dt1d:
        prepare_dt1d_inference_cache(model)
    meter = SegmentationMeter(num_classes=num_classes, binary=binary)
    total_loss = 0.0
    samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = segmentation_loss(outputs, targets, binary)
        meter.update(outputs["out"], targets)
        total_loss += float(loss) * images.shape[0]
        samples += images.shape[0]
    result = meter.compute()
    result["loss"] = total_loss / max(1, samples)
    return result


def train_detection_epoch(model, loader, optimizer, device, use_amp: bool) -> Dict[str, float]:
    model.train()
    enforce_frozen_norm_eval(model)
    total_loss = 0.0
    images_seen = 0
    start = time.perf_counter()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and device.type == "cuda"))
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in target.items()}
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(images)
        images_seen += len(images)
    return {
        "loss": total_loss / max(1, images_seen),
        "epoch_time_sec": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate_detection(model, loader, device, cache_dt1d: bool) -> Dict[str, float]:
    model.eval()
    if cache_dt1d:
        prepare_dt1d_inference_cache(model)
    meter = DetectionAP50Meter(0.5)
    for images, targets in loader:
        device_images = [image.to(device, non_blocking=True) for image in images]
        outputs = model(device_images)
        meter.update(outputs, targets)
    return meter.compute()


@torch.no_grad()
def measure_latency(model, loader, device, task: str, warmup: int = 3, iterations: int = 10) -> Dict[str, float]:
    model.eval()
    prepare_dt1d_inference_cache(model)
    batch = next(iter(loader))
    if task == "detection":
        inputs = [image.to(device) for image in batch[0]]
        call = lambda: model(inputs)
        batch_size = len(inputs)
    else:
        inputs = batch[0].to(device)
        call = lambda: model(inputs)
        batch_size = inputs.shape[0]
    for _ in range(max(0, warmup)):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(max(1, iterations)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0 / max(1, batch_size))
    values = torch.tensor(timings)
    result = {
        "latency_ms_per_image": float(values.median().item()),
        "latency_p95_ms_per_image": float(torch.quantile(values, 0.95).item()),
        "fps": float(1000.0 / values.median().clamp_min(1e-9).item()),
    }
    if device.type == "cuda":
        result["peak_inference_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    return result


__all__ = [
    "train_segmentation_epoch",
    "evaluate_segmentation",
    "train_detection_epoch",
    "evaluate_detection",
    "measure_latency",
]
