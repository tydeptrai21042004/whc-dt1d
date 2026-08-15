from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch


class SegmentationMeter:
    def __init__(self, num_classes: int, binary: bool = False) -> None:
        self.binary = bool(binary)
        self.num_classes = 2 if self.binary else int(num_classes)
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if self.binary:
            prediction = (torch.sigmoid(logits[:, 0]) >= 0.5).long()
        else:
            prediction = logits.argmax(dim=1)
        target = target.long()
        valid = (target >= 0) & (target < self.num_classes)
        encoded = target[valid] * self.num_classes + prediction[valid]
        bincount = torch.bincount(encoded.cpu(), minlength=self.num_classes ** 2)
        self.confusion += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        cm = self.confusion
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        union = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        iou = torch.where(union > 0, tp / union, torch.nan)
        dice = torch.where(denom_dice > 0, 2 * tp / denom_dice, torch.nan)
        pixel_acc = tp.sum() / cm.sum().clamp_min(1)
        if self.binary:
            return {
                "iou": float(torch.nan_to_num(iou[1]).item()),
                "dice": float(torch.nan_to_num(dice[1]).item()),
                "pixel_accuracy": float(pixel_acc.item()),
            }
        return {
            "miou": float(torch.nanmean(iou).item()),
            "dice": float(torch.nanmean(dice).item()),
            "pixel_accuracy": float(pixel_acc.item()),
        }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), dtype=torch.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter).clamp_min(1e-12)


@dataclass
class DetectionRecord:
    score: float
    true_positive: int
    false_positive: int


class DetectionAP50Meter:
    """Dependency-free one/multi-class AP50 evaluator for manuscript sanity checks."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = float(iou_threshold)
        self.records: List[DetectionRecord] = []
        self.gt_count = 0
        self.pred_count = 0

    @torch.no_grad()
    def update(self, outputs: Iterable[dict], targets: Iterable[dict]) -> None:
        for output, target in zip(outputs, targets):
            pred_boxes = output.get("boxes", torch.zeros((0, 4))).detach().cpu().float()
            pred_scores = output.get("scores", torch.zeros((len(pred_boxes),))).detach().cpu().float()
            pred_labels = output.get("labels", torch.ones((len(pred_boxes),), dtype=torch.long)).detach().cpu().long()
            gt_boxes = target.get("boxes", torch.zeros((0, 4))).detach().cpu().float()
            gt_labels = target.get("labels", torch.ones((len(gt_boxes),), dtype=torch.long)).detach().cpu().long()
            self.gt_count += len(gt_boxes)
            self.pred_count += len(pred_boxes)
            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            order = pred_scores.argsort(descending=True)
            for pred_index in order.tolist():
                label = pred_labels[pred_index]
                candidates = torch.where((gt_labels == label) & (~matched))[0]
                tp = 0
                if candidates.numel() > 0:
                    ious = box_iou(pred_boxes[pred_index : pred_index + 1], gt_boxes[candidates]).squeeze(0)
                    best_value, best_local = ious.max(dim=0)
                    if float(best_value) >= self.iou_threshold:
                        matched[candidates[int(best_local)]] = True
                        tp = 1
                self.records.append(
                    DetectionRecord(float(pred_scores[pred_index]), tp, 1 - tp)
                )

    def compute(self) -> Dict[str, float]:
        if not self.records:
            return {"ap50": 0.0, "recall50": 0.0, "gt_count": float(self.gt_count), "pred_count": 0.0}
        records = sorted(self.records, key=lambda record: record.score, reverse=True)
        tp = torch.tensor([record.true_positive for record in records], dtype=torch.float64).cumsum(0)
        fp = torch.tensor([record.false_positive for record in records], dtype=torch.float64).cumsum(0)
        recall = tp / max(1, self.gt_count)
        precision = tp / (tp + fp).clamp_min(1e-12)
        recall_ext = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
        precision_ext = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))
        for index in range(len(precision_ext) - 2, -1, -1):
            precision_ext[index] = torch.maximum(precision_ext[index], precision_ext[index + 1])
        changes = torch.where(recall_ext[1:] != recall_ext[:-1])[0]
        ap = torch.sum(
            (recall_ext[changes + 1] - recall_ext[changes]) * precision_ext[changes + 1]
        )
        return {
            "ap50": float(ap.item()),
            "recall50": float(recall[-1].item()) if len(recall) else 0.0,
            "gt_count": float(self.gt_count),
            "pred_count": float(self.pred_count),
        }


__all__ = ["SegmentationMeter", "DetectionAP50Meter", "box_iou"]
