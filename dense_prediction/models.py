from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Large_Weights, ViT_B_16_Weights, vit_b_16
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from models.dt1d_adapter import DT1DAdapter
from models.plain_axial_depthwise_adapter import PlainAxialDepthwiseAdapter
from models.tuning_modules.bam_adapter import BAMAdapter
from models.tuning_modules.lora_conv import LoRAConv2d, apply_lora_conv2d
from models.tuning_modules.ssf import SSF


class FeatureAdapterWrapper(nn.Module):
    """Apply an adapter to a frozen block output while preserving metadata."""

    def __init__(self, block: nn.Module, adapter: nn.Module, channels: int) -> None:
        super().__init__()
        self.block = block
        self.adapter = adapter
        self.out_channels = int(channels)
        if hasattr(block, "_is_cn"):
            self._is_cn = block._is_cn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.block(x))


class DenseConvAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, gate_init: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, channels // max(1, int(reduction)))
        self.branch = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        nn.init.zeros_(self.branch[-1].weight)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.is_dense_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate * self.branch(x)


class DenseResidualAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, gate_init: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, channels // max(1, int(reduction)))
        self.branch = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        nn.init.zeros_(self.branch[-1].weight)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.is_dense_adapter = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate * self.branch(x)




class TinyDetectionModel(nn.Module):
    """Small execution-only detector used by tests and ``--smoke`` runs.

    Real experiments always use Faster R-CNN MobileNetV3-FPN. This model keeps
    the same train/eval call contract so the training engine can be tested
    quickly and deterministically on CPU.
    """

    def __init__(self, num_classes: int, adapter: nn.Module | None = None) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.adapter = adapter if adapter is not None else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rpn = nn.Identity()
        self.roi_heads = nn.ModuleDict({
            "classifier": nn.Linear(32, int(num_classes)),
            "box_regressor": nn.Linear(32, 4),
        })
        self.dense_pipeline = "fasterrcnn_mobilenet_v3_fpn"
        self.is_tiny_detection_smoke = True

    def forward(self, images, targets=None):
        batch = torch.stack(list(images), dim=0)
        features = self.adapter(self.features(batch))
        pooled = self.pool(features).flatten(1)
        logits = self.roi_heads["classifier"](pooled)
        boxes_raw = self.roi_heads["box_regressor"](pooled)
        if self.training and targets is not None:
            labels = torch.stack([target["labels"][0] for target in targets]).to(logits.device)
            target_boxes = torch.stack([target["boxes"][0] for target in targets]).to(boxes_raw.device)
            image_scales = torch.tensor(
                [[image.shape[-1], image.shape[-2], image.shape[-1], image.shape[-2]] for image in images],
                device=boxes_raw.device, dtype=boxes_raw.dtype,
            ).clamp_min(1.0)
            normalized_target = target_boxes / image_scales
            predicted = torch.sigmoid(boxes_raw)
            return {
                "loss_classifier": F.cross_entropy(logits, labels),
                "loss_box_reg": F.l1_loss(predicted, normalized_target),
            }
        probabilities = torch.softmax(logits, dim=1)
        outputs = []
        for index, image in enumerate(images):
            height, width = image.shape[-2:]
            raw = torch.sigmoid(boxes_raw[index])
            x1 = torch.minimum(raw[0], raw[2]) * width
            x2 = torch.maximum(raw[0], raw[2]) * width
            y1 = torch.minimum(raw[1], raw[3]) * height
            y2 = torch.maximum(raw[1], raw[3]) * height
            label = int(torch.argmax(probabilities[index, 1:]).item()) + 1 if probabilities.shape[1] > 1 else 0
            score = probabilities[index, label]
            outputs.append({
                "boxes": torch.stack((x1, y1, x2, y2)).reshape(1, 4),
                "labels": torch.tensor([label], device=logits.device, dtype=torch.long),
                "scores": score.reshape(1),
            })
        return outputs


class TinyPatchTransformer(nn.Module):
    """Small test-only patch transformer; real experiments use ViT-B/16."""

    def __init__(self, hidden_dim: int = 64, patch_size: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        h, w = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(x.shape[0], self.hidden_dim, h, w)


class ViTDenseDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool,
        adapter: nn.Module | None,
        variant: str = "vit_b16",
    ) -> None:
        super().__init__()
        self.variant = str(variant)
        if self.variant == "tiny":
            self.encoder = TinyPatchTransformer()
            self.hidden_dim = self.encoder.hidden_dim
            self.input_image_size = None
        elif self.variant == "vit_b16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.encoder = vit_b_16(weights=weights)
            self.hidden_dim = int(self.encoder.hidden_dim)
            self.input_image_size = int(self.encoder.image_size)
            self.encoder.heads = nn.Identity()
        else:
            raise ValueError(f"Unsupported ViT variant: {variant}")
        self.adapter = adapter if adapter is not None else nn.Identity()
        self.decoder = nn.Conv2d(self.hidden_dim, int(num_classes), kernel_size=1)
        self.is_dense_vit = True

    def _vit_features(self, x: torch.Tensor) -> torch.Tensor:
        original = x.shape[-2:]
        if original != (self.input_image_size, self.input_image_size):
            x = F.interpolate(
                x, size=(self.input_image_size, self.input_image_size), mode="bilinear", align_corners=False
            )
        x = self.encoder._process_input(x)
        n = x.shape[0]
        cls = self.encoder.class_token.expand(n, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.encoder.encoder(x)[:, 1:]
        side = int(math.sqrt(x.shape[1]))
        if side * side != x.shape[1]:
            raise RuntimeError("ViT patch-token count is not square")
        return x.transpose(1, 2).reshape(n, self.hidden_dim, side, side)

    def forward(self, x: torch.Tensor):
        output_size = x.shape[-2:]
        if self.variant == "tiny":
            features = self.encoder.forward_features(x)
        else:
            features = self._vit_features(x)
        features = self.adapter(features)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return {"out": logits}


def _parse_stage_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ("4", "7", "13", "16")
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace(";", ",").split(",") if part.strip())
    return tuple(str(part) for part in value)


def _adapter_for(method: str, channels: int, args) -> nn.Module:
    if method == "dt1d":
        return DT1DAdapter(
            channels,
            cache_kernel=args.cache_dt1d,
        )
    if method == "plain_axial":
        return PlainAxialDepthwiseAdapter(
            channels,
            axis=args.dt_axis,
            kernel_mode=args.axial_kernel_mode,
            gate_mode=args.dt_gate_mode,
            padding_mode=args.dt_padding,
            use_pointwise=args.dt_use_pointwise,
            project_l1=args.axial_project_l1,
        )
    if method == "conv_adapter":
        return DenseConvAdapter(channels, reduction=args.adapter_reduction)
    if method == "residual_adapter":
        return DenseResidualAdapter(channels, reduction=args.adapter_reduction)
    if method == "ssf":
        module = SSF(channels)
        module.is_dense_adapter = True
        return module
    if method == "bam":
        module = BAMAdapter(channels, reduction=args.adapter_reduction, gate_init=0.0)
        module.is_dense_adapter = True
        return module
    raise ValueError(f"No feature adapter for method {method!r}")


def _attach_feature_adapters(container: nn.Module, method: str, args) -> int:
    inserted = 0
    for name in _parse_stage_names(args.adapter_stages):
        if name not in container._modules:
            available = ", ".join(container._modules.keys())
            raise KeyError(f"Adapter stage {name!r} not found; available stages: {available}")
        block = container._modules[name]
        channels = getattr(block, "out_channels", None)
        if channels is None:
            raise TypeError(f"Cannot infer output channels for stage {name}: {type(block).__name__}")
        container._modules[name] = FeatureAdapterWrapper(block, _adapter_for(method, channels, args), channels)
        inserted += 1
    return inserted


def _vit_adapter(method: str, channels: int, args) -> nn.Module | None:
    if method in {"dt1d", "plain_axial", "conv_adapter", "residual_adapter", "ssf", "bam"}:
        return _adapter_for(method, channels, args)
    return None


def build_dense_model(args) -> nn.Module:
    pipeline = str(args.pipeline).lower()
    method = str(args.tuning_method).lower()
    pretrained = bool(args.pretrained)

    if pipeline == "deeplab_mobilenet_v3":
        backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=int(args.num_classes),
            aux_loss=False,
        )
        if method == "lora_conv":
            apply_lora_conv2d(model.backbone, r=args.lora_rank, alpha=args.lora_alpha, target=args.lora_target)
        elif method in {"dt1d", "plain_axial", "conv_adapter", "residual_adapter", "ssf", "bam"}:
            _attach_feature_adapters(model.backbone, method, args)
        model.dense_pipeline = pipeline
        return model

    if pipeline == "vit_b16_dense":
        hidden_dim = 64 if args.vit_variant == "tiny" else 768
        adapter = _vit_adapter(method, hidden_dim, args)
        model = ViTDenseDecoder(
            int(args.num_classes), pretrained=pretrained, adapter=adapter, variant=args.vit_variant
        )
        model.dense_pipeline = pipeline
        return model

    if pipeline == "fasterrcnn_mobilenet_v3_fpn":
        if getattr(args, "detector_variant", "mobilenet_v3_fpn") == "tiny":
            adapter = _adapter_for(method, 32, args) if method in {"dt1d", "plain_axial", "conv_adapter", "residual_adapter", "ssf", "bam"} else None
            model = TinyDetectionModel(int(args.num_classes), adapter=adapter)
            model.dense_pipeline = pipeline
            return model
        backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=int(args.num_classes),
            min_size=int(args.input_size),
            max_size=int(args.input_size),
        )
        if method == "lora_conv":
            apply_lora_conv2d(model.backbone.body, r=args.lora_rank, alpha=args.lora_alpha, target=args.lora_target)
        elif method in {"dt1d", "plain_axial", "conv_adapter", "residual_adapter", "ssf", "bam"}:
            _attach_feature_adapters(model.backbone.body, method, args)
        model.dense_pipeline = pipeline
        return model

    raise ValueError(f"Unsupported dense pipeline: {pipeline}")


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = bool(enabled)


def _enable_adapter_parameters(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (DT1DAdapter, PlainAxialDepthwiseAdapter, DenseConvAdapter, DenseResidualAdapter, SSF, BAMAdapter)):
            _set_requires_grad(module, True)
        if isinstance(module, LoRAConv2d):
            if module.lora_down is not None:
                _set_requires_grad(module.lora_down, True)
                _set_requires_grad(module.lora_up, True)


def _enable_task_head(model: nn.Module, pipeline: str) -> None:
    if pipeline == "deeplab_mobilenet_v3":
        _set_requires_grad(model.classifier, True)
        if getattr(model, "aux_classifier", None) is not None:
            _set_requires_grad(model.aux_classifier, True)
    elif pipeline == "vit_b16_dense":
        _set_requires_grad(model.decoder, True)
    elif pipeline == "fasterrcnn_mobilenet_v3_fpn":
        _set_requires_grad(model.rpn, True)
        _set_requires_grad(model.roi_heads, True)
    else:
        raise ValueError(pipeline)


def configure_dense_trainability(model: nn.Module, args) -> dict[str, int]:
    method = str(args.tuning_method).lower()
    pipeline = str(args.pipeline).lower()
    if method == "full":
        _set_requires_grad(model, True)
    else:
        _set_requires_grad(model, False)
        _enable_task_head(model, pipeline)
        if method in {"dt1d", "plain_axial", "conv_adapter", "residual_adapter", "ssf", "bam", "lora_conv"}:
            _enable_adapter_parameters(model)
        elif method == "bitfit":
            for name, parameter in model.named_parameters():
                if name.endswith("bias"):
                    parameter.requires_grad = True
        elif method not in {"linear", "head_only"}:
            raise ValueError(f"Unsupported tuning method for dense prediction: {method}")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    adapter = 0
    for module in model.modules():
        if isinstance(module, (DT1DAdapter, PlainAxialDepthwiseAdapter, DenseConvAdapter, DenseResidualAdapter, SSF, BAMAdapter)):
            adapter += sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        elif isinstance(module, LoRAConv2d):
            if module.lora_down is not None:
                adapter += sum(parameter.numel() for parameter in module.lora_down.parameters())
                adapter += sum(parameter.numel() for parameter in module.lora_up.parameters())
    return {"total_params": int(total), "trainable_params": int(trainable), "adapter_params": int(adapter)}


def enforce_frozen_norm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            direct = list(module.parameters(recurse=False))
            if direct and not any(parameter.requires_grad for parameter in direct):
                module.eval()


def prepare_dt1d_inference_cache(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, (DT1DAdapter,)) and module.cache_kernel:
            module.prepare_for_inference()
            count += 1
    return count


__all__ = [
    "FeatureAdapterWrapper",
    "ViTDenseDecoder",
    "build_dense_model",
    "configure_dense_trainability",
    "enforce_frozen_norm_eval",
    "prepare_dt1d_inference_cache",
]
