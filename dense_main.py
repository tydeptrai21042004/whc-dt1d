from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dense_prediction.datasets import build_dense_datasets, detection_collate_fn
from dense_prediction.engine import (
    evaluate_detection,
    evaluate_segmentation,
    measure_latency,
    train_detection_epoch,
    train_segmentation_epoch,
)
from proposal_contract import runtime_metadata
from dense_prediction.models import build_dense_model, configure_dense_trainability


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DT1D dense-prediction training")
    parser.add_argument("--task", choices=["binary_segmentation", "semantic_segmentation", "detection"], required=True)
    parser.add_argument(
        "--pipeline",
        choices=["deeplab_mobilenet_v3", "vit_b16_dense", "fasterrcnn_mobilenet_v3_fpn"],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "pennfudan",
            "drive",
            "oxford_pet_segmentation",
            "oxford_pet_detection",
            "fake_binary",
            "fake_semantic",
            "fake_detection",
        ],
        required=True,
    )
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--download", type=str2bool, default=False)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--fake_train_size", type=int, default=8)
    parser.add_argument("--fake_val_size", type=int, default=4)
    parser.add_argument("--fake_test_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)

    parser.add_argument(
        "--tuning_method",
        choices=[
            "dt1d",
            "plain_axial",
            "full",
            "linear",
            "head_only",
            "bitfit",
            "conv_adapter",
            "residual_adapter",
            "ssf",
            "bam",
            "lora_conv",
        ],
        default="dt1d",
    )
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--adapter_stages", default="4,7,13,16")
    parser.add_argument("--adapter_reduction", type=int, default=16)
    parser.add_argument("--vit_variant", choices=["vit_b16", "tiny"], default="vit_b16")
    parser.add_argument("--detector_variant", choices=["mobilenet_v3_fpn", "tiny"], default="mobilenet_v3_fpn")

    parser.add_argument("--dt_axis", choices=["h", "w", "hw"], default="hw")
    parser.add_argument("--dt_alpha_group", type=int, default=16)
    parser.add_argument("--dt_detail_basis", choices=["orth", "raw"], default="orth")
    parser.add_argument("--dt_detail_components", choices=["both", "offset4", "offset8", "none"], default="offset4")
    parser.add_argument("--dt_active_offsets", default="1,2,4,8")
    parser.add_argument("--dt_gate_mode", choices=["learned", "fixed"], default="learned")
    parser.add_argument("--dt_padding", choices=["replicate", "reflect", "zeros"], default="replicate")
    parser.add_argument("--dt_contrast_split", type=int, default=8)
    parser.add_argument("--dt_use_pointwise", type=str2bool, default=False)
    parser.add_argument("--dt_project_l1", type=str2bool, default=True)
    parser.add_argument("--cache_dt1d", type=str2bool, default=True)
    parser.add_argument("--axial_kernel_mode", choices=["unrestricted", "direct_symmetric", "reduced_symmetric"], default="unrestricted")
    parser.add_argument("--axial_project_l1", type=str2bool, default=False)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_target", choices=["all", "1x1", "3x3"], default="1x1")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_mem", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", type=str2bool, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--save_checkpoint", type=str2bool, default=True)
    parser.add_argument("--profile_latency", type=str2bool, default=True)
    parser.add_argument("--latency_warmup", type=int, default=3)
    parser.add_argument("--latency_iterations", type=int, default=10)
    parser.add_argument("--output_dir", default="outputs/dense_prediction")
    return parser


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def infer_num_classes(args) -> int:
    if args.num_classes is not None:
        return int(args.num_classes)
    if args.task == "binary_segmentation":
        return 1
    if args.task == "semantic_segmentation":
        return 3
    return 2


def make_loaders(args, train_dataset, val_dataset, test_dataset):
    detection = args.task == "detection"
    collate = detection_collate_fn if detection else None
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=collate,
    )
    train = DataLoader(train_dataset, shuffle=True, drop_last=False, **common)
    val = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    test = DataLoader(test_dataset, shuffle=False, drop_last=False, **common)
    return train, val, test


def state_dict_cpu(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def task_specific_state_dict(model):
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = model.state_dict()
    result = {}
    for name, value in state.items():
        if name in trainable_names or any(name.startswith(prefix.rsplit(".", 1)[0]) for prefix in trainable_names):
            result[name] = value.detach().cpu()
    return result


def primary_metric(task: str, metrics: dict) -> float:
    if task == "binary_segmentation":
        return float(metrics["iou"])
    if task == "semantic_segmentation":
        return float(metrics["miou"])
    return float(metrics["ap50"])


def evaluate(model, loader, args, device):
    if args.task == "detection":
        return evaluate_detection(model, loader, device, args.cache_dt1d)
    return evaluate_segmentation(
        model,
        loader,
        device,
        num_classes=args.num_classes,
        binary=args.task == "binary_segmentation",
        cache_dt1d=args.cache_dt1d,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.num_classes = infer_num_classes(args)
    if args.task == "detection" and args.pipeline != "fasterrcnn_mobilenet_v3_fpn":
        raise ValueError("Detection requires fasterrcnn_mobilenet_v3_fpn")
    if args.task != "detection" and args.pipeline == "fasterrcnn_mobilenet_v3_fpn":
        raise ValueError("Faster R-CNN is only valid for detection")
    if args.dataset == "oxford_pet_detection" and args.task != "detection":
        raise ValueError("oxford_pet_detection requires --task detection")
    if args.dataset == "oxford_pet_segmentation" and args.task != "semantic_segmentation":
        raise ValueError("oxford_pet_segmentation requires semantic_segmentation")

    set_seed(args.seed, args.deterministic)
    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    args.pin_mem = bool(args.pin_mem and device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf8")

    train_dataset, val_dataset, test_dataset = build_dense_datasets(args)
    train_loader, val_loader, test_loader = make_loaders(args, train_dataset, val_dataset, test_dataset)
    model = build_dense_model(args).to(device)
    counts = configure_dense_trainability(model, args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters were selected")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history = []
    best_metric = float("-inf")
    best_epoch = -1
    best_state = None
    train_start = time.perf_counter()
    peak_train_memory = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(args.epochs):
        if args.task == "detection":
            train_metrics = train_detection_epoch(model, train_loader, optimizer, device, args.use_amp)
        else:
            train_metrics = train_segmentation_epoch(
                model, train_loader, optimizer, device, args.task == "binary_segmentation", args.use_amp
            )
        val_metrics = evaluate(model, val_loader, args, device)
        score = primary_metric(args.task, val_metrics)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_state = state_dict_cpu(model)
        if device.type == "cuda":
            peak_train_memory = max(peak_train_memory, torch.cuda.max_memory_allocated(device) / 1024**2)
        print(json.dumps(row, sort_keys=True))

    total_train_time = time.perf_counter() - train_start
    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, args, device)

    efficiency = {
        **counts,
        "trainable_fraction": counts["trainable_params"] / max(1, counts["total_params"]),
        "task_storage_fp32_mb": counts["trainable_params"] * 4 / 1024**2,
        "task_storage_fp16_mb": counts["trainable_params"] * 2 / 1024**2,
        "task_head_or_other_trainable_params": int(counts["trainable_params"] - counts.get("adapter_params", 0)),
        "adapter_fraction_of_total": counts.get("adapter_params", 0) / max(1, counts["total_params"]),
        "peak_train_memory_mb": peak_train_memory,
        "total_train_time_sec": total_train_time,
        "mean_epoch_time_sec": float(np.mean([row["train_epoch_time_sec"] for row in history])),
    }
    if args.profile_latency:
        efficiency.update(
            measure_latency(
                model,
                test_loader,
                device,
                args.task,
                warmup=args.latency_warmup,
                iterations=args.latency_iterations,
            )
        )

    summary = {
        "task": args.task,
        "pipeline": args.pipeline,
        "dataset": args.dataset,
        "method": args.tuning_method,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "checkpoint_selection_rule": "highest validation IoU/mIoU/AP50; test evaluated once at that checkpoint",
        "test_metrics": test_metrics,
        "efficiency": efficiency,
        "metric_definition": (
            "binary: foreground-class IoU/Dice and global pixel accuracy from the accumulated confusion matrix"
            if args.task == "binary_segmentation" else
            "semantic: macro class mIoU/macro Dice and global pixel accuracy from the accumulated confusion matrix"
            if args.task == "semantic_segmentation" else
            "detection sanity check: dependency-free score-ranked AP50/Recall50 at IoU=0.5"
        ),
        **runtime_metadata(Path(__file__).resolve().parent),
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in history for key in row}))
        writer.writeheader()
        writer.writerows(history)
    if args.save_checkpoint:
        torch.save(
            {
                "model": best_state,
                "task_specific": task_specific_state_dict(model),
                "summary": summary,
            },
            output_dir / "best_checkpoint.pt",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
