"""评估 UNet / DeepLabV3-like CNN 裂缝分割基线。

该脚本输出与 SAM+LoRA 相同口径的 JSON 指标，便于论文表格统一汇总。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.cnn_segmentation import build_cnn_baseline
from baselines.cnn_segmentation.losses import segmentation_loss
from utils.data_loader import DatasetConfig, TunnelCrackDataset
from utils.metrics import SegmentationMetricMeter, ensure_mask_shape, to_binary


def parse_args() -> argparse.Namespace:
    """解析评估命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate CNN baseline on tunnel crack dataset")
    parser.add_argument("--data_root", type=str, default="data", help="数据集根目录")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="评估划分")
    parser.add_argument("--config", type=str, required=True, help="CNN baseline YAML 配置")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练得到的 checkpoint")
    parser.add_argument("--output", type=str, required=True, help="评估 JSON 输出路径")
    parser.add_argument("--device", type=str, default="", help="cuda / cpu；留空自动选择")
    parser.add_argument("--batch_size", type=int, default=0, help="覆盖配置中的 batch size")
    parser.add_argument("--num_workers", type=int, default=-1, help="覆盖配置中的 num_workers")
    parser.add_argument("--threshold", type=float, default=0.5, help="概率图二值化阈值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，确保评估过程可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(device_arg: str) -> torch.device:
    """选择评估设备。"""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_eval_loader(
    data_root: str,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """构建评估 DataLoader。

    输入:
    - data_root/split: 数据路径；
    - img_size: resize 后输入尺寸；
    - batch_size/num_workers: 评估加载参数。

    输出:
    - 不做随机增强的 DataLoader。
    """
    dataset = TunnelCrackDataset(
        DatasetConfig(
            data_root=data_root,
            split=split,
            img_size=img_size,
            use_augment=False,
            horizontal_flip_p=0.0,
            vertical_flip_p=0.0,
        )
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> int:
    """加载 CNN baseline checkpoint，并返回 epoch。"""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return int(ckpt.get("epoch", -1))
    model.load_state_dict(ckpt, strict=True)
    return -1


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    threshold: float,
) -> Dict[str, float]:
    """执行完整评估并返回统一指标。

    指标通过 `SegmentationMetricMeter` 计算，因此与 SAM+LoRA 的 JSON 口径一致。
    """
    model.eval()
    loss_cfg = cfg.get("loss", {})
    boundary_radius = int(loss_cfg.get("boundary_radius", 3))
    meter = SegmentationMetricMeter(boundary_radius=boundary_radius)

    progress = tqdm(loader, total=len(loader), desc="Eval", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(images)["logits"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time_sec = time.perf_counter() - start

        loss_dict = segmentation_loss(
            logits,
            masks,
            w_bce=float(loss_cfg.get("w_bce", 1.0)),
            w_dice=float(loss_cfg.get("w_dice", 2.0)),
            pos_weight=float(loss_cfg.get("pos_weight", 10.0)),
        )
        pred_bin = to_binary(torch.sigmoid(logits), threshold=threshold)
        masks = ensure_mask_shape(masks, target_hw=pred_bin.shape[-2:])
        target_bin = to_binary(masks, threshold=0.5)
        meter.update(
            pred_bin=pred_bin,
            target_bin=target_bin,
            infer_time_sec=infer_time_sec,
            loss_value=float(loss_dict["loss"].item()),
        )

    return meter.summary()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = select_device(args.device)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    img_size = int(model_cfg.get("img_size", 512))
    batch_size = int(args.batch_size) if args.batch_size > 0 else int(train_cfg.get("batch_size", 4))
    num_workers = int(args.num_workers) if args.num_workers >= 0 else int(train_cfg.get("num_workers", 0))

    loader = build_eval_loader(args.data_root, args.split, img_size, batch_size, num_workers)
    model = build_cnn_baseline(model_cfg).to(device)
    loaded_epoch = load_checkpoint(model, args.checkpoint, device)
    metrics = evaluate(model, loader, device, cfg, threshold=float(args.threshold))

    payload = {
        "model": str(model_cfg.get("name", "")),
        "split": args.split,
        "data_root": args.data_root,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "loaded_epoch": loaded_epoch,
        "threshold": float(args.threshold),
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] model={model.__class__.__name__}, split={args.split}, samples={len(loader.dataset)}")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")
    print(f"[INFO] saved: {output}")


if __name__ == "__main__":
    main()
