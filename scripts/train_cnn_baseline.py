"""训练 UNet / DeepLabV3-like CNN 裂缝分割基线。

该脚本用于补充论文对比实验：在同一份 `data/train|val`、同一套 mask
归一化逻辑、同一套 mIoU/Dice/Precision/Recall/Boundary-IoU 指标口径下，
训练传统深度分割模型，与 SAM-AMG、SAM-Box、SAM+LoRA 做横向对比。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.cnn_segmentation import build_cnn_baseline
from baselines.cnn_segmentation.losses import segmentation_loss
from utils.data_loader import build_dataloaders


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入来自终端，输出为 argparse 命名空间。
    这样做是为了保持与 `scripts/train.py` 类似的使用方式，便于实验记录。
    """
    parser = argparse.ArgumentParser(description="Train CNN baselines for tunnel crack segmentation")
    parser.add_argument("--data_root", type=str, default="data", help="数据集根目录")
    parser.add_argument("--config", type=str, required=True, help="CNN baseline YAML 配置")
    parser.add_argument("--output_dir", type=str, required=True, help="checkpoint 输出目录")
    parser.add_argument("--device", type=str, default="", help="cuda / cpu；留空自动选择")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default="", help="可选：断点恢复 checkpoint")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，提升训练可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(device_arg: str) -> torch.device:
    """选择训练设备。

    优先使用用户显式指定的设备；否则自动优先 CUDA。
    """
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> Optimizer:
    """根据配置构建优化器。

    输入:
    - model: CNN 分割模型；
    - cfg: 配置字典，读取 `training.lr`、`training.weight_decay` 和 `optimizer.type`。

    输出:
    - PyTorch optimizer。
    """
    train_cfg = cfg["training"]
    opt_cfg = cfg.get("optimizer", {"type": "adamw"})
    params = [param for param in model.parameters() if param.requires_grad]
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    opt_type = str(opt_cfg.get("type", "adamw")).lower()

    if opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"不支持的优化器类型: {opt_type}")


def build_scheduler(optimizer: Optimizer, cfg: Dict[str, Any], epochs: int):
    """构建学习率调度器。"""
    sched_cfg = cfg.get("scheduler", {"type": "cosine"})
    sched_type = str(sched_cfg.get("type", "cosine")).lower()
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(sched_cfg.get("min_lr", 1e-6)),
        )
    if sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )
    if sched_type == "none":
        return None
    raise ValueError(f"不支持的调度器类型: {sched_type}")


def compute_batch_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """计算 batch 级 IoU，用于训练过程快速监控。

    输入:
    - logits: `[B,1,H,W]`；
    - targets: `[B,1,H,W]`。

    输出:
    - 当前 batch 的平均 IoU。
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    if targets.shape[-2:] != pred.shape[-2:]:
        targets = torch.nn.functional.interpolate(targets.float(), size=pred.shape[-2:], mode="nearest")
    target = (targets >= 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    return float((inter / (union + 1e-6)).mean().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    cfg: Dict[str, Any],
    log_interval: int,
) -> Dict[str, float]:
    """训练一个 epoch。

    输入:
    - model/loader/optimizer/device: 标准 PyTorch 训练组件；
    - cfg: 从中读取 loss 权重；
    - log_interval: tqdm 显示间隔。

    输出:
    - `loss` 与 `iou` 的 epoch 平均值。
    """
    model.train()
    loss_cfg = cfg.get("loss", {})
    running_loss = 0.0
    running_iou = 0.0

    progress = tqdm(enumerate(loader), total=len(loader), desc="Train", leave=False)
    for step, batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        logits = outputs["logits"]
        loss_dict = segmentation_loss(
            logits,
            masks,
            w_bce=float(loss_cfg.get("w_bce", 1.0)),
            w_dice=float(loss_cfg.get("w_dice", 2.0)),
            pos_weight=float(loss_cfg.get("pos_weight", 10.0)),
        )
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        batch_iou = compute_batch_iou(logits.detach(), masks)
        running_loss += float(loss.item())
        running_iou += batch_iou
        if step % max(log_interval, 1) == 0:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{batch_iou:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    n = max(len(loader), 1)
    return {"loss": running_loss / n, "iou": running_iou / n}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """验证一个 epoch。

    如果没有 val 集，则返回 NaN；当前项目有 val，因此正式实验会正常计算。
    """
    if loader is None:
        return {"loss": float("nan"), "iou": float("nan")}

    model.eval()
    loss_cfg = cfg.get("loss", {})
    running_loss = 0.0
    running_iou = 0.0
    progress = tqdm(loader, total=len(loader), desc="Val", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        outputs = model(images)
        logits = outputs["logits"]
        loss_dict = segmentation_loss(
            logits,
            masks,
            w_bce=float(loss_cfg.get("w_bce", 1.0)),
            w_dice=float(loss_cfg.get("w_dice", 2.0)),
            pos_weight=float(loss_cfg.get("pos_weight", 10.0)),
        )
        running_loss += float(loss_dict["loss"].item())
        running_iou += compute_batch_iou(logits, masks)
    n = max(len(loader), 1)
    return {"loss": running_loss / n, "iou": running_iou / n}


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    best_score: float,
    is_best: bool,
) -> None:
    """保存 last/best checkpoint。

    checkpoint 中同时保存模型、优化器、调度器和 best 分数，方便断点恢复。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_score": float(best_score),
    }
    torch.save(payload, output_dir / "last_model.pth")
    if is_best:
        torch.save(payload, output_dir / "best_model.pth")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = select_device(args.device)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    log_cfg = cfg.get("logging", {})
    early_cfg = cfg.get("early_stopping", {})

    img_size = int(model_cfg.get("img_size", 512))
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg.get("epochs", 50))

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        hflip_p=float(aug_cfg.get("horizontal_flip", 0.5)),
        vflip_p=float(aug_cfg.get("vertical_flip", 0.5)),
    )
    model = build_cnn_baseline(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs=epochs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = {
        "args": vars(args),
        "config": cfg,
        "model_class": model.__class__.__name__,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
    }
    with (output_dir / f"run_config_{run_tag}.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    start_epoch = 1
    best_score = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", -1.0))

    print(f"[INFO] model={model.__class__.__name__}, device={device}")
    print(f"[INFO] train={len(train_loader.dataset)}, val={len(val_loader.dataset) if val_loader else 0}")
    print(f"[INFO] output_dir={output_dir}")

    patience = int(early_cfg.get("patience", 10))
    min_delta = float(early_cfg.get("min_delta", 1e-4))
    early_enabled = bool(early_cfg.get("enabled", True))
    no_improve_count = 0
    save_interval = int(ckpt_cfg.get("save_interval", 1))
    log_interval = int(log_cfg.get("log_interval", 10))

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg, log_interval)
        val_metrics = validate_one_epoch(model, val_loader, device, cfg)
        if scheduler is not None:
            scheduler.step()

        current_score = val_metrics["iou"] if not np.isnan(val_metrics["iou"]) else train_metrics["iou"]
        is_best = current_score > best_score + min_delta
        if is_best:
            best_score = current_score
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (epoch % save_interval == 0) or is_best or epoch == epochs:
            save_checkpoint(output_dir, epoch, model, optimizer, scheduler, best_score, is_best)

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f}, train_iou={train_metrics['iou']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_iou={val_metrics['iou']:.4f}, "
            f"best={best_score:.4f}"
        )
        if early_enabled and no_improve_count >= patience:
            print(f"[INFO] early stopping: no improvement for {no_improve_count} epochs")
            break

    print(f"[INFO] training finished, best_val_iou={best_score:.4f}")


if __name__ == "__main__":
    main()
