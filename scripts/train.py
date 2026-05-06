"""训练脚本（最小可跑闭环版本）。

本脚本重点解决三件事：
1. 读取配置和数据；
2. 构建 EnhancedSAM（SAM + LoRA + Boundary）；
3. 跑通训练/验证并保存 checkpoint。

你可以把它理解为：项目从“只有模型代码”走向“能真正训练”的第一步。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# 让脚本在任何工作目录下都能找到项目根目录模块。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.enhanced_sam import EnhancedSAM, EnhancedSAMConfig, build_enhanced_sam
from models.sam_base import load_sam_model, patch_sam_for_img_size
from utils.data_loader import build_dataloaders


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这里的参数基本覆盖“最小训练闭环”：
    - 数据路径
    - SAM 预训练权重路径
    - 配置文件路径
    - 输出目录
    - 设备/随机种子/断点恢复
    """
    parser = argparse.ArgumentParser(description="Train EnhancedSAM for tunnel crack segmentation")
    parser.add_argument("--data_root", type=str, default="data", help="数据集根目录")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM 预训练权重路径")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="训练配置文件路径")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="checkpoint 输出目录")
    parser.add_argument("--device", type=str, default="", help="训练设备，例如 cuda / cpu；留空则自动选择")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default="", help="断点恢复 checkpoint 路径（可选）")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，提升结果复现性。

    为什么要这样做：
    - 同样代码、同样数据、同样参数，结果尽量接近；
    - 便于论文实验复现和问题排查。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置。

    返回的是 Python 字典，后续通过 key 访问各模块配置。
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(device_arg: str) -> torch.device:
    """选择训练设备。

    规则：
    - 如果用户显式传入 device，就按用户要求；
    - 否则自动优先用 CUDA。
    """
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: Dict[str, Any], sam_checkpoint: str, device: torch.device) -> EnhancedSAM:
    """根据配置构建 EnhancedSAM 模型。

    构建顺序：
    1) 加载 SAM 主体；
    2) 按训练分辨率对 SAM 做尺寸适配；
    3) 组装 LoRA + Boundary 配置；
    4) 生成 EnhancedSAM 并移动到 device。
    """
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    boundary_cfg = cfg["boundary"]
    loss_cfg = cfg["loss"]

    # 1) 先加载 SAM 主体。
    #    这一步会把官方预训练权重读进来。
    sam = load_sam_model(
        model_type=model_cfg["type"],
        checkpoint_path=sam_checkpoint,
        device=str(device),
    )

    # 2) 如果训练分辨率不是 SAM 默认 1024，则需要做位置编码插值适配。
    #    目的是让 ViT 的位置编码和新输入尺寸匹配。
    img_size = int(model_cfg["img_size"])
    if img_size != 1024:
        patch_sam_for_img_size(sam, img_size)

    # 3) 组装 EnhancedSAM 配置。
    #    注意：这里把 yaml 中的参数显式转成 int/float/bool，
    #    可以降低配置类型不一致导致的隐式错误风险。
    enhanced_cfg = EnhancedSAMConfig(
        vit_embed_dim=int(model_cfg["vit_embed_dim"]),
        prompt_embed_dim=int(model_cfg["prompt_embed_dim"]),
        use_lora=bool(lora_cfg["use_lora"]),
        lora_rank=int(lora_cfg["rank"]),
        lora_alpha=float(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        lora_target_modules=list(lora_cfg["target_modules"]),
        use_boundary=bool(boundary_cfg["use_boundary"]),
        boundary_mid_channels=int(boundary_cfg["mid_channels"]),
        boundary_num_iterations=int(boundary_cfg["num_iterations"]),
        boundary_detector_mid=int(boundary_cfg["detector_mid"]),
        loss_w_bce=float(loss_cfg["w_bce"]),
        loss_w_dice=float(loss_cfg["w_dice"]),
        loss_w_bound=float(loss_cfg["w_bound"]),
        loss_pos_weight=float(loss_cfg["pos_weight"]),
        loss_boundary_radius=int(loss_cfg["boundary_radius"]),
    )

    # 4) 构建并放到设备上。
    #    device 可能是 cuda 或 cpu。
    model = build_enhanced_sam(sam, config=enhanced_cfg).to(device)
    return model


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> Optimizer:
    """根据配置构建优化器。

    只会优化 requires_grad=True 的参数：
    - 典型是 LoRA 参数 + 边界细化分支参数；
    - 冻结的 SAM 主体参数不会参与更新。
    """
    optimizer_cfg = cfg["optimizer"]
    train_cfg = cfg["training"]

    # 只优化 requires_grad=True 的参数（例如 LoRA 与边界分支）。
    params = [p for p in model.parameters() if p.requires_grad]

    opt_type = optimizer_cfg["type"].lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
    eps = float(optimizer_cfg.get("eps", 1e-8))

    if opt_type == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if opt_type == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if opt_type == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"不支持的优化器类型: {opt_type}")


def build_scheduler(optimizer: Optimizer, cfg: Dict[str, Any], epochs: int):
    """构建学习率调度器。"""
    scheduler_cfg = cfg["scheduler"]
    scheduler_type = scheduler_cfg["type"].lower()

    if scheduler_type == "cosine":
        min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    raise ValueError(f"不支持的调度器类型: {scheduler_type}")


def select_logits(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """从模型输出中选出用于评估的 logits。

    逻辑：
    - 有 refined_mask（边界细化输出）时，优先使用它；
    - 否则从 masks 中选 IoU 最高的那张。
    """
    if "refined_mask" in outputs:
        # refined_mask 形状一般是 [B, 1, H, W]
        return outputs["refined_mask"]

    masks = outputs["masks"]
    iou_pred = outputs["iou_pred"]
    best_idx = iou_pred.argmax(dim=1, keepdim=True)
    h, w = masks.shape[-2:]
    # masks 形状通常是 [B, M, H, W]（M=候选掩码数量）
    # best_idx 形状是 [B,1]，代表每张图选哪一个候选掩码
    # gather 后得到 [B,1,H,W]
    return masks.gather(1, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))


def compute_batch_iou(pred_logits: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """计算一个 batch 的平均 IoU（用于训练监控）。

    输入：
    - pred_logits: [B,1,H,W] 的原始 logits（未过 sigmoid）
    - target_mask: [B,1,H,W] 或同等可插值尺寸

    流程：
    1) sigmoid -> 概率
    2) 阈值二值化 -> 0/1
    3) 计算交并比并在 batch 上取均值
    """
    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob >= threshold).float()

    if target_mask.shape[-2:] != pred_bin.shape[-2:]:
        # 预测和标签尺寸不一致时，把标签插值到预测尺寸
        # 最近邻插值可保持标签离散值，不会产生中间灰度
        target_mask = torch.nn.functional.interpolate(
            target_mask.float(),
            size=pred_bin.shape[-2:],
            mode="nearest",
        )
    target_bin = (target_mask >= 0.5).float()

    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = (pred_bin + target_bin).clamp(max=1).sum(dim=(1, 2, 3))
    iou = (inter / (union + 1e-6)).mean().item()
    return float(iou)


def train_one_epoch(
    model: EnhancedSAM,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    log_interval: int,
) -> Dict[str, float]:
    """训练一个 epoch。

    一个 epoch = 把训练集完整过一遍。
    返回该 epoch 的平均 loss 和平均 iou。
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    # tqdm 只是进度条显示，不影响训练逻辑
    progress = tqdm(enumerate(loader), total=len(loader), desc="Train", leave=False)
    for step, batch in progress:
        # image 形状: [B,3,H,W]
        # mask  形状: [B,1,H,W]
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        # 清空上一轮梯度，避免梯度累积污染本轮更新
        optimizer.zero_grad(set_to_none=True)

        # 前向传播：得到模型输出（masks/iou_pred/refined_mask 等）
        outputs = model(image=images, multimask=False)
        # 计算损失（内部会自动处理 refined_mask 优先策略）
        loss_dict = model.compute_loss(outputs, masks)
        loss = loss_dict["loss"]
        # 反向传播 + 参数更新
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # 训练监控指标：这里用 batch IoU
            pred_logits = select_logits(outputs)
            batch_iou = compute_batch_iou(pred_logits, masks)

        running_loss += float(loss.item())
        running_iou += batch_iou

        if step % max(log_interval, 1) == 0:
            # 实时打印 loss/iou/lr，便于看训练是否正常
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{batch_iou:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    n = max(len(loader), 1)
    return {
        "loss": running_loss / n,
        "iou": running_iou / n,
    }


@torch.no_grad()
def validate_one_epoch(
    model: EnhancedSAM,
    loader: Optional[DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    """验证一个 epoch（如果没有验证集，返回空指标）。

    与训练的关键差异：
    - model.eval()：关闭 dropout 等训练态行为
    - @torch.no_grad()：不计算梯度，节省显存与计算
    """
    if loader is None:
        return {"loss": float("nan"), "iou": float("nan")}

    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    progress = tqdm(loader, total=len(loader), desc="Val", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        outputs = model(image=images, multimask=False)
        loss_dict = model.compute_loss(outputs, masks)
        pred_logits = select_logits(outputs)
        batch_iou = compute_batch_iou(pred_logits, masks)

        running_loss += float(loss_dict["loss"].item())
        running_iou += batch_iou

    n = max(len(loader), 1)
    return {
        "loss": running_loss / n,
        "iou": running_iou / n,
    }


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: EnhancedSAM,
    optimizer: Optimizer,
    scheduler,
    best_score: float,
    is_best: bool,
) -> Path:
    """保存 checkpoint。

    保存内容包括：
    - 当前 epoch
    - 模型参数
    - 优化器状态
    - 调度器状态
    - 当前最优分数

    这样做的意义：
    - 可以断点恢复训练；
    - 可以完整复现实验过程。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
    }

    last_path = output_dir / "last_model.pth"
    torch.save(ckpt, last_path)

    if is_best:
        best_path = output_dir / "best_model.pth"
        torch.save(ckpt, best_path)

    return last_path


def main() -> None:
    # -------------------------------
    # 1) 准备阶段：参数、配置、随机种子、设备
    # -------------------------------
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    device = select_device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 读取配置中的主要训练参数，后续会传给各个子模块。
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    ckpt_cfg = cfg["checkpoint"]
    log_cfg = cfg["logging"]
    early_stop_cfg = cfg["early_stopping"]

    img_size = int(model_cfg["img_size"])
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    epochs = int(train_cfg["epochs"])

    # -------------------------------
    # 2) 数据阶段：构建 DataLoader
    # -------------------------------
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        hflip_p=float(aug_cfg.get("horizontal_flip", 0.5)),
        vflip_p=float(aug_cfg.get("vertical_flip", 0.5)),
    )
    print(f"[INFO] 训练样本数: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"[INFO] 验证样本数: {len(val_loader.dataset)}")
    else:
        print("[WARN] 未发现 val 集，将跳过验证。")

    # -------------------------------
    # 3) 模型阶段：构建模型、优化器、调度器
    # -------------------------------
    model = build_model(cfg=cfg, sam_checkpoint=args.sam_checkpoint, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs=epochs)

    # 记录一些基础信息，方便你后续排查和写论文实验设置。
    print(model.param_report())
    print(f"[INFO] 可训练参数组数: {len(optimizer.param_groups)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 4) 记录阶段：保存配置快照，保证实验可追踪
    # -------------------------------
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = output_dir / f"run_config_{run_tag}.json"
    snapshot = {
        "args": vars(args),
        "config": cfg,
        "enhanced_sam_config": asdict(model.config),
    }
    with snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 配置快照已保存: {snapshot_path}")

    # -------------------------------
    # 5) 恢复阶段：可选断点恢复
    # -------------------------------
    start_epoch = 1
    best_score = -1.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume 文件不存在: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        print(f"[INFO] 从 {resume_path} 恢复，起始 epoch={start_epoch}")

    # -------------------------------
    # 6) 训练控制参数：早停、监控指标、保存频率
    # -------------------------------
    early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
    patience = int(early_stop_cfg.get("patience", 10))
    min_delta = float(early_stop_cfg.get("min_delta", 1e-4))
    no_improve_count = 0

    monitor_name = ckpt_cfg.get("monitor", "val_iou")
    log_interval = int(log_cfg.get("log_interval", 10))
    save_interval = int(ckpt_cfg.get("save_interval", 1))

    # -------------------------------
    # 7) 主循环：epoch 级训练与验证
    # -------------------------------
    print("[INFO] 开始训练...")
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            log_interval=log_interval,
        )
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
        )

        # 处理调度器步进：
        # - plateau 需要喂入监控指标；
        # - 其他调度器直接 step 即可。
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric_for_scheduler = val_metrics["iou"] if not np.isnan(val_metrics["iou"]) else train_metrics["iou"]
            scheduler.step(metric_for_scheduler)
        else:
            scheduler.step()

        # 监控指标优先看 val_iou；若没有验证集，则退化为 train_iou。
        # current_score 会用于“最佳模型判断”和“早停判断”。
        if monitor_name == "val_iou":
            current_score = val_metrics["iou"] if not np.isnan(val_metrics["iou"]) else train_metrics["iou"]
        else:
            # 如果后续你要监控别的指标，可以在这里扩展。
            current_score = val_metrics["iou"] if not np.isnan(val_metrics["iou"]) else train_metrics["iou"]

        is_best = current_score > (best_score + min_delta)
        if is_best:
            best_score = current_score
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 按间隔保存 last_model，同时若是最佳则保存 best_model。
        if (epoch % save_interval == 0) or is_best or (epoch == epochs):
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
                is_best=is_best,
            )

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f}, train_iou={train_metrics['iou']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_iou={val_metrics['iou']:.4f}, "
            f"best={best_score:.4f}"
        )

        # 早停判断：如果连续 patience 个 epoch 没提升，就提前结束。
        if early_stop_enabled and (no_improve_count >= patience):
            print(
                f"[INFO] 触发早停：连续 {no_improve_count} 个 epoch 无提升（patience={patience}）。"
            )
            break

    # -------------------------------
    # 8) 收尾：打印最终结果
    # -------------------------------
    print("[INFO] 训练结束。")
    print(f"[INFO] 最优监控指标({monitor_name}) = {best_score:.4f}")


if __name__ == "__main__":
    main()
