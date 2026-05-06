"""评估脚本（LoRA/SAM 分割模型）。

这个脚本负责把“训练好的 checkpoint”在指定数据集划分（val/test）上做统一评估，
并输出你论文常用的核心指标：
- mIoU
- Dice
- Precision
- Recall
- Boundary-IoU
- 推理速度（ms/图、FPS）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# 保证从任意工作目录运行脚本时，仍能导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.boundary_refinement import BoundaryLoss
from models.enhanced_sam import EnhancedSAM, EnhancedSAMConfig, build_enhanced_sam
from models.sam_base import load_sam_model, patch_sam_for_img_size
from utils.data_loader import DatasetConfig, TunnelCrackDataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate EnhancedSAM on tunnel crack dataset")
    parser.add_argument("--data_root", type=str, default="data", help="数据集根目录")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="评估数据划分")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM 预训练权重路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练得到的模型 checkpoint 路径")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="训练配置文件路径")
    parser.add_argument("--device", type=str, default="", help="评估设备，如 cuda/cpu；留空自动选择")
    parser.add_argument("--batch_size", type=int, default=0, help="评估 batch size，0 表示沿用配置文件")
    parser.add_argument("--num_workers", type=int, default=-1, help="评估 num_workers，-1 表示沿用配置文件")
    parser.add_argument("--threshold", type=float, default=0.5, help="概率图二值化阈值")
    parser.add_argument("--output", type=str, default="results/eval_metrics.json", help="评估结果输出 JSON 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，保证可复现。"""
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

    评估阶段通常不做随机增强，因此 use_augment=False。
    """
    cfg = DatasetConfig(
        data_root=data_root,
        split=split,
        img_size=img_size,
        use_augment=False,
        horizontal_flip_p=0.0,
        vertical_flip_p=0.0,
    )
    dataset = TunnelCrackDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def build_model(cfg: Dict[str, Any], sam_checkpoint: str, device: torch.device) -> EnhancedSAM:
    """按配置构建评估模型。"""
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    boundary_cfg = cfg["boundary"]
    loss_cfg = cfg["loss"]

    sam = load_sam_model(
        model_type=model_cfg["type"],
        checkpoint_path=sam_checkpoint,
        device=str(device),
    )

    img_size = int(model_cfg["img_size"])
    if img_size != 1024:
        patch_sam_for_img_size(sam, img_size)

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
    return build_enhanced_sam(sam, config=enhanced_cfg).to(device)


def load_trained_weights(model: EnhancedSAM, checkpoint_path: str, device: torch.device) -> int:
    """加载训练权重，返回 epoch 信息。"""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return int(ckpt.get("epoch", -1))

    # 兼容“纯 state_dict 文件”格式
    model.load_state_dict(ckpt, strict=True)
    return -1


def select_logits(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """从模型输出中选最终预测 logits。"""
    if "refined_mask" in outputs:
        return outputs["refined_mask"]  # [B,1,H,W]
    masks = outputs["masks"]            # [B,M,H,W]
    iou_pred = outputs["iou_pred"]      # [B,M]
    best_idx = iou_pred.argmax(dim=1, keepdim=True)
    h, w = masks.shape[-2:]
    return masks.gather(1, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))


def ensure_mask_shape(mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """把标签 mask 调整到目标分辨率。"""
    if mask.shape[-2:] == target_hw:
        return mask
    return torch.nn.functional.interpolate(mask.float(), size=target_hw, mode="nearest")


def compute_confusion_counts(pred_bin: torch.Tensor, target_bin: torch.Tensor) -> Dict[str, float]:
    """统计 TP/FP/FN/TN（按像素级）。"""
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1.0 - target_bin)).sum().item()
    fn = ((1.0 - pred_bin) * target_bin).sum().item()
    tn = ((1.0 - pred_bin) * (1.0 - target_bin)).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_boundary_iou(pred_bin: torch.Tensor, target_bin: torch.Tensor, radius: int) -> float:
    """计算 Boundary-IoU（像素级边界区域 IoU）。"""
    pred_boundary = BoundaryLoss.get_boundary_mask(pred_bin, radius=radius)
    gt_boundary = BoundaryLoss.get_boundary_mask(target_bin, radius=radius)

    inter = (pred_boundary * gt_boundary).sum().item()
    union = ((pred_boundary + gt_boundary) > 0).float().sum().item()
    return float(inter / (union + 1e-6))


@torch.no_grad()
def evaluate(
    model: EnhancedSAM,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    boundary_radius: int,
) -> Dict[str, float]:
    """执行完整评估并返回指标。"""
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_boundary_iou = 0.0
    total_infer_time = 0.0

    # 全局累积 TP/FP/FN/TN，最后算“数据集级”指标。
    global_tp = 0.0
    global_fp = 0.0
    global_fn = 0.0
    global_tn = 0.0

    progress = tqdm(loader, total=len(loader), desc="Eval", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bsz = images.shape[0]

        # 计时前向推理（不含后处理），用于统计速度。
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(image=images, multimask=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_infer_time += (time.perf_counter() - t0)

        loss_dict = model.compute_loss(outputs, masks)
        pred_logits = select_logits(outputs)

        # 概率化 + 二值化
        pred_prob = torch.sigmoid(pred_logits)
        pred_bin = (pred_prob >= threshold).float()

        # 标签尺寸对齐
        masks = ensure_mask_shape(masks, target_hw=pred_bin.shape[-2:])
        target_bin = (masks >= 0.5).float()

        # 统计混淆矩阵元素
        counts = compute_confusion_counts(pred_bin, target_bin)
        global_tp += counts["tp"]
        global_fp += counts["fp"]
        global_fn += counts["fn"]
        global_tn += counts["tn"]

        # Boundary-IoU（按 batch 求值后再按样本加权）
        bnd_iou = compute_boundary_iou(pred_bin, target_bin, radius=boundary_radius)
        total_boundary_iou += bnd_iou * bsz

        total_loss += float(loss_dict["loss"].item()) * bsz
        total_samples += bsz

    eps = 1e-6
    precision = global_tp / (global_tp + global_fp + eps)
    recall = global_tp / (global_tp + global_fn + eps)
    iou = global_tp / (global_tp + global_fp + global_fn + eps)
    dice = (2.0 * global_tp) / (2.0 * global_tp + global_fp + global_fn + eps)

    avg_loss = total_loss / max(total_samples, 1)
    avg_boundary_iou = total_boundary_iou / max(total_samples, 1)
    ms_per_image = (total_infer_time / max(total_samples, 1)) * 1000.0
    fps = 1.0 / max(total_infer_time / max(total_samples, 1), 1e-9)

    return {
        "loss": float(avg_loss),
        "mIoU": float(iou),  # 二分类前景任务里 mIoU 与前景 IoU 一致使用
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "Boundary-IoU": float(avg_boundary_iou),
        "ms_per_image": float(ms_per_image),
        "FPS": float(fps),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = select_device(args.device)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]

    img_size = int(model_cfg["img_size"])
    batch_size = int(args.batch_size) if args.batch_size > 0 else int(train_cfg["batch_size"])
    num_workers = int(args.num_workers) if args.num_workers >= 0 else int(train_cfg["num_workers"])
    boundary_radius = int(loss_cfg.get("boundary_radius", 3))

    loader = build_eval_loader(
        data_root=args.data_root,
        split=args.split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = build_model(cfg=cfg, sam_checkpoint=args.sam_checkpoint, device=device)
    loaded_epoch = load_trained_weights(model, args.checkpoint, device=device)

    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 评估 split: {args.split}")
    print(f"[INFO] 样本数量: {len(loader.dataset)}")
    print(f"[INFO] 加载 checkpoint: {args.checkpoint} (epoch={loaded_epoch})")

    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        threshold=float(args.threshold),
        boundary_radius=boundary_radius,
    )

    # 输出到终端
    print("[INFO] 评估结果:")
    for k, v in metrics.items():
        if k in {"ms_per_image", "FPS"}:
            print(f"  - {k}: {v:.4f}")
        else:
            print(f"  - {k}: {v:.6f}")

    # 输出到 JSON 文件
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_root": args.data_root,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "sam_checkpoint": args.sam_checkpoint,
        "config": args.config,
        "threshold": args.threshold,
        "metrics": metrics,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 评估结果已保存: {out_path}")


if __name__ == "__main__":
    main()

