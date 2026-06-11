"""分割任务评估工具（统一指标口径）。

本模块提供：
1) 从模型输出中抽取最终 mask logits；
2) 预测/标签尺寸对齐与二值化；
3) TP/FP/FN/TN 累积统计；
4) mIoU/Dice（micro 全局像素聚合）+ mIoU_macro/Dice_macro（逐图平均）
   / Precision / Recall / Boundary-IoU / 速度指标汇总。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def select_final_logits(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """从模型输出中选取最终用于评估的 logits。

    规则：
    - 若有边界细化分支输出 `refined_mask`，优先使用；
    - 否则从 `masks` 中按 `iou_pred` 选择最佳候选 mask。
    """
    if "refined_mask" in outputs:
        return outputs["refined_mask"]  # [B, 1, H, W]

    masks = outputs["masks"]        # [B, M, H, W]
    iou_pred = outputs["iou_pred"]  # [B, M]
    best_idx = iou_pred.argmax(dim=1, keepdim=True)  # [B,1]
    h, w = masks.shape[-2:]
    return masks.gather(1, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))


def ensure_mask_shape(mask: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """把标签 mask 调整到目标分辨率。"""
    if mask.shape[-2:] == target_hw:
        return mask
    return F.interpolate(mask.float(), size=target_hw, mode="nearest")


def to_binary(prob_or_mask: torch.Tensor, threshold: float) -> torch.Tensor:
    """按阈值转二值（0/1 float）。"""
    return (prob_or_mask >= threshold).float()


def _boundary_mask(mask01: torch.Tensor, radius: int) -> torch.Tensor:
    """把二值 mask 转为边界区域 mask（0/1）。"""
    if radius <= 0:
        return mask01
    k = 2 * int(radius) + 1
    dilated = F.max_pool2d(mask01.float(), kernel_size=k, stride=1, padding=radius)
    boundary = (dilated - mask01.float()).clamp(0.0, 1.0)
    return boundary


def boundary_iou(pred01: torch.Tensor, gt01: torch.Tensor, radius: int) -> float:
    """计算 Boundary-IoU。"""
    pred_b = _boundary_mask(pred01, radius=radius)
    gt_b = _boundary_mask(gt01, radius=radius)
    inter = (pred_b * gt_b).sum().item()
    union = ((pred_b + gt_b) > 0).float().sum().item()
    return float(inter / (union + 1e-6))


@dataclass
class _Confusion:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0


class SegmentationMetricMeter:
    """分割指标累积器（数据集级）。"""

    def __init__(self, boundary_radius: int = 3) -> None:
        self.boundary_radius = int(boundary_radius)
        self.conf = _Confusion()
        # 逐图(macro)累积器：每张图各自算 IoU/Dice 再累加，summary 时除以样本数得样本平均。
        # 与全局(micro)混淆矩阵并列：micro 易被大前景样本主导，macro 更贴近“平均每张图”的观感。
        self.total_img_iou = 0.0
        self.total_img_dice = 0.0
        self.total_boundary_iou = 0.0
        self.total_samples = 0
        self.total_infer_time_sec = 0.0
        self.total_loss = 0.0
        self.loss_count = 0

    def update(
        self,
        pred_bin: torch.Tensor,
        target_bin: torch.Tensor,
        infer_time_sec: float,
        loss_value: Optional[float] = None,
    ) -> None:
        """更新一个 batch 的统计。"""
        bsz = int(pred_bin.shape[0])

        self.conf.tp += float((pred_bin * target_bin).sum().item())
        self.conf.fp += float((pred_bin * (1.0 - target_bin)).sum().item())
        self.conf.fn += float(((1.0 - pred_bin) * target_bin).sum().item())
        self.conf.tn += float(((1.0 - pred_bin) * (1.0 - target_bin)).sum().item())

        # 逐图(macro)IoU/Dice：对每个样本在 (C,H,W) 维度各自求 tp/fp/fn，
        # 单独算 IoU/Dice 后累加；这样占多数的细裂缝小图也能等权参与平均，
        # 不会像 micro 那样被少数大前景样本淹没。
        eps = 1e-6
        per_img_dims = tuple(range(1, pred_bin.ndim))
        tp_i = (pred_bin * target_bin).sum(dim=per_img_dims)
        fp_i = (pred_bin * (1.0 - target_bin)).sum(dim=per_img_dims)
        fn_i = ((1.0 - pred_bin) * target_bin).sum(dim=per_img_dims)
        iou_i = tp_i / (tp_i + fp_i + fn_i + eps)
        dice_i = (2.0 * tp_i) / (2.0 * tp_i + fp_i + fn_i + eps)
        self.total_img_iou += float(iou_i.sum().item())
        self.total_img_dice += float(dice_i.sum().item())

        self.total_boundary_iou += boundary_iou(pred_bin, target_bin, radius=self.boundary_radius) * bsz
        self.total_infer_time_sec += float(infer_time_sec)
        self.total_samples += bsz

        if loss_value is not None:
            self.total_loss += float(loss_value) * bsz
            self.loss_count += bsz

    def summary(self) -> Dict[str, float]:
        """返回汇总指标。"""
        eps = 1e-6
        tp = self.conf.tp
        fp = self.conf.fp
        fn = self.conf.fn

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        miou = tp / (tp + fp + fn + eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)

        sec_per_image = self.total_infer_time_sec / max(self.total_samples, 1)
        ms_per_image = sec_per_image * 1000.0
        fps = 1.0 / max(sec_per_image, 1e-9)

        metrics = {
            # micro（全局像素聚合）口径：本质是“裂缝前景类”的全局 IoU/Dice，
            # 非跨类别平均，且会被大前景样本主导。保留原 key 不变以兼容历史结果。
            "mIoU": float(miou),
            "Dice": float(dice),
            # macro（逐图平均）口径：更贴近“平均每张图分得怎样”，论文建议同时报告。
            "mIoU_macro": float(self.total_img_iou / max(self.total_samples, 1)),
            "Dice_macro": float(self.total_img_dice / max(self.total_samples, 1)),
            "Precision": float(precision),
            "Recall": float(recall),
            "Boundary-IoU": float(self.total_boundary_iou / max(self.total_samples, 1)),
            "ms_per_image": float(ms_per_image),
            "FPS": float(fps),
        }
        if self.loss_count > 0:
            metrics["loss"] = float(self.total_loss / max(self.loss_count, 1))
        return metrics
