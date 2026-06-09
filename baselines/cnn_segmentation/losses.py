"""CNN 分割基线使用的损失函数。

本模块与 SAM+LoRA 主线保持同一类二分类分割口径：BCEWithLogits +
Dice Loss。输入 logits 为 `[B,1,H,W]`，标签 mask 为 `[B,1,H,W]`。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """计算二分类 Dice Loss。

    输入:
    - logits: 模型原始输出 `[B,1,H,W]`；
    - targets: 二值 mask `[B,1,H,W]`，裂缝为 1，背景为 0。

    输出:
    - 标量 Dice Loss，值越小表示预测与标签重叠越好。

    为什么这样做:
    - 裂缝像素占比很低，仅用 BCE 容易被背景主导；
    - Dice Loss 直接优化前景区域重叠，对细长裂缝更友好。
    """
    probs = torch.sigmoid(logits)
    if targets.shape[-2:] != logits.shape[-2:]:
        targets = F.interpolate(targets.float(), size=logits.shape[-2:], mode="nearest")
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_bce: float = 1.0,
    w_dice: float = 2.0,
    pos_weight: float = 10.0,
) -> Dict[str, torch.Tensor]:
    """计算 CNN baseline 总损失。

    输入:
    - logits: `[B,1,H,W]`，未经过 sigmoid 的裂缝预测；
    - targets: `[B,1,H,W]`，二值裂缝标签；
    - w_bce/w_dice: 两项损失权重；
    - pos_weight: BCE 正样本权重，用于缓解裂缝像素稀疏问题。

    输出:
    - `loss`: 总损失；
    - `bce`: BCEWithLogits 损失；
    - `dice`: Dice Loss。

    为什么这样做:
    - 与当前 SAM+LoRA 的训练思想保持一致，便于论文中公平说明；
    - `pos_weight` 继续沿用主线实验设置，降低对比实验的变量数量。
    """
    if targets.shape[-2:] != logits.shape[-2:]:
        targets = F.interpolate(targets.float(), size=logits.shape[-2:], mode="nearest")
    targets = targets.float()
    weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
    dice = dice_loss_with_logits(logits, targets)
    loss = float(w_bce) * bce + float(w_dice) * dice
    return {"loss": loss, "bce": bce, "dice": dice}
