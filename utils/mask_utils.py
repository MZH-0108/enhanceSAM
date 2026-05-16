"""Mask polarity utilities for crack segmentation."""

from __future__ import annotations

import numpy as np


def normalize_crack_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """把原始标注统一转换为“裂缝=1、背景=0”的二值 mask。

    输入:
    - mask: OpenCV 读取的灰度标注图，shape 为 [H, W]，通常像素值为 0/255。
    - threshold: 二值化阈值，大于该阈值先视为白色区域。

    输出:
    - crack01: uint8 二值图，shape 为 [H, W]，取值只包含 {0, 1}。

    为什么这样做:
    - 当前数据集同时存在“黑底白裂缝”和“白底黑裂缝”两种 annotation。
    - 裂缝在该数据集中始终是少数像素；因此先按阈值得到 white01，
      若 white01 占比超过 50%，说明白色更可能是背景，需要取反。
    - 统一极性后，训练、评估、baseline box prompt 和论文可视化才在同一语义下工作。
    """
    if mask.ndim != 2:
        raise ValueError(f"mask 必须是二维灰度图，实际 shape={mask.shape}")

    white01 = (mask > threshold).astype(np.uint8)
    if float(white01.mean()) > 0.5:
        return (1 - white01).astype(np.uint8)
    return white01

