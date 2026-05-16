"""可视化工具（推理/评估共用）。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def probability_to_u8(prob: np.ndarray) -> np.ndarray:
    """把 [0,1] 概率图转成 uint8 灰度图。"""
    return np.clip(prob * 255.0, 0, 255).astype(np.uint8)


def binary01_to_u8(mask01: np.ndarray) -> np.ndarray:
    """把 0/1 二值图转成 0/255 uint8 图。"""
    return (mask01.astype(np.uint8) * 255).astype(np.uint8)


def overlay_mask_on_bgr(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """把二值 mask 覆盖到 BGR 图像。"""
    overlay = image_bgr.copy()
    color_layer = np.zeros_like(overlay, dtype=np.uint8)
    color_layer[:, :] = np.array(color_bgr, dtype=np.uint8)

    region = mask_u8 > 0
    mixed = cv2.addWeighted(overlay, 1.0 - alpha, color_layer, alpha, 0.0)
    overlay[region] = mixed[region]
    return overlay


def save_predict_artifacts(
    output_dir: Path,
    stem: str,
    mask_u8: np.ndarray,
    save_overlay: bool = False,
    overlay_bgr: Optional[np.ndarray] = None,
    save_prob: bool = False,
    prob_float01: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """统一保存推理输出，并返回文件路径字典。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask_u8)

    overlay_path = output_dir / f"{stem}_overlay.png"
    if save_overlay and overlay_bgr is not None:
        cv2.imwrite(str(overlay_path), overlay_bgr)

    prob_path = output_dir / f"{stem}_prob.png"
    if save_prob and prob_float01 is not None:
        cv2.imwrite(str(prob_path), probability_to_u8(prob_float01))

    return {
        "mask": str(mask_path),
        "overlay": str(overlay_path) if (save_overlay and overlay_bgr is not None) else "",
        "prob": str(prob_path) if (save_prob and prob_float01 is not None) else "",
    }
