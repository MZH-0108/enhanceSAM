"""SAM Vanilla Baseline 公共工具。

本文件提供两类能力：
1. 数据样本配对与读取（image + annotation）
2. 指标统计与结果保存

这样做的目的：
- 避免 eval_amg.py / eval_box_oracle.py 复制大量重复代码；
- 保证两种 baseline 的评估口径完全一致，便于论文横向对比。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

# 常见图像后缀，支持混合格式（jpg/png/tif...）
IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class Sample:
    """一个数据样本（原图路径 + 标注路径）。"""

    image_path: Path
    mask_path: Path


def _list_files_by_suffix(folder: Path) -> List[Path]:
    """列出目录中所有支持后缀的文件。"""
    files: List[Path] = []
    for suffix in IMAGE_SUFFIXES:
        files.extend(folder.glob(f"*{suffix}"))
        files.extend(folder.glob(f"*{suffix.upper()}"))
    return sorted(set(files))


def collect_samples(data_root: str, split: str) -> List[Sample]:
    """从 data/<split>/images + annotations 收集可配对样本。

    配对规则：
    - 以 stem（不含后缀的文件名）做匹配；
    - 例如 `abc.jpg` 对 `abc.png` 视为一对。
    """
    split_dir = Path(data_root) / split
    image_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"

    if not image_dir.exists():
        raise FileNotFoundError(f"未找到图像目录: {image_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"未找到标注目录: {ann_dir}")

    images = _list_files_by_suffix(image_dir)
    anns = _list_files_by_suffix(ann_dir)

    ann_map = {p.stem: p for p in anns}
    samples: List[Sample] = []
    for img in images:
        ann = ann_map.get(img.stem)
        if ann is not None:
            samples.append(Sample(image_path=img, mask_path=ann))

    if not samples:
        raise RuntimeError(
            f"在 {image_dir} 与 {ann_dir} 中没有找到可配对样本。"
        )
    return samples


def load_image_bgr(path: Path) -> np.ndarray:
    """读取 BGR 原图。"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"图像读取失败: {path}")
    return img


def load_gt_mask_binary(path: Path) -> np.ndarray:
    """读取 GT mask 并转为 0/1 二值图。

    输出：
    - shape: [H, W]
    - dtype: uint8
    - values: {0, 1}
    """
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"标注读取失败: {path}")
    return (m > 127).astype(np.uint8)


def mask_to_bbox_xyxy(mask01: np.ndarray, padding: int = 0) -> Optional[List[int]]:
    """把单个二值 mask 转成最小外接框 [x1,y1,x2,y2]。

    规则：
    - 若无前景像素，返回 None；
    - 可选 padding，避免框太紧。
    """
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    if padding > 0:
        h, w = mask01.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w - 1, x2 + padding)
        y2 = min(h - 1, y2 + padding)
    return [x1, y1, x2, y2]


def split_connected_components(
    mask01: np.ndarray,
    min_area: int,
    padding: int,
) -> List[List[int]]:
    """把一个 mask 拆成多个连通域框（可用于多裂缝目标）。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    bboxes: List[List[int]] = []
    h, w = mask01.shape[:2]

    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        x1, y1 = x, y
        x2, y2 = x + bw - 1, y + bh - 1

        if padding > 0:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w - 1, x2 + padding)
            y2 = min(h - 1, y2 + padding)
        bboxes.append([x1, y1, x2, y2])
    return bboxes


class MetricMeter:
    """指标累积器（数据集级评估）。"""

    def __init__(self, boundary_radius: int = 3) -> None:
        self.boundary_radius = int(boundary_radius)
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.boundary_iou_sum = 0.0
        self.num_images = 0
        self.total_infer_time_sec = 0.0

    def update(
        self,
        pred01: np.ndarray,
        gt01: np.ndarray,
        infer_time_sec: float,
    ) -> None:
        """更新一张图像的统计。"""
        pred = (pred01 > 0).astype(np.uint8)
        gt = (gt01 > 0).astype(np.uint8)

        self.tp += float(np.logical_and(pred == 1, gt == 1).sum())
        self.fp += float(np.logical_and(pred == 1, gt == 0).sum())
        self.fn += float(np.logical_and(pred == 0, gt == 1).sum())
        self.tn += float(np.logical_and(pred == 0, gt == 0).sum())

        self.boundary_iou_sum += boundary_iou(pred, gt, radius=self.boundary_radius)
        self.total_infer_time_sec += float(infer_time_sec)
        self.num_images += 1

    def summary(self) -> Dict[str, float]:
        """返回最终汇总指标。"""
        eps = 1e-6
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        boundary_iou_mean = self.boundary_iou_sum / max(self.num_images, 1)

        sec_per_img = self.total_infer_time_sec / max(self.num_images, 1)
        ms_per_img = sec_per_img * 1000.0
        fps = 1.0 / max(sec_per_img, 1e-9)

        return {
            "mIoU": float(iou),
            "Dice": float(dice),
            "Precision": float(precision),
            "Recall": float(recall),
            "Boundary-IoU": float(boundary_iou_mean),
            "ms_per_image": float(ms_per_img),
            "FPS": float(fps),
        }


def boundary_iou(pred01: np.ndarray, gt01: np.ndarray, radius: int = 3) -> float:
    """边界 IoU（Boundary-IoU）。"""
    pred_b = boundary_mask(pred01, radius=radius)
    gt_b = boundary_mask(gt01, radius=radius)

    inter = float(np.logical_and(pred_b > 0, gt_b > 0).sum())
    union = float(np.logical_or(pred_b > 0, gt_b > 0).sum())
    return inter / (union + 1e-6)


def boundary_mask(mask01: np.ndarray, radius: int) -> np.ndarray:
    """通过膨胀 - 原图得到边界区域。"""
    mask_u8 = (mask01 > 0).astype(np.uint8)
    k = 2 * int(radius) + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    b = np.clip(dilated - mask_u8, 0, 1).astype(np.uint8)
    return b


def save_json(path: str, payload: Dict) -> None:
    """保存 JSON（自动创建父目录）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

