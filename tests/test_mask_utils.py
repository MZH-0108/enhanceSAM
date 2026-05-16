"""mask 极性归一化测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from baselines.sam_vanilla.common import load_gt_mask_binary
from utils.data_loader import DatasetConfig, TunnelCrackDataset
from utils.mask_utils import normalize_crack_mask


def _write_png(path: Path, array: np.ndarray) -> None:
    """写入测试 PNG，输入为 numpy 图像，输出为磁盘文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), array)
    assert ok


def test_normalize_crack_mask_supports_white_crack() -> None:
    """黑底白裂缝应保持白色区域为前景。

    输入:
    - 8x8 灰度 mask，背景为 0，裂缝列为 255。

    输出:
    - 归一化后裂缝列为 1，背景为 0。

    为什么这样做:
    - mosaic 样本属于这种极性，不能被错误取反。
    """
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 3] = 255

    normalized = normalize_crack_mask(mask)

    assert normalized.dtype == np.uint8
    assert normalized[:, 3].sum() == 8
    assert normalized.sum() == 8


def test_normalize_crack_mask_supports_black_crack() -> None:
    """白底黑裂缝应取反，让黑色裂缝成为前景。

    输入:
    - 8x8 灰度 mask，背景为 255，裂缝列为 0。

    输出:
    - 归一化后裂缝列为 1，背景为 0。

    为什么这样做:
    - 非 mosaic 样本属于这种极性；若不取反，模型会学习背景而不是裂缝。
    """
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[:, 3] = 0

    normalized = normalize_crack_mask(mask)

    assert normalized.dtype == np.uint8
    assert normalized[:, 3].sum() == 8
    assert normalized.sum() == 8


def test_dataset_and_baseline_share_mask_polarity(tmp_path: Path) -> None:
    """DataLoader 与 baseline 读取同一白底黑裂缝标注时应得到同一前景。

    输入:
    - synthetic image: 16x16 RGB。
    - synthetic annotation: 白底黑色竖线裂缝。

    输出:
    - DataLoader resize 后的 mask 非空且前景较少。
    - baseline 读取出的原始尺寸 mask 中，黑色裂缝列为 1。

    为什么这样做:
    - 训练、评估、SAM-Box-Oracle 若极性不一致，会得到不可比较的指标和图。
    """
    image_dir = tmp_path / "train" / "images"
    mask_dir = tmp_path / "train" / "annotations"
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    mask[4:12, 7:9] = 0
    _write_png(image_dir / "sample_01.png", image)
    _write_png(mask_dir / "sample_01.png", mask)

    dataset = TunnelCrackDataset(
        DatasetConfig(data_root=str(tmp_path), split="train", img_size=16, use_augment=False)
    )
    sample = dataset[0]
    baseline_mask = load_gt_mask_binary(mask_dir / "sample_01.png")

    assert sample["mask"].shape == (1, 16, 16)
    assert int(sample["mask"].sum().item()) == 16
    assert baseline_mask[4:12, 7:9].sum() == 16
    assert baseline_mask.sum() == 16

