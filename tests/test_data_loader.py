"""数据加载器测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import utils.data_loader as data_loader_module
from utils.data_loader import DatasetConfig, TunnelCrackDataset


def _write_png(path: Path, array: np.ndarray) -> None:
    """写入 PNG 测试文件，失败时直接让测试中断。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), array)
    assert ok


def test_dataset_skips_invalid_image_header(tmp_path: Path) -> None:
    """坏图像文件应在构建样本时被跳过，而不是训练中途崩溃。

    场景：
    - good_01.png 是正常图像和正常标注；
    - bad_01.png 文件大小非 0，但内容全 0，不是合法 PNG。

    断言：
    - Dataset 初始化阶段会发出 RuntimeWarning；
    - 最终只保留 good_01 这一对样本；
    - 返回张量 shape 与训练脚本约定一致：image=[3,H,W]，mask=[1,H,W]。
    """
    image_dir = tmp_path / "train" / "images"
    mask_dir = tmp_path / "train" / "annotations"

    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 7:9] = 255
    _write_png(image_dir / "good_01.png", image)
    _write_png(mask_dir / "good_01.png", mask)

    # 该文件模拟真实数据中的损坏 PNG：扩展名正确、大小非 0，但文件头不是 PNG 魔数。
    (image_dir / "bad_01.png").write_bytes(b"\x00" * 128)
    _write_png(mask_dir / "bad_01.png", mask)

    cfg = DatasetConfig(data_root=str(tmp_path), split="train", img_size=32, use_augment=False)
    with pytest.warns(RuntimeWarning, match="跳过不可识别图像文件"):
        dataset = TunnelCrackDataset(cfg)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert sample["image_path"].endswith("good_01.png")


def test_dataset_skips_corrupt_png_with_valid_header(tmp_path: Path) -> None:
    """文件头合法但内容损坏的 PNG 也应提前跳过。

    场景：
    - good_01.png 是正常图像和正常标注；
    - corrupt_01.png 以 PNG 魔数开头，但后续 IDAT/数据内容不完整，OpenCV 无法解码。

    断言：
    - Dataset 初始化阶段会发出 RuntimeWarning；
    - 损坏样本不会进入 samples；
    - 后续训练迭代只会看到可正常读取的样本。
    """
    image_dir = tmp_path / "train" / "images"
    mask_dir = tmp_path / "train" / "annotations"

    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 7:9] = 255
    _write_png(image_dir / "good_01.png", image)
    _write_png(mask_dir / "good_01.png", mask)

    # 该文件模拟真实训练中遇到的损坏 PNG：
    # - 前 8 字节是合法 PNG 魔数；
    # - 后续内容不足以构成完整 PNG，cv2.imdecode 会返回 None。
    (image_dir / "corrupt_01.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
    _write_png(mask_dir / "corrupt_01.png", mask)

    cfg = DatasetConfig(data_root=str(tmp_path), split="train", img_size=32, use_augment=False)
    with pytest.warns(RuntimeWarning, match="跳过无法解码图像文件"):
        dataset = TunnelCrackDataset(cfg)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert sample["image_path"].endswith("good_01.png")


def test_getitem_falls_back_when_runtime_decode_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """运行时单个样本读取失败时，应跳到后续可读样本。

    场景：
    - 初始化时 first_01.png 和 second_01.png 都是正常文件；
    - 初始化后模拟 first_01.png 在训练迭代中解码失败。

    断言：
    - `dataset[0]` 不抛异常；
    - 返回的是后续可读样本 second_01.png；
    - 这样长训练不会因为一次 libpng 解码失败直接中断。
    """
    image_dir = tmp_path / "train" / "images"
    mask_dir = tmp_path / "train" / "annotations"

    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 7:9] = 255
    _write_png(image_dir / "first_01.png", image)
    _write_png(mask_dir / "first_01.png", mask)
    _write_png(image_dir / "second_01.png", image)
    _write_png(mask_dir / "second_01.png", mask)

    cfg = DatasetConfig(data_root=str(tmp_path), split="train", img_size=32, use_augment=False)
    dataset = TunnelCrackDataset(cfg)

    original_read = data_loader_module._read_image_with_imdecode

    def flaky_read(path: Path, flags: int):
        """模拟训练运行时第一张图瞬时解码失败。"""
        if path.name == "first_01.png" and path.parent.name == "images":
            return None
        return original_read(path, flags)

    monkeypatch.setattr(data_loader_module, "_read_image_with_imdecode", flaky_read)

    with pytest.warns(RuntimeWarning, match="运行时跳过图像读取失败样本"):
        sample = dataset[0]

    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert sample["image_path"].endswith("second_01.png")
