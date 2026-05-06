"""数据加载模块（隧道裂缝分割任务）。

本模块的目标是把磁盘中的图像/标注，转换成模型训练可直接使用的张量。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Albumentations 是常用图像增强库，requirements 里已列出。
# 这里做了“可选导入”：
# - 如果环境里有它，就使用更规范的增强流水线；
# - 如果没有，也能走基础的 resize + 归一化逻辑，避免代码直接崩溃。
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except Exception:
    A = None  # type: ignore[assignment]
    ToTensorV2 = None  # type: ignore[assignment]
    HAS_ALBUMENTATIONS = False


# 常见图像后缀列表，用于自动发现样本文件。
IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class DatasetConfig:
    """数据集配置对象。

    这样做的好处是：参数集中管理，后续扩展（比如加入 cutmix/mosaic）更方便。
    """

    data_root: str
    split: str
    img_size: int = 512
    use_augment: bool = False
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    image_dir_name: str = "images"
    mask_dir_name: str = "annotations"


class TunnelCrackDataset(Dataset):
    """隧道裂缝分割数据集。

    目录约定：
        data_root/
          train/
            images/
            annotations/
          val/
            images/
            annotations/
          test/
            images/
            annotations/

    标注约定：
        - 二值分割图，背景=0，裂缝=255（或1）。
        - 代码中统一映射为 0/1 的 float 张量。
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg
        self.split_dir = Path(cfg.data_root) / cfg.split
        self.image_dir = self.split_dir / cfg.image_dir_name
        self.mask_dir = self.split_dir / cfg.mask_dir_name

        # 先做路径合法性检查，避免训练跑到中途才发现路径错误。
        if not self.image_dir.exists():
            raise FileNotFoundError(f"未找到图像目录: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"未找到标注目录: {self.mask_dir}")

        # 自动构建“图像-标注”配对列表。
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(
                f"在 {self.image_dir} 和 {self.mask_dir} 中没有找到可配对的样本，请检查文件命名是否一致。"
            )

        # 根据 split 决定是否启用增强。
        self.transform = self._build_transform(
            img_size=cfg.img_size,
            use_augment=cfg.use_augment,
            hflip_p=cfg.horizontal_flip_p,
            vflip_p=cfg.vertical_flip_p,
        )

    def _build_samples(self) -> List[Tuple[Path, Path]]:
        """构建样本配对列表。

        配对规则：图像文件名去掉后缀后的 stem 必须与标注 stem 一致。
        示例：
            images/0001.jpg  <-> annotations/0001.png
        """
        image_paths: List[Path] = []
        for suffix in IMAGE_SUFFIXES:
            image_paths.extend(self.image_dir.glob(f"*{suffix}"))
            image_paths.extend(self.image_dir.glob(f"*{suffix.upper()}"))

        image_paths = sorted(set(image_paths))
        samples: List[Tuple[Path, Path]] = []

        for img_path in image_paths:
            stem = img_path.stem
            mask_path = self._find_mask_by_stem(stem)
            if mask_path is not None:
                samples.append((img_path, mask_path))

        return samples

    def _find_mask_by_stem(self, stem: str) -> Optional[Path]:
        """根据 stem 在标注目录中找对应文件。

        常见情况是标注统一为 .png，这里也同时兼容其他常见后缀。
        """
        for suffix in IMAGE_SUFFIXES:
            candidate = self.mask_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
            candidate_upper = self.mask_dir / f"{stem}{suffix.upper()}"
            if candidate_upper.exists():
                return candidate_upper
        return None

    @staticmethod
    def _build_transform(
        img_size: int,
        use_augment: bool,
        hflip_p: float,
        vflip_p: float,
    ):
        """构建预处理/增强流水线。"""
        if HAS_ALBUMENTATIONS:
            ops: List = [A.Resize(height=img_size, width=img_size)]
            if use_augment:
                # 训练阶段常用的几种轻量增强：
                # - 水平/垂直翻转：增强方向鲁棒性；
                # - 亮度对比度扰动：增强光照鲁棒性。
                ops.extend(
                    [
                        A.HorizontalFlip(p=hflip_p),
                        A.VerticalFlip(p=vflip_p),
                        A.RandomBrightnessContrast(p=0.2),
                    ]
                )

            # Normalize + ToTensorV2 是深度学习常见组合：
            # - Normalize：把像素缩放到标准分布，提升训练稳定性；
            # - ToTensorV2：HWC -> CHW，并转为 torch.Tensor。
            ops.extend(
                [
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=255.0,
                    ),
                    ToTensorV2(),
                ]
            )
            return A.Compose(ops)

        # 没有 albumentations 的兜底分支：返回 None，后面走手工预处理。
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------
        # A. 先通过索引拿到“图像路径 + 标注路径”
        # -------------------------------
        # self.samples 的元素形态是 (Path, Path)
        # 例如: (train/images/0001.jpg, train/annotations/0001.png)
        img_path, mask_path = self.samples[index]

        # -------------------------------
        # B. 读取原始图像
        # -------------------------------
        # cv2.imread(..., IMREAD_COLOR) 得到形状: (H, W, 3)
        # 像素范围通常是 uint8 的 [0, 255]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"图像读取失败: {img_path}")
        # OpenCV 默认通道顺序是 BGR，但大多数深度学习模型按 RGB 训练
        # 所以这里必须转色彩通道顺序，避免训练效果异常
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # -------------------------------
        # C. 读取分割标注（mask）
        # -------------------------------
        # IMREAD_GRAYSCALE 会得到形状: (H, W)
        # 像素通常是 0 或 255（也可能是 0/1）
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"标注读取失败: {mask_path}")

        # 把标注统一成 float32 的二值图:
        # - 大于 127 的像素视为裂缝 -> 1.0
        # - 其余视为背景 -> 0.0
        # 处理后 mask 形状仍是 (H, W)，值域变成 {0.0, 1.0}
        mask = (mask > 127).astype(np.float32)

        # -------------------------------
        # D. 做预处理 / 数据增强
        # -------------------------------
        if self.transform is not None:
            # Albumentations 同时处理 image 和 mask，避免几何变换错位
            # transformed["image"] -> torch.Tensor, 形状 (3, H, W)
            # transformed["mask"]  -> torch.Tensor, 形状 (H, W)
            transformed = self.transform(image=image, mask=mask)
            image_tensor = transformed["image"].float()  # (3, H, W)
            mask_tensor = transformed["mask"].float()    # (H, W)
        else:
            # 兜底分支：手工 resize + normalize + tensor 化
            # 图像 resize 使用双线性插值，保持视觉平滑
            image = cv2.resize(image, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_LINEAR)
            # mask resize 使用最近邻插值，避免标签被插值成非 0/1 的脏值
            mask = cv2.resize(mask, (self.cfg.img_size, self.cfg.img_size), interpolation=cv2.INTER_NEAREST)

            # 先把像素从 [0,255] 缩放到 [0,1]
            image = image.astype(np.float32) / 255.0
            # 再按 ImageNet 统计量做标准化，提升训练稳定性
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            image = (image - mean) / std

            # numpy(H, W, C) -> torch(C, H, W)
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            # mask 转 tensor，当前仍是 (H, W)
            mask_tensor = torch.from_numpy(mask).float()

        # 分割任务里，mask 通常显式保留通道维度，统一为 (1, H, W)。
        # 这样后面和网络输出对齐更直接（输出一般也是 [B,1,H,W]）
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # -------------------------------
        # E. 返回一个样本字典
        # -------------------------------
        # image: (3, H, W) float32
        # mask : (1, H, W) float32, 值域 {0,1}
        # image_path/mask_path: 方便排查错误样本
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        }


def _build_dataset(
    data_root: str,
    split: str,
    img_size: int,
    use_augment: bool,
    hflip_p: float,
    vflip_p: float,
) -> Optional[TunnelCrackDataset]:
    """按 split 创建数据集对象。

    注意：
        如果 split 目录不存在（例如暂时没有 val 集），返回 None，
        由上层决定是否跳过验证流程。
    """
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        return None

    cfg = DatasetConfig(
        data_root=data_root,
        split=split,
        img_size=img_size,
        use_augment=use_augment,
        horizontal_flip_p=hflip_p,
        vertical_flip_p=vflip_p,
    )
    return TunnelCrackDataset(cfg)


def build_dataloaders(
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """构建训练/验证 DataLoader。

    返回：
        train_loader: 必须存在
        val_loader: 允许为 None（当 val 目录不存在时）
    """
    # 1) 构建训练集（启用增强）
    train_dataset = _build_dataset(
        data_root=data_root,
        split="train",
        img_size=img_size,
        use_augment=True,
        hflip_p=hflip_p,
        vflip_p=vflip_p,
    )
    if train_dataset is None:
        raise FileNotFoundError(
            f"未找到训练集目录: {Path(data_root) / 'train'}"
        )

    # 2) 构建验证集（默认不做增强，保证评估稳定）
    val_dataset = _build_dataset(
        data_root=data_root,
        split="val",
        img_size=img_size,
        use_augment=False,
        hflip_p=0.0,
        vflip_p=0.0,
    )

    # Windows 下多进程加载有时会遇到句柄/序列化问题，因此提供 pin_memory 和 persistent_workers 控制。
    # 3) 构建训练 DataLoader
    # shuffle=True：每个 epoch 打乱样本顺序，降低训练偏差
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        # 4) 构建验证 DataLoader
        # shuffle=False：验证集顺序固定，方便复现实验结果
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )

    return train_loader, val_loader
