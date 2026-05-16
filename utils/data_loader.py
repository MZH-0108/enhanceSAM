"""数据加载模块（隧道裂缝分割任务）。

本模块的目标是把磁盘中的图像/标注，转换成模型训练可直接使用的张量。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.mask_utils import normalize_crack_mask

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


# 文件头魔数校验表：
# - 输入：图像/标注文件路径；
# - 输出：快速判断文件内容是否至少符合扩展名对应的常见图像格式；
# - 为什么这样做：真实数据集中可能存在大小非 0 但内容全 0 或截断的坏文件，
#   若等到 DataLoader worker 中才发现，会导致训练跑到中途崩溃且定位成本高。
IMAGE_MAGIC_PREFIXES: Dict[str, Tuple[bytes, ...]] = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8",),
    ".jpeg": (b"\xff\xd8",),
    ".bmp": (b"BM",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
}


def _has_valid_image_header(path: Path) -> bool:
    """轻量检查图像文件头是否与扩展名匹配。

    输入：
    - path: 图像或标注文件路径。

    输出：
    - True: 文件头与扩展名对应格式匹配；
    - False: 文件为空、读取失败或文件头明显非法。

    说明：
    - 这里只读前 16 个字节，不完整解码整张图，避免数据集初始化过慢；
    - 它不能替代完整解码校验，但能拦截当前训练中遇到的“全 0 PNG”坏样本。
    """
    prefixes = IMAGE_MAGIC_PREFIXES.get(path.suffix.lower())
    if prefixes is None:
        return False

    try:
        with path.open("rb") as f:
            header = f.read(16)
    except OSError:
        return False

    return any(header.startswith(prefix) for prefix in prefixes)


def _read_image_with_imdecode(path: Path, flags: int) -> Optional[np.ndarray]:
    """稳健读取图像。

    输入：
    - path: 图像路径；
    - flags: OpenCV 解码模式，如 `cv2.IMREAD_COLOR` 或 `cv2.IMREAD_GRAYSCALE`。

    输出：
    - 解码后的 numpy 数组；若读取或解码失败则返回 None。

    为什么这样做：
    - `cv2.imread(str(path))` 在 Windows 上遇到中文路径或特殊路径时偶尔会失败；
    - `np.fromfile + cv2.imdecode` 对路径编码更稳健，训练和可视化脚本保持同一读取策略。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _can_decode_image_file(path: Path, flags: int) -> bool:
    """完整校验图像文件是否可被 OpenCV 解码。

    输入：
    - path: 图像或标注文件路径；
    - flags: OpenCV 解码模式，彩色图像用 `cv2.IMREAD_COLOR`，标注用 `cv2.IMREAD_GRAYSCALE`。

    输出：
    - True: 文件头合法，且 `cv2.imdecode` 能实际返回图像数组；
    - False: 文件头非法、文件不可读、或文件内容损坏导致解码失败。

    为什么这样做：
    - 仅检查 PNG/JPG 文件头只能发现“全 0 文件”等明显坏样本；
    - 真实训练中还可能出现文件头正确但压缩数据损坏的 PNG，例如 libpng 报
      `IDAT: incorrect data check`；
    - 训练前完整解码一次可以把这类样本提前过滤，避免第 N 个 epoch 才中断。
    """
    if not _has_valid_image_header(path):
        return False
    return _read_image_with_imdecode(path, flags) is not None


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
                # 训练集样本数较大，不能在 worker 内部才发现坏文件。
                # 这里提前过滤文件头明显非法的样本：
                # - 输入: image/mask 路径；
                # - 输出: 仅保留可被正常解码概率较高的样本；
                # - 原因: 坏样本会导致单个 DataLoader worker 抛异常，从而中断整轮训练。
                if not _has_valid_image_header(img_path):
                    warnings.warn(
                        f"跳过不可识别图像文件: {img_path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                if not _has_valid_image_header(mask_path):
                    warnings.warn(
                        f"跳过不可识别标注文件: {mask_path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                # 文件头合法不代表图像内容完整。这里再做一次真实解码：
                # - 图像输入使用彩色模式，期望输出形状为 [H,W,3]；
                # - 标注输入使用灰度模式，期望输出形状为 [H,W]；
                # - 若解码失败，说明该样本训练时一定会在 __getitem__ 中失败，因此直接跳过。
                if not _can_decode_image_file(img_path, cv2.IMREAD_COLOR):
                    warnings.warn(
                        f"跳过无法解码图像文件: {img_path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                if not _can_decode_image_file(mask_path, cv2.IMREAD_GRAYSCALE):
                    warnings.warn(
                        f"跳过无法解码标注文件: {mask_path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
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
        # A. 先通过索引拿到“图像路径 + 标注路径”，并完成稳健读取
        # -------------------------------
        # self.samples 的元素形态是 (Path, Path)，例如：
        # (train/images/0001.jpg, train/annotations/0001.png)。
        #
        # 为什么这里不是读取失败就直接 raise：
        # - 长训练可能在第 N 个 epoch 才遇到 libpng 的瞬时解码失败；
        # - 直接 raise 会丢掉本轮训练进度，尤其在 Windows 后台训练中恢复成本高；
        # - 因此这里按顺序尝试后续样本，找到第一对可解码的 image/mask 后继续训练。
        # 输入输出：
        # - 输入 index 是 DataLoader 给出的样本索引；
        # - 输出 image/mask 是已成功解码的 numpy 数组；
        # - 若整个数据集都不可读，才抛 RuntimeError，这是数据集级别故障。
        image: Optional[np.ndarray] = None
        mask: Optional[np.ndarray] = None
        img_path: Path
        mask_path: Path
        for offset in range(len(self.samples)):
            sample_index = (index + offset) % len(self.samples)
            img_path, mask_path = self.samples[sample_index]

            # cv2 解码后图像形状: (H, W, 3)，像素范围通常是 uint8 的 [0, 255]。
            image = _read_image_with_imdecode(img_path, cv2.IMREAD_COLOR)
            if image is None:
                warnings.warn(
                    f"运行时跳过图像读取失败样本: {img_path}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            # 标注用灰度读取，形状为 (H, W)。
            mask = _read_image_with_imdecode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                warnings.warn(
                    f"运行时跳过标注读取失败样本: {mask_path}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            break
        else:
            raise RuntimeError("数据集中所有样本均无法读取，请检查 data 目录中的图像/标注文件。")

        assert image is not None
        assert mask is not None
        # OpenCV 默认通道顺序是 BGR，但大多数深度学习模型按 RGB 训练
        # 所以这里必须转色彩通道顺序，避免训练效果异常
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 把标注统一成 float32 的二值图:
        # - 大于 127 的像素视为裂缝 -> 1.0
        # - 其余视为背景 -> 0.0
        # 处理后 mask 形状仍是 (H, W)，值域变成 {0.0, 1.0}
        # 输入: 原始灰度 annotation，shape=[H,W]，可能是白裂缝或黑裂缝。
        # 输出: 裂缝=1、背景=0 的 float32 mask，shape=[H,W]。
        # 原因: 数据集中 mosaic 与非 mosaic 的标注极性相反，直接用 mask > 127
        # 会把大量非 mosaic 样本的背景当成前景，导致指标虚高。
        mask = normalize_crack_mask(mask).astype(np.float32)

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
