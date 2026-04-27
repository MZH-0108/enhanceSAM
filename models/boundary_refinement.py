"""边界精细化模块 - 用于精确的裂缝边缘检测。

本模块实现以下组件:
    - BoundaryDetector: 使用 Sobel 初始化的卷积进行边界检测
    - BoundaryRefineNet: 在边界图引导下迭代精细化掩码
    - BoundaryLoss: 边界感知损失函数
    - BoundaryMetrics: 边界质量评估指标

设计动机:
    裂缝具有细长、连续的特点，边界精度对分割效果影响很大。
    通过专门的边界检测和精细化模块，可以显著提升边界 IoU。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BoundaryDetector(nn.Module):
    """轻量级边界检测分支。

    使用 Sobel 算子初始化的卷积层来检测特征图中的边缘。

    Args:
        in_channels: 输入特征通道数（SAM 通常为 256）
        mid_channels: 中间层通道数（默认: 64）

    输入:
        特征图 (B, C, H, W)

    输出:
        边界概率图 (B, 1, H, W)，值域 [0, 1]

    Example:
        >>> detector = BoundaryDetector(in_channels=256, mid_channels=64)
        >>> features = torch.randn(2, 256, 64, 64)
        >>> boundary_map = detector(features)
        >>> print(boundary_map.shape)  # (2, 1, 64, 64)
    """

    def __init__(self, in_channels: int, mid_channels: int = 64) -> None:
        super().__init__()

        # 边缘检测卷积层（使用 Sobel 算子初始化）
        self.edge_conv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )

        # 边界检测头：BN -> ReLU -> Conv -> BN -> ReLU -> Conv (输出 1 通道)
        self.head = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, mid_channels // 2, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            # 最后一层输出 1 通道（边界概率）
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1, bias=True),
        )

        # 用 Sobel 算子初始化边缘卷积权重
        self._init_edge_weights(in_channels, mid_channels)

    def _init_edge_weights(self, in_c: int, out_c: int) -> None:
        """使用 Sobel 算子初始化边缘检测卷积。

        将一半通道初始化为水平 Sobel，另一半初始化为垂直 Sobel，
        这样模型从一开始就具备边缘检测能力。

        Args:
            in_c: 输入通道数
            out_c: 输出通道数
        """
        with torch.no_grad():
            w = self.edge_conv.weight.data

            # 先用 Kaiming 初始化作为基础（保留随机性）
            nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")

            # 一半通道用水平 Sobel，一半用垂直 Sobel
            half = out_c // 2

            # 水平 Sobel 算子（检测水平边缘）
            sobel_h = (
                torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
                .view(1, 3, 3)
                / 4.0  # 归一化避免数值过大
            )

            # 垂直 Sobel 算子（检测垂直边缘）
            sobel_v = (
                torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                .view(1, 3, 3)
                / 4.0
            )

            # 将 Sobel 算子叠加到权重上（保留 Kaiming 随机部分 + 边缘检测能力）
            for i in range(min(half, out_c)):
                w[i] = w[i] + sobel_h.expand(in_c, -1, -1)
            for i in range(half, out_c):
                w[i] = w[i] + sobel_v.expand(in_c, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            边界概率图 (B, 1, H, W)
        """
        # 边缘卷积 -> 检测头 -> Sigmoid（输出概率）
        x = self.edge_conv(x)
        x = self.head(x)
        return torch.sigmoid(x)


class _RefineBlock(nn.Module):
    """残差精细化块（内部使用）。

    结构: Conv -> BN -> ReLU -> Conv -> BN -> 残差连接 -> ReLU

    Args:
        channels: 通道数
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 双卷积块
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带残差连接的前向传播。"""
        # 残差连接：output = ReLU(x + Conv(x))
        return self.relu(x + self.conv(x))


class BoundaryRefineNet(nn.Module):
    """基于边界引导的迭代式掩码精细化网络。

    通过多次迭代精细化，结合边界图信息，提升掩码边缘的准确性。

    Args:
        feat_channels: 特征图通道数（默认: 256）
        mid_channels: 中间层通道数（默认: 64）
        num_iterations: 精细化迭代次数（默认: 3）

    输入:
        coarse_mask: 初始掩码 logits (B, 1, H, W)
        boundary_map: 边界概率图 (B, 1, H, W)
        features: 编码器特征 (B, C, H', W')

    输出:
        refined_mask: 精细化后的掩码 logits (B, 1, H, W)

    Example:
        >>> refiner = BoundaryRefineNet(feat_channels=256, mid_channels=64)
        >>> coarse = torch.randn(2, 1, 128, 128)
        >>> boundary = torch.randn(2, 1, 128, 128)
        >>> features = torch.randn(2, 256, 128, 128)
        >>> refined = refiner(coarse, boundary, features)
    """

    def __init__(
        self,
        feat_channels: int = 256,
        mid_channels: int = 64,
        num_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations

        # 输入通道数 = 粗掩码(1) + 边界图(1) + 特征图(feat_channels)
        in_c = 1 + 1 + feat_channels
        # 输入投影层：将拼接的输入映射到中间通道数
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_c, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 多个残差精细化块（迭代精细化）
        self.refine_blocks = nn.ModuleList(
            [_RefineBlock(mid_channels) for _ in range(num_iterations)]
        )

        # 输出头：将中间特征映射回 1 通道（残差连接到原始掩码）
        self.out_head = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)

        # 输出头初始化为零，保证训练初期等价于原始掩码
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        coarse_mask: torch.Tensor,
        boundary_map: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """迭代精细化前向传播。

        Args:
            coarse_mask: 初始掩码 logits (B, 1, H, W)
            boundary_map: 边界概率图 (B, 1, H, W)
            features: 编码器特征 (B, C, H', W')

        Returns:
            精细化后的掩码 logits (B, 1, H, W)
        """
        # 获取目标分辨率（以粗掩码为准）
        H, W = coarse_mask.shape[-2:]

        # 调整特征图到目标分辨率
        if features.shape[-2:] != (H, W):
            features = F.interpolate(
                features, size=(H, W), mode="bilinear", align_corners=False
            )

        # 调整边界图到目标分辨率
        if boundary_map.shape[-2:] != (H, W):
            boundary_map = F.interpolate(
                boundary_map, size=(H, W), mode="bilinear", align_corners=False
            )

        # 在通道维度拼接：粗掩码 + 边界图 + 特征
        x = torch.cat([coarse_mask, boundary_map, features], dim=1)
        # 投影到中间通道数
        x = self.input_proj(x)

        # 迭代精细化
        for block in self.refine_blocks:
            x = block(x)

        # 残差连接：refined = coarse + delta
        return coarse_mask + self.out_head(x)


class BoundaryLoss(nn.Module):
    """组合式边界感知分割损失。

    损失公式:
        L = w_bce * BCE + w_dice * Dice + w_bound * BoundaryBCE

    其中:
        - BCE: 标准二元交叉熵损失
        - Dice: Dice 损失（处理类别不平衡）
        - BoundaryBCE: 边界区域加权 BCE（强调边界精度）

    Args:
        w_bce: BCE 损失权重（默认: 1.0）
        w_dice: Dice 损失权重（默认: 2.0）
        w_bound: 边界加权 BCE 损失权重（默认: 2.0）
        boundary_radius: 边界区域膨胀半径（默认: 3）
        pos_weight: 正样本权重，处理类别不平衡（默认: 10.0）

    Example:
        >>> criterion = BoundaryLoss(w_bce=1.0, w_dice=2.0, w_bound=2.0)
        >>> pred = torch.randn(2, 1, 128, 128)
        >>> target = torch.randint(0, 2, (2, 1, 128, 128)).float()
        >>> loss_dict = criterion(pred, target)
        >>> print(loss_dict['loss'])
    """

    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 2.0,
        w_bound: float = 2.0,
        boundary_radius: int = 3,
        pos_weight: float = 10.0,
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_bound = w_bound
        self.radius = boundary_radius
        # 注册为 buffer，会随模型一起移动到设备但不参与梯度更新
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    @staticmethod
    def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """计算 Dice 损失。

        Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
        DiceLoss = 1 - Dice

        Args:
            pred: 预测概率 (B, 1, H, W)
            target: 真值掩码 (B, 1, H, W)
            eps: 数值稳定性常数

        Returns:
            Dice 损失（标量）
        """
        # 展平为 (B, H*W) 便于按样本计算
        pred_f = pred.flatten(1)
        tgt_f = target.flatten(1)
        # 交集
        inter = (pred_f * tgt_f).sum(dim=1)
        # 并集（不去重）
        union = pred_f.sum(dim=1) + tgt_f.sum(dim=1)
        # Dice 系数
        dice = (2.0 * inter + eps) / (union + eps)
        # 损失 = 1 - Dice，对 batch 求平均
        return (1.0 - dice).mean()

    @staticmethod
    def get_boundary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        """通过膨胀提取掩码的边界区域。

        原理: 边界 = 膨胀后的掩码 - 原始掩码

        Args:
            mask: 二��掩码 (B, 1, H, W)
            radius: 膨胀半径

        Returns:
            边界掩码 (B, 1, H, W)，边界区域为 1，其他为 0
        """
        # 构造膨胀核（全 1 的方形核）
        k = 2 * radius + 1
        kernel = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype)
        # 卷积实现膨胀（结果 > 0 即被覆盖）
        dilated = F.conv2d(mask.float(), kernel, padding=radius)
        # 边界 = 膨胀掩码 - 原始掩码
        boundary = (dilated > 0).float() - mask.float()
        return boundary

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        boundary_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算组合损失。

        Args:
            pred_logits: 预测 logits (B, 1, H, W)
            target: 真值二值掩码 (B, 1, H, W)，值为 {0, 1}
            boundary_gt: 可选的预计算边界掩码

        Returns:
            包含以下键的字典:
                - loss: 总损失
                - bce: BCE 损失分量
                - dice: Dice 损失分量
                - boundary: 边界 BCE 分量
        """
        # 将 logits 转换为概率
        pred_prob = torch.sigmoid(pred_logits)
        tgt = target.float()

        # 标准 BCE 损失（带正样本权重）
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, tgt, pos_weight=self.pos_weight
        )

        # Dice 损失
        dice = self._dice_loss(pred_prob, tgt)

        # 边界加权 BCE
        # 如果未提供边界图，则从真值计算
        if boundary_gt is None:
            boundary_gt = self.get_boundary_mask(target, self.radius)

        # 权重图：背景区域权重 1，边界区域权重 5（强调边界）
        boundary_weight = 1.0 + boundary_gt.float() * 4.0

        # 加权 BCE（边界区域损失被放大 5 倍）
        bnd = F.binary_cross_entropy_with_logits(
            pred_logits, tgt, weight=boundary_weight, pos_weight=self.pos_weight
        )

        # 组合总损失
        total = self.w_bce * bce + self.w_dice * dice + self.w_bound * bnd

        # 返回所有损失分量（detach 用于日志记录，不影响梯度）
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice": dice.detach(),
            "boundary": bnd.detach(),
        }


class BoundaryMetrics:
    """边界质量评估指标。

    提供以下指标:
        - IoU (交并比)
        - Dice 系数
        - Boundary F-measure (边界 F 值)
    """

    @staticmethod
    def iou(
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """计算 IoU (交并比)。

        IoU = |X ∩ Y| / |X ∪ Y|

        Args:
            pred_mask: 预测概率 (B, 1, H, W)
            gt_mask: 真值掩码 (B, 1, H, W)
            threshold: 二值化阈值
            eps: 数值稳定性常数

        Returns:
            每个样本的 IoU (B,)
        """
        # 二值化预测
        pred = (pred_mask >= threshold).float()
        gt = gt_mask.float()

        # 交集：A 和 B 都为 1 的像素数
        inter = (pred * gt).sum(dim=(1, 2, 3))
        # 并集：A 或 B 为 1 的像素数（clamp 防止重复计数）
        union = (pred + gt).clamp(max=1).sum(dim=(1, 2, 3))

        return inter / (union + eps)

    @staticmethod
    def dice(
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """计算 Dice 系数。

        Dice = 2 * |X ∩ Y| / (|X| + |Y|)

        Args:
            pred_mask: 预测概率 (B, 1, H, W)
            gt_mask: 真值掩码 (B, 1, H, W)
            threshold: 二值化阈值
            eps: 数值稳定性常数

        Returns:
            每个样本的 Dice (B,)
        """
        pred = (pred_mask >= threshold).float()
        gt = gt_mask.float()

        # 交集
        inter = (pred * gt).sum(dim=(1, 2, 3))
        # 分母 = |X| + |Y|
        denom = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))

        return (2.0 * inter + eps) / (denom + eps)

    @staticmethod
    def boundary_f_measure(
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        radius: int = 3,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """计算边界 F 值 (BF-score)。

        通过比较预测边界和真值边界的吻合程度，评估边界质量。
        允许一定的边界容差（由 radius 控制）。

        Args:
            pred_mask: 预测概率 (B, 1, H, W)
            gt_mask: 真值掩码 (B, 1, H, W)
            radius: 边界容差半径
            threshold: 二值化阈值

        Returns:
            每个样本的边界 F 值 (B,)
        """
        # 二值化预测
        pred_bin = (pred_mask >= threshold).float()

        # 提取预测和真值的边界
        pred_bound = BoundaryLoss.get_boundary_mask(pred_bin, radius)
        gt_bound = BoundaryLoss.get_boundary_mask(gt_mask, radius)

        # 对边界进行膨胀，引入容差
        k = 2 * radius + 1
        kernel = torch.ones(1, 1, k, k, device=pred_mask.device, dtype=pred_mask.dtype)

        # 膨胀后的边界（用于计算容差范围内的匹配）
        pred_dil = (F.conv2d(pred_bound, kernel, padding=radius) > 0).float()
        gt_dil = (F.conv2d(gt_bound, kernel, padding=radius) > 0).float()

        # 精确率：预测边界中有多少在真值容差范围内
        tp_p = (pred_bound * gt_dil).sum(dim=(1, 2, 3))
        # 召回率：真值边界中有多少在预测容差范围内
        tp_r = (gt_bound * pred_dil).sum(dim=(1, 2, 3))

        prec = tp_p / (pred_bound.sum(dim=(1, 2, 3)) + 1e-6)
        rec = tp_r / (gt_bound.sum(dim=(1, 2, 3)) + 1e-6)

        # F 值 = 2 * P * R / (P + R)
        return 2 * prec * rec / (prec + rec + 1e-6)
