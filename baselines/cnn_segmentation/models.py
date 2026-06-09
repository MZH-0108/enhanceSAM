"""UNet / DeepLabV3-like 裂缝分割基线模型。

本文件只依赖 PyTorch，不下载外部预训练权重，目的是让论文对比实验在
当前离线/受限环境中也能完整复现。两个模型都接收归一化后的 RGB 图像
`[B, 3, H, W]`，输出二分类 logits `[B, 1, H, W]`，后续通过 sigmoid
得到裂缝概率图。
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


def _make_divisible_channels(channels: int) -> int:
    """把通道数转换为正整数。

    输入:
    - channels: 配置文件传入的通道数。

    输出:
    - 至少为 1 的整数通道数。

    为什么这样做:
    - 配置文件可能被写成字符串或较小数值，统一清洗可避免卷积层构造时报错。
    """
    return max(1, int(channels))


class ConvBNReLU(nn.Module):
    """卷积 + BatchNorm + ReLU 基础块。

    输入输出:
    - 输入张量形状 `[B, C_in, H, W]`；
    - 输出张量形状 `[B, C_out, H_out, W_out]`。

    为什么这样做:
    - UNet 和 DeepLab 的主体都大量复用该模式；
    - 封装成模块可以保持网络定义简洁，并统一归一化与激活策略。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """UNet 中的双卷积块。

    输入输出:
    - 输入 `[B, C_in, H, W]`；
    - 输出 `[B, C_out, H, W]`，空间尺寸保持不变。

    为什么这样做:
    - 原始 UNet 每个尺度通常连续堆叠两次 3x3 卷积；
    - 对裂缝这类细长目标，连续局部卷积有利于提取局部纹理和边缘模式。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """轻量 UNet 二分类裂缝分割基线。

    输入:
    - `image`: `[B, 3, H, W]` RGB 图像张量，已由 `utils.data_loader` 完成归一化。

    输出:
    - 字典 `{"logits": logits, "masks": logits}`；
    - `logits` 形状为 `[B, 1, H, W]`，表示每个像素属于裂缝的未归一化分数。

    为什么这样做:
    - UNet 是医学/工业细粒度分割中最常见的强基线；
    - 编码器-解码器与跳连结构适合保留裂缝细边界；
    - 返回字典是为了与项目已有 SAM 训练/评估习惯尽量一致。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        base = _make_divisible_channels(base_channels)

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, kernel_size=1)

    @staticmethod
    def _concat_skip(decoder_feature: torch.Tensor, encoder_feature: torch.Tensor) -> torch.Tensor:
        """拼接解码器特征与编码器跳连特征。

        输入:
        - decoder_feature: 上采样后的特征 `[B, C_d, H_d, W_d]`；
        - encoder_feature: 对应尺度编码特征 `[B, C_e, H_e, W_e]`。

        输出:
        - 拼接后的张量 `[B, C_d + C_e, H_e, W_e]`。

        为什么这样做:
        - 输入尺寸如果不是 16 的整数倍，反卷积上采样后可能出现 1 像素误差；
        - 先插值到跳连特征尺寸，可以避免 concat 时 shape 不匹配。
        """
        if decoder_feature.shape[-2:] != encoder_feature.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=encoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([decoder_feature, encoder_feature], dim=1)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_hw = image.shape[-2:]

        e1 = self.enc1(image)          # [B, base, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 2base, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 4base, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 8base, H/8, W/8]
        b = self.bottleneck(self.pool(e4))  # [B, 16base, H/16, W/16]

        d4 = self.dec4(self._concat_skip(self.up4(b), e4))
        d3 = self.dec3(self._concat_skip(self.up3(d4), e3))
        d2 = self.dec2(self._concat_skip(self.up2(d3), e2))
        d1 = self.dec1(self._concat_skip(self.up1(d2), e1))
        logits = self.head(d1)
        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return {"logits": logits, "masks": logits}


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling，多尺度空洞卷积模块。

    输入:
    - `[B, C_in, H, W]` 深层语义特征。

    输出:
    - `[B, C_out, H, W]` 多尺度上下文融合特征。

    为什么这样做:
    - DeepLabV3 的核心思想是用不同 dilation rate 的空洞卷积捕获多尺度上下文；
    - 裂缝既有局部细纹理，也可能有较长连续结构，多尺度感受野有助于稳定识别。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rates: Iterable[int] = (1, 6, 12, 18),
    ) -> None:
        super().__init__()
        branches: List[nn.Module] = []
        for rate in rates:
            if int(rate) == 1:
                branches.append(ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0))
            else:
                branches.append(
                    ConvBNReLU(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=int(rate),
                        dilation=int(rate),
                    )
                )
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            ConvBNReLU(out_channels * len(branches), out_channels, kernel_size=1, padding=0),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class DeepLabV3Like(nn.Module):
    """DeepLabV3-like 二分类裂缝分割基线。

    输入:
    - `image`: `[B, 3, H, W]`。

    输出:
    - 字典 `{"logits": logits, "masks": logits}`，其中 logits 为 `[B, 1, H, W]`。

    说明:
    - 这里实现的是“DeepLabV3 风格”而不是 torchvision 预训练 DeepLabV3；
    - 原因是当前环境网络受限，不能依赖外部权重下载；
    - 它保留 DeepLabV3 的 ASPP 多尺度空洞卷积核心，适合作为可复现实验基线。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        aspp_channels: int = 128,
        aspp_rates: Iterable[int] = (1, 6, 12, 18),
    ) -> None:
        super().__init__()
        base = _make_divisible_channels(base_channels)
        aspp_out = _make_divisible_channels(aspp_channels)

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, base, stride=2),
            ConvBNReLU(base, base),
        )
        self.layer1 = nn.Sequential(
            ConvBNReLU(base, base * 2, stride=2),
            ConvBNReLU(base * 2, base * 2),
        )
        self.layer2 = nn.Sequential(
            ConvBNReLU(base * 2, base * 4, stride=2),
            ConvBNReLU(base * 4, base * 4),
        )
        # 保持 stride=1，并通过 dilation 扩大感受野，这是 DeepLab 系列的典型做法。
        self.layer3 = nn.Sequential(
            ConvBNReLU(base * 4, base * 8, stride=1, padding=2, dilation=2),
            ConvBNReLU(base * 8, base * 8, stride=1, padding=4, dilation=4),
        )
        self.aspp = ASPP(base * 8, aspp_out, rates=aspp_rates)
        self.decoder = nn.Sequential(
            ConvBNReLU(aspp_out + base, base * 2),
            ConvBNReLU(base * 2, base),
            nn.Conv2d(base, out_channels, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_hw = image.shape[-2:]
        low_level = self.stem(image)       # [B, base, H/2, W/2]
        x = self.layer1(low_level)         # [B, 2base, H/4, W/4]
        x = self.layer2(x)                 # [B, 4base, H/8, W/8]
        x = self.layer3(x)                 # [B, 8base, H/8, W/8]
        x = self.aspp(x)                   # [B, aspp_channels, H/8, W/8]
        x = F.interpolate(x, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, low_level], dim=1)
        logits = self.decoder(x)
        logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return {"logits": logits, "masks": logits}


def build_cnn_baseline(cfg: Dict[str, object]) -> nn.Module:
    """根据配置构建 CNN 分割基线。

    输入:
    - cfg: `model` 配置字典，至少包含 `name`，可选 `base_channels` 等字段。

    输出:
    - `UNet` 或 `DeepLabV3Like` 实例。

    为什么这样做:
    - 训练/评估脚本只需要调用一个工厂函数；
    - 后续如果加入 `SegNet`、`FCN` 等模型，也只需扩展这里。
    """
    name = str(cfg.get("name", "unet")).lower()
    in_channels = int(cfg.get("in_channels", 3))
    out_channels = int(cfg.get("out_channels", 1))
    base_channels = int(cfg.get("base_channels", 32))

    if name in {"unet", "u-net"}:
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
    if name in {"deeplab", "deeplabv3", "deeplabv3_like", "deeplabv3-like"}:
        return DeepLabV3Like(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            aspp_channels=int(cfg.get("aspp_channels", max(64, base_channels * 4))),
            aspp_rates=tuple(int(rate) for rate in cfg.get("aspp_rates", [1, 6, 12, 18])),
        )
    raise ValueError(f"不支持的 CNN baseline 模型: {name}")
