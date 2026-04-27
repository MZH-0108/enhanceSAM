"""SAM 基础模型封装 - 用于隧道裂缝分割。

本模块提供加载和使用 SAM 模型的简洁接口，支持自定义图像尺寸和预处理。

主要功能:
    - load_sam_model: 加载 SAM 预训练模型
    - patch_sam_for_img_size: 适配自定义图像尺寸（位置编码插值）
    - SAMBase: SAM 模型的统一封装类
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple
from segment_anything import sam_model_registry


def load_sam_model(
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> nn.Module:
    """加载 SAM 预训练模型。

    Args:
        model_type: SAM 模型类型，可选 ['vit_b', 'vit_l', 'vit_h']
        checkpoint_path: SAM 预训练权重文件路径
        device: 模型加载到的设备 ('cuda' 或 'cpu')

    Returns:
        加载完成的 SAM 模型

    Raises:
        ValueError: 当 model_type 无效或 checkpoint_path 为 None 时
        FileNotFoundError: 当 checkpoint_path 文件不存在时

    Example:
        >>> sam = load_sam_model('vit_b', 'sam_vit_b_01ec64.pth')
        >>> print(sam)
    """
    # 验证模型类型是否在官方注册表中
    if model_type not in sam_model_registry:
        raise ValueError(
            f"无效的模型类型: {model_type}。"
            f"请从 {list(sam_model_registry.keys())} 中选择"
        )

    # 验证权重路径是否提供
    if checkpoint_path is None:
        raise ValueError("必须提供 checkpoint_path 参数")

    # 调用 SAM 官方 API 加载模型并恢复预训练权重
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    # 将模型移动到指定设备 (GPU/CPU)
    sam = sam.to(device)
    # 设置为评估模式（关闭 Dropout 和 BatchNorm 的训练行为）
    sam.eval()

    return sam


def patch_sam_for_img_size(sam: nn.Module, img_size: int) -> None:
    """适配 SAM 模型到自定义图像尺寸。

    SAM 默认在 1024x1024 图像上预训练。本函数通过插值位置编码，
    使模型能够支持其他尺寸（如 512x512）的输入。

    Args:
        sam: 待适配的 SAM 模型
        img_size: 目标图像尺寸（必须是 16 的倍数）

    Raises:
        ValueError: 当 img_size 不能被 16 整除时

    Example:
        >>> sam = load_sam_model('vit_b', 'sam_vit_b_01ec64.pth')
        >>> patch_sam_for_img_size(sam, 512)
    """
    # ViT 将图像分成 16x16 的 patch，因此尺寸必须是 16 的倍数
    if img_size % 16 != 0:
        raise ValueError(f"img_size 必须能被 16 整除，当前值为 {img_size}")

    # 更新图像编码器的 img_size 属性
    if hasattr(sam, "image_encoder"):
        sam.image_encoder.img_size = img_size

        # 对位置编码进行插值适配
        if hasattr(sam.image_encoder, "pos_embed"):
            # 计算原始网格尺寸（如 64x64 = 4096 个位置）
            old_size = int(sam.image_encoder.pos_embed.shape[1] ** 0.5)
            # 计算新网格尺寸（如 32x32 = 1024 个位置）
            new_size = img_size // 16

            # 仅当尺寸不同时才需要插值
            if old_size != new_size:
                pos_embed = sam.image_encoder.pos_embed
                # 步骤 1: 从 (1, N, D) 重塑为 (1, H, W, D)
                pos_embed = pos_embed.reshape(1, old_size, old_size, -1)
                # 步骤 2: 转换为图像格式 (1, D, H, W) 以便插值
                pos_embed = pos_embed.permute(0, 3, 1, 2)

                # 步骤 3: 使用双三次插值缩放位置编码
                pos_embed = torch.nn.functional.interpolate(
                    pos_embed,
                    size=(new_size, new_size),
                    mode="bicubic",  # 双三次插值，比双线性更平滑
                    align_corners=False,
                )

                # 步骤 4: 转回原始格式 (1, N', D)
                pos_embed = pos_embed.permute(0, 2, 3, 1)
                pos_embed = pos_embed.reshape(1, -1, pos_embed.shape[-1])
                # 更新模型的位置编码
                sam.image_encoder.pos_embed = pos_embed


class SAMBase(nn.Module):
    """SAM 基础封装类，支持自定义图像尺寸。

    本类对原始 SAM 模型进行封装，提供以下功能：
    - 支持自定义图像尺寸（512x512, 1024x1024 等）
    - 简化的前向传播接口
    - 预处理工具

    Args:
        sam_model: 从 segment_anything 加载的 SAM 模型
        img_size: 目标图像尺寸（默认: 1024）

    Example:
        >>> sam = load_sam_model('vit_b', 'sam_vit_b_01ec64.pth')
        >>> sam_base = SAMBase(sam, img_size=512)
        >>> image = torch.randn(2, 3, 512, 512)
        >>> embeddings = sam_base.encode_image(image)
    """

    def __init__(self, sam_model: nn.Module, img_size: int = 1024) -> None:
        super().__init__()
        # 保存 SAM 模型实例
        self.sam = sam_model
        # 保存目标图像尺寸
        self.img_size = img_size

        # 如果不是默认尺寸 1024，则进行适配
        if img_size != 1024:
            patch_sam_for_img_size(self.sam, img_size)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """将图像编码为特征嵌入。

        Args:
            image: 输入图像张量 (B, 3, H, W)，已归一化

        Returns:
            图像特征嵌入 (B, 256, H/16, W/16)

        Raises:
            ValueError: 当图像尺寸与期望尺寸不匹配时
        """
        # 验证输入图像尺寸是否符合要求
        if image.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"期望图像尺寸 {self.img_size}x{self.img_size}，"
                f"实际为 {image.shape[-2]}x{image.shape[-1]}"
            )

        # 调用 SAM 的 Image Encoder (ViT) 进行特征提取
        return self.sam.image_encoder(image)

    def forward(
        self,
        image: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SAM 完整前向传播。

        Args:
            image: 输入图像 (B, 3, H, W)
            points: 可选的点提示 (坐标, 标签)
            boxes: 可选的框提示 (B, 4)
            masks: 可选的掩码提示 (B, 1, H, W)
            multimask_output: 是否输出多个候选掩码

        Returns:
            masks: 预测的掩码 (B, num_masks, H/4, W/4)
            iou_predictions: IoU 预测分数 (B, num_masks)
        """
        # 步骤 1: 图像编码 - 提取特征
        image_embeddings = self.encode_image(image)

        # 步骤 2: 提示编码 - 处理点/框/掩码提示
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # 步骤 3: 掩码解码 - 生成最终掩码
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return low_res_masks, iou_predictions

    @property
    def device(self) -> torch.device:
        """获取模型所在设备。"""
        return next(self.parameters()).device

    def __repr__(self) -> str:
        return f"SAMBase(img_size={self.img_size}, device={self.device})"
