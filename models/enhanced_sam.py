"""增强版 SAM - 集成 SAM + LoRA + 边界精细化。

本模块整合三个核心组件:
    1. SAM 基础模型（图像编码器 + 掩码解码器）
    2. LoRA 适配器（参数高效微调）
    3. 边界精细化（精确的边缘检测）

主要类:
    - EnhancedSAMConfig: 完整配置
    - EnhancedSAM: 集成模型
    - build_enhanced_sam: 工厂函数
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_adapter import LoRAAdapter, LoRAConfig, apply_lora_to_sam
from .boundary_refinement import (
    BoundaryDetector,
    BoundaryRefineNet,
    BoundaryLoss,
)


@dataclass
class EnhancedSAMConfig:
    """EnhancedSAM 完整配置。

    Args:
        vit_embed_dim: ViT 内部维度 (vit_b/l/h 分别为 768/1024/1280)
        prompt_embed_dim: SAM neck 输出维度（固定为 256）
        use_lora: 是否启用 LoRA 微调
        lora_rank: LoRA 秩 r
        lora_alpha: LoRA 缩放系数
        lora_dropout: LoRA Dropout 概率
        lora_target_modules: 注入 LoRA 的目标模块名称
        use_boundary: 是否启用边界精细化
        boundary_mid_channels: 边界精细化中间层通道数
        boundary_num_iterations: 精细化迭代次数
        boundary_detector_mid: 边界检测器中间层通道数
        loss_w_bce: BCE 损失权重
        loss_w_dice: Dice 损失权重
        loss_w_bound: 边界损失权重
        loss_pos_weight: 类别不平衡的正样本权重
        loss_boundary_radius: 边界区域半径

    Example:
        >>> config = EnhancedSAMConfig(
        ...     vit_embed_dim=768,
        ...     use_lora=True,
        ...     use_boundary=True
        ... )
    """

    # 模型维度配置
    vit_embed_dim: int = 768       # ViT 内部维度: 768(vit_b)/1024(vit_l)/1280(vit_h)
    prompt_embed_dim: int = 256    # SAM 固定为 256

    # LoRA 配置
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "qkv",
            "proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "lin1",
            "lin2",
        ]
    )

    # 边界精细化配置
    use_boundary: bool = True
    boundary_mid_channels: int = 64
    boundary_num_iterations: int = 3
    boundary_detector_mid: int = 64

    # 损失函数配置
    loss_w_bce: float = 1.0
    loss_w_dice: float = 2.0
    loss_w_bound: float = 2.0
    loss_pos_weight: float = 10.0  # 裂缝像素少，需要较大正样本权重
    loss_boundary_radius: int = 3


class EnhancedSAM(nn.Module):
    """增强版 SAM，用于隧道裂缝分割。

    整合 SAM、LoRA 微调和边界精细化模块。

    Args:
        sam: 来自 segment_anything 的原始 SAM 模型
        config: EnhancedSAMConfig 配置

    前向传播输入:
        image: 输入图像 (B, 3, H, W)，已预处理
        points: 可选的点提示 (坐标, 标签)
        boxes: 可选的框提示 (B, 4)
        masks: 可选的掩码提示 (B, 1, H, W)
        multimask: 是否输出多个候选掩码

    前向传播输出:
        包含以下键的字典:
            - masks: 预测掩码 (B, num_masks, H/4, W/4) logits
            - iou_pred: IoU 预测 (B, num_masks)
            - refined_mask: 精细化掩码 (B, 1, H/4, W/4) logits（启用边界时）
            - boundary_map: 边界概率图 (B, 1, H/4, W/4)（启用边界时）

    Example:
        >>> from segment_anything import sam_model_registry
        >>> sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        >>> config = EnhancedSAMConfig()
        >>> model = EnhancedSAM(sam, config)
        >>> image = torch.randn(2, 3, 512, 512)
        >>> outputs = model(image=image)
    """

    def __init__(self, sam: nn.Module, config: EnhancedSAMConfig) -> None:
        super().__init__()
        self.config = config
        self.sam = sam

        # 步骤 1: 注入 LoRA 到 SAM
        self.lora_adapter: Optional[LoRAAdapter] = None
        if config.use_lora:
            # 构建 LoRA 配置
            lora_cfg = LoRAConfig(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
            # 注入 LoRA 并冻结基础权重
            self.lora_adapter = apply_lora_to_sam(sam, config=lora_cfg, freeze=True)

        # 步骤 2: 初始化边界精细化模块
        self.boundary_detector: Optional[BoundaryDetector] = None
        self.boundary_refiner: Optional[BoundaryRefineNet] = None

        if config.use_boundary:
            # 边界检测器：从特征图预测边界概率
            self.boundary_detector = BoundaryDetector(
                in_channels=config.prompt_embed_dim,
                mid_channels=config.boundary_detector_mid,
            )
            # 边界精细化网络：根据边界图迭代优化掩码
            self.boundary_refiner = BoundaryRefineNet(
                feat_channels=config.prompt_embed_dim,
                mid_channels=config.boundary_mid_channels,
                num_iterations=config.boundary_num_iterations,
            )

        # 步骤 3: 初始化损失函数
        self.criterion = BoundaryLoss(
            w_bce=config.loss_w_bce,
            w_dice=config.loss_w_dice,
            w_bound=config.loss_w_bound,
            boundary_radius=config.loss_boundary_radius,
            pos_weight=config.loss_pos_weight,
        )

    def forward(
        self,
        image: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """EnhancedSAM 完整前向传播。

        Args:
            image: 输入图像 (B, 3, H, W)
            points: 可选的 (坐标, 标签) 元组
            boxes: 可选的框提示 (B, 4)
            masks: 可选的掩码提示 (B, 1, H, W)
            multimask: 是否输出多个候选掩码

        Returns:
            包含 masks、iou_pred 以及（可选）refined_mask、boundary_map 的字典
        """
        # 步骤 1: 图像编码 -> 特征图
        image_embeddings = self.sam.image_encoder(image)  # (B, 256, H/16, W/16)

        # 步骤 2: 提示编码 -> sparse + dense 嵌入
        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # 步骤 3: 掩码解码 -> 粗掩码 + IoU 预测
        low_res_masks, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask,
        )

        # 输出字典（基础部分）
        out = {
            "masks": low_res_masks,  # (B, num_masks, H/4, W/4)
            "iou_pred": iou_pred,    # (B, num_masks)
        }

        # 步骤 4: 边界精细化（可选）
        if self.boundary_detector is not None and self.boundary_refiner is not None:
            # 4.1 检测边界
            boundary_map = self.boundary_detector(image_embeddings)

            # 4.2 选择 IoU 最高的掩码进行精细化
            best_idx = iou_pred.argmax(dim=1, keepdim=True)  # (B, 1)
            best_mask = low_res_masks.gather(
                1,
                best_idx.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, *low_res_masks.shape[-2:]),
            )  # (B, 1, H/4, W/4)

            # 4.3 在边界引导下精细化掩码
            refined = self.boundary_refiner(
                coarse_mask=best_mask,
                boundary_map=boundary_map,
                features=image_embeddings,
            )

            # 添加精细化结果到输出
            out["refined_mask"] = refined
            out["boundary_map"] = boundary_map

        return out

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算训练损失。

        Args:
            outputs: forward() 的输出字典
            targets: 真值掩码 (B, 1, H, W)，二值

        Returns:
            包含各损失分量的字典
        """
        # 优先使用精细化掩码计算损失
        if "refined_mask" in outputs:
            pred = outputs["refined_mask"]
        else:
            # 没有精细化时，使用多掩码输出中 IoU 最高的那个
            best_idx = outputs["iou_pred"].argmax(dim=1, keepdim=True)
            H, W = outputs["masks"].shape[-2:]
            pred = outputs["masks"].gather(
                1, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            )

        # 将真值掩码下采样到与预测相同的分辨率
        H, W = pred.shape[-2:]
        # 使用最近邻插值保持二值性
        tgt_resized = F.interpolate(targets.float(), size=(H, W), mode="nearest")

        # 计算组合损失（BCE + Dice + Boundary）
        return self.criterion(pred, tgt_resized)

    def trainable_params(self) -> List[nn.Parameter]:
        """获取所有可训练参数列表。"""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_count(self) -> int:
        """统计可训练参数数量。"""
        return sum(p.numel() for p in self.trainable_params())

    def total_count(self) -> int:
        """统计模型总参数数量。"""
        return sum(p.numel() for p in self.parameters())

    def param_report(self) -> str:
        """生成参数统计报告。

        Returns:
            格式化的报告字符串
        """
        total = self.total_count()
        trainable = self.trainable_count()
        # 计算可训练参数比例
        pct = trainable / total * 100 if total > 0 else 0
        # LoRA 参数数量
        lora_n = self.lora_adapter.lora_count() if self.lora_adapter else 0

        # 启用的模块列表
        modules = []
        if self.config.use_lora:
            modules.append("LoRA")
        if self.config.use_boundary:
            modules.append("Boundary")

        lines = [
            "=" * 60,
            "EnhancedSAM 参数报告",
            "=" * 60,
            f"  总参数量      : {total:,}",
            f"  可训练参数量  : {trainable:,}  ({pct:.3f}%)",
            f"  LoRA 参数量   : {lora_n:,}",
            f"  启用模块      : SAM + {' + '.join(modules)}",
            "=" * 60,
        ]
        return "\n".join(lines)


def build_enhanced_sam(
    sam: nn.Module,
    config: Optional[EnhancedSAMConfig] = None,
) -> EnhancedSAM:
    """从 SAM 模型构建 EnhancedSAM（工厂函数）。

    Args:
        sam: 来自 segment_anything 的 SAM 模型
        config: EnhancedSAMConfig 配置（None 则使用默认配置）

    Returns:
        EnhancedSAM 实例

    Example:
        >>> from segment_anything import sam_model_registry
        >>> sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        >>> model = build_enhanced_sam(sam)
        >>> print(model.param_report())
    """
    # 如果未提供配置，使用默认配置
    if config is None:
        config = EnhancedSAMConfig()

    return EnhancedSAM(sam, config)
