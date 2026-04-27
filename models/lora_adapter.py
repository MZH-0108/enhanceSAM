"""LoRA (低秩适应) 适配器 - 用于参数高效微调。

本模块为 SAM 实现 LoRA 微调，仅用 1-2% 的可训练参数即可达到良好效果。
LoRA 通过在线性层中注入可训练的低秩矩阵实现微调。

数学原理:
    h = Wx + (α/r) · BAx
    其中:
        W: 冻结的原始权重 (out_features, in_features)
        A: 可训练的低秩矩阵 (rank, in_features)
        B: 可训练的低秩矩阵 (out_features, rank)
        α/r: 缩放系数

参考文献:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """LoRA 超参数配置。

    Args:
        rank: 低秩维度 r (通常为 4, 8 或 16)
        alpha: 缩放系数；实际缩放因子 = alpha / rank
        dropout: LoRA 路径上的 Dropout 概率
        target_modules: 用于匹配模块名称的子字符串列表（决定哪些层注入 LoRA）

    Example:
        >>> config = LoRAConfig(rank=8, alpha=16.0, dropout=0.1)
        >>> print(config.scale)  # 输出 2.0 (= 16.0 / 8)
    """

    rank: int = 8                  # 低秩维度，越大表达能力越强但参数也越多
    alpha: float = 16.0            # 缩放系数，控制 LoRA 路径的影响程度
    dropout: float = 0.1           # Dropout 概率，防止过拟合
    target_modules: List[str] = field(
        default_factory=lambda: [
            "qkv",       # 自注意力的 QKV 投影
            "proj",      # 注意力输出投影
            "q_proj",    # Q 投影（独立）
            "k_proj",    # K 投影（独立）
            "v_proj",    # V 投影（独立）
            "out_proj",  # 输出投影
            "lin1",      # MLP 第一层
            "lin2",      # MLP 第二层
        ]
    )

    @property
    def scale(self) -> float:
        """计算实际的缩放因子 (alpha / rank)。"""
        return self.alpha / self.rank


# 针对 SAM 不同子模块的推荐配置
LORA_CONFIGS: Dict[str, LoRAConfig] = {
    # 图像编码器：注入到注意力的 QKV 和投影层
    "image_encoder": LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["qkv", "proj"],
    ),
    # 提示编码器：仅注入投影层（参数较少）
    "prompt_encoder": LoRAConfig(
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=["proj"],
    ),
    # 掩码解码器：注入到所有线性层
    "mask_decoder": LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "lin1", "lin2"],
    ),
    # 完整模型：使用默认目标模块列表
    "full": LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
    ),
}


class LoRALinear(nn.Module):
    """带 LoRA 旁路的线性层。

    前向计算公式:
        y = W @ x + scale * (B @ A @ x)

    其中:
        - W: 冻结的原始权重 (out_features, in_features)
        - A: 可训练的低秩矩阵 (rank, in_features) - Kaiming 初始化
        - B: 可训练的低秩矩阵 (out_features, rank) - 零初始化
        - scale: 缩放因子 = alpha / rank

    设计要点:
        - B 初始化为零，使得训练初期 LoRA 路径输出为零，等价于原始模型
        - 仅 A 和 B 可训练，原始权重 W 保持冻结
        - 推理时可将 LoRA 合并到 W 中，实现零开销

    Args:
        in_features: 输入维度
        out_features: 输出维度
        rank: LoRA 秩 r
        alpha: 缩放系数
        dropout: LoRA 输入路径的 Dropout
        bias: 是否包含偏置项

    Example:
        >>> layer = LoRALinear(256, 512, rank=8, alpha=16.0)
        >>> x = torch.randn(4, 256)
        >>> y = layer(x)
        >>> print(y.shape)  # (4, 512)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank
        # 标记 LoRA 是否已合并到原始权重（推理优化）
        self.merged = False

        # 冻结的原始权重（不参与梯度更新）
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )

        # 冻结的偏置（如果存在）
        self.bias_param = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias
            else None
        )

        # 可训练的 LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # LoRA 路径上的 Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # 初始化 LoRA 矩阵
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化 LoRA 矩阵权重。

        - A: Kaiming 均匀初始化（与 nn.Linear 默认相同）
        - B: 零初始化（保证训练初期 LoRA 输出为零，等价于原始模型）
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带 LoRA 的前向传播。

        Args:
            x: 输入张量 (..., in_features)

        Returns:
            输出张量 (..., out_features)
        """
        # 基础线性变换：y = W @ x + b
        base = F.linear(x, self.weight, self.bias_param)

        # 如果已合并，LoRA 已经在权重中，直接返回基础输出
        if self.merged:
            return base

        # LoRA 路径: x → dropout → A → B → 缩放
        lora = self.lora_dropout(x)
        lora = F.linear(lora, self.lora_A)  # (..., rank)
        lora = F.linear(lora, self.lora_B)  # (..., out_features)

        # 合并基础输出和 LoRA 输出
        return base + self.scale * lora

    def merge_lora(self) -> None:
        """将 LoRA 权重合并到基础权重，用于推理加速。

        合并后:
            W_merged = W + scale * (B @ A)

        这样推理时无需计算 LoRA 路径，零额外开销。
        """
        if not self.merged:
            self.weight.data += self.scale * (self.lora_B @ self.lora_A)
            self.merged = True

    def unmerge_lora(self) -> None:
        """取消合并 LoRA 权重，恢复独立训练状态。

        恢复:
            W = W_merged - scale * (B @ A)
        """
        if self.merged:
            self.weight.data -= self.scale * (self.lora_B @ self.lora_A)
            self.merged = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> "LoRALinear":
        """从现有的 nn.Linear 层创建 LoRALinear。

        Args:
            linear: 源线性层
            rank: LoRA 秩
            alpha: LoRA 缩放系数
            dropout: LoRA Dropout

        Returns:
            包含原始权重的 LoRALinear 层

        Example:
            >>> linear = nn.Linear(256, 512)
            >>> lora_linear = LoRALinear.from_linear(linear, rank=8)
        """
        # 检查源层是否包含偏置
        has_bias = linear.bias is not None
        # 创建新的 LoRALinear 层
        layer = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
        )

        # 复制原始权重和偏置
        layer.weight.data.copy_(linear.weight.data)
        if has_bias and linear.bias is not None:
            layer.bias_param.data.copy_(linear.bias.data)

        # 移动到与源层相同的设备
        layer = layer.to(linear.weight.device)

        return layer

    def extra_repr(self) -> str:
        """打印额外的层信息（用于 print(model)）。"""
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.3f}, merged={self.merged}"
        )


class LoRAAdapter:
    """LoRA 注入和生命周期管理器。

    本类负责:
        - 识别需要注入 LoRA 的目标线性层
        - 用 LoRALinear 替换原始 Linear 层
        - 管理参数冻结/解冻
        - 提供参数统计信息

    Args:
        model: 待注入 LoRA 的模型
        config: LoRA 配置

    Example:
        >>> model = MyModel()
        >>> config = LoRAConfig(rank=8, alpha=16.0)
        >>> adapter = LoRAAdapter(model, config)
        >>> adapter.inject()
        >>> adapter.freeze_base()
        >>> print(adapter.param_report())
    """

    def __init__(self, model: nn.Module, config: LoRAConfig) -> None:
        self.model = model
        self.config = config
        # 存储已注入的 LoRA 层（名称 -> LoRALinear 实例）
        self._layers: Dict[str, LoRALinear] = {}
        # 标记是否已注入
        self._injected = False

    def _should_inject(self, name: str, module: nn.Module) -> bool:
        """判断模块是否应该被替换为 LoRA。

        Args:
            name: 模块的完整名称
            module: 模块实例

        Returns:
            True 表示应该注入 LoRA
        """
        # 必须是 nn.Linear 类型
        if not isinstance(module, nn.Linear):
            return False

        # 如果未指定目标模块，则注入所有 Linear 层
        if not self.config.target_modules:
            return True

        # 检查名称是否包含任何目标模块的子字符串
        return any(target in name for target in self.config.target_modules)

    def inject(self) -> "LoRAAdapter":
        """将 LoRA 注入到目标线性层。

        Returns:
            自身（支持链式调用）
        """
        # 防止重复注入
        if self._injected:
            return self

        replacements = []

        # 第一步：找到所有目标模块
        for full_name, module in self.model.named_modules():
            if not self._should_inject(full_name, module):
                continue

            # 解析父模块名和子模块名
            parts = full_name.rsplit(".", 1)
            parent_name = parts[0] if len(parts) > 1 else ""
            child_name = parts[-1]

            # 获取父模块对象
            parent = self.model
            if parent_name:
                for p in parent_name.split("."):
                    parent = getattr(parent, p)

            replacements.append((parent, child_name, module, full_name))

        # 第二步：用 LoRALinear 替换原始 Linear 层
        for parent, child_name, linear, full_name in replacements:
            lora_layer = LoRALinear.from_linear(
                linear,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
            )
            # 通过 setattr 替换父模块中的子模块
            setattr(parent, child_name, lora_layer)
            # 记录已注入的层
            self._layers[full_name] = lora_layer

        self._injected = True
        return self

    def freeze_base(self) -> "LoRAAdapter":
        """冻结基础模型权重，仅训练 LoRA 参数。

        Returns:
            自身（支持链式调用）
        """
        # 遍历所有参数，仅 lora_A 和 lora_B 可训练
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_A" in name or "lora_B" in name
        return self

    def unfreeze_all(self) -> "LoRAAdapter":
        """解冻所有参数（用于完整微调）。

        Returns:
            自身（支持链式调用）
        """
        for param in self.model.parameters():
            param.requires_grad = True
        return self

    def merge_all(self) -> "LoRAAdapter":
        """合并所有 LoRA 权重到基础权重（推理优化）。

        Returns:
            自身（支持链式调用）
        """
        for layer in self._layers.values():
            layer.merge_lora()
        return self

    def unmerge_all(self) -> "LoRAAdapter":
        """取消合并所有 LoRA 权重。

        Returns:
            自身（支持链式调用）
        """
        for layer in self._layers.values():
            layer.unmerge_lora()
        return self

    def trainable_params(self) -> List[nn.Parameter]:
        """获取所有可训练参数列表。"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_count(self) -> int:
        """统计可训练参数数量。"""
        return sum(p.numel() for p in self.trainable_params())

    def total_count(self) -> int:
        """统计模型总参数数量。"""
        return sum(p.numel() for p in self.model.parameters())

    def lora_count(self) -> int:
        """统计 LoRA 参数数量 (A + B 矩阵)。"""
        return sum(
            layer.lora_A.numel() + layer.lora_B.numel()
            for layer in self._layers.values()
        )

    def injected_layers(self) -> Dict[str, LoRALinear]:
        """获取已注入的 LoRA 层字典。"""
        return dict(self._layers)

    def param_report(self) -> str:
        """生成参数效率报告。

        Returns:
            格式化的报告字符串
        """
        total = self.total_count()
        lora_n = self.lora_count()
        trainable = self.trainable_count()
        # 计算 LoRA 参数占总参数的比例
        pct = lora_n / total * 100 if total > 0 else 0

        lines = [
            "=" * 55,
            "LoRA 参数效率报告",
            "=" * 55,
            f"  注入层数      : {len(self._layers)}",
            f"  总参数量      : {total:,}",
            f"  LoRA 参数量   : {lora_n:,}  (占总参数 {pct:.3f}%)",
            f"  可训练参数量  : {trainable:,}",
            f"  LoRA 秩       : {self.config.rank}",
            f"  LoRA alpha    : {self.config.alpha}",
            f"  缩放因子      : {self.config.scale:.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)


def apply_lora_to_sam(
    sam_model: nn.Module,
    config: Optional[LoRAConfig] = None,
    preset: str = "full",
    freeze: bool = True,
) -> LoRAAdapter:
    """一键将 LoRA 应用到 SAM 模型。

    Args:
        sam_model: 待注入 LoRA 的 SAM 模型
        config: 自定义 LoRA 配置（优先级高于 preset）
        preset: 预设配置名称 ('full', 'image_encoder' 等)
        freeze: 是否在注入后冻结基础权重

    Returns:
        LoRAAdapter 实例

    Example:
        >>> from segment_anything import sam_model_registry
        >>> sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        >>> adapter = apply_lora_to_sam(sam, preset='full', freeze=True)
        >>> print(adapter.param_report())
    """
    # 选择配置：优先使用自定义配置，否则使用预设
    cfg = config or LORA_CONFIGS.get(preset, LORA_CONFIGS["full"])
    # 创建适配器
    adapter = LoRAAdapter(sam_model, cfg)
    # 注入 LoRA
    adapter.inject()

    # 可选：冻结基础权重
    if freeze:
        adapter.freeze_base()

    return adapter
