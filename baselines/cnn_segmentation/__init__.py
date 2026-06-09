"""CNN 语义分割基线包。

该包用于承载与 SAM+LoRA 做论文对比的传统深度分割模型，例如 UNet
和 DeepLabV3-like。所有模型均输出二分类裂缝 logits，便于复用项目中
已经统一的 mask 归一化、loss 和指标口径。
"""

from baselines.cnn_segmentation.models import (
    DeepLabV3Like,
    UNet,
    build_cnn_baseline,
)

__all__ = ["DeepLabV3Like", "UNet", "build_cnn_baseline"]
