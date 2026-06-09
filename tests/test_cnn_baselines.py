"""UNet / DeepLabV3-like baseline smoke tests。

这些测试不验证最终精度，只验证论文对比实验新增基线的关键工程契约：
1. 模型能接收 `[B,3,H,W]` 图像；
2. 输出 logits 为 `[B,1,H,W]`；
3. BCE + Dice 损失为有限值且可反向传播。
"""

from __future__ import annotations

import pytest
import torch

from baselines.cnn_segmentation import DeepLabV3Like, UNet, build_cnn_baseline
from baselines.cnn_segmentation.losses import segmentation_loss


@pytest.mark.parametrize(
    "model",
    [
        UNet(base_channels=8),
        DeepLabV3Like(base_channels=8, aspp_channels=16),
    ],
)
def test_cnn_baseline_forward_shape(model: torch.nn.Module) -> None:
    """验证两个 CNN baseline 的输入输出 shape。

    输入是两张 64x64 RGB 图像；输出必须是同尺寸单通道裂缝 logits。
    """
    images = torch.randn(2, 3, 64, 64)
    outputs = model(images)
    assert set(outputs.keys()) >= {"logits", "masks"}
    assert outputs["logits"].shape == (2, 1, 64, 64)
    assert outputs["masks"].shape == (2, 1, 64, 64)


def test_cnn_baseline_loss_backward() -> None:
    """验证损失函数有限且能够回传梯度。

    这能提前发现 logits/target 尺寸不匹配、pos_weight 设备不匹配等常见错误。
    """
    model = UNet(base_channels=8)
    images = torch.randn(2, 3, 64, 64)
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 16:48, 30:34] = 1.0
    logits = model(images)["logits"]
    loss_dict = segmentation_loss(logits, masks, pos_weight=5.0)
    assert torch.isfinite(loss_dict["loss"])
    loss_dict["loss"].backward()
    grad_norm = sum(
        float(param.grad.abs().sum().item())
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_norm > 0.0


def test_build_cnn_baseline_factory() -> None:
    """验证配置工厂能按名称构建 UNet 和 DeepLabV3-like。"""
    unet = build_cnn_baseline({"name": "unet", "base_channels": 8})
    deeplab = build_cnn_baseline({"name": "deeplabv3_like", "base_channels": 8, "aspp_channels": 16})
    assert isinstance(unet, UNet)
    assert isinstance(deeplab, DeepLabV3Like)
