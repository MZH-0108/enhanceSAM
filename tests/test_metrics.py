"""SegmentationMetricMeter 的 micro / macro 口径测试。

重点验证：当样本前景面积差异很大时，micro（全局像素聚合）会被大前景样本
主导而显得很高，而 macro（逐图平均）更贴近“平均每张图分得怎样”。这正是
项目里“指标漂亮但细裂缝效果图差”的机制，必须用 macro 才能暴露。
"""

from __future__ import annotations

import torch

from utils.metrics import SegmentationMetricMeter


def _block(h: int, w: int, fg_pixels: int) -> torch.Tensor:
    """构造 [1,1,h,w] 二值图，前 fg_pixels 个像素置 1，其余为 0。"""
    t = torch.zeros(1, 1, h, w)
    t.view(-1)[:fg_pixels] = 1.0
    return t


def test_macro_and_micro_keys_present() -> None:
    """summary 应同时输出 micro 与 macro 两套口径的 key。"""
    meter = SegmentationMetricMeter(boundary_radius=0)
    pred = _block(8, 8, 10)
    meter.update(pred_bin=pred, target_bin=pred.clone(), infer_time_sec=0.0)
    s = meter.summary()
    for k in ["mIoU", "Dice", "mIoU_macro", "Dice_macro", "Precision", "Recall"]:
        assert k in s, f"summary 缺少指标 key: {k}"


def test_macro_diverges_from_micro_on_imbalanced_sizes() -> None:
    """大前景完美 + 小前景全错：macro=0.5，micro 被大前景主导接近 1。"""
    meter = SegmentationMetricMeter(boundary_radius=0)

    # 图1：100 像素全前景，完美预测 → 逐图 IoU = 1
    tgt1 = _block(10, 10, 100)
    meter.update(pred_bin=tgt1.clone(), target_bin=tgt1, infer_time_sec=0.0)

    # 图2：仅 1 像素前景，预测全 0 → 逐图 IoU = 0
    tgt2 = _block(10, 10, 1)
    meter.update(pred_bin=torch.zeros_like(tgt2), target_bin=tgt2, infer_time_sec=0.0)

    s = meter.summary()
    # macro：逐图 (1 + 0) / 2 = 0.5
    assert abs(s["mIoU_macro"] - 0.5) < 1e-4
    # micro：tp=100, fp=0, fn=1 → 100/101 ≈ 0.990，被大前景样本主导
    assert s["mIoU"] > 0.95
    # 两口径明显分歧，正是“整体指标高但多数细裂缝图差”的信号
    assert s["mIoU"] - s["mIoU_macro"] > 0.4
