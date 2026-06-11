"""诊断脚本：量化 normalize_crack_mask 的极性启发式误判规模。

目的（只读分析，不改数据、不需要 torch/GPU）：
- 极性审计已证明：mosaic_* = 黑底白裂缝；非 mosaic = 白底黑裂缝。
  因此“文件名分组”是极性的可靠金标准。
- 当前 utils/mask_utils.normalize_crack_mask 用的是“白色像素占比 > 0.5 就取反”
  的启发式。本脚本对比两者，统计有多少样本会被启发式判错，
  以及这些样本的前景占比分布，从而验证“指标虚高 / 效果图难看”的根因。

用法：
    python scripts/diagnose_mask_polarity.py --data_root data --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Windows 终端默认 GBK，强制 UTF-8 输出避免中文乱码。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def heuristic_inverts(white_ratio: float) -> bool:
    """复刻 normalize_crack_mask 的判断：白占比 > 0.5 时取反。"""
    return white_ratio > 0.5


def true_polarity_is_white_crack(stem: str) -> bool:
    """金标准：mosaic_* 是白裂缝（不应取反），其余是黑裂缝（应取反）。"""
    return stem.startswith("mosaic_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--threshold", type=int, default=127)
    args = parser.parse_args()

    mask_dir = Path(args.data_root) / args.split / "annotations"
    paths = sorted(mask_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"未找到标注文件: {mask_dir}")

    total = 0
    misjudged = 0
    danger_zone = 0  # 白占比落在 [0.4, 0.6] 的“近临界”样本
    fg_pixels = []   # 每张图的裂缝前景【像素绝对数量】，用于看 micro 聚合的集中度
    fg_ratios_all = []

    for p in paths:
        with Image.open(p) as im:
            mask = np.asarray(im.convert("L"))
        total += 1

        white_ratio = float((mask > args.threshold).mean())

        # 启发式给出的“裂缝”是哪一极
        inv = heuristic_inverts(white_ratio)
        fg_ratio = (1.0 - white_ratio) if inv else white_ratio

        # 金标准：非 mosaic 应取反（黑裂缝），mosaic 不应取反
        should_invert = not true_polarity_is_white_crack(p.stem)
        if inv != should_invert:
            misjudged += 1
        if 0.4 <= white_ratio <= 0.6:
            danger_zone += 1

        fg_ratios_all.append(fg_ratio)
        fg_pixels.append(fg_ratio * mask.size)

    fr = np.array(fg_ratios_all)
    fp = np.array(fg_pixels)

    print(f"\n===== 1) 极性启发式可靠性 ({args.split}, n={total}) =====")
    print(f"被判错样本数: {misjudged} ({misjudged/total*100:.2f}%) | "
          f"近临界区[0.40,0.60]样本数: {danger_zone} ({danger_zone/total*100:.2f}%)")

    print(f"\n===== 2) 类别不平衡：裂缝前景占比分布 =====")
    for q in [50, 75, 90, 95, 99]:
        print(f"  P{q}: {np.percentile(fr, q)*100:.3f}%", end="   ")
    print(f"  max: {fr.max()*100:.3f}%")
    print(f"  平均前景占比: {fr.mean()*100:.3f}%  （前景像素极度稀疏）")

    print(f"\n===== 3) micro(全局像素)聚合的集中度 =====")
    order = np.argsort(fp)[::-1]
    cum = np.cumsum(fp[order]) / fp.sum()
    for pct in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(total * pct))
        print(f"  前 {pct*100:>4.0f}% 样本（{k:>4d}张，前景最大的图）贡献了 "
              f"{cum[k-1]*100:5.1f}% 的全部裂缝像素")
    print("  → micro mIoU = tp/(tp+fp+fn) 由这些大前景样本主导；"
          "占多数的细裂缝小样本即使分错，对该数字几乎无影响。")


if __name__ == "__main__":
    main()
