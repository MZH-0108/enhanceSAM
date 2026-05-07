"""原始 SAM 基线评估：Automatic Mask Generator (AMG)。

说明：
- 该脚本不使用 GT 提示（点/框），是“真实自动分割”基线；
- AMG 会输出多个候选区域，本脚本通过简单几何规则筛选“更像裂缝”的区域。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from baselines.sam_vanilla.common import (
    MetricMeter,
    collect_samples,
    load_gt_mask_binary,
    load_image_bgr,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM vanilla baseline with AutomaticMaskGenerator")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM 预训练权重")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/baselines/sam_amg_val.json")

    # 裂缝候选筛选参数（可按数据集微调）
    parser.add_argument("--min_area", type=int, default=30, help="最小候选区域面积")
    parser.add_argument("--max_area_ratio", type=float, default=0.2, help="最大候选区域面积占整图比例")
    parser.add_argument("--min_elongation", type=float, default=2.5, help="最小细长比（max(w/h, h/w)）")
    parser.add_argument("--min_pred_iou", type=float, default=0.75, help="AMG predicted_iou 下限")
    parser.add_argument("--boundary_radius", type=int, default=3)
    return parser.parse_args()


def build_amg(args: argparse.Namespace) -> SamAutomaticMaskGenerator:
    """构建 AMG 生成器。"""
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).to(args.device)
    sam.eval()

    # 这里采用一组偏保守参数，重点是生成稳定 mask。
    # 后续你可以针对裂缝数据调 points_per_side / pred_iou_thresh 等参数。
    amg = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=20,
    )
    return amg


def mask_elongation(mask01: np.ndarray) -> float:
    """计算候选区域细长比。"""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return 0.0
    w = float(xs.max() - xs.min() + 1)
    h = float(ys.max() - ys.min() + 1)
    return float(max(w / max(h, 1e-6), h / max(w, 1e-6)))


def candidate_pass(
    cand: Dict,
    img_area: int,
    min_area: int,
    max_area_ratio: float,
    min_elongation: float,
    min_pred_iou: float,
) -> bool:
    """判断候选掩码是否通过“裂缝形态”筛选。"""
    area = int(cand.get("area", 0))
    if area < min_area:
        return False
    if area > int(img_area * max_area_ratio):
        return False

    pred_iou = float(cand.get("predicted_iou", 0.0))
    if pred_iou < min_pred_iou:
        return False

    seg = cand["segmentation"].astype(np.uint8)
    elong = mask_elongation(seg)
    if elong < min_elongation:
        return False
    return True


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.data_root, args.split)
    amg = build_amg(args)
    meter = MetricMeter(boundary_radius=args.boundary_radius)

    progress = tqdm(samples, total=len(samples), desc=f"SAM-AMG-{args.split}")
    for sp in progress:
        image_bgr = load_image_bgr(sp.image_path)
        gt01 = load_gt_mask_binary(sp.mask_path)
        h, w = gt01.shape[:2]
        img_area = h * w

        # AMG 要求 RGB 输入
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        candidates = amg.generate(image_rgb)
        infer_t = time.perf_counter() - t0

        pred01 = np.zeros((h, w), dtype=np.uint8)
        # 把通过筛选的候选掩码做并集，得到最终裂缝预测
        for cand in candidates:
            if candidate_pass(
                cand=cand,
                img_area=img_area,
                min_area=args.min_area,
                max_area_ratio=args.max_area_ratio,
                min_elongation=args.min_elongation,
                min_pred_iou=args.min_pred_iou,
            ):
                seg = cand["segmentation"].astype(np.uint8)
                pred01 = np.maximum(pred01, seg)

        meter.update(pred01=pred01, gt01=gt01, infer_time_sec=infer_t)

    metrics = meter.summary()
    payload = {
        "method": "SAM-AutomaticMaskGenerator",
        "data_root": args.data_root,
        "split": args.split,
        "model_type": args.model_type,
        "sam_checkpoint": args.sam_checkpoint,
        "params": {
            "min_area": args.min_area,
            "max_area_ratio": args.max_area_ratio,
            "min_elongation": args.min_elongation,
            "min_pred_iou": args.min_pred_iou,
            "boundary_radius": args.boundary_radius,
        },
        "num_samples": len(samples),
        "metrics": metrics,
    }
    save_json(args.output, payload)

    print("[INFO] SAM-AMG baseline done.")
    for k, v in metrics.items():
        if k in {"ms_per_image", "FPS"}:
            print(f"  - {k}: {v:.4f}")
        else:
            print(f"  - {k}: {v:.6f}")
    print(f"[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()

