"""原始 SAM 基线上限评估：Box Oracle。

核心思路：
- 使用 GT mask 计算外接框（这是“理想提示”）；
- 把外接框喂给 SAM predictor，观察 SAM 在强提示下的上限性能。

注意：
- 这是上限参考，不是实际部署能力。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segment_anything import SamPredictor, sam_model_registry

from baselines.sam_vanilla.common import (
    MetricMeter,
    collect_samples,
    load_gt_mask_binary,
    load_image_bgr,
    mask_to_bbox_xyxy,
    save_json,
    split_connected_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM vanilla baseline with Box Oracle prompts")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/baselines/sam_box_oracle_val.json")

    parser.add_argument("--padding", type=int, default=8, help="外接框扩边像素")
    parser.add_argument("--use_components", action="store_true", help="是否按连通域拆分为多个框")
    parser.add_argument("--min_component_area", type=int, default=20, help="连通域最小面积阈值")
    parser.add_argument("--boundary_radius", type=int, default=3)
    return parser.parse_args()


def build_predictor(args: argparse.Namespace) -> SamPredictor:
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint).to(args.device)
    sam.eval()
    return SamPredictor(sam)


def predict_with_boxes(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    bboxes_xyxy: List[List[int]],
) -> np.ndarray:
    """给定一组框，做 SAM 预测并并集。"""
    h, w = image_rgb.shape[:2]
    pred01 = np.zeros((h, w), dtype=np.uint8)

    predictor.set_image(image_rgb)

    for box in bboxes_xyxy:
        box_np = np.array(box, dtype=np.float32)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],
            multimask_output=False,
        )
        # masks: [1,H,W] bool
        pred01 = np.maximum(pred01, masks[0].astype(np.uint8))
    return pred01


def main() -> None:
    args = parse_args()
    samples = collect_samples(args.data_root, args.split)
    predictor = build_predictor(args)
    meter = MetricMeter(boundary_radius=args.boundary_radius)

    progress = tqdm(samples, total=len(samples), desc=f"SAM-BoxOracle-{args.split}")
    for sp in progress:
        image_bgr = load_image_bgr(sp.image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt01 = load_gt_mask_binary(sp.mask_path)

        # 从 GT 生成“理想框”
        if args.use_components:
            bboxes = split_connected_components(
                mask01=gt01,
                min_area=args.min_component_area,
                padding=args.padding,
            )
        else:
            one_box = mask_to_bbox_xyxy(gt01, padding=args.padding)
            bboxes = [one_box] if one_box is not None else []

        # 无前景时，预测全 0（该样本对裂缝任务等价于无目标）
        if not bboxes:
            pred01 = np.zeros_like(gt01, dtype=np.uint8)
            infer_t = 0.0
        else:
            t0 = time.perf_counter()
            pred01 = predict_with_boxes(
                predictor=predictor,
                image_rgb=image_rgb,
                bboxes_xyxy=bboxes,
            )
            infer_t = time.perf_counter() - t0

        meter.update(pred01=pred01, gt01=gt01, infer_time_sec=infer_t)

    metrics = meter.summary()
    payload = {
        "method": "SAM-Box-Oracle",
        "data_root": args.data_root,
        "split": args.split,
        "model_type": args.model_type,
        "sam_checkpoint": args.sam_checkpoint,
        "params": {
            "padding": args.padding,
            "use_components": bool(args.use_components),
            "min_component_area": args.min_component_area,
            "boundary_radius": args.boundary_radius,
        },
        "num_samples": len(samples),
        "metrics": metrics,
    }
    save_json(args.output, payload)

    print("[INFO] SAM-Box-Oracle baseline done.")
    for k, v in metrics.items():
        if k in {"ms_per_image", "FPS"}:
            print(f"  - {k}: {v:.4f}")
        else:
            print(f"  - {k}: {v:.6f}")
    print(f"[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()

