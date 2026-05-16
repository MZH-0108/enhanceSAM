"""生成 M2 论文定性对比图。

本脚本用于把同一批 val 样本的 `GT`、`SAM-AMG`、`SAM-Box-Oracle`
与 `SAM+LoRA` 预测结果拼成论文可用的多列对比图。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.sam_vanilla.common import (  # noqa: E402
    Sample,
    collect_samples,
    mask_to_bbox_xyxy,
    split_connected_components,
)
from baselines.sam_vanilla.eval_amg import build_amg, candidate_pass  # noqa: E402
from baselines.sam_vanilla.eval_box_oracle import predict_with_boxes  # noqa: E402
from scripts.predict import (  # noqa: E402
    build_model,
    load_config,
    load_trained_weights,
    preprocess_image,
    select_device,
)
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402
from utils.metrics import select_final_logits  # noqa: E402
from utils.mask_utils import normalize_crack_mask  # noqa: E402
from utils.visualization import binary01_to_u8, overlay_mask_on_bgr  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate M2 qualitative comparison figures")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--m2_config", type=str, default="configs/m2_eval_config.yaml")
    parser.add_argument("--train_config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--lora_checkpoint", type=str, default="checkpoints/m2_lora/best_model.pth")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument(
        "--sample_stems",
        type=str,
        default="",
        help="逗号分隔的样本 stem；为空时自动选择前景面积较大的样本。",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--panel_width", type=int, default=360)
    parser.add_argument("--output_dir", type=str, default="results/visualizations/m2_qualitative")
    parser.add_argument("--paper_dir", type=str, default="paper/figures/m2_qualitative")
    parser.add_argument(
        "--allow_missing_lora",
        action="store_true",
        help="仅用于临时检查 baseline 图；论文正式图不应启用该参数。",
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="只生成 SAM-AMG 与 SAM-Box-Oracle 的 mask/叠图/panel，不加载 SAM+LoRA。",
    )
    return parser.parse_args()


def imread_bgr(path: Path) -> np.ndarray:
    """稳健读取 BGR 图像。

    输入：
    - path: 图像路径。
    输出：
    - BGR 图像，shape 为 [H,W,3]。
    为什么这样做：
    - Windows + OpenCV 在部分路径或文件编码场景下 `cv2.imread` 可能返回 None；
      `np.fromfile + cv2.imdecode` 对中文路径与特殊路径更稳健。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"图像读取失败: {path}")
    return image


def imread_mask01(path: Path) -> np.ndarray:
    """读取 GT mask 并转成“裂缝=1、背景=0”的 0/1 二值图。

    输入:
    - path: 原始 annotation 路径，灰度图可能是白裂缝或黑裂缝。

    输出:
    - crack01: uint8 二值图，shape=[H,W]，取值为 {0,1}。

    为什么这样做:
    - M2 可视化要和训练、baseline 评估使用同一目标语义。
    - 数据集中存在两套 mask 极性，统一归一化后 GT、Box-Oracle 和 LoRA
      对比图才真正表示裂缝区域。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"标注读取失败: {path}")
    return normalize_crack_mask(mask)


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置；文件不存在时返回空字典，便于脚本使用默认参数。"""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def choose_samples(samples: Sequence[Sample], sample_stems: str, num_samples: int) -> List[Sample]:
    """选择用于论文可视化的样本。

    输入：
    - samples: 当前 split 中所有 image/mask 配对样本。
    - sample_stems: 用户指定的样本名列表；非空时严格按该顺序选择。
    - num_samples: 自动选择时的样本数量。
    输出：
    - 固定顺序的样本列表。
    为什么这样做：
    - 论文图需要可复现；自动模式按 GT 前景面积降序选择，避免样本裂缝过少导致对比不可见。
    """
    if sample_stems.strip():
        wanted = [s.strip() for s in sample_stems.split(",") if s.strip()]
        sample_map = {s.image_path.stem: s for s in samples}
        missing = [stem for stem in wanted if stem not in sample_map]
        if missing:
            raise FileNotFoundError(f"指定样本不存在: {missing}")
        return [sample_map[stem] for stem in wanted]

    ranked: List[Tuple[int, Sample]] = []
    for sample in samples:
        gt01 = imread_mask01(sample.mask_path)
        ranked.append((int(gt01.sum()), sample))
    ranked.sort(key=lambda item: (-item[0], item[1].image_path.stem))
    return [sample for area, sample in ranked if area > 0][: max(1, int(num_samples))]


def build_box_predictor(model_type: str, checkpoint: str, device: torch.device) -> SamPredictor:
    """构建 SAM Box-Oracle predictor，输入官方 SAM 权重，输出可复用预测器。"""
    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(str(device))
    sam.eval()
    return SamPredictor(sam)


def predict_amg_mask(
    amg: Any,
    image_bgr: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """运行 SAM-AMG 并按裂缝形态规则筛选候选 mask。

    输入：
    - image_bgr: 原图，shape [H,W,3]。
    - params: 与 `baselines/sam_vanilla/eval_amg.py` 一致的筛选参数。
    输出：
    - pred01: 预测二值图，shape [H,W]，值域 {0,1}。
    """
    h, w = image_bgr.shape[:2]
    pred01 = np.zeros((h, w), dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    candidates = amg.generate(image_rgb)

    for cand in candidates:
        if candidate_pass(
            cand=cand,
            img_area=h * w,
            min_area=int(params.get("min_area", 30)),
            max_area_ratio=float(params.get("max_area_ratio", 0.2)),
            min_elongation=float(params.get("min_elongation", 2.5)),
            min_pred_iou=float(params.get("min_pred_iou", 0.75)),
        ):
            pred01 = np.maximum(pred01, cand["segmentation"].astype(np.uint8))
    return pred01


def predict_box_mask(
    predictor: SamPredictor,
    image_bgr: np.ndarray,
    gt01: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """运行 SAM-Box-Oracle，框由 GT mask 生成。

    输入：
    - predictor: 已加载官方 SAM 权重的 predictor。
    - image_bgr: 原图，shape [H,W,3]。
    - gt01: GT 二值图，shape [H,W]。
    - params: padding/use_components/min_component_area 等参数。
    输出：
    - pred01: SAM 在理想框提示下的预测二值图。
    """
    use_components = bool(params.get("use_components", True))
    padding = int(params.get("padding", 8))
    min_component_area = int(params.get("min_component_area", 20))

    if use_components:
        bboxes = split_connected_components(
            mask01=gt01,
            min_area=min_component_area,
            padding=padding,
        )
    else:
        one_box = mask_to_bbox_xyxy(gt01, padding=padding)
        bboxes = [one_box] if one_box is not None else []

    if not bboxes:
        return np.zeros_like(gt01, dtype=np.uint8)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return predict_with_boxes(predictor=predictor, image_rgb=image_rgb, bboxes_xyxy=bboxes)


@torch.no_grad()
def predict_lora_mask(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    img_size: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    """运行 SAM+LoRA 预测并恢复到原图尺寸。

    输入：
    - image_bgr: 原始 BGR 图像，shape [H,W,3]。
    - img_size: 训练配置中的模型输入尺寸。
    - threshold: 概率二值化阈值。
    输出：
    - pred01: 原图尺寸二值图，shape [H,W]，值域 {0,1}。
    复杂张量变化：
    - preprocess 后 image_tensor: [1,3,img_size,img_size]；
    - 模型输出 logits: [1,1,h,w] 或 [1,M,h,w] 经 `select_final_logits` 统一为 [1,1,h,w]；
    - 插值到 [1,1,img_size,img_size]，sigmoid 后再 resize 回原图 [H,W]。
    """
    orig_h, orig_w = image_bgr.shape[:2]
    image_tensor, _ = preprocess_image(image_bgr, img_size=img_size)
    image_tensor = image_tensor.to(device, non_blocking=True)
    outputs = model(image=image_tensor, multimask=False)
    logits = select_final_logits(outputs)
    logits_up = F.interpolate(logits, size=(img_size, img_size), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits_up)[0, 0].detach().cpu().numpy()
    mask_u8 = (prob >= threshold).astype(np.uint8) * 255
    mask_orig = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return (mask_orig > 0).astype(np.uint8)


def resize_keep_width(image_bgr: np.ndarray, width: int) -> np.ndarray:
    """按固定宽度等比例缩放单列图，保持各方法展示尺度一致。"""
    h, w = image_bgr.shape[:2]
    scale = float(width) / max(w, 1)
    height = max(1, int(round(h * scale)))
    return cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)


def draw_title(tile_bgr: np.ndarray, title: str) -> np.ndarray:
    """给单列图增加标题栏，输出仍为 BGR 图像。"""
    header_h = 42
    canvas = np.full((tile_bgr.shape[0] + header_h, tile_bgr.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h:, :, :] = tile_bgr
    cv2.putText(
        canvas,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_panel(
    image_bgr: np.ndarray,
    gt01: np.ndarray,
    amg01: np.ndarray,
    box01: np.ndarray,
    lora01: Optional[np.ndarray],
    panel_width: int,
    include_lora: bool = True,
) -> np.ndarray:
    """拼接论文多列对比图。

    输出列顺序：
    - baseline_only: Original / GT / SAM-AMG / SAM-Box；
    - 完整 M2: Original / GT / SAM-AMG / SAM-Box / SAM+LoRA。
    GT 使用绿色叠图，预测结果使用红色叠图，便于读者快速区分。
    """
    gt_overlay = overlay_mask_on_bgr(image_bgr, binary01_to_u8(gt01), color_bgr=(0, 180, 0), alpha=0.55)
    amg_overlay = overlay_mask_on_bgr(image_bgr, binary01_to_u8(amg01), color_bgr=(0, 0, 255), alpha=0.55)
    box_overlay = overlay_mask_on_bgr(image_bgr, binary01_to_u8(box01), color_bgr=(0, 0, 255), alpha=0.55)
    if lora01 is None:
        lora_overlay = image_bgr.copy()
        cv2.putText(
            lora_overlay,
            "LoRA checkpoint missing",
            (24, max(36, lora_overlay.shape[0] // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        lora_overlay = overlay_mask_on_bgr(image_bgr, binary01_to_u8(lora01), color_bgr=(0, 0, 255), alpha=0.55)

    columns = [
        ("Original", image_bgr),
        ("GT", gt_overlay),
        ("SAM-AMG", amg_overlay),
        ("SAM-Box", box_overlay),
    ]
    if include_lora:
        columns.append(("SAM+LoRA", lora_overlay))
    titled = [draw_title(resize_keep_width(img, panel_width), title) for title, img in columns]
    max_h = max(tile.shape[0] for tile in titled)
    padded: List[np.ndarray] = []
    for tile in titled:
        if tile.shape[0] < max_h:
            pad = np.full((max_h - tile.shape[0], tile.shape[1], 3), 255, dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)
    return np.hstack(padded)


def save_mask(path: Path, mask01: np.ndarray) -> None:
    """保存 0/1 mask 为 0/255 PNG。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), binary01_to_u8(mask01))


def save_overlay(path: Path, image_bgr: np.ndarray, mask01: np.ndarray, color_bgr: Tuple[int, int, int]) -> None:
    """保存检测结果叠图。

    输入：
    - image_bgr: 原图，shape [H,W,3]。
    - mask01: 预测或 GT 二值图，shape [H,W]。
    - color_bgr: 叠加颜色，GT 使用绿色，预测使用红色。
    输出：
    - PNG 叠图，便于肉眼检查裂缝位置。
    为什么这样做：
    - 单独 mask 能看像素级输出，但缺少原图语义；叠图能快速判断裂缝检测是否贴合真实裂缝。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_mask_on_bgr(image_bgr, binary01_to_u8(mask01), color_bgr=color_bgr, alpha=0.55)
    cv2.imwrite(str(path), overlay)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    m2_cfg = load_yaml(args.m2_config)
    amg_params = dict(m2_cfg.get("sam_amg", {}).get("params", {}))
    box_params = dict(m2_cfg.get("sam_box_oracle", {}).get("params", {}))

    lora_checkpoint = Path(args.lora_checkpoint)
    lora_ready = lora_checkpoint.exists()
    if (not args.baseline_only) and (not lora_ready) and (not args.allow_missing_lora):
        raise FileNotFoundError(
            f"未找到 SAM+LoRA checkpoint: {lora_checkpoint}。"
            "请先完成 LoRA 训练，或仅临时检查 baseline 图时添加 --allow_missing_lora。"
        )

    samples = collect_samples(args.data_root, args.split)
    selected = choose_samples(samples, args.sample_stems, args.num_samples)

    # A. 构建两条 SAM baseline。AMG 与 Box-Oracle 都使用官方 SAM 权重，
    #    但输入提示不同：AMG 无提示自动分割，Box-Oracle 使用 GT 外接框作为理想提示。
    amg_args = argparse.Namespace(
        model_type=args.model_type,
        sam_checkpoint=args.sam_checkpoint,
        device=str(device),
        points_per_side=int(amg_params.get("points_per_side", 32)),
        pred_iou_thresh=float(amg_params.get("pred_iou_thresh", 0.86)),
        stability_score_thresh=float(amg_params.get("stability_score_thresh", 0.92)),
        crop_n_layers=int(amg_params.get("crop_n_layers", 1)),
        crop_n_points_downscale_factor=int(amg_params.get("crop_n_points_downscale_factor", 2)),
        min_mask_region_area=int(amg_params.get("min_mask_region_area", 20)),
    )
    amg = build_amg(amg_args)
    box_predictor = build_box_predictor(args.model_type, args.sam_checkpoint, device)

    # B. 如果 LoRA 权重存在且不是 baseline_only，则加载增强模型；
    #    baseline_only 专门用于先查看两条 SAM baseline 的裂缝检测结果。
    lora_model: Optional[torch.nn.Module] = None
    img_size = 512
    loaded_epoch = None
    if (not args.baseline_only) and lora_ready:
        train_cfg = load_config(args.train_config)
        img_size = int(train_cfg["model"]["img_size"])
        lora_model = build_model(train_cfg, args.sam_checkpoint, device)
        loaded_epoch = load_trained_weights(lora_model, str(lora_checkpoint), device)
        lora_model.eval()

    output_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    paper_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for sample in selected:
        stem = sample.image_path.stem
        image_bgr = imread_bgr(sample.image_path)
        gt01 = imread_mask01(sample.mask_path)
        amg01 = predict_amg_mask(amg, image_bgr, amg_params)
        box01 = predict_box_mask(box_predictor, image_bgr, gt01, box_params)
        lora01 = None
        if lora_model is not None:
            lora01 = predict_lora_mask(lora_model, image_bgr, img_size, float(args.threshold), device)

        save_mask(mask_dir / "gt" / f"{stem}.png", gt01)
        save_mask(mask_dir / "sam_amg" / f"{stem}.png", amg01)
        save_mask(mask_dir / "sam_box_oracle" / f"{stem}.png", box01)
        save_overlay(overlay_dir / "gt" / f"{stem}.png", image_bgr, gt01, color_bgr=(0, 180, 0))
        save_overlay(overlay_dir / "sam_amg" / f"{stem}.png", image_bgr, amg01, color_bgr=(0, 0, 255))
        save_overlay(overlay_dir / "sam_box_oracle" / f"{stem}.png", image_bgr, box01, color_bgr=(0, 0, 255))
        if lora01 is not None:
            save_mask(mask_dir / "sam_lora" / f"{stem}.png", lora01)
            save_overlay(overlay_dir / "sam_lora" / f"{stem}.png", image_bgr, lora01, color_bgr=(0, 0, 255))

        panel = make_panel(
            image_bgr=image_bgr,
            gt01=gt01,
            amg01=amg01,
            box01=box01,
            lora01=lora01,
            panel_width=int(args.panel_width),
            include_lora=not bool(args.baseline_only),
        )
        panel_path = paper_dir / f"{stem}_comparison.png"
        cv2.imwrite(str(panel_path), panel)

        records.append(
            {
                "stem": stem,
                "image": str(sample.image_path),
                "annotation": str(sample.mask_path),
                "paper_panel": str(panel_path),
                "gt_mask": str(mask_dir / "gt" / f"{stem}.png"),
                "sam_amg_mask": str(mask_dir / "sam_amg" / f"{stem}.png"),
                "sam_box_oracle_mask": str(mask_dir / "sam_box_oracle" / f"{stem}.png"),
                "sam_lora_mask": str(mask_dir / "sam_lora" / f"{stem}.png") if lora01 is not None else "",
                "gt_overlay": str(overlay_dir / "gt" / f"{stem}.png"),
                "sam_amg_overlay": str(overlay_dir / "sam_amg" / f"{stem}.png"),
                "sam_box_oracle_overlay": str(overlay_dir / "sam_box_oracle" / f"{stem}.png"),
                "sam_lora_overlay": str(overlay_dir / "sam_lora" / f"{stem}.png") if lora01 is not None else "",
            }
        )
        print(f"[OK] {stem} -> {panel_path}")

    index = {
        "data_root": args.data_root,
        "split": args.split,
        "sam_checkpoint": args.sam_checkpoint,
        "lora_checkpoint": str(lora_checkpoint),
        "lora_checkpoint_ready": bool(lora_ready),
        "lora_loaded_epoch": loaded_epoch,
        "baseline_only": bool(args.baseline_only),
        "threshold": float(args.threshold),
        "num_samples": len(records),
        "records": records,
    }
    index_path = output_dir / "m2_qualitative_index.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 可视化索引: {index_path}")
    print(f"[INFO] 论文图目录: {paper_dir}")


if __name__ == "__main__":
    main()
