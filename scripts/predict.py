"""推理脚本（单图 / 批量图像预测）。

本脚本的定位：
1. 加载训练好的模型权重；
2. 对新图像做裂缝分割预测；
3. 输出二值 mask 与可视化叠图。

和项目闭环的关系：
- train.py 负责训练模型；
- eval.py 负责算指标；
- predict.py 负责在“真实新图”上实际使用模型。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

# 让脚本在任意工作目录下运行时，都能找到项目模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.enhanced_sam import EnhancedSAM, EnhancedSAMConfig, build_enhanced_sam
from models.sam_base import load_sam_model, patch_sam_for_img_size


IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Predict crack masks using trained EnhancedSAM model")
    parser.add_argument("--image", type=str, default="", help="单张图像路径（与 --input_dir 二选一）")
    parser.add_argument("--input_dir", type=str, default="", help="批量图像目录（与 --image 二选一）")
    parser.add_argument("--output", type=str, default="outputs/predict", help="输出目录")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="SAM 预训练权重路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练得到的模型 checkpoint 路径")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="训练配置文件路径")
    parser.add_argument("--device", type=str, default="", help="推理设备，如 cuda/cpu；留空自动选择")
    parser.add_argument("--threshold", type=float, default=0.5, help="概率图二值化阈值")
    parser.add_argument("--save_overlay", action="store_true", help="是否额外保存叠图可视化")
    parser.add_argument("--save_prob", action="store_true", help="是否保存概率热力图（灰度）")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(device_arg: str) -> torch.device:
    """选择推理设备。"""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: Dict[str, Any], sam_checkpoint: str, device: torch.device) -> EnhancedSAM:
    """按配置构建模型。"""
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    boundary_cfg = cfg["boundary"]
    loss_cfg = cfg["loss"]

    sam = load_sam_model(
        model_type=model_cfg["type"],
        checkpoint_path=sam_checkpoint,
        device=str(device),
    )

    img_size = int(model_cfg["img_size"])
    if img_size != 1024:
        patch_sam_for_img_size(sam, img_size)

    enhanced_cfg = EnhancedSAMConfig(
        vit_embed_dim=int(model_cfg["vit_embed_dim"]),
        prompt_embed_dim=int(model_cfg["prompt_embed_dim"]),
        use_lora=bool(lora_cfg["use_lora"]),
        lora_rank=int(lora_cfg["rank"]),
        lora_alpha=float(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        lora_target_modules=list(lora_cfg["target_modules"]),
        use_boundary=bool(boundary_cfg["use_boundary"]),
        boundary_mid_channels=int(boundary_cfg["mid_channels"]),
        boundary_num_iterations=int(boundary_cfg["num_iterations"]),
        boundary_detector_mid=int(boundary_cfg["detector_mid"]),
        loss_w_bce=float(loss_cfg["w_bce"]),
        loss_w_dice=float(loss_cfg["w_dice"]),
        loss_w_bound=float(loss_cfg["w_bound"]),
        loss_pos_weight=float(loss_cfg["pos_weight"]),
        loss_boundary_radius=int(loss_cfg["boundary_radius"]),
    )
    return build_enhanced_sam(sam, config=enhanced_cfg).to(device)


def load_trained_weights(model: EnhancedSAM, checkpoint_path: str, device: torch.device) -> int:
    """加载训练权重，返回 checkpoint 中记录的 epoch。"""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        return int(ckpt.get("epoch", -1))

    # 兼容直接保存 state_dict 的情况
    model.load_state_dict(ckpt, strict=True)
    return -1


def list_images(input_dir: Path) -> List[Path]:
    """列出目录中的图像文件。"""
    images: List[Path] = []
    for suffix in IMAGE_SUFFIXES:
        images.extend(input_dir.glob(f"*{suffix}"))
        images.extend(input_dir.glob(f"*{suffix.upper()}"))
    return sorted(set(images))


def preprocess_image(image_bgr: np.ndarray, img_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """图像预处理：BGR->RGB、resize、归一化、转张量。

    返回：
    - image_tensor: [1,3,H,W]，可直接送模型
    - image_rgb_resized: [H,W,3]，用于生成可视化叠图
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb_resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # 与训练阶段保持一致：ImageNet 归一化
    image = image_rgb_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std

    # numpy(H,W,C) -> torch(1,C,H,W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float()
    return image_tensor, image_rgb_resized


def select_logits(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """从模型输出中选最终预测 logits。"""
    if "refined_mask" in outputs:
        return outputs["refined_mask"]  # [B,1,H,W]
    masks = outputs["masks"]            # [B,M,H,W]
    iou_pred = outputs["iou_pred"]      # [B,M]
    best_idx = iou_pred.argmax(dim=1, keepdim=True)
    h, w = masks.shape[-2:]
    return masks.gather(1, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w))


@torch.no_grad()
def predict_one(
    model: EnhancedSAM,
    image_path: Path,
    img_size: int,
    threshold: float,
    device: torch.device,
) -> Dict[str, Any]:
    """对单张图像做预测。"""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"图像读取失败: {image_path}")

    orig_h, orig_w = image_bgr.shape[:2]

    image_tensor, image_rgb_resized = preprocess_image(image_bgr, img_size=img_size)
    image_tensor = image_tensor.to(device, non_blocking=True)

    # 前向推理
    outputs = model(image=image_tensor, multimask=False)
    logits = select_logits(outputs)  # [1,1,h,w]

    # logits 分辨率可能小于输入尺寸（如 SAM 的 1/4），插值到 img_size
    logits_up = torch.nn.functional.interpolate(
        logits,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    prob = torch.sigmoid(logits_up)[0, 0].detach().cpu().numpy()  # [H,W], 值域 [0,1]
    mask_bin = (prob >= threshold).astype(np.uint8) * 255         # [H,W], 值域 {0,255}

    # 为了与原图对齐显示，再 resize 回原图大小
    mask_bin_orig = cv2.resize(mask_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    prob_orig = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 叠图（红色区域表示裂缝）
    overlay = image_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = 0.45
    crack_region = mask_bin_orig > 0
    overlay[crack_region] = cv2.addWeighted(overlay, 1.0 - alpha, red, alpha, 0)[crack_region]

    return {
        "image_path": str(image_path),
        "orig_size": [int(orig_h), int(orig_w)],
        "mask_bin_orig": mask_bin_orig,   # uint8, {0,255}, 原图尺寸
        "overlay_bgr": overlay,           # BGR, 原图尺寸
        "prob_orig": prob_orig,           # float32, [0,1], 原图尺寸
        "resized_rgb": image_rgb_resized, # RGB, img_size 尺寸
    }


def ensure_inputs(image: str, input_dir: str) -> Tuple[Optional[Path], Optional[Path]]:
    """检查输入参数合法性。"""
    image_path = Path(image) if image else None
    dir_path = Path(input_dir) if input_dir else None

    if image_path and dir_path:
        raise ValueError("参数冲突：--image 和 --input_dir 只能二选一。")
    if (image_path is None) and (dir_path is None):
        raise ValueError("参数缺失：--image 和 --input_dir 至少需要提供一个。")

    if image_path and (not image_path.exists()):
        raise FileNotFoundError(f"未找到图像文件: {image_path}")
    if dir_path and (not dir_path.exists()):
        raise FileNotFoundError(f"未找到输入目录: {dir_path}")
    return image_path, dir_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = select_device(args.device)

    image_path, input_dir = ensure_inputs(args.image, args.input_dir)

    img_size = int(cfg["model"]["img_size"])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg=cfg, sam_checkpoint=args.sam_checkpoint, device=device)
    loaded_epoch = load_trained_weights(model, args.checkpoint, device=device)
    model.eval()

    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 模型输入尺寸: {img_size}")
    print(f"[INFO] 加载 checkpoint: {args.checkpoint} (epoch={loaded_epoch})")

    if image_path is not None:
        images = [image_path]
    else:
        images = list_images(input_dir)  # type: ignore[arg-type]
        if not images:
            raise RuntimeError(f"目录中未发现图像: {input_dir}")

    records: List[Dict[str, Any]] = []
    for path in images:
        result = predict_one(
            model=model,
            image_path=path,
            img_size=img_size,
            threshold=float(args.threshold),
            device=device,
        )

        stem = path.stem
        mask_path = output_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), result["mask_bin_orig"])

        overlay_path = output_dir / f"{stem}_overlay.png"
        if args.save_overlay:
            cv2.imwrite(str(overlay_path), result["overlay_bgr"])

        prob_path = output_dir / f"{stem}_prob.png"
        if args.save_prob:
            prob_u8 = np.clip(result["prob_orig"] * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(prob_path), prob_u8)

        rec = {
            "image": str(path),
            "mask": str(mask_path),
            "overlay": str(overlay_path) if args.save_overlay else "",
            "prob": str(prob_path) if args.save_prob else "",
            "orig_size": result["orig_size"],
        }
        records.append(rec)

        print(f"[OK] {path.name} -> mask saved")

    # 保存一次推理索引，方便你后续做论文图表或错误案例追踪
    index_path = output_dir / "predict_index.json"
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "sam_checkpoint": args.sam_checkpoint,
        "threshold": args.threshold,
        "num_images": len(records),
        "records": records,
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 推理完成，共 {len(records)} 张。")
    print(f"[INFO] 输出目录: {output_dir}")
    print(f"[INFO] 索引文件: {index_path}")


if __name__ == "__main__":
    main()

