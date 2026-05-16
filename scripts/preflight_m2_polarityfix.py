"""M2 极性修正后正式训练前预检。

这个脚本在长时间训练前运行，目标是把数据、配置、权重和 mask 语义问题提前暴露，
避免训练完成后才发现结果不可用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import DatasetConfig, TunnelCrackDataset  # noqa: E402
from utils.mask_utils import normalize_crack_mask  # noqa: E402


EXPECTED_SAM_VIT_B_SHA256 = "EC2DF62732614E57411CDCF32A23FFDF28910380D03139EE0F4FCBE91EB8C912"


def parse_args() -> argparse.Namespace:
    """解析预检参数。

    输入:
    - data_root/config/sam_checkpoint/output: 分别指向数据、训练配置、SAM 权重和报告输出。

    输出:
    - argparse.Namespace，供 main 使用。

    为什么这样做:
    - 预检需要和正式训练使用同一组路径，避免检查的是一套文件、训练的是另一套文件。
    """
    parser = argparse.ArgumentParser(description="Preflight checks before M2 polarity-fix training")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--config", type=str, default="configs/m2_lora_polarityfix_config.yaml")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--output", type=str, default="analysis/m2_polarityfix_preflight_2026-05-15.json")
    parser.add_argument("--expected_sam_sha256", type=str, default=EXPECTED_SAM_VIT_B_SHA256)
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置并保证结果是 dict。"""
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件不是 YAML dict: {path}")
    return loaded


def sha256_file(path: Path) -> str:
    """计算文件 SHA256。

    输入:
    - path: 权重文件路径。

    输出:
    - 大写 SHA256 字符串。

    为什么这样做:
    - SAM 权重若损坏或换版本，长训练得到的结果不可复现。
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def imdecode_gray(path: Path) -> np.ndarray:
    """用 Windows 友好的方式读取灰度图。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"灰度图解码失败: {path}")
    return image


def annotation_stats(data_root: Path, split: str) -> Dict[str, Any]:
    """统计某个 split 的原始标注极性和归一化后裂缝比例。

    输入:
    - data_root/split/annotations 下的标注 PNG。

    输出:
    - 包含样本数量、原始白色比例、归一化裂缝比例、异常样本列表的 dict。

    为什么这样做:
    - 本次返工的根因是标注极性混用。这里直接验证归一化后所有样本都变成少量裂缝前景。
    """
    ann_dir = data_root / split / "annotations"
    if not ann_dir.exists():
        raise FileNotFoundError(f"标注目录不存在: {ann_dir}")

    raw_white_ratios: List[float] = []
    crack_ratios: List[float] = []
    empty_after_normalize: List[str] = []
    too_large_after_normalize: List[Tuple[str, float]] = []

    for path in sorted(ann_dir.glob("*.png")):
        raw = imdecode_gray(path)
        raw_white = float((raw > 127).mean())
        crack = normalize_crack_mask(raw)
        crack_ratio = float(crack.mean())

        raw_white_ratios.append(raw_white)
        crack_ratios.append(crack_ratio)

        if crack_ratio <= 0.0:
            empty_after_normalize.append(path.name)
        if crack_ratio > 0.5:
            too_large_after_normalize.append((path.name, crack_ratio))

    raw_arr = np.asarray(raw_white_ratios, dtype=np.float64)
    crack_arr = np.asarray(crack_ratios, dtype=np.float64)
    if raw_arr.size == 0:
        raise RuntimeError(f"{split} 没有 annotation 文件")

    return {
        "count": int(raw_arr.size),
        "raw_white_majority": int((raw_arr > 0.5).sum()),
        "raw_black_majority": int((raw_arr < 0.5).sum()),
        "raw_white_ratio_min": float(raw_arr.min()),
        "raw_white_ratio_median": float(np.median(raw_arr)),
        "raw_white_ratio_max": float(raw_arr.max()),
        "normalized_crack_ratio_min": float(crack_arr.min()),
        "normalized_crack_ratio_median": float(np.median(crack_arr)),
        "normalized_crack_ratio_max": float(crack_arr.max()),
        "empty_after_normalize": empty_after_normalize[:20],
        "too_large_after_normalize": too_large_after_normalize[:20],
    }


def dataset_stats(data_root: str, split: str, img_size: int) -> Dict[str, Any]:
    """构建真实 Dataset，验证图像/标注配对、坏图过滤和 tensor 输出。

    输入:
    - data_root/split/images 与 annotations。

    输出:
    - dataset 样本数，以及第一条样本的 image/mask shape 和 mask 前景数量。

    为什么这样做:
    - 直接走训练会使用的 Dataset 类，能提前暴露解码、配对和 tensor shape 问题。
    """
    dataset = TunnelCrackDataset(
        DatasetConfig(data_root=data_root, split=split, img_size=img_size, use_augment=False)
    )
    sample = dataset[0]
    return {
        "dataset_count": int(len(dataset)),
        "first_image_shape": list(sample["image"].shape),
        "first_mask_shape": list(sample["mask"].shape),
        "first_mask_sum": float(sample["mask"].sum().item()),
        "first_image_path": sample["image_path"],
        "first_mask_path": sample["mask_path"],
    }


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """验证训练配置中最容易造成长训练浪费的字段。

    输出:
    - 关键配置摘要。

    为什么这样做:
    - Windows 上 num_workers 必须为 0；输出目录由训练命令控制；模型尺寸和 batch size
      需要在训练前明确记录。
    """
    required_top = ["model", "lora", "boundary", "loss", "training", "optimizer", "scheduler"]
    missing = [key for key in required_top if key not in cfg]
    if missing:
        raise KeyError(f"配置缺少顶层字段: {missing}")

    training = cfg["training"]
    model = cfg["model"]
    if int(training.get("num_workers", -1)) != 0:
        raise ValueError("Windows 正式训练要求 training.num_workers = 0")
    if int(model.get("img_size", 0)) <= 0:
        raise ValueError("model.img_size 必须为正数")
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs 必须为正数")

    return {
        "model_type": model.get("type"),
        "img_size": int(model["img_size"]),
        "batch_size": int(training["batch_size"]),
        "epochs": int(training["epochs"]),
        "num_workers": int(training["num_workers"]),
        "lr": float(training["lr"]),
        "boundary_enabled": bool(cfg["boundary"].get("use_boundary", False)),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    config_path = Path(args.config)
    sam_path = Path(args.sam_checkpoint)

    cfg = read_yaml(config_path)
    config_summary = validate_config(cfg)

    if not sam_path.exists():
        raise FileNotFoundError(f"SAM 权重不存在: {sam_path}")
    actual_sha = sha256_file(sam_path)
    if args.expected_sam_sha256 and actual_sha != args.expected_sam_sha256.upper():
        raise RuntimeError(
            f"SAM 权重 SHA256 不匹配: actual={actual_sha}, expected={args.expected_sam_sha256.upper()}"
        )

    split_reports: Dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        ann = annotation_stats(data_root, split)
        ds = dataset_stats(args.data_root, split, int(config_summary["img_size"]))
        if ann["empty_after_normalize"]:
            raise RuntimeError(f"{split} 存在归一化后空 mask: {ann['empty_after_normalize'][:5]}")
        if ann["too_large_after_normalize"]:
            raise RuntimeError(f"{split} 存在归一化后前景过大 mask: {ann['too_large_after_normalize'][:5]}")
        split_reports[split] = {"annotations": ann, "dataset": ds}

    representatives = {}
    for stem in ["214_05_01", "mosaic_1561"]:
        path = data_root / "val" / "annotations" / f"{stem}.png"
        if path.exists():
            raw = imdecode_gray(path)
            representatives[stem] = {
                "raw_white_ratio": float((raw > 127).mean()),
                "normalized_crack_ratio": float(normalize_crack_mask(raw).mean()),
            }

    payload = {
        "status": "ok",
        "data_root": str(data_root),
        "config": str(config_path),
        "sam_checkpoint": str(sam_path),
        "sam_sha256": actual_sha,
        "config_summary": config_summary,
        "splits": split_reports,
        "representatives": representatives,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[OK] M2 polarity-fix preflight passed")
    print(f"[INFO] report: {out_path}")
    for split, report in split_reports.items():
        ann = report["annotations"]
        ds = report["dataset"]
        print(
            f"[INFO] {split}: dataset={ds['dataset_count']}, "
            f"raw_white_majority={ann['raw_white_majority']}, "
            f"raw_black_majority={ann['raw_black_majority']}, "
            f"normalized_crack_median={ann['normalized_crack_ratio_median']:.6f}, "
            f"normalized_crack_max={ann['normalized_crack_ratio_max']:.6f}"
        )


if __name__ == "__main__":
    main()

