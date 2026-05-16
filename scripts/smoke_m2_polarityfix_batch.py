"""M2 polarity-fix 真实数据单 batch 训练冒烟测试。

该脚本在正式长训练前加载真实 SAM 权重、真实数据和正式配置，执行一个 train batch
的 forward/loss/backward/optimizer step，再执行一个 val batch forward。它不替代正式训练，
只用于提前发现 GPU、模型、loss、mask shape 或数据语义错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import (  # noqa: E402
    build_model,
    build_optimizer,
    compute_batch_iou,
    load_config,
    select_device,
    select_logits,
    set_seed,
)
from utils.data_loader import build_dataloaders  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析 smoke 参数。"""
    parser = argparse.ArgumentParser(description="Run one real train/val batch before long M2 training")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--config", type=str, default="configs/m2_lora_polarityfix_config.yaml")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="analysis/m2_polarityfix_batch_smoke_2026-05-15.json")
    return parser.parse_args()


def mask_summary(mask: torch.Tensor) -> Dict[str, Any]:
    """汇总 batch mask，确认前景比例处于裂缝分割合理范围。

    输入:
    - mask: [B,1,H,W]，来自正式 DataLoader，裂缝应为 1。

    输出:
    - shape、sum、foreground_ratio。

    为什么这样做:
    - 如果 mask 极性又错了，foreground_ratio 会接近 1，正式训练前即可失败。
    """
    ratio = float(mask.float().mean().item())
    if ratio <= 0.0 or ratio > 0.5:
        raise RuntimeError(f"mask 前景比例异常: {ratio:.6f}")
    return {
        "shape": list(mask.shape),
        "sum": float(mask.sum().item()),
        "foreground_ratio": ratio,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(args.seed))
    device = select_device(args.device)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        img_size=int(model_cfg["img_size"]),
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(train_cfg["num_workers"]),
        hflip_p=float(aug_cfg.get("horizontal_flip", 0.5)),
        vflip_p=float(aug_cfg.get("vertical_flip", 0.5)),
    )
    if val_loader is None:
        raise RuntimeError("val_loader 为空，不能启动正式训练")

    model = build_model(cfg=cfg, sam_checkpoint=args.sam_checkpoint, device=device)
    optimizer = build_optimizer(model, cfg)

    train_batch = next(iter(train_loader))
    train_images = train_batch["image"].to(device, non_blocking=True)
    train_masks = train_batch["mask"].to(device, non_blocking=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_outputs = model(image=train_images, multimask=False)
    train_loss_dict = model.compute_loss(train_outputs, train_masks)
    train_loss = train_loss_dict["loss"]
    train_loss.backward()
    optimizer.step()
    train_iou = compute_batch_iou(select_logits(train_outputs), train_masks)

    val_batch = next(iter(val_loader))
    val_images = val_batch["image"].to(device, non_blocking=True)
    val_masks = val_batch["mask"].to(device, non_blocking=True)
    model.eval()
    with torch.no_grad():
        val_outputs = model(image=val_images, multimask=False)
        val_loss_dict = model.compute_loss(val_outputs, val_masks)
        val_iou = compute_batch_iou(select_logits(val_outputs), val_masks)

    payload = {
        "status": "ok",
        "device": str(device),
        "config": args.config,
        "sam_checkpoint": args.sam_checkpoint,
        "train_dataset_count": int(len(train_loader.dataset)),
        "val_dataset_count": int(len(val_loader.dataset)),
        "train_mask": mask_summary(train_masks.detach().cpu()),
        "val_mask": mask_summary(val_masks.detach().cpu()),
        "train_loss": float(train_loss.item()),
        "train_iou": float(train_iou),
        "val_loss": float(val_loss_dict["loss"].item()),
        "val_iou": float(val_iou),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[OK] M2 polarity-fix one-batch smoke passed")
    print(f"[INFO] train_loss={payload['train_loss']:.6f}, train_iou={payload['train_iou']:.6f}")
    print(f"[INFO] val_loss={payload['val_loss']:.6f}, val_iou={payload['val_iou']:.6f}")
    print(f"[INFO] report: {out_path}")


if __name__ == "__main__":
    main()

