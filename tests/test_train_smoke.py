"""训练脚本 smoke test（1 个 epoch，CPU 可跑）。

这个测试的目标不是验证模型精度，
而是验证“训练主流程”是否完整可执行：
1) 能进入 train/val 循环；
2) 能完成优化器与调度器步进；
3) 能正常输出 checkpoint 与配置快照。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

import scripts.train as train_script


class _TinySegDataset(Dataset):
    """极小分割数据集（合成数据）。

    说明：
    - 返回字段与真实训练一致：`image` / `mask`；
    - 图像尺寸很小（64x64），减少测试耗时；
    - mask 用固定几何区域构造，保证 loss 非零且稳定。
    """

    def __init__(self, n: int = 4, h: int = 64, w: int = 64) -> None:
        self.n = int(n)
        self.h = int(h)
        self.w = int(w)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 构造一个简单图像：底噪 + 与索引相关的偏置，避免样本完全相同。
        image = torch.randn(3, self.h, self.w) * 0.1 + (idx % 3) * 0.05

        # 构造一个“裂缝样式”的细长前景区域（矩形条带）。
        mask = torch.zeros(1, self.h, self.w, dtype=torch.float32)
        x0 = 8 + (idx % 4) * 4
        mask[:, 10:54, x0 : x0 + 3] = 1.0
        return {"image": image.float(), "mask": mask}


@dataclass
class _DummyConfig:
    """用于替代真实 EnhancedSAMConfig，满足 asdict(model.config) 调用。"""

    use_lora: bool = True
    use_boundary: bool = False
    note: str = "smoke_test_dummy_config"


class _TinyTrainModel(nn.Module):
    """极简可训练模型，接口对齐 train.py 期望。

    关键对齐点：
    - `forward(image, multimask=False)` 返回 `masks` 与 `iou_pred`；
    - `compute_loss(outputs, targets)` 返回包含 `loss` 的字典；
    - `param_report()` 返回可打印字符串。
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )
        self.config = _DummyConfig()

    def forward(self, image: torch.Tensor, multimask: bool = False) -> Dict[str, torch.Tensor]:
        logits = self.encoder(image)          # [B,1,H,W]
        iou_pred = torch.zeros(image.size(0), 1, device=image.device)
        return {"masks": logits, "iou_pred": iou_pred}

    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = outputs["masks"]
        if targets.shape[-2:] != logits.shape[-2:]:
            targets = F.interpolate(targets.float(), size=logits.shape[-2:], mode="nearest")
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return {"loss": bce, "bce": bce, "dice": bce, "boundary": torch.tensor(0.0, device=logits.device)}

    def param_report(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Dummy Parameter Report: total={total}, trainable={trainable}"


def _make_tiny_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """构建极小 train/val loader，保证测试快且稳定。"""
    train_ds = _TinySegDataset(n=4)
    val_ds = _TinySegDataset(n=2)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    return train_loader, val_loader


def test_train_main_smoke_one_epoch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """smoke test：验证 train.main() 能跑完 1 个 epoch 并落盘关键产物。"""

    # 1) 输出目录放到 pytest 临时目录，避免污染真实项目目录。
    output_dir = tmp_path / "ckpt_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2) 构造 main() 所需参数（通过 monkeypatch 替换 parse_args）。
    args = SimpleNamespace(
        data_root="ignored_in_smoke",
        sam_checkpoint="ignored_in_smoke",
        config="ignored_in_smoke",
        output_dir=str(output_dir),
        device="cpu",
        seed=123,
        resume="",
    )

    # 3) 构造最小配置，只保留 main/optimizer/scheduler 真正会访问的字段。
    cfg = {
        "model": {"img_size": 64},
        "training": {"batch_size": 2, "num_workers": 0, "epochs": 1, "lr": 1e-3, "weight_decay": 0.0},
        "augmentation": {"horizontal_flip": 0.0, "vertical_flip": 0.0},
        "checkpoint": {"monitor": "val_iou", "save_interval": 1},
        "logging": {"log_interval": 1},
        "early_stopping": {"enabled": False, "patience": 3, "min_delta": 1e-4},
        "optimizer": {"type": "adamw", "betas": [0.9, 0.999], "eps": 1e-8},
        "scheduler": {"type": "cosine", "min_lr": 1e-6},
    }

    # 4) 用轻量替身替换重路径（SAM加载/真实数据读取），聚焦流程完整性。
    monkeypatch.setattr(train_script, "parse_args", lambda: args)
    monkeypatch.setattr(train_script, "load_config", lambda _: cfg)
    monkeypatch.setattr(train_script, "build_dataloaders", lambda **_: _make_tiny_dataloaders())
    monkeypatch.setattr(train_script, "build_model", lambda **_: _TinyTrainModel().to(torch.device("cpu")))

    # 5) 执行主流程。
    train_script.main()

    # 6) 断言关键输出文件存在：last_model + best_model + run_config 快照。
    assert (output_dir / "last_model.pth").exists()
    assert (output_dir / "best_model.pth").exists()
    run_cfg_files = list(output_dir.glob("run_config_*.json"))
    assert len(run_cfg_files) == 1
