"""顺序运行 UNet / DeepLabV3-like 基线训练与评估。

这个 Python 版 runner 用于替代 PowerShell 版一键脚本，原因是当前 Windows
环境里的 `Start-Process -RedirectStandardOutput` 会触发 `Path/PATH` 环境变量
冲突。Python `subprocess.run` 的日志重定向更稳定，适合后台长任务。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析 runner 参数。"""
    parser = argparse.ArgumentParser(description="Run CNN baseline experiments sequentially")
    parser.add_argument("--device", default="cuda", help="训练/评估设备")
    parser.add_argument("--data_root", default="data", help="数据集根目录")
    parser.add_argument("--stamp", default="", help="日志时间戳；留空自动生成")
    return parser.parse_args()


def build_commands(python_exe: Path, device: str, data_root: str) -> Dict[str, List[str]]:
    """构建需要顺序执行的命令。

    输入:
    - python_exe: 当前虚拟环境 Python；
    - device/data_root: 训练设备和数据根目录。

    输出:
    - 命令名称到命令参数列表的映射。

    为什么这样做:
    - 所有命令集中定义，便于记录和复现；
    - 训练和评估严格顺序执行，避免两个模型同时抢 GPU。
    """
    return {
        "unet_train": [
            str(python_exe),
            "scripts/train_cnn_baseline.py",
            "--data_root",
            data_root,
            "--config",
            "configs/baselines/unet_polarityfix.yaml",
            "--output_dir",
            "checkpoints/baselines/unet_polarityfix",
            "--device",
            device,
        ],
        "unet_val": [
            str(python_exe),
            "scripts/eval_cnn_baseline.py",
            "--data_root",
            data_root,
            "--split",
            "val",
            "--config",
            "configs/baselines/unet_polarityfix.yaml",
            "--checkpoint",
            "checkpoints/baselines/unet_polarityfix/best_model.pth",
            "--output",
            "results/baselines/unet_polarityfix_val.json",
            "--device",
            device,
        ],
        "unet_test": [
            str(python_exe),
            "scripts/eval_cnn_baseline.py",
            "--data_root",
            data_root,
            "--split",
            "test",
            "--config",
            "configs/baselines/unet_polarityfix.yaml",
            "--checkpoint",
            "checkpoints/baselines/unet_polarityfix/best_model.pth",
            "--output",
            "results/baselines/unet_polarityfix_test.json",
            "--device",
            device,
        ],
        "deeplabv3_like_train": [
            str(python_exe),
            "scripts/train_cnn_baseline.py",
            "--data_root",
            data_root,
            "--config",
            "configs/baselines/deeplabv3_polarityfix.yaml",
            "--output_dir",
            "checkpoints/baselines/deeplabv3_polarityfix",
            "--device",
            device,
        ],
        "deeplabv3_like_val": [
            str(python_exe),
            "scripts/eval_cnn_baseline.py",
            "--data_root",
            data_root,
            "--split",
            "val",
            "--config",
            "configs/baselines/deeplabv3_polarityfix.yaml",
            "--checkpoint",
            "checkpoints/baselines/deeplabv3_polarityfix/best_model.pth",
            "--output",
            "results/baselines/deeplabv3_polarityfix_val.json",
            "--device",
            device,
        ],
        "deeplabv3_like_test": [
            str(python_exe),
            "scripts/eval_cnn_baseline.py",
            "--data_root",
            data_root,
            "--split",
            "test",
            "--config",
            "configs/baselines/deeplabv3_polarityfix.yaml",
            "--checkpoint",
            "checkpoints/baselines/deeplabv3_polarityfix/best_model.pth",
            "--output",
            "results/baselines/deeplabv3_polarityfix_test.json",
            "--device",
            device,
        ],
    }


def main() -> int:
    """执行完整 CNN baseline 实验流水线。"""
    args = parse_args()
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    python_exe = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    log_dir = PROJECT_ROOT / "logs" / "baselines"
    result_dir = PROJECT_ROOT / "results" / "baselines"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "baselines"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"

    status_path = log_dir / f"cnn_baseline_runner_{stamp}.status.txt"
    exit_path = log_dir / f"cnn_baseline_runner_{stamp}.exit.txt"
    command_path = log_dir / f"cnn_baseline_runner_{stamp}.commands.json"
    commands = build_commands(python_exe, args.device, args.data_root)
    command_path.write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")

    with status_path.open("w", encoding="utf-8") as status:
        status.write(f"START {datetime.now().isoformat(timespec='seconds')}\n")
        status.write(f"DEVICE {args.device}\n")
        status.write(f"DATA_ROOT {args.data_root}\n")
        status.flush()

        for name, command in commands.items():
            stdout_path = log_dir / f"{name}_{stamp}.out.log"
            stderr_path = log_dir / f"{name}_{stamp}.err.log"
            status.write(f"{name.upper()} START {datetime.now().isoformat(timespec='seconds')}\n")
            status.write(f"{name.upper()} STDOUT {stdout_path}\n")
            status.write(f"{name.upper()} STDERR {stderr_path}\n")
            status.flush()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                )
            status.write(f"{name.upper()} EXIT {completed.returncode} {datetime.now().isoformat(timespec='seconds')}\n")
            status.flush()
            if completed.returncode != 0:
                exit_path.write_text(str(completed.returncode), encoding="ascii")
                status.write("FAILED\n")
                return int(completed.returncode)

        status.write(f"SUCCESS {datetime.now().isoformat(timespec='seconds')}\n")
    exit_path.write_text("0", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
