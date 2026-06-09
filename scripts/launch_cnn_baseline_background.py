"""后台启动 CNN baseline 实验 runner。

该脚本只负责“点火”：启动后立即返回，真正的长时间训练由
`scripts/run_cnn_baseline_experiments.py` 在独立后台进程中顺序执行。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析后台启动参数。"""
    parser = argparse.ArgumentParser(description="Launch CNN baseline runner in background")
    parser.add_argument("--device", default="cuda", help="训练/评估设备")
    parser.add_argument("--data_root", default="data", help="数据集根目录")
    return parser.parse_args()


def main() -> None:
    """创建 detached 后台进程并记录 PID。"""
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / "baselines"
    log_dir.mkdir(parents=True, exist_ok=True)

    python_exe = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    runner = PROJECT_ROOT / "scripts" / "run_cnn_baseline_experiments.py"
    launcher_stdout = log_dir / f"cnn_baseline_background_{stamp}.out.log"
    launcher_stderr = log_dir / f"cnn_baseline_background_{stamp}.err.log"

    command = [
        str(python_exe),
        str(runner),
        "--device",
        args.device,
        "--data_root",
        args.data_root,
        "--stamp",
        stamp,
    ]

    creationflags = 0
    if sys.platform.startswith("win"):
        # 不使用 DETACHED_PROCESS：当前 Codex 桌面沙箱下该标志可能导致子进程
        # 静默退出且不写入任何日志。CREATE_NO_WINDOW 可以隐藏窗口，
        # CREATE_NEW_PROCESS_GROUP 则让训练进程脱离当前控制组，父进程退出后仍可继续。
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

    with launcher_stdout.open("w", encoding="utf-8") as stdout, launcher_stderr.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )

    record = {
        "launch_time": datetime.now().isoformat(timespec="seconds"),
        "runner_pid": process.pid,
        "stamp": stamp,
        "command": command,
        "launcher_stdout": str(launcher_stdout),
        "launcher_stderr": str(launcher_stderr),
        "runner_status": str(log_dir / f"cnn_baseline_runner_{stamp}.status.txt"),
        "runner_exit": str(log_dir / f"cnn_baseline_runner_{stamp}.exit.txt"),
        "runner_commands": str(log_dir / f"cnn_baseline_runner_{stamp}.commands.json"),
        "checkpoints": str(PROJECT_ROOT / "checkpoints" / "baselines"),
        "results": str(PROJECT_ROOT / "results" / "baselines"),
    }
    record_path = log_dir / f"cnn_baseline_background_{stamp}.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (log_dir / f"cnn_baseline_background_{stamp}.pid.txt").write_text(str(process.pid), encoding="ascii")

    print(f"RUNNER_PID={process.pid}")
    print(f"STAMP={stamp}")
    print(f"RECORD={record_path}")
    print(f"STATUS={record['runner_status']}")


if __name__ == "__main__":
    main()
