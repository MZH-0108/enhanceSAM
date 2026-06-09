<#
运行 UNet 与 DeepLabV3-like 对比实验。

用途：
- 统一设置 Windows + torch 稳定性环境变量；
- 按顺序训练 UNet、DeepLabV3-like；
- 分别在 val/test 上评估，并把日志和 JSON 指标落盘。

说明：
- 本脚本会执行较长时间训练，建议在确认 GPU 空闲后运行；
- 输出目录位于 checkpoints/baselines、results/baselines、logs/baselines；
- 不会自动 git add / commit / push。
#>

param(
    [string]$Python = ".\venv\Scripts\python.exe",
    [string]$Device = "cuda",
    [string]$DataRoot = "data"
)

$ErrorActionPreference = "Stop"
$env:TORCHDYNAMO_DISABLE = "1"
$env:NO_ALBUMENTATIONS_UPDATE = "1"

New-Item -ItemType Directory -Force -Path "logs\baselines" | Out-Null
New-Item -ItemType Directory -Force -Path "results\baselines" | Out-Null
New-Item -ItemType Directory -Force -Path "checkpoints\baselines" | Out-Null

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = "logs\baselines\${Name}_${timestamp}.out.log"
    $stderr = "logs\baselines\${Name}_${timestamp}.err.log"
    Write-Host "[INFO] Start $Name"
    Write-Host "[INFO] stdout: $stdout"
    Write-Host "[INFO] stderr: $stderr"

    & $Python @CommandArgs 1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Name, exit=$LASTEXITCODE"
    }
}

Invoke-LoggedCommand -Name "unet_train" -CommandArgs @(
    "scripts\train_cnn_baseline.py",
    "--data_root", $DataRoot,
    "--config", "configs\baselines\unet_polarityfix.yaml",
    "--output_dir", "checkpoints\baselines\unet_polarityfix",
    "--device", $Device
)

Invoke-LoggedCommand -Name "unet_val" -CommandArgs @(
    "scripts\eval_cnn_baseline.py",
    "--data_root", $DataRoot,
    "--split", "val",
    "--config", "configs\baselines\unet_polarityfix.yaml",
    "--checkpoint", "checkpoints\baselines\unet_polarityfix\best_model.pth",
    "--output", "results\baselines\unet_polarityfix_val.json",
    "--device", $Device
)

Invoke-LoggedCommand -Name "unet_test" -CommandArgs @(
    "scripts\eval_cnn_baseline.py",
    "--data_root", $DataRoot,
    "--split", "test",
    "--config", "configs\baselines\unet_polarityfix.yaml",
    "--checkpoint", "checkpoints\baselines\unet_polarityfix\best_model.pth",
    "--output", "results\baselines\unet_polarityfix_test.json",
    "--device", $Device
)

Invoke-LoggedCommand -Name "deeplabv3_like_train" -CommandArgs @(
    "scripts\train_cnn_baseline.py",
    "--data_root", $DataRoot,
    "--config", "configs\baselines\deeplabv3_polarityfix.yaml",
    "--output_dir", "checkpoints\baselines\deeplabv3_polarityfix",
    "--device", $Device
)

Invoke-LoggedCommand -Name "deeplabv3_like_val" -CommandArgs @(
    "scripts\eval_cnn_baseline.py",
    "--data_root", $DataRoot,
    "--split", "val",
    "--config", "configs\baselines\deeplabv3_polarityfix.yaml",
    "--checkpoint", "checkpoints\baselines\deeplabv3_polarityfix\best_model.pth",
    "--output", "results\baselines\deeplabv3_polarityfix_val.json",
    "--device", $Device
)

Invoke-LoggedCommand -Name "deeplabv3_like_test" -CommandArgs @(
    "scripts\eval_cnn_baseline.py",
    "--data_root", $DataRoot,
    "--split", "test",
    "--config", "configs\baselines\deeplabv3_polarityfix.yaml",
    "--checkpoint", "checkpoints\baselines\deeplabv3_polarityfix\best_model.pth",
    "--output", "results\baselines\deeplabv3_polarityfix_test.json",
    "--device", $Device
)

Write-Host "[INFO] CNN baseline experiments completed."
