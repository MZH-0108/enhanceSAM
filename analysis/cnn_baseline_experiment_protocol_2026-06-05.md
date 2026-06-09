# UNet / DeepLab 对比实验方案（2026-06-05）

## 目的
- 补充传统深度分割模型基线，与 `SAM-AMG`、`SAM-Box-Oracle`、`SAM+LoRA` 形成论文对比。
- 所有模型统一使用当前 polarity-fixed 数据口径：裂缝为 1、背景为 0。

## 已新增模型
- `UNet`：编码器-解码器 + 跳连结构，适合细粒度裂缝分割。
- `DeepLabV3-like`：轻量 CNN 编码器 + ASPP 空洞卷积模块，不依赖外部预训练权重。

## 统一实验口径
- 输入尺寸：`512 x 512`
- 数据目录：`data/train|val|test`
- 损失：`BCEWithLogits + Dice Loss`
- 正样本权重：`pos_weight=10`
- 指标：`mIoU`、`Dice`、`Precision`、`Recall`、`Boundary-IoU`、`FPS`
- Windows 稳定性环境变量：
  - `TORCHDYNAMO_DISABLE=1`
  - `NO_ALBUMENTATIONS_UPDATE=1`

## 正式运行命令
```powershell
.\scripts\run_cnn_baseline_experiments.ps1 -Device cuda
```

## 单模型运行命令
```powershell
$env:TORCHDYNAMO_DISABLE="1"
$env:NO_ALBUMENTATIONS_UPDATE="1"

.\venv\Scripts\python.exe scripts\train_cnn_baseline.py `
  --data_root data `
  --config configs\baselines\unet_polarityfix.yaml `
  --output_dir checkpoints\baselines\unet_polarityfix `
  --device cuda

.\venv\Scripts\python.exe scripts\eval_cnn_baseline.py `
  --data_root data `
  --split val `
  --config configs\baselines\unet_polarityfix.yaml `
  --checkpoint checkpoints\baselines\unet_polarityfix\best_model.pth `
  --output results\baselines\unet_polarityfix_val.json `
  --device cuda
```

## 预期输出
- UNet checkpoint：`checkpoints/baselines/unet_polarityfix/`
- DeepLabV3-like checkpoint：`checkpoints/baselines/deeplabv3_polarityfix/`
- UNet 指标：
  - `results/baselines/unet_polarityfix_val.json`
  - `results/baselines/unet_polarityfix_test.json`
- DeepLabV3-like 指标：
  - `results/baselines/deeplabv3_polarityfix_val.json`
  - `results/baselines/deeplabv3_polarityfix_test.json`
- 训练日志：`logs/baselines/`

## 论文表述注意
- 当前 `DeepLabV3-like` 为从零训练的轻量复现，不是 ImageNet/COCO 预训练的官方 DeepLabV3。
- 如果论文需要更强的 CNN 基线，可后续在网络允许时补充 `torchvision DeepLabV3-ResNet50` 或 `segmentation_models_pytorch` 预训练版本。
