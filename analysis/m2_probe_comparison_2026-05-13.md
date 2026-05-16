# M2 val 对比结果（LoRA probe，2026-05-13）

## 说明

- 本表用于验证 M2 评估链路已经跑通。
- `SAM+LoRA-probe` 仅训练 1 个 epoch，checkpoint 为 `checkpoints/m2_lora_probe/best_model.pth`。
- 该结果不能替代正式 50 epoch LoRA 训练结果；正式论文表格应等待 `checkpoints/m2_lora/best_model.pth` 产出后重跑。

## 数据与设置

- 数据集划分：`val`
- 样本数：1591
- SAM 权重：`checkpoints/sam_vit_b_01ec64.pth`
- LoRA probe 配置：`configs/m2_lora_probe_config.yaml`
- LoRA probe 指标 JSON：`results/enhanced/sam_lora_probe_val_gpu.json`

## 指标对比

| 方法 | mIoU | Dice | Precision | Recall | Boundary-IoU | FPS |
|---|---:|---:|---:|---:|---:|---:|
| SAM-AMG | 0.012200 | 0.024107 | 0.480311 | 0.012364 | 0.017539 | 0.317026 |
| SAM-Box-Oracle | 0.890213 | 0.941918 | 0.988996 | 0.899119 | 0.132016 | 8.909702 |
| SAM+LoRA-probe (1 epoch) | 0.986607 | 0.993258 | 0.989088 | 0.997464 | 0.490773 | 53.844907 |

## 初步结论

- 原始 `SAM-AMG` 自动分割几乎无法有效覆盖裂缝，召回率仅 `0.012364`。
- `SAM-Box-Oracle` 在理想框提示下 mIoU 较高，但依赖 GT 框，不代表真实自动部署能力。
- `SAM+LoRA-probe` 已证明训练、评估、落盘链路可用；正式结论仍需使用完整训练 checkpoint 重跑。
