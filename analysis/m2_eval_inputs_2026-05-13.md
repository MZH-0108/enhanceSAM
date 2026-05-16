# M2 val 对比评估输入确认（2026-05-13）

## 1. 环境状态
- `venv` 中 PyTorch 已是 CUDA 版本：`2.11.0+cu128`。
- `torch.version.cuda`: `12.8`。
- `torch.cuda.is_available()`: `True`。
- 本机 GPU：`NVIDIA GeForce RTX 5080`，显存约 `16GB`。

## 2. 数据与权重
- 数据根目录：`data/`。
- 训练集图像数：`11059`。
- 验证集图像数：`1591`。
- 测试集图像数：`1594`。
- 官方 SAM 权重：`checkpoints/sam_vit_b_01ec64.pth`。

## 3. 三组评估输入
| 方法 | 入口 | 权重依赖 | 输出路径 | 状态 |
|---|---|---|---|---|
| SAM-AMG | `baselines/sam_vanilla/eval_amg.py` | 官方 SAM 权重 | `results/baselines/sam_amg_val_gpu.json` | 已具备，已有结果 |
| SAM-Box-Oracle | `baselines/sam_vanilla/eval_box_oracle.py` | 官方 SAM 权重 | `results/baselines/sam_box_oracle_val_gpu.json` | 已具备，已有结果 |
| SAM+LoRA | `scripts/eval.py` | 官方 SAM 权重 + 训练 checkpoint | `results/enhanced/sam_lora_val_gpu.json` | 阻塞：缺少 `checkpoints/m2_lora/best_model.pth` |

## 4. 已有 baseline 指标
| 方法 | mIoU | Dice | Precision | Recall | Boundary-IoU | FPS |
|---|---:|---:|---:|---:|---:|---:|
| SAM-AMG | 0.012200 | 0.024107 | 0.480311 | 0.012364 | 0.017539 | 0.317026 |
| SAM-Box-Oracle | 0.890213 | 0.941918 | 0.988996 | 0.899119 | 0.132016 | 8.909702 |

## 5. 当前结论
M2 统一评估配置已固化到 `configs/m2_eval_config.yaml`。当前无法完成三组完整 val 对比，因为 `SAM+LoRA` 缺少训练得到的 checkpoint。下一步应在 M2 范围内先启动 LoRA 主线训练，产出 `checkpoints/m2_lora/best_model.pth` 后再执行 `scripts/eval.py`。
