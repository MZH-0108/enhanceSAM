# SAM Vanilla Baselines

本目录提供原始 SAM 的两种基线评估方式：

1. `eval_amg.py`  
   自动掩码模式（AMG），不使用 GT 提示，代表真实自动分割能力。

2. `eval_box_oracle.py`  
   外接框提示模式（Box Oracle），框由 GT mask 计算，代表 SAM 在理想提示下的上限能力。

---

## 1) AMG 基线（推荐先跑）

```bash
python baselines/sam_vanilla/eval_amg.py ^
  --data_root data ^
  --split val ^
  --model_type vit_b ^
  --sam_checkpoint checkpoints/pretrained/sam_vit_b_01ec64.pth ^
  --device cuda ^
  --output results/baselines/sam_amg_val.json
```

---

## 2) Box Oracle 基线

```bash
python baselines/sam_vanilla/eval_box_oracle.py ^
  --data_root data ^
  --split val ^
  --model_type vit_b ^
  --sam_checkpoint checkpoints/pretrained/sam_vit_b_01ec64.pth ^
  --device cuda ^
  --padding 8 ^
  --use_components ^
  --output results/baselines/sam_box_oracle_val.json
```

---

## 输出指标

- `mIoU`
- `Dice`
- `Precision`
- `Recall`
- `Boundary-IoU`
- `ms_per_image`
- `FPS`

---

## 论文建议

建议在表格中同时报告：

- `SAM-AMG`（真实自动模式）
- `SAM-Box-Oracle`（上限参考）
- `SAM + LoRA`（你的方法）

这样能清楚展示：原始 SAM 基线 -> 提示上限 -> 微调改进 的完整逻辑链。

