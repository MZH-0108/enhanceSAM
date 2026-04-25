# Experiments 目录说明

本目录用于系统化管理所有实验，每个实验一个子目录。

---

## 📁 目录结构

```
experiments/
├── exp001_baseline/          # 实验 1: SAM 基线
├── exp002_lora/              # 实验 2: SAM + LoRA
├── exp003_lora_boundary/     # 实验 3: SAM + LoRA + Boundary
└── exp004_ablation/          # 实验 4: 消融实验
```

---

## 🎯 实验命名规范

```
exp{编号}_{简短描述}/
```

**示例**:
- `exp001_baseline` - SAM 基线实验
- `exp002_lora_r8` - LoRA rank=8
- `exp003_lora_r16` - LoRA rank=16
- `exp004_full_model` - 完整模型

---

## 📋 每个实验目录包含

```
exp001_baseline/
├── config.yaml              # 实验配置
├── results.json             # 实验结果
├── logs/                    # 训练日志
│   ├── train.log
│   └── tensorboard/
├── checkpoints/             # 模型检查点
│   ├── best_model.pth
│   └── epoch_*.pth
└── README.md                # 实验说明
```

---

## 📝 实验记录模板

每个实验的 `README.md` 应包含：

```markdown
# Experiment 001: SAM Baseline

## 目标
验证 SAM 在隧道裂缝数据集上的基线性能

## 配置
- Model: SAM ViT-B
- Image Size: 512x512
- Batch Size: 4
- Epochs: 50
- Learning Rate: 1e-4

## 结果
- mIoU: 0.267
- Dice: 0.421
- Boundary-IoU: 0.854

## 观察
- SAM 在裂缝检测上表现一般
- 边界质量较差
- 需要微调

## 下一步
- 尝试 LoRA 微调
```

---

## 🔬 实验管理最佳实践

### 1. 实验前
- 创建新的实验目录
- 复制并修改 config.yaml
- 编写实验目标

### 2. 实验中
- 记录训练日志
- 保存检查点
- 监控 TensorBoard

### 3. 实验后
- 记录结果到 results.json
- 分析结果，写 README
- 对比之前的实验

---

## 📊 结果对比

使用脚本自动生成对比表格：

```bash
python analysis/compare_experiments.py \
  --exp1 exp001_baseline \
  --exp2 exp002_lora \
  --exp3 exp003_lora_boundary
```

---

**维护**: 每次实验后更新此文档
