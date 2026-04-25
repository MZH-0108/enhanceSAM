# Analysis 目录说明

本目录包含深入分析脚本，用于理解模型行为、数据特性和错误模式。

---

## 📁 分析脚本

### 1. dataset_statistics.py
**功能**: 数据集统计分析

**输出**:
- 图像尺寸分布
- 裂缝像素占比
- 裂缝宽度分布
- 裂缝长度分布
- 数据集平衡性

**使用**:
```bash
python analysis/dataset_statistics.py --data_root data/
```

---

### 2. error_analysis.py
**功能**: 错误分析

**分析内容**:
- 假阳性（误检）分析
- 假阴性（漏检）分析
- 困难样本识别
- 错误类型分类

**使用**:
```bash
python analysis/error_analysis.py \
  --checkpoint checkpoints/best_model.pth \
  --data_root data/test/
```

---

### 3. attention_visualization.py
**功能**: 注意力可视化

**可视化内容**:
- SAM Image Encoder 的注意力图
- LoRA 学到的特征
- 边界检测器的激活图

**使用**:
```bash
python analysis/attention_visualization.py \
  --checkpoint checkpoints/best_model.pth \
  --image data/test/images/sample.jpg
```

---

### 4. lora_rank_analysis.py
**功能**: LoRA rank 分析

**分析内容**:
- 不同 rank (4, 8, 16, 32) 的性能
- 参数量 vs 性能权衡
- LoRA 矩阵的秩分析

**使用**:
```bash
python analysis/lora_rank_analysis.py \
  --ranks 4 8 16 32 \
  --data_root data/val/
```

---

## 📊 分析结果

所有分析结果保存在：
```
analysis/results/
├── dataset_stats.json
├── error_analysis.json
├── attention_maps/
└── lora_rank_comparison.csv
```

---

## 🎯 用途

### 论文撰写
- 数据集统计 → Introduction/Experiments
- 错误分析 → Discussion
- 注意力可视化 → Qualitative Results
- LoRA 分析 → Ablation Study

### 模型改进
- 识别困难样本 → 数据增强策略
- 分析错误模式 → 模型改进方向
- LoRA rank 选择 → 最优配置

---

**待实现**: 根据需要添加更多分析脚本
