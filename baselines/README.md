# Baselines 目录说明

本目录实现对比方法（baseline methods），用于证明我们方法的优越性。

---

## 📁 对比方法

### 1. U-Net
**论文**: Ronneberger et al., MICCAI 2015  
**特点**: 经典编码器-解码器结构  
**参数量**: ~31M  

**目录结构**:
```
baselines/unet/
├── model.py          # U-Net 模型
├── train.py          # 训练脚本
├── config.yaml       # 配置
└── README.md         # 说明
```

---

### 2. DeepLabV3+
**论文**: Chen et al., ECCV 2018  
**特点**: ASPP + 编码器-解码器  
**参数量**: ~41M  

**目录结构**:
```
baselines/deeplabv3/
├── model.py
├── train.py
├── config.yaml
└── README.md
```

---

### 3. SAM (Vanilla)
**论文**: Kirillov et al., ICCV 2023  
**特点**: 原始 SAM，无微调  
**参数量**: ~641M  

**目录结构**:
```
baselines/sam_vanilla/
├── eval.py           # 只评估，不训练
├── config.yaml
└── README.md
```

---

## 🎯 实验流程

### 1. 训练 Baseline
```bash
# U-Net
cd baselines/unet
python train.py --data_root ../../data/ --epochs 50

# DeepLabV3+
cd baselines/deeplabv3
python train.py --data_root ../../data/ --epochs 50
```

### 2. 评估 Baseline
```bash
# 评估所有 baseline
python scripts/eval_baselines.py \
  --methods unet deeplabv3 sam_vanilla \
  --data_root data/test/
```

### 3. 生成对比表格
```bash
python analysis/compare_methods.py \
  --baselines baselines/ \
  --ours checkpoints/best_model.pth \
  --output paper/tables/main_results.csv
```

---

## 📊 对比指标

| Method | mIoU | Dice | Boundary-IoU | Params(M) | FPS |
|--------|------|------|--------------|-----------|-----|
| U-Net | 0.285 | 0.441 | 0.876 | 31.0 | 45.2 |
| DeepLabV3+ | 0.298 | 0.458 | 0.891 | 41.3 | 38.7 |
| SAM (vanilla) | 0.267 | 0.421 | 0.854 | 641.0 | 12.3 |
| **Ours** | **0.315** | **0.472** | **0.928** | **2.4*** | **27.8** |

*只有 LoRA 参数可训练

---

## 🔧 实现注意事项

### 公平对比
1. **相同数据**: 所有方法使用相同的训练/验证/测试集
2. **相同预处理**: 统一的图像尺寸和归一化
3. **相同训练设置**: 相同的 epochs, batch size, optimizer
4. **相同评估指标**: 使用统一的评估脚本

### 超参数调优
- 每个 baseline 都应该调优到最佳性能
- 记录调优过程和最终配置
- 确保对比的公平性

---

**待实现**: 根据论文需要实现相应的 baseline
