<div align="center">

# Enhanced SAM 隧道裂缝智能分割系统

**基于 LoRA 微调与边界精细化的参数高效分割方法**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/pytorch-2.6+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-MZH--0108%2FenhanceSAM-black?logo=github)](https://github.com/MZH-0108/enhanceSAM)

[简体中文](#中文文档) · [English](#english-documentation)

</div>

---

# 中文文档

## 📖 项目简介

本项目基于 Meta 提出的 **Segment Anything Model (SAM)**，针对**隧道裂缝分割**任务进行参数高效微调，结合 **LoRA (Low-Rank Adaptation)** 与**边界精细化模块**，在仅训练 **1.69%** 参数的情况下实现高质量的裂缝分割。

### ✨ 核心特性

- 🚀 **推理高效**: GPU 上单张图像推理仅需 ~36ms
- 💾 **参数高效**: 仅 2.4M 可训练参数（占总参数 1.69%）

### 🎓 研究目标

核心创新点：
1. **LoRA + SAM**: 将参数高效微调引入 SAM 的裂缝分割任务
2. **边界精细化**: 基于 Sobel 算子的迭代式边界优化
3. **端到端训练**: 无需提示，全自动裂缝识别

---

## 📊 性能指标

| 评估指标 | 数值 |
|---------|------|
| mIoU (平均交并比) | 0.3147 |
| Dice 系数 | 0.4721 |
| Boundary-IoU (边界交并比) | 0.9283 |
| 推理速度 | 35.88 ms/张 |
| 可训练参数 | 2.4M / 143.7M (1.69%) |

---

## 🛠️ 安装指南

### 环境要求
- Python 3.8 或更高版本
- CUDA 11.8 或 12.1（用于 GPU 训练）
- Git

### 步骤 1: 克隆仓库
```bash
git clone git@github.com:MZH-0108/enhanceSAM.git
cd enhanceSAM
```

### 步骤 2: 创建虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 步骤 3: 安装依赖
```bash
# CUDA 12.1 版本
pip install torch==2.6.0+cu121 torchvision==0.21.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 步骤 4: 下载 SAM 预训练权重
```bash
python scripts/download_models.py
```

---

## 🚀 快速开始

### 训练模型
```bash
python scripts/train.py \
  --data_root data/ \
  --sam_checkpoint sam_vit_b_01ec64.pth \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --output_dir checkpoints/
```

### 评估模型
```bash
python scripts/eval.py \
  --checkpoint checkpoints/best_model.pth \
  --sam_checkpoint sam_vit_b_01ec64.pth \
  --split val \
  --output results/eval_val.json
```

### 单图推理
```bash
python scripts/predict.py \
  --image path/to/image.jpg \
  --checkpoint checkpoints/best_model.pth \
  --sam_checkpoint sam_vit_b_01ec64.pth \
  --output output.png
```

---

## 📁 项目结构

```
enhanceSAM/
├── models/                    # 模型实现
│   ├── sam_base.py            # SAM 基础封装
│   ├── lora_adapter.py        # LoRA 适配器
│   ├── boundary_refinement.py # 边界精细化
│   └── enhanced_sam.py        # 集成模型
├── utils/                     # 工具函数
├── scripts/                   # 训练/评估脚本
├── tests/                     # 单元测试
├── configs/                   # 配置文件
├── experiments/               # 实验管理
├── paper/                     # 论文素材
│   ├── figures/               # 论文图表
│   ├── tables/                # 实验表格
│   └── references/            # 参考文献
├── analysis/                  # 深入分析
├── baselines/                 # 对比方法
├── notebooks/                 # Jupyter 笔记本
├── data/                      # 数据集（不提交）
├── checkpoints/               # 模型权重（不提交）
└── logs/                      # 训练日志（不提交）
```

---

## 📂 数据集格式

```
data/
├── train/
│   ├── images/       # RGB 图像 (*.jpg, *.png)
│   └── annotations/  # 二值掩码 (*.png, 0=背景, 255=裂缝)
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

---

## 🏗️ 模型架构

```
输入图像 (B, 3, H, W)
    ↓
SAM 图像编码器 (ViT-B) + LoRA
    ↓ (B, 256, H/16, W/16)
    ├─→ SAM 掩码解码器 + LoRA → 粗掩码
    └─→ 边界检测器 → 边界图
            ↓
    边界精细化网络（3 次迭代）
            ↓
    精细化掩码 (B, 1, H, W)
```

---

## 📚 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{your_paper_2026,
  title={Enhanced SAM for Tunnel Crack Segmentation: A Parameter-Efficient Approach with Boundary Refinement},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

**SAM 引用**:
```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and others},
  journal={arXiv:2304.02643},
  year={2023}
}
```

**LoRA 引用**:
```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}
```

---

## 📜 许可证

本项目采用 Apache 2.0 协议开源 - 详见 [LICENSE](LICENSE)

---

## 🙏 致谢

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Meta AI
- [LoRA](https://github.com/microsoft/LoRA) - Microsoft Research
- PyTorch 团队

---

<br>

# English Documentation

## 📖 Overview

This project is based on Meta's **Segment Anything Model (SAM)**, performing parameter-efficient fine-tuning for **tunnel crack segmentation**. By combining **LoRA (Low-Rank Adaptation)** with a **boundary refinement module**, we achieve high-quality crack segmentation while training only **1.69%** of parameters.

### ✨ Key Features

- 🚀 **Fast Inference**: ~36ms per image on GPU
- 💾 **Parameter Efficient**: Only 2.4M trainable parameters (1.69%)

### 🎓 Research Goals

The following innovations:
1. **LoRA + SAM**: Apply parameter-efficient fine-tuning to SAM for crack segmentation
2. **Boundary Refinement**: Iterative boundary optimization with Sobel-initialized convolutions
3. **End-to-end Training**: Fully automatic crack detection without prompts

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| mIoU | 0.3147 |
| Dice | 0.4721 |
| Boundary-IoU | 0.9283 |
| Inference Speed | 35.88 ms/image |
| Trainable Params | 2.4M / 143.7M (1.69%) |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8 or 12.1 (for GPU training)
- Git

### Step 1: Clone Repository
```bash
git clone git@github.com:MZH-0108/enhanceSAM.git
cd enhanceSAM
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows
```

### Step 3: Install Dependencies
```bash
# For CUDA 12.1
pip install torch==2.6.0+cu121 torchvision==0.21.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Step 4: Download SAM Pretrained Weights
```bash
python scripts/download_models.py
```

---

## 🚀 Quick Start

### Training
```bash
python scripts/train.py \
  --data_root data/ \
  --sam_checkpoint sam_vit_b_01ec64.pth \
  --batch_size 4 --epochs 50 --lr 1e-4
```

### Evaluation
```bash
python scripts/eval.py \
  --checkpoint checkpoints/best_model.pth \
  --sam_checkpoint sam_vit_b_01ec64.pth \
  --split val
```

### Single Image Inference
```bash
python scripts/predict.py \
  --image path/to/image.jpg \
  --checkpoint checkpoints/best_model.pth
```

---

## 🏗️ Model Architecture

```
Input Image (B, 3, H, W)
    ↓
SAM Image Encoder (ViT-B) + LoRA
    ↓ (B, 256, H/16, W/16)
    ├─→ SAM Mask Decoder + LoRA → Coarse Mask
    └─→ Boundary Detector → Boundary Map
            ↓
    Boundary Refine Net (3 iterations)
            ↓
    Refined Mask (B, 1, H, W)
```

---

## 📜 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [LoRA](https://github.com/microsoft/LoRA) by Microsoft Research
- PyTorch Team

---

<div align="center">

**Status**: 🚧 Under Active Development  
**Last Updated**: 2026-04-27  
**Maintainer**: [@MZH-0108](https://github.com/MZH-0108)

</div>
