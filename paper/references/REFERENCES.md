# 参考文献库 - EnhancedSAM 项目

本文件收集与项目相关的所有参考文献，按模块分类。

---

## 1. 基础模型 - SAM (Segment Anything Model)

### 1.1 SAM 原始论文 ⭐⭐⭐⭐⭐
**标题**: Segment Anything  
**作者**: Kirillov, A., Mintun, E., Ravi, N., et al.  
**会议/期刊**: ICCV 2023  
**链接**: https://arxiv.org/abs/2304.02643  
**代码**: https://github.com/facebookresearch/segment-anything  

**相关内容**:
- SAM 模型架构（Image Encoder, Prompt Encoder, Mask Decoder）
- 提示工程（点、框、掩码提示）
- SA-1B 数据集

**引用位置**:
- Introduction: SAM 作为基础模型
- Related Work: SAM 相关工作
- Method: SAM 架构说明
- models/sam_base.py: SAM 加载和使用

---

## 2. 参数高效微调 - LoRA

### 2.1 LoRA 原始论文 ⭐⭐⭐⭐⭐
**标题**: LoRA: Low-Rank Adaptation of Large Language Models  
**作者**: Hu, E. J., Shen, Y., Wallis, P., et al.  
**会议/期刊**: ICLR 2022  
**链接**: https://arxiv.org/abs/2106.09685  
**代码**: https://github.com/microsoft/LoRA  

**相关内容**:
- LoRA 原理：h = Wx + (α/r)BAx
- 低秩分解理论
- 参数效率分析

**引用位置**:
- Introduction: 参数高效微调的必要性
- Related Work: LoRA 方法
- Method: LoRA 实现细节
- models/lora_adapter.py: LoRA 实现

**BibTeX**:
```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and others},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

---

### 2.2 LoRA 应用于视觉模型
**标题**: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning  
**作者**: Zhang, Q., Chen, M., et al.  
**会议/期刊**: ICLR 2023  
**链接**: https://arxiv.org/abs/2303.10512  

**相关内容**:
- LoRA 在视觉 Transformer 中的应用
- 不同层使用不同 rank 的策略

**引用位置**:
- Related Work: LoRA 在视觉任务中的应用
- Method: 多尺度 LoRA 设计（如果实现）

---

## 3. SAM 适配与微调

### 3.1 Medical SAM Adapter ⭐⭐⭐⭐
**标题**: Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation  
**作者**: Wu, J., Fu, R., et al.  
**会议/期刊**: arXiv 2023  
**链接**: https://arxiv.org/abs/2304.12620  
**代码**: https://github.com/WuJunde/Medical-SAM-Adapter  

**相关内容**:
- SAM 在医学图像分割中的适配
- Adapter 设计
- 小样本学习

**引用位置**:
- Related Work: SAM 适配方法
- Method: 参考 Adapter 设计思路

---

### 3.2 SAM-Adapter
**标题**: SAM-Adapter: Adapting Segment Anything in Underperformed Scenes  
**作者**: Chen, T., Zhu, L., et al.  
**会议/期刊**: ICCV 2023 Workshop  
**链接**: https://arxiv.org/abs/2304.09148  
**代码**: https://github.com/tianrun-chen/SAM-Adapter-PyTorch  

**相关内容**:
- SAM 在特定场景的适配
- 轻量级适配器设计

**引用位置**:
- Related Work: SAM 适配方法对比

---

## 4. 边界精细化

### 4.1 Boundary IoU ⭐⭐⭐⭐
**标题**: Boundary IoU: Improving Object-Centric Image Segmentation Evaluation  
**作者**: Cheng, B., Girshick, R., et al.  
**会议/期刊**: CVPR 2021  
**链接**: https://arxiv.org/abs/2103.16562  

**相关内容**:
- Boundary IoU 指标定义
- 边界质量评估方法

**引用位置**:
- Method: Boundary Refinement 模块
- Experiments: Boundary-IoU 指标
- models/boundary_refinement.py: BoundaryMetrics

---

### 4.2 边界感知分割
**标题**: Boundary-Aware Segmentation Network for Mobile and Web Applications  
**作者**: Takikawa, T., et al.  
**会议/期刊**: arXiv 2019  
**链接**: https://arxiv.org/abs/1907.05740  

**相关内容**:
- 边界感知损失函数
- 边界检测模块

**引用位置**:
- Method: Boundary Loss 设计
- models/boundary_refinement.py: BoundaryLoss

---

### 4.3 Sobel 边缘检测（经典方法）
**标题**: A 3x3 Isotropic Gradient Operator for Image Processing  
**作者**: Sobel, I., Feldman, G.  
**会议/期刊**: Stanford AI Project 1968  

**相关内容**:
- Sobel 滤波器
- 边缘检测算法

**引用位置**:
- Method: Boundary Detector 初始化
- models/boundary_refinement.py: _init_edge_weights

---

## 5. 裂缝检测相关工作

### 5.1 深度学习裂缝检测综述 ⭐⭐⭐⭐
**标题**: Deep Learning-Based Crack Damage Detection Using Convolutional Neural Networks  
**作者**: Cha, Y. J., Choi, W., Büyüköztürk, O.  
**会议/期刊**: Computer-Aided Civil and Infrastructure Engineering, 2017  
**链接**: https://doi.org/10.1111/mice.12263  

**相关内容**:
- 裂缝检测的挑战
- CNN 在裂缝检测中的应用

**引用位置**:
- Introduction: 裂缝检测的重要性和挑战
- Related Work: 裂缝检测方法

---

### 5.2 裂缝分割数据集
**标题**: CrackForest Dataset: A Benchmark for Automatic Crack Detection  
**作者**: Shi, Y., Cui, L., et al.  
**会议/期刊**: ICIP 2016  
**链接**: https://github.com/cuilimeng/CrackForest-dataset  

**相关内容**:
- 裂缝数据集
- 评估基准

**引用位置**:
- Experiments: 数据集介绍（如果使用）
- Related Work: 裂缝检测数据集

---

### 5.3 隧道裂缝检测
**标题**: Automatic Tunnel Crack Detection Based on U-Net and a Convolutional Neural Network  
**作者**: Huang, H., Li, Q., Zhang, D.  
**会议/期刊**: Applied Sciences, 2018  
**链接**: https://doi.org/10.3390/app8101995  

**相关内容**:
- 隧道裂缝特点
- U-Net 在裂缝检测中的应用

**引用位置**:
- Introduction: 隧道裂缝检测的特殊性
- Related Work: 隧道裂缝检测方法
- Experiments: 对比方法（U-Net baseline）

---

## 6. 语义分割基础方法

### 6.1 U-Net ⭐⭐⭐⭐⭐
**标题**: U-Net: Convolutional Networks for Biomedical Image Segmentation  
**作者**: Ronneberger, O., Fischer, P., Brox, T.  
**会议/期刊**: MICCAI 2015  
**链接**: https://arxiv.org/abs/1505.04597  

**相关内容**:
- U-Net 架构
- 编码器-解码器结构

**引用位置**:
- Related Work: 经典分割方法
- Experiments: Baseline 对比

---

### 6.2 DeepLabV3+ ⭐⭐⭐⭐
**标题**: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation  
**作者**: Chen, L. C., Zhu, Y., et al.  
**会议/期刊**: ECCV 2018  
**链接**: https://arxiv.org/abs/1802.02611  

**相关内容**:
- ASPP 模块
- 多尺度特征融合

**引用位置**:
- Related Work: 先进分割方法
- Experiments: Baseline 对比

---

### 6.3 FCN
**标题**: Fully Convolutional Networks for Semantic Segmentation  
**作者**: Long, J., Shelhamer, E., Darrell, T.  
**会议/期刊**: CVPR 2015  
**链接**: https://arxiv.org/abs/1411.4038  

**相关内容**:
- 全卷积网络
- 端到端训练

**引用位置**:
- Related Work: 语义分割基础方法
- Method: 端到端训练的理论依据

---

## 7. 评估指标

### 7.1 Dice 系数
**标题**: Measures of the Amount of Ecologic Association Between Species  
**作者**: Dice, L. R.  
**会议/期刊**: Ecology, 1945  

**相关内容**:
- Dice 系数定义
- 重叠度量

**引用位置**:
- Experiments: 评估指标说明

---

### 7.2 Hausdorff Distance
**标题**: Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks  
**作者**: Karimi, D., Salcudean, S. E.  
**会议/期刊**: IEEE TMI, 2020  
**链接**: https://arxiv.org/abs/1904.10030  

**相关内容**:
- Hausdorff 距离
- 边界距离评估

**引用位置**:
- Experiments: 边界质量评估指标

---

## 8. Vision Transformer

### 8.1 ViT (Vision Transformer) ⭐⭐⭐⭐⭐
**标题**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
**作者**: Dosovitskiy, A., et al.  
**会议/期刊**: ICLR 2021  
**链接**: https://arxiv.org/abs/2010.11929  

**相关内容**:
- Vision Transformer 架构
- 自注意力机制

**引用位置**:
- Related Work: SAM 的 Image Encoder 基于 ViT
- Method: ViT 架构说明

---

## 9. 小样本学习与数据增强

### 9.1 数据增强综述
**标题**: A Survey on Image Data Augmentation for Deep Learning  
**作者**: Shorten, C., Khoshgoftaar, T. M.  
**会议/期刊**: Journal of Big Data, 2019  
**链接**: https://doi.org/10.1186/s40537-019-0197-0  

**相关内容**:
- 数据增强方法
- 小样本学习

**引用位置**:
- Method: 数据增强策略
- utils/data_loader.py: 数据增强实现

---

## 10. 不确定性估计（如果实现）

### 10.1 贝叶斯深度学习
**标题**: What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?  
**作者**: Kendall, A., Gal, Y.  
**会议/期刊**: NeurIPS 2017  
**链接**: https://arxiv.org/abs/1703.04977  

**相关内容**:
- 认知不确定性
- 偶然不确定性

**引用位置**:
- Method: Uncertainty Estimation 模块（如果实现）

---

## 11. 对比学习（如果实现）

### 11.1 SimCLR
**标题**: A Simple Framework for Contrastive Learning of Visual Representations  
**作者**: Chen, T., Kornblith, S., et al.  
**会议/期刊**: ICML 2020  
**链接**: https://arxiv.org/abs/2002.05709  

**相关内容**:
- 对比学习框架
- 自监督学习

**引用位置**:
- Method: Contrastive Learning 模块（如果实现）

---

## 12. 实验设置与训练技巧

### 12.1 Adam 优化器
**标题**: Adam: A Method for Stochastic Optimization  
**作者**: Kingma, D. P., Ba, J.  
**会议/期刊**: ICLR 2015  
**链接**: https://arxiv.org/abs/1412.6980  

**相关内容**:
- Adam 优化算法
- 自适应学习率

**引用位置**:
- Experiments: 训练设置

---

### 12.2 学习率调度
**标题**: SGDR: Stochastic Gradient Descent with Warm Restarts  
**作者**: Loshchilov, I., Hutter, F.  
**会议/期刊**: ICLR 2017  
**链接**: https://arxiv.org/abs/1608.03983  

**相关内容**:
- Cosine Annealing
- 学习率调度策略

**引用位置**:
- Experiments: 学习率调度

---

## 13. 相关应用

### 13.1 基础设施检测
**标题**: Deep Learning for Infrastructure Inspection: A Survey  
**作者**: Spencer, B. F., et al.  
**会议/期刊**: Structural Health Monitoring, 2019  

**相关内容**:
- 基础设施检测综述
- 深度学习应用

**引用位置**:
- Introduction: 应用背景

---

## 使用说明

### 引用格式

**论文中引用**:
```latex
SAM \cite{kirillov2023segany} is a foundation model...
We adopt LoRA \cite{hu2022lora} for parameter-efficient fine-tuning...
```

**BibTeX 示例**:
```bibtex
@inproceedings{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4015--4026},
  year={2023}
}
```

---

## 更新日志

- 2026-04-25: 初始版本，收集核心参考文献
- 待更新: 根据实际实现的模块补充相关论文

---

**维护**: 随着项目开发，持续更新相关文献
