# CLAUDE.md - 代码规范与AI辅助开发指南

**项目**: Tunnel Crack Segmentation with Enhanced SAM  
**版本**: v1.0.0  
**用途**: 为AI助手（Claude）提供项目代码规范和开发指导

---

## 1. 项目概述

本项目是一个基于SAM的隧道裂缝分割系统，采用LoRA微调和边界精细化技术。

**核心模块**:
- SAM基础模型 (Segment Anything Model)
- LoRA适配器 (参数高效微调)
- 边界精细化模块 (Boundary Refinement)

**已移除模块**:
- FSA (FreqSpatialAttention) - 导致IoU下降
- DAP (DefectAwarePrompt) - 导致IoU下降

---

## 2. 代码风格规范

### 2.1 Python代码规范

**遵循PEP 8标准**:
- 缩进: 4个空格
- 行长度: 最大88字符 (Black默认)
- 命名:
  - 类名: `PascalCase` (如 `LoRAAdapter`)
  - 函数/变量: `snake_case` (如 `load_sam_model`)
  - 常量: `UPPER_SNAKE_CASE` (如 `DEFAULT_RANK`)
  - 私有成员: `_leading_underscore` (如 `_inject_lora`)

**类型注解**:
```python
from typing import Optional, Tuple, Dict
import torch

def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """前向传播
    
    Args:
        x: 输入张量 (B, C, H, W)
    
    Returns:
        masks: 分割掩码 (B, 1, H, W)
        iou_pred: IoU预测 (B,)
    """
    pass
```

**Docstring格式** (Google风格):
```python
def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """计算边界感知损失
    
    Args:
        pred: 预测掩码 (B, 1, H, W)
        target: 真值掩码 (B, 1, H, W)
    
    Returns:
        包含以下键的字典:
            - loss: 总损失
            - bce: BCE损失
            - dice: Dice损失
            - boundary: 边界损失
    
    Raises:
        ValueError: 如果pred和target形状不匹配
    """
    pass
```

### 2.2 导入顺序
```python
# 1. 标准库
import os
import sys
from typing import Optional

# 2. 第三方库
import torch
import torch.nn as nn
import numpy as np

# 3. 本地模块
from models.lora_adapter import LoRAAdapter
from utils.metrics import compute_iou
```

### 2.3 代码组织
```python
class EnhancedSAM(nn.Module):
    """类定义
    
    1. 类文档字符串
    2. 类变量
    3. __init__
    4. 公共方法
    5. 私有方法
    6. 静态方法/类方法
    """
    
    # 类变量
    DEFAULT_CONFIG = {...}
    
    def __init__(self, ...):
        """初始化"""
        pass
    
    def forward(self, ...):
        """公共方法"""
        pass
    
    def _preprocess(self, ...):
        """私有方法"""
        pass
    
    @staticmethod
    def from_pretrained(...):
        """静态方法"""
        pass
```

---

## 3. 模型开发规范

### 3.1 模型定义
```python
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """配置类使用dataclass"""
    embed_dim: int = 768
    num_layers: int = 12
    dropout: float = 0.1

class MyModel(nn.Module):
    """模型类继承nn.Module"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # 初始化层
        self.layer = nn.Linear(config.embed_dim, config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播必须实现"""
        return self.layer(x)
```

### 3.2 权重初始化
```python
def _init_weights(self):
    """权重初始化"""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
```

### 3.3 设备管理
```python
# 推荐方式：使用.to(device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
images = images.to(device)

# 避免硬编码.cuda()
# 错误: model.cuda()
```

---

## 4. 训练代码规范

### 4.1 训练循环
```python
def train_one_epoch(model, loader, optimizer, device):
    """训练一个epoch
    
    Args:
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        device: 设备
    
    Returns:
        平均损失
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 前向传播
        outputs = model(image=images)
        loss_dict = model.compute_loss(outputs, masks)
        loss = loss_dict['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

### 4.2 检查点保存
```python
def save_checkpoint(model, optimizer, epoch, val_iou, path):
    """保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        val_iou: 验证集IoU
        path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_iou': val_iou,
        'config': model.config.__dict__,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
```

### 4.3 评估模式
```python
@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型
    
    使用@torch.no_grad()装饰器禁用梯度计算
    """
    model.eval()  # 切换到评估模式
    
    metrics = []
    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        outputs = model(image=images)
        # 计算指标
        iou = compute_iou(outputs['masks'], masks)
        metrics.append(iou)
    
    return np.mean(metrics)
```

---

## 5. 数据处理规范

### 5.1 数据集类
```python
from torch.utils.data import Dataset

class TunnelCrackDataset(Dataset):
    """隧道裂缝数据集
    
    目录结构:
        root/
          images/       *.jpg, *.png
          annotations/  *.png (二值掩码)
    """
    
    def __init__(self, root: str, img_size: int = 1024, augment: bool = False):
        self.root = root
        self.img_size = img_size
        self.augment = augment
        self.samples = self._load_samples()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """返回字典格式，包含image和mask"""
        img_path, mask_path = self.samples[idx]
        # 加载和预处理
        return {'image': image, 'mask': mask, 'path': img_path}
```

### 5.2 数据增强
```python
import albumentations as A

def get_train_transforms(img_size=512):
    """训练集数据增强"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

---

## 6. 测试规范

### 6.1 单元测试
```python
import pytest
import torch

def test_lora_linear_forward():
    """测试LoRA线性层前向传播"""
    layer = LoRALinear(in_features=256, out_features=512, rank=8)
    x = torch.randn(4, 256)
    y = layer(x)
    
    assert y.shape == (4, 512), f"Expected shape (4, 512), got {y.shape}"
    assert not torch.isnan(y).any(), "Output contains NaN"

def test_lora_merge():
    """测试LoRA权重合并"""
    layer = LoRALinear(256, 512, rank=8)
    
    # 合并前
    x = torch.randn(4, 256)
    y_before = layer(x)
    
    # 合并
    layer.merge_lora()
    assert layer.merged == True
    
    # 合并后输出应相同
    y_after = layer(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)
```

### 6.2 测试覆盖
```bash
# 运行测试并生成覆盖率报告
pytest tests/ -v --cov=models --cov=utils --cov-report=html

# 目标：覆盖率 ≥ 80%
```

---

## 7. Git提交规范

### 7.1 提交信息格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型 (type)**:
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

**示例**:
```
feat(lora): implement LoRA adapter for SAM

- Add LoRALinear layer with kaiming initialization
- Implement LoRAAdapter injection logic
- Support weight merging for inference

Closes #1
```

### 7.2 分支管理
- `main`: 稳定版本
- `develop`: 开发分支
- `feature/*`: 功能分支
- `hotfix/*`: 紧急修复

---

## 8. 文件命名规范

### 8.1 Python文件
- 模块: `snake_case.py` (如 `lora_adapter.py`)
- 测试: `test_*.py` (如 `test_lora_adapter.py`)
- 脚本: `动词_名词.py` (如 `train.py`, `eval.py`)

### 8.2 配置文件
- YAML: `*_config.yaml` (如 `train_config.yaml`)
- JSON: `*.json` (如 `results.json`)

### 8.3 检查点
- 格式: `epoch_{epoch}.pth` 或 `best_model.pth`
- 包含: epoch, model_state, optimizer_state, metrics

---

## 9. 性能优化指南

### 9.1 推理优化
```python
# 1. LoRA权重合并
model.lora_adapter.merge_all()

# 2. 评估模式
model.eval()

# 3. 禁用梯度
with torch.no_grad():
    outputs = model(images)

# 4. 混合精度
with torch.cuda.amp.autocast():
    outputs = model(images)
```

### 9.2 显存优化
```python
# 1. 梯度累积
accumulation_steps = 4
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. 清理缓存
torch.cuda.empty_cache()
```

---

## 10. 错误处理

### 10.1 异常处理
```python
def load_checkpoint(path: str) -> dict:
    """加载检查点
    
    Args:
        path: 检查点路径
    
    Returns:
        检查点字典
    
    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 加载失败
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
```

### 10.2 输入验证
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """前向传播
    
    Args:
        x: 输入张量 (B, C, H, W)
    
    Returns:
        输出张量 (B, C', H', W')
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D input, got {x.dim()}D")
    
    if x.size(1) != self.in_channels:
        raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
    
    return self.layer(x)
```

---

## 11. 日志规范

### 11.1 日志级别
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEBUG: 详细调试信息
logger.debug(f"Input shape: {x.shape}")

# INFO: 一般信息
logger.info(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")

# WARNING: 警告信息
logger.warning(f"Learning rate is very high: {lr}")

# ERROR: 错误信息
logger.error(f"Failed to load checkpoint: {e}")
```

### 11.2 训练日志
```python
# 使用tqdm显示进度
from tqdm import tqdm

for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in pbar:
        loss = train_step(batch)
        pbar.set_postfix({'loss': f'{loss:.4f}'})
```

---

## 12. AI辅助开发指南

### 12.1 代码生成提示
当请求AI生成代码时，提供以下信息：
- 功能描述
- 输入/输出格式
- 性能要求
- 参考现有代码

**示例**:
```
请实现一个边界检测器模块：
- 输入: 特征图 (B, 256, H, W)
- 输出: 边界概率图 (B, 1, H, W)
- 使用Sobel初始化卷积核
- 参考models/boundary_refinement.py中的BoundaryDetector
```

### 12.2 代码审查提示
请AI审查代码时，关注：
- 是否符合本文档规范
- 是否有潜在bug
- 性能优化建议
- 测试覆盖

### 12.3 调试提示
遇到问题时，提供：
- 错误信息完整堆栈
- 相关代码片段
- 输入数据形状
- 期望行为

---

## 13. 常见问题

### 13.1 LoRA相关
**Q: LoRA权重何时合并？**
A: 训练时不合并，推理时可选择合并以提升速度。

**Q: 如何调整LoRA rank？**
A: rank越大表达能力越强，但参数量增加。推荐8-16。

### 13.2 训练相关
**Q: 如何处理类别不平衡？**
A: 使用pos_weight参数，推荐10.0（裂缝像素占比约10%）。

**Q: 显存不足怎么办？**
A: 降低batch_size，使用梯度累积，或降低图像尺寸。

### 13.3 评估相关
**Q: 为什么mIoU较低？**
A: 裂缝分割是困难任务，0.3-0.4是合理范围。关注Boundary-IoU。

---

## 14. 参考资源

### 14.1 官方文档
- PyTorch: https://pytorch.org/docs/
- Segment Anything: https://github.com/facebookresearch/segment-anything

### 14.2 论文
- SAM: "Segment Anything", Kirillov et al., ICCV 2023
- LoRA: "LoRA: Low-Rank Adaptation of Large Language Models", Hu et al., ICLR 2022

---

**文档版本**: v1.0  
**最后更新**: 2026-04-25  
**维护者**: 项目团队
