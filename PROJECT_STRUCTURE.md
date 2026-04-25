# EnhancedSAM 项目目录结构说明

**更新时间**: 2026-04-25  
**版本**: v1.1 (添加实验管理、论文素材等目录)

---

## 📁 完整目录结构

```
E:\SCIopus\enhanceSAM/
│
├── 📁 .git/                          # Git 版本控制
├── 📁 .vscode/                       # VS Code 配置
│   └── launch.json                   # 调试配置（5个）
│
├── 📄 .gitignore                     # Git 忽略规则
├── 📄 README.md                      # 项目说明
├── 📄 CLAUDE.md                      # AI 辅助开发规范
├── 📄 requirements.txt               # Python 依赖
├── 📄 __init__.py                    # 根包初始化
│
├── 📁 configs/                       # ✅ 配置文件
│   ├── train_config.yaml            # 训练配置
│   ├── eval_config.yaml             # 评估配置（待添加）
│   └── model_config.yaml            # 模型配置（待添加）
│
├── 📁 models/                        # ✅ 模型实现（Phase 2 完成）
│   ├── __init__.py
│   ├── sam_base.py                  # SAM 基础封装 (195行)
│   ├── lora_adapter.py              # LoRA 适配器 (467行)
│   ├── boundary_refinement.py       # 边界精细化 (453行)
│   └── enhanced_sam.py              # 集成模型 (332行)
│
├── 📁 utils/                         # ⏳ 工具函数（Phase 3 待开发）
│   ├── __init__.py
│   ├── data_loader.py               # 数据加载器（待实现）
│   ├── metrics.py                   # 评估指标（待实现）
│   ├── visualization.py             # 可视化工具（待实现）
│   └── logger.py                    # 日志工具（待实现）
│
├── 📁 scripts/                       # ⏳ 可执行脚本（Phase 3 待开发）
│   ├── train.py                     # 训练脚本（待实现）
│   ├── eval.py                      # 评估脚本（待实现）
│   ├── predict.py                   # 推理脚本（待实现）
│   ├── download_models.py           # 下载 SAM 权重（待实现）
│   └── export_onnx.py               # 模型导出（可选）
│
├── 📁 tests/                         # ✅ 单元测试（Phase 2 完成）
│   ├── __init__.py
│   ├── test_sam_base.py             # SAM 基础测试 (126行)
│   ├── test_lora_adapter.py         # LoRA 测试 (261行)
│   ├── test_boundary_refinement.py  # 边界精细化测试 (222行)
│   └── test_enhanced_sam.py         # 集成测试 (298行)
│
├── 📁 experiments/                   # 🆕 实验管理
│   ├── README.md                    # 实验说明
│   ├── exp001_baseline/             # 实验 1: SAM 基线
│   │   ├── config.yaml
│   │   ├── results.json
│   │   ├── logs/
│   │   └── checkpoints/
│   ├── exp002_lora/                 # 实验 2: SAM + LoRA
│   ├── exp003_lora_boundary/        # 实验 3: SAM + LoRA + Boundary
│   └── exp004_ablation/             # 实验 4: 消融实验
│
├── 📁 paper/                         # 🆕 论文相关
│   ├── figures/                     # 论文图表
│   │   ├── architecture.pdf         # 模型架构图
│   │   ├── results_comparison.pdf   # 结果对比图
│   │   ├── qualitative_results.pdf  # 定性结果
│   │   └── training_curves.pdf      # 训练曲线
│   ├── tables/                      # 论文表格数据
│   │   ├── main_results.csv         # 主要结果
│   │   ├── ablation_study.csv       # 消融实验
│   │   └── comparison_methods.csv   # 方法对比
│   └── references/                  # 🆕 参考文献
│       ├── REFERENCES.md            # 参考文献库
│       └── bibtex.bib               # BibTeX 文件（待添加）
│
├── 📁 analysis/                      # 🆕 深入分析
│   ├── dataset_statistics.py        # 数据集统计分析
│   ├── error_analysis.py            # 错误分析
│   ├── attention_visualization.py   # 注意力可视化
│   └── lora_rank_analysis.py        # LoRA rank 分析
│
├── 📁 baselines/                     # 🆕 对比方法
│   ├── unet/                        # U-Net 基线
│   │   ├── model.py
│   │   └── train.py
│   ├── deeplabv3/                   # DeepLabV3+ 基线
│   └── sam_vanilla/                 # 原始 SAM（无微调）
│
├── 📁 notebooks/                     # 🆕 Jupyter 分析
│   ├── 01_data_exploration.ipynb    # 数据探索
│   ├── 02_model_analysis.ipynb      # 模型分析
│   ├── 03_results_visualization.ipynb # 结果可视化
│   └── 04_paper_figures.ipynb       # 论文图表生成
│
├── 📁 data/                          # 数据集（不提交 Git）
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   ├── val/
│   │   ├── images/
│   │   └── annotations/
│   └── test/
│       ├── images/
│       └── annotations/
│
├── 📁 checkpoints/                   # 模型检查点（不提交 Git）
│   ├── best_model.pth
│   ├── epoch_10.pth
│   └── sam_vit_b_01ec64.pth         # SAM 预训练权重
│
└── 📁 logs/                          # 训练日志（不提交 Git）
    ├── tensorboard/
    ├── train.log
    └── eval.log
```

---

## 📋 目录功能说明

### 核心代码目录

| 目录 | 状态 | 说明 | 文件数 |
|------|------|------|--------|
| `models/` | ✅ 完成 | 模型实现 | 4 个 Python 文件 |
| `utils/` | ⏳ 待开发 | 工具函数 | 待实现 |
| `scripts/` | ⏳ 待开发 | 可执行脚本 | 待实现 |
| `tests/` | ✅ 完成 | 单元测试 | 4 个测试文件 |

### 实验与论文目录

| 目录 | 状态 | 说明 | 用途 |
|------|------|------|------|
| `experiments/` | 🆕 新增 | 实验管理 | 系统化记录每个实验 |
| `paper/` | 🆕 新增 | 论文素材 | 图表、表格、参考文献 |
| `analysis/` | 🆕 新增 | 深入分析 | 错误分析、可视化 |
| `baselines/` | 🆕 新增 | 对比方法 | U-Net, DeepLabV3+ 等 |
| `notebooks/` | 🆕 新增 | 交互式分析 | Jupyter 笔记本 |

### 数据与输出目录

| 目录 | 说明 | Git 跟踪 |
|------|------|----------|
| `data/` | 数据集 | ❌ 不跟踪 |
| `checkpoints/` | 模型权重 | ❌ 不跟踪 |
| `logs/` | 训练日志 | ❌ 不跟踪 |

---

## 🎯 各目录的使用场景

### 1. 日常开发
```
models/      → 实现新模块
utils/       → 添加工具函数
scripts/     → 编写训练/评估脚本
tests/       → 编写单元测试
```

### 2. 实验管理
```
experiments/ → 每个实验一个子目录
  ├── config.yaml    → 实验配置
  ├── results.json   → 实验结果
  └── logs/          → 实验日志
```

### 3. 论文撰写
```
paper/
  ├── figures/       → 生成论文图表
  ├── tables/        → 整理实验数据
  └── references/    → 管理参考文献
```

### 4. 结果分析
```
analysis/            → 深入分析脚本
notebooks/           → 交互式分析
```

### 5. 对比实验
```
baselines/           → 实现对比方法
  ├── unet/
  ├── deeplabv3/
  └── sam_vanilla/
```

---

## 📝 目录创建状态

### ✅ 已创建
- models/, utils/, scripts/, tests/
- configs/, data/, checkpoints/, logs/
- .vscode/, .git/

### 🆕 新增（已创建）
- experiments/
- paper/figures/, paper/tables/, paper/references/
- analysis/
- baselines/
- notebooks/

---

## 🔄 下一步工作

### Phase 3: 数据与训练
1. 实现 `utils/data_loader.py`
2. 实现 `utils/metrics.py`
3. 实现 `scripts/train.py`
4. 实现 `scripts/eval.py`

### Phase 4: 实验与分析
5. 运行基线实验 → `experiments/exp001_baseline/`
6. 运行 LoRA 实验 → `experiments/exp002_lora/`
7. 分析结果 → `analysis/` 和 `notebooks/`

### Phase 5: 论文准备
8. 生成论文图表 → `paper/figures/`
9. 整理实验数据 → `paper/tables/`
10. 撰写论文

---

**维护**: 随着项目开发，持续更新此文档
