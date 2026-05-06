# 隧道裂缝识别项目计划（需求与技术方案）

## 1. 项目定位
- 研究方向：隧道裂缝图像分割（医学/工业分割范式，语义分割任务）。
- 基线模型：SAM（ViT-B）；
- 当前核心方案：LoRA 微调 + 边界细化（Boundary Refinement）。
- 目标：形成可复现实验系统，产出可投稿北大核心期刊的实验与方法结果。

## 2. 当前目录审计结论（2026-05-05）
### 合理部分
- `models/` + `tests/` 已形成“实现-单测”对应关系，结构清晰。
- `configs/train_config.yaml` 已覆盖模型、损失、训练超参，具备可配置性。
- `data/`、`checkpoints/`、`logs/` 已被 `.gitignore` 忽略，符合大文件管理规范。

### 不足部分
- `scripts/`、`utils/` 仍是空壳，训练/评估/推理流水线未闭环。
- 研究文档分散在多个子目录，项目入口信息不统一。
- 缺少实验登记规范（实验ID、配置快照、结果追踪）与回滚操作手册。

## 3. 规范化后的目录策略
保留当前主干，不做大规模重构，避免影响既有代码与测试。

### 根目录保留
- `README.md`（GitHub 入口）
- `requirements.txt`
- `configs/ models/ tests/ scripts/ utils/`
- `plan.md`（当前执行计划）

### 文档归档（已执行）
- 说明类 `.md` 已迁移到上一级外置文档目录 `E:\SCIcodex\doc\enhanceSAM\`：
  - `project/AGENTS.md`
  - `project/PRODUCT_SPECS.md`
  - `project/PROJECT_STRUCTURE.md`
  - `research/*`

### 已清理冗余文件（已执行）
- `project_tree.txt`
- `.github_setup_commands.txt`
- `setup_github.sh`

## 4. 项目需求
## 4.1 功能需求
1. 支持 SAM + LoRA 的可配置训练流程（冻结策略、rank、alpha、dropout）。
2. 支持边界细化分支的可开关训练与推理。
3. 支持统一评估输出：mIoU、Dice、Boundary-IoU、Precision、Recall、FPS。
4. 支持单图推理与批量推理，输出 mask 与可视化叠图。
5. 支持消融实验：`SAM`、`SAM+LoRA`、`SAM+LoRA+Boundary`。

## 4.2 工程需求
1. 全流程命令行化（train/eval/predict/export）。
2. 核心模块具备单元测试，关键路径覆盖率 >= 80%。
3. 实验可复现：固定随机种子、记录配置与commit哈希。
4. Git 可回滚：分支策略、提交规范、版本标签。

## 5. 技术方案
## 5.1 模型路线
- 主干：SAM Image Encoder + Mask Decoder。
- 参数高效微调：LoRA 注入 attention/MLP 线性层。
- 边界增强：BoundaryDetector + BoundaryRefineNet + Boundary-aware Loss。

## 5.2 训练策略
- 损失函数：`BCE + Dice + BoundaryWeightedBCE`。
- 训练技巧：AMP、梯度裁剪、余弦退火、早停。
- 数据增强：翻转、尺度抖动、亮度对比度扰动、模糊噪声模拟。

## 5.3 评估与统计
- 主指标：mIoU、Dice、Boundary-IoU。
- 辅指标：参数量、显存占用、吞吐/FPS、单图时延。
- 输出工件：`results/*.json`、`paper/tables/*.csv`、可视化图。

## 6. 可发表导向的创新点规划（后续迭代）
1. **Crack-Aware Prompt Generator（建议优先）**  
   自动从纹理与边缘候选生成 prompt token，提高细小裂缝召回。
2. **Topology-Preserving Loss**  
   对细长连通结构增加连通性约束，降低裂缝断裂预测。
3. **Multi-Scale Boundary Distillation**  
   用高分辨边界教师分支蒸馏主分支，提升边缘稳定性。
4. **Domain Generalization**  
   多隧道域颜色/光照扰动训练，验证跨工况泛化能力。

> 投稿策略：创新点至少落地 1 个主创新 + 1 个辅助创新，并完成消融与跨域验证。

## 7. 里程碑（建议）
1. **M1（2026-05-05 ~ 2026-05-20）**：补齐 `scripts/` 与 `utils/` 训练闭环。
2. **M2（2026-05-21 ~ 2026-06-10）**：跑通三组基线+消融，固化日志模板。
3. **M3（2026-06-11 ~ 2026-07-10）**：落地主创新（Prompt/Topology 二选一）。
4. **M4（2026-07-11 ~ 2026-08-10）**：论文图表、误差分析、初稿撰写。

## 8. Git 管理与回滚机制
## 8.1 分支策略
- `main`：仅接收通过测试的稳定版本。
- `develop`：日常集成分支。
- `feature/*`：功能开发分支。
- `exp/*`：实验性方案分支（可快速丢弃）。

## 8.2 提交规范
- 格式：`type(scope): summary`
- 示例：`feat(models): add crack-aware prompt generator`

## 8.3 回滚流程
1. 查看历史：`git log --oneline --graph`
2. 回滚单次提交：`git revert <commit>`
3. 恢复文件版本：`git restore --source <commit> <path>`
4. 发布节点：`git tag -a v0.2.0 -m "ablation complete"`

## 9. 下一步执行清单
- [ ] 实现 `scripts/train.py`、`scripts/eval.py`、`scripts/predict.py`
- [ ] 实现 `utils/data_loader.py`、`utils/metrics.py`、`utils/visualization.py`
- [ ] 增加集成测试（训练1个epoch的smoke test）
- [ ] 固化实验记录模板（配置+结果+结论）
- [ ] 启动主创新模块原型开发
