# enhanceSAM 执行计划（2026-05-13）

## 0. 强制执行规则
1. 每次执行任务前，先阅读 `AGENTS.md` 与 `PROJECT_STATE.md`。  
2. 先在本文件写明“本次任务计划”，再开始执行。  
3. 每次只执行 `PROJECT_STATE.md` 中“当前进行中”的事项，避免并行扩散。  
4. 任务完成后同步更新 `PROJECT_STATE.md`（已完成 / 下一步 / 产物路径）。  

## 1. 项目现状分析（本次审计结论）
- 代码闭环已成型：`models/`、`scripts/train.py`、`scripts/eval.py`、`scripts/predict.py`、`utils/`、`tests/` 已齐全。  
- 数据组织完整：`data/train|val|test` 已落地，目录结构满足训练/验证/测试流程。  
- 当前主要阻塞是环境：本机缺少 `pytest`（且 `PROJECT_STATE.md` 也记录缺 `torch + pytest`），因此无法执行 smoke test。  
- 测试存在一致性风险：`tests/test_lora_adapter.py`、`tests/test_enhanced_sam.py` 断言英文参数报告文案，但实现返回中文报告字符串，后续在可跑环境中大概率失败。  

## 2. 本次任务计划（对齐 PROJECT_STATE 当前进行中）
1. [x] 补齐可执行环境：安装 `torch` 与 `pytest`。  
2. [x] 执行 smoke test：`pytest tests/test_train_smoke.py -v`。  
3. [x] 执行核心单测：`pytest tests -v`，记录失败明细。  
4. [x] 修复测试与实现不一致项（优先处理参数报告文案断言）。  
5. [x] 固化当前结果到 `analysis/test_reports/2026-05-12_smoke_and_unit.md`。  

## 3. 产物约定
- 测试记录：`analysis/test_reports/2026-05-12_smoke_and_unit.md`  
- 如有修复：对应模块与测试文件同步提交（模型代码 + `tests/test_*.py`）  

## 4. 本次追加任务（2026-05-12）
1. [x] 下载 `sam_vit_b_01ec64.pth` 到 `checkpoints/`。  
2. [x] 校验文件存在性与文件大小，并记录路径。  

## 5. M2 对比评估任务（当前）
1. [x] 切换 `venv` 为 CUDA 版 PyTorch，并验证 `torch.cuda.is_available()`。  
   - 2026-05-13 验证结果：`torch 2.11.0+cu128`，`torch.version.cuda=12.8`，`torch.cuda.is_available()=True`，GPU 为 RTX 5080。
2. [ ] 确认三组评估输入：`SAM-AMG`、`SAM-Box-Oracle`、`SAM+LoRA`。  
3. [ ] 在 `val` 集执行三组评估并保存 JSON 结果。  
4. [ ] 产出统一对比表到 `analysis/` 与 `paper/tables/`。  

## 6. 本次任务计划（2026-05-13，M2 继续）
1. [x] 读取 `AGENTS.md` 与 `PROJECT_STATE.md`，确认本轮只推进 M2 val 对比评估。  
2. [x] 验证 CUDA PyTorch 与本机 GPU 可用性。  
3. [ ] 检查 `SAM-AMG` / `SAM-Box-Oracle` / `SAM+LoRA` 三组评估入口、配置和权重依赖。  
4. [ ] 若 `SAM+LoRA` 缺少可评估权重，先记录为 M2 阻塞并更新 `PROJECT_STATE.md`，不跳到其他阶段。  

## 7. 本次追加任务计划（2026-05-13，M2 论文可视化）
1. [x] 确认当前 baseline 只有 JSON 指标结果，尚无论文可用 PNG/JPG 可视化结果。  
2. [x] 新增 M2 定性对比图生成脚本，统一输出 `GT`、`SAM-AMG`、`SAM-Box-Oracle`、`SAM+LoRA`。  
3. [x] 在 `SAM+LoRA` checkpoint 缺失时给出明确阻塞提示，避免生成不完整论文图。  
4. [x] 语法检查脚本，并回写 `PROJECT_STATE.md` 的最新产物路径。  

## 8. 本次追加任务计划（2026-05-13，SAM baseline 可视化）
1. [x] 为 `scripts/visualize_m2_comparison.py` 增加 `--baseline_only` 模式，仅生成 `SAM-AMG` 与 `SAM-Box-Oracle` 的 mask/叠图/panel。
2. [x] 在 val 集自动选择裂缝前景较明显的样本，生成 baseline mask 图和可视化叠图。
3. [x] 记录输出目录，并同步更新 `PROJECT_STATE.md`。

## 9. 本次任务计划（2026-05-13，继续 LoRA 主线）
1. [x] 定位 `data/train/images/226_01_01.png` 读取失败原因，确认是文件损坏、OpenCV 路径读取问题还是 DataLoader worker 问题。
2. [x] 修复数据读取稳健性或异常样本处理，保证训练阶段不因单张图像读取失败中断。
3. [x] 用短训/单 epoch 命令验证 LoRA 训练能越过此前失败点并生成 checkpoint。
4. [x] 记录验证结果与下一步完整训练命令到 `PROJECT_STATE.md`。

## 10. 本次任务计划（2026-05-13，正式 LoRA 训练）
1. [x] 恢复检查上次正式 LoRA 训练：确认是否仍有进程、是否已生成 `checkpoints/m2_lora/best_model.pth`。
2. [x] 若未完成，则用 stdout/stderr 重定向重新启动后台训练：`configs/train_config.yaml` -> `checkpoints/m2_lora/`。
3. [x] 将日志、PID 与启动命令写入 `logs/`，便于断开会话后继续追踪。
4. [x] 确认训练进程已启动并写入初始日志。
5. [x] 更新 `PROJECT_STATE.md`，把下一步改为等待正式训练完成后评估。

## 11. 本次任务计划（2026-05-13，断开后重新执行正式 LoRA 训练）
1. [x] 重新读取 `PROJECT_STATE.md`，确认当前只推进 M2 val 对比评估中的正式 LoRA 训练前置任务。
2. [x] 检查上次正式训练状态：未发现活跃 Python 训练进程，且 `checkpoints/m2_lora/best_model.pth` 尚未生成。
3. [x] 重新启动正式 LoRA 后台训练，命令仍使用 `configs/train_config.yaml`、`checkpoints/sam_vit_b_01ec64.pth` 与 `checkpoints/m2_lora/`。
4. [x] 记录新的日志、PID 与启动命令到 `logs/`。
5. [x] 验证新训练进程已存在，并检查初始日志没有立即失败。
6. [x] 更新 `PROJECT_STATE.md`，记录本次重启产物路径与下一步。

### 执行记录
- 正式训练因 Windows 后台进程与多进程 DataLoader 稳定性问题，将 `configs/train_config.yaml` 的 `training.num_workers` 调整为 `0`。
- 当前存活训练进程 PID: `10944`。
- 日志路径：`logs/m2_lora_train_formal_detached_20260513_215158.out.log`、`logs/m2_lora_train_formal_detached_20260513_215158.err.log`。
- PID/命令记录：`logs/m2_lora_train_formal_detached_20260513_215158.pid.txt`、`logs/m2_lora_train_formal_detached_20260513_215158.command.txt`。
- 启动配置快照：`checkpoints/m2_lora/run_config_20260513_215204.json`。

## 12. 本次任务计划（2026-05-14，修复正式训练图像读取失败）
1. [x] 读取 `PROJECT_STATE.md`，确认当前仍在正式 LoRA 训练闭环内处理阻塞。
2. [x] 检查正式训练日志，定位失败样本为 `data/train/images/940_05_01.png`，错误为 `libpng error: IDAT: incorrect data check`。
3. [x] 将 DataLoader 初始化校验从“文件头魔数”升级为“实际解码校验”，提前跳过文件头合法但内容损坏的图像/标注。
4. [x] 增加测试覆盖“文件头合法但解码失败”的损坏 PNG。
5. [x] 扫描 train/val 图像与标注，确认是否还有其他解码失败文件。
6. [x] 从 `checkpoints/m2_lora/last_model.pth` 断点恢复正式 LoRA 训练。
7. [x] 更新 `PROJECT_STATE.md`，记录修复、坏样本、恢复训练日志与下一步。

### 执行记录
- 修复文件：`utils/data_loader.py`、`tests/test_data_loader.py`。
- 新增运行时兜底：单个 image/mask 解码失败时跳到后续可读样本，避免长训练中断。
- 测试命令：`venv\Scripts\python.exe -m pytest tests\test_data_loader.py -v --basetemp analysis\pytest_tmp_decode`，结果 `3 passed`。
- 解码扫描范围：`data/train/images`、`data/train/annotations`、`data/val/images`、`data/val/annotations`，共 `25300` 个文件，仅发现 `data/train/images/226_01_01.png` 固定不可解码。
- 恢复训练 PID：`27500`。
- 恢复训练日志：`logs/m2_lora_train_resume_decodefix_20260514_152808.out.log`、`logs/m2_lora_train_resume_decodefix_20260514_152808.err.log`。
- 恢复配置快照：`checkpoints/m2_lora/run_config_20260514_153121.json`。

## 13. 本次任务计划（2026-05-15，训练后过拟合与结果验证）
1. [x] 确认正式 LoRA 训练已完成，读取 train/val epoch 日志。
2. [x] 用 `checkpoints/m2_lora/best_model.pth` 在独立 `test` 集评估，输出 `results/enhanced/sam_lora_test_gpu.json`。
3. [x] 用同一权重重跑 `val` 集正式指标，输出 `results/enhanced/sam_lora_val_gpu.json`。
4. [x] 汇总 train/val 曲线、val/test 指标差距、baseline 对比，判断是否有明显过拟合或数据问题。
5. [x] 更新 `PROJECT_STATE.md`，记录正式训练结果与下一步论文图/表任务。

### 执行记录
- 正式最佳 checkpoint：`checkpoints/m2_lora/best_model.pth`，epoch=`44`，训练日志最佳 `val_iou=0.8455`。
- 最后一轮训练日志：epoch `50`，`train_iou=0.9150`，`val_iou=0.8453`。
- 正式 val 指标：`results/enhanced/sam_lora_val_gpu.json`，mIoU=`0.991246`，Dice=`0.995604`，Boundary-IoU=`0.658638`，FPS=`50.27`。
- 正式 test 指标：`results/enhanced/sam_lora_test_gpu.json`，mIoU=`0.991093`，Dice=`0.995526`，Boundary-IoU=`0.654690`，FPS=`49.83`。
- 数据切分重名检查：`train-val=0`，`train-test=0`，`val-test=0`。

## 14. 本次任务计划（2026-05-15，M2 正式表格与定性图）
1. [x] 生成正式 M2 val 对比表：`analysis/m2_formal_comparison_2026-05-15.md`、`paper/tables/m2_formal_comparison.csv`。
2. [x] 运行 `scripts/visualize_m2_comparison.py`，使用正式 `checkpoints/m2_lora/best_model.pth` 生成 M2 定性对比图。
3. [x] 检查正式定性图、mask、overlay 和索引文件是否落盘。
4. [x] 更新 `PROJECT_STATE.md`，记录正式表格和定性图路径。

### 执行记录
- 正式表格：`analysis/m2_formal_comparison_2026-05-15.md`、`paper/tables/m2_formal_comparison.csv`。
- 默认定性图：`paper/figures/m2_qualitative/`，对应 mask/overlay/index 位于 `results/visualizations/m2_qualitative/`。
- 补充混合前景样本图：`paper/figures/m2_qualitative_mixed/`，对应 mask/overlay/index 位于 `results/visualizations/m2_qualitative_mixed/`。
- 补充检查发现默认 6 个样本 GT 前景比例接近全图，已额外选择 mosaic 与高前景样本覆盖约 0.1% 到 97% 前景比例，用于排查高指标是否由空 mask 或全图前景驱动。

## 15. 本次任务计划（2026-05-15，M2 定性图质量复查）
1. [x] 扩展候选图数量，生成分层候选：`paper/figures/m2_qualitative_candidates/`。
2. [x] 计算候选图 GT/预测面积比例与 IoU，定位定性图质量差的原因。
3. [x] 抽查原始 annotation，确认存在“白底黑裂缝”和“黑底白裂缝”两种标注极性。
4. [x] 生成 mask 极性审计记录，并更新 `PROJECT_STATE.md`。

### 执行记录
- 候选图：`paper/figures/m2_qualitative_candidates/`。
- 候选输出索引：`results/visualizations/m2_qualitative_candidates/m2_qualitative_index.json`。
- 联系表：`analysis/figures/m2_candidate_contact_sheet_1.png`、`analysis/figures/m2_candidate_contact_sheet_2.png`。
- 极性审计：`analysis/m2_mask_polarity_audit_2026-05-15.md`。
- 结论：当前 M2 指标和论文图不能作为最终结果；下一步必须先统一 mask 极性，再重新训练、评估和出图。

## 16. 本次任务计划（2026-05-15，mask 极性修正）
1. [x] 新增统一 mask 极性归一化函数，输出“裂缝=1、背景=0”。
2. [x] 将训练 DataLoader、SAM baseline GT 读取、M2 可视化 GT 读取接入同一归一化逻辑。
3. [x] 增加单元测试覆盖白底黑裂缝、黑底白裂缝和 DataLoader/baseline 一致性。
4. [x] 运行相关测试并更新 `PROJECT_STATE.md`。

### 执行记录
- 新增：`utils/mask_utils.py`。
- 修改：`utils/data_loader.py`、`baselines/sam_vanilla/common.py`、`scripts/visualize_m2_comparison.py`。
- 新增测试：`tests/test_mask_utils.py`。
- 测试命令：`.\venv\Scripts\python.exe -m pytest tests\test_mask_utils.py tests\test_data_loader.py -v --basetemp analysis\pytest_tmp_mask_polarity`。
- 测试结果：`6 passed`。
- 注意：默认 pytest 临时目录 `C:\Users\22702\AppData\Local\Temp\pytest-of-22702` 当前无权限，需继续使用仓库内 `--basetemp`。

## 17. 本次任务计划（2026-05-15，极性修正后重新训练前自检与启动）
1. [x] 复读 `AGENTS.md`、`PROJECT_STATE.md`、`plan.md`，确认当前只执行 M2 polarityfix 训练任务。
2. [x] 自检训练入口、配置、数据读取、mask 极性、坏图过滤、SAM 权重、输出目录隔离。
3. [x] 补齐必要的预检脚本/配置，避免训练一天后才发现数据或配置错误。
4. [x] 运行单元测试、数据预检和短训练 smoke/probe。
5. [x] 全部通过后，启动 `checkpoints/m2_lora_polarityfix/` 正式训练并记录日志/PID。

### 执行记录
- 训练前预检脚本：`scripts/preflight_m2_polarityfix.py`。
- polarityfix 正式配置：`configs/m2_lora_polarityfix_config.yaml`。
- 真实 batch 冒烟脚本：`scripts/smoke_m2_polarityfix_batch.py`。
- 单元/冒烟测试：`tests/test_mask_utils.py`、`tests/test_data_loader.py`、`tests/test_train_smoke.py`，命令使用 `--basetemp analysis\pytest_tmp_preflight`，结果 `7 passed`。
- 数据预检报告：`analysis/m2_polarityfix_preflight_2026-05-15.json`，结果通过；归一化后 train/val/test 最大裂缝前景比例分别为 `0.096022`、`0.095681`、`0.095681`。
- 真实 batch smoke 报告：`analysis/m2_polarityfix_batch_smoke_2026-05-15.json`，结果通过。
- 正式训练已启动：PID=`8768`（cmd 守护进程），训练 Python 子进程已运行。
- 训练日志：`logs/m2_lora_polarityfix_train_20260515_092421.out.log`、`logs/m2_lora_polarityfix_train_20260515_092421.err.log`。
- PID/命令：`logs/m2_lora_polarityfix_train_20260515_092421.pid.txt`、`logs/m2_lora_polarityfix_train_20260515_092421.command.cmd`。
- checkpoint 目录：`checkpoints/m2_lora_polarityfix/`；本次正式训练配置快照：`checkpoints/m2_lora_polarityfix/run_config_20260515_092707.json`。
## 18. M2 polarityfix training completion and next evaluation
1. [x] Confirm formal polarityfix training finished all 50 epochs.
2. [x] Confirm `checkpoints/m2_lora_polarityfix/best_model.pth` and `last_model.pth` exist.
3. [x] Record best training-time validation metric: `val_iou=0.6330` at epoch 44.
4. [x] Run corrected formal `val` evaluation with `best_model.pth`: `results/enhanced/sam_lora_polarityfix_val_gpu.json`.
5. [x] Run corrected formal `test` evaluation with `best_model.pth`: `results/enhanced/sam_lora_polarityfix_test_gpu.json`.
6. [x] Regenerate corrected M2 comparison tables and qualitative figures.

## 19. M2 polarityfix corrected comparison artifacts
1. [x] Resume missing SAM-AMG polarityfix validation range `start_index=1200` through the end of the val split.
2. [x] Merge all SAM-AMG part JSON files into `results/baselines/sam_amg_polarityfix_val_gpu.json`.
3. [x] Update corrected comparison table and analysis report.
4. [x] Regenerate default and mixed qualitative figures with `checkpoints/m2_lora_polarityfix/best_model.pth`.
5. [x] Update `PROJECT_STATE.md` with completed work, next steps, and artifact paths.

### 执行记录
- SAM-AMG merged result: `results/baselines/sam_amg_polarityfix_val_gpu.json`，覆盖 `1591/1591` val samples，mIoU=`0.044840`，Boundary-IoU=`0.029975`。
- Corrected table/report: `paper/tables/m2_formal_comparison.csv`、`analysis/m2_formal_comparison_2026-05-16.md`。
- Corrected default figures: `paper/figures/m2_qualitative/`，mask/overlay/index 位于 `results/visualizations/m2_qualitative/`。
- Corrected mixed figures: `paper/figures/m2_qualitative_mixed/`，mask/overlay/index 位于 `results/visualizations/m2_qualitative_mixed/`。
- 下一步：人工抽查 corrected 定性图并筛选 3-4 张论文正文代表图。

## 20. M2 polarityfix qualitative figure screening
1. [x] 汇总 corrected default/mixed 定性图的 GT、SAM+LoRA 面积比例与 IoU，辅助排除空 mask、全图粘连和明显错位样本。
2. [x] 人工查看候选 panel，记录各样本可用性与问题。
3. [x] 选出 3-4 张论文正文代表图，并写入 `analysis/m2_qualitative_selection_2026-05-16.md`。
4. [x] 更新 `PROJECT_STATE.md` 的已完成、下一步和产物路径。

### 执行记录
- Contact sheet: `analysis/figures/m2_polarityfix_contact_sheet_2026-05-16.png`。
- 筛选记录：`analysis/m2_qualitative_selection_2026-05-16.md`。
- 正文候选图复制目录：`paper/figures/m2_selected/`。
- 选定正文候选 stem：`214_01_01`、`1615_07_01`、`347_01_01`、`1220_04_01`。
- `mosaic_*` 样本因拼接痕迹明显，暂作为补充/审计图，不优先放论文正文。

## 21. M2 polarityfix formal result wording
1. [x] 基于 corrected val/test 指标与 qualitative selection，整理论文结果段落草稿。
2. [x] 明确旧 `m2_lora/` 结果只作诊断，不进入最终正文结果。
3. [x] 更新 `PROJECT_STATE.md` 的当前进行中与下一步。

### 执行记录
- 结果表述草稿：`analysis/m2_result_wording_2026-05-16.md`。
- 最终 M2 主线材料应使用 polarity-fixed checkpoint、表格和图：`checkpoints/m2_lora_polarityfix/best_model.pth`、`paper/tables/m2_formal_comparison.csv`、`paper/figures/m2_selected/`。
- 旧 `checkpoints/m2_lora/` 结果只保留为问题定位记录，不进入最终论文结果。

## 22. M2 corrected result archive and ablation planning
1. [x] 将 corrected M2 定量表、正文候选图、筛选记录与结果表述集中归档到论文材料目录。
2. [x] 制定 M2 消融实验计划表，明确实验变量、配置路径、输出路径、预计成本和停止条件。
3. [x] 更新 `PROJECT_STATE.md`，把下一步推进到“创建首批消融配置并做预检”，但不启动训练。

### 执行记录
- Corrected M2 论文材料归档目录：`paper/m2_corrected/`。
- 归档索引：`paper/m2_corrected/README.md`。
- 消融计划：`analysis/m2_ablation_plan_2026-05-16.md`。
- 消融计划 CSV：`paper/tables/m2_ablation_plan.csv`。
- 下一步只做 P0 配置与预检：`m2_lora_r4`、`m2_lora_r16`；暂不启动完整训练。

## 23. M2 P0 ablation config and smoke preflight
1. [x] 创建 `configs/ablations/m2_lora_r4_polarityfix.yaml`，仅将 `lora.rank=4`、`lora.alpha=8.0`。
2. [x] 创建 `configs/ablations/m2_lora_r16_polarityfix.yaml`，仅将 `lora.rank=16`、`lora.alpha=32.0`。
3. [x] 校验两个配置相对 `configs/m2_lora_polarityfix_config.yaml` 只变更 rank/alpha。
4. [x] 运行预检测试：`tests/test_mask_utils.py`、`tests/test_data_loader.py`、`tests/test_train_smoke.py`，结果 `7 passed`。
5. [x] 对 `m2_lora_r4` 运行真实数据 one-batch smoke。
6. [x] 对 `m2_lora_r16` 运行真实数据 one-batch smoke。
7. [ ] 更新 `PROJECT_STATE.md` 并提交预检报告。

### 执行记录
- `r4` smoke 报告：`analysis/m2_ablation_lora_r4_batch_smoke_2026-05-16.json`，train_loss=`4.315238`，val_loss=`5.097844`。
- `r16` smoke 报告：`analysis/m2_ablation_lora_r16_batch_smoke_2026-05-16.json`，train_loss=`4.412747`，val_loss=`4.947453`。
- 两个 smoke 均使用真实 SAM 权重、真实 train/val batch 和 CUDA，均通过。

## 24. M2 P0 r4 full training launch
1. [x] 确认没有其他 Python 训练进程运行，避免两个 CUDA 长任务并发。
2. [x] 使用 `configs/ablations/m2_lora_r4_polarityfix.yaml` 启动完整 50 epoch 训练。
3. [x] 输出目录固定为 `checkpoints/m2_ablation_lora_r4/`，日志固定写入 `logs/m2_ablation_lora_r4_train_*.out.log` 与 `.err.log`。
4. [x] 启动后检查进程存在、日志开始写入、没有立即异常退出。
5. [x] 更新 `PROJECT_STATE.md` 并提交 Git，保留可回滚记录。

### 执行记录
- 启动方式：独立 Windows cmd 窗口，避免当前工具会话结束后清理隐藏子进程。
- 启动时间戳：`20260516_170116`。
- cmd PID：`12588`。
- 训练配置：`configs/ablations/m2_lora_r4_polarityfix.yaml`。
- checkpoint 目录：`checkpoints/m2_ablation_lora_r4/`。
- 命令记录：`logs/m2_ablation_lora_r4_train_20260516_170116.command.cmd`。
- PID/路径记录：`logs/m2_ablation_lora_r4_train_20260516_170116.pid.txt`。
- stdout：`logs/m2_ablation_lora_r4_train_20260516_170116.out.log`。
- stderr：`logs/m2_ablation_lora_r4_train_20260516_170116.err.log`。
- 初始检查：cmd 与 Python 进程均存在，日志已写出 `使用设备: cuda`；stderr 仅出现已知坏图 `226_01_01.png` 跳过警告。

## 25. M2 P0 r4 first checkpoint monitoring
1. [x] Confirm `m2_lora_r4` training process is still alive.
2. [x] Confirm first epoch summary is available in stdout.
3. [x] Confirm `best_model.pth` and `last_model.pth` exist under `checkpoints/m2_ablation_lora_r4/`.
4. [x] Record epoch 1 metrics in `PROJECT_STATE.md`.
5. [x] Clean transient Python cache directories generated by recent runs.
6. [ ] Continue monitoring until all 50 epochs complete.
7. [ ] Run val/test evaluation for `m2_lora_r4` after training completes.
8. [ ] Start `m2_lora_r16` only after r4 completion and evaluation are recorded.

### Artifacts
- Training log: `logs/m2_ablation_lora_r4_train_20260516_170116.out.log`
- Progress/stderr log: `logs/m2_ablation_lora_r4_train_20260516_170116.err.log`
- Checkpoint directory: `checkpoints/m2_ablation_lora_r4/`
- Epoch 1 metrics: `train_loss=1.1755`, `train_iou=0.5381`, `val_loss=0.9691`, `val_iou=0.5715`, `best=0.5715`
