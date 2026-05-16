# PROJECT STATE（动态核心）

## 0) 使用规则（每次任务必读）
1. 先读 `AGENTS.md` 与本文件，再开始执行。  
2. 每次只做“当前进行中”里的任务，不并行扩散。  
3. 完成后必须回写：`已完成`、`下一步`、`最新产物路径`。  
4. 若需求变更，先更新“冻结决策”，再改代码。  

## 1) 当前总目标（一句话）
完成隧道裂缝分割项目的可复现实验闭环，并产出可用于北大核心投稿的对比与消融结果。

## 2) 当前里程碑
M1（2026-05-05 ~ 2026-05-20）：工程闭环与规范化。

## 3) 已完成（最近）
- SAM baseline 两条线已实现：`SAM-AMG`、`SAM-Box-Oracle`。  
- `scripts/train.py`、`scripts/eval.py`、`scripts/predict.py` 已补齐。  
- `utils/data_loader.py` 已补齐并添加中文注释。  
- `utils/metrics.py`、`utils/visualization.py` 已补齐并接入 `eval/predict`。  
- 已新增训练 1-epoch smoke test：`tests/test_train_smoke.py`（轻量假模型 + 小样本）。  
- 数据集已合并并拆分到 `data/train|val|test`。  
- 项目清理完成（临时目录基本清除，仅 `.tmp` 受权限残留）。
- 已完成项目结构与代码审计，识别当前测试阻塞与一致性风险。  
- 已将执行规则与当前阶段计划固化到 `plan.md`。  
- 已在项目 `venv` 中完成依赖安装（含 `torch + pytest`），并通过 `tests/test_train_smoke.py`。  
- 已固化本次测试产物：`analysis/test_reports/2026-05-12_smoke_and_unit.md`。  
- 已完成全量单测回归：`venv` 环境下 `pytest tests -v` 结果为 `48 passed`。  
- 已修复测试用例与当前实现不一致项（`tests/test_enhanced_sam.py`、`tests/test_lora_adapter.py`）。  
- 已下载 SAM 权重 `sam_vit_b_01ec64.pth` 到 `checkpoints/` 并完成 SHA256 校验。  
- 已确认 SAM 两个 baseline 当前只有 JSON 指标结果，尚无论文可用可视化 PNG/JPG 产物。  
- 已新增 M2 论文定性对比图脚本：`scripts/visualize_m2_comparison.py`，用于统一生成 `GT`、`SAM-AMG`、`SAM-Box-Oracle`、`SAM+LoRA` 对比图。  
- 已生成 SAM 两个 baseline 的 val 定性可视化结果（6 张样本）：纯 mask、原图叠图、论文拼接图均已落盘。  
- 已定位 LoRA 训练中断原因：`data/train/images/226_01_01.png` 文件头全 0，属于损坏 PNG；train/val 图像与标注共 25300 个文件中仅发现这一处非法文件头。  
- 已修复 `utils/data_loader.py`：构建样本时跳过文件头非法的图像/标注，并使用 `np.fromfile + cv2.imdecode` 提升 Windows 路径读取稳健性。  
- 已新增 `tests/test_data_loader.py` 覆盖坏图像跳过逻辑；相关测试通过。  
- 已完成 LoRA 真实数据 1 epoch probe 训练，输出 `checkpoints/m2_lora_probe/best_model.pth`，val_iou=`0.7993`。  
- 已用 LoRA probe 权重完成 val 评估，输出 `results/enhanced/sam_lora_probe_val_gpu.json`。  
- 已产出 LoRA probe 临时对比表与定性图，作为 M2 链路跑通证明。  
- 已重新执行正式 LoRA 训练：`configs/train_config.yaml` -> `checkpoints/m2_lora/`，当前后台 PID=`10944`。  
- 为保证 Windows 断开会话后的训练稳定性，已将正式配置 `training.num_workers` 调整为 `0`，避免 DataLoader worker 子进程静默退出。  
- 正式 LoRA 训练在第 9 个 epoch 遇到 `data/train/images/940_05_01.png` 的一次 libpng 解码失败后中断；已修复 DataLoader：初始化阶段完整解码校验，运行时单样本读取失败会跳过到后续可读样本。  
- 已从 `checkpoints/m2_lora/last_model.pth` 恢复正式 LoRA 训练，当前后台 PID=`27500`。  
- 正式 LoRA 训练已完成 50 epoch，最佳 checkpoint 为 `checkpoints/m2_lora/best_model.pth`（epoch=44，训练日志最佳 `val_iou=0.8455`）。  
- 已完成正式 `SAM+LoRA` val/test 同口径评估：val mIoU=`0.991246`、Boundary-IoU=`0.658638`；test mIoU=`0.991093`、Boundary-IoU=`0.654690`。  
- 已检查数据切分文件名交叉：`train-val=0`、`train-test=0`、`val-test=0`，未发现重名泄漏。  

## 4) 当前进行中（只保留1-2项）
- 监控 M2 P0 `m2_lora_r4` 完整训练，等待首个 checkpoint 与 epoch 指标落盘；不要启动 `r16`。  

## 5) 下一步（严格按顺序）
1. 监控 `logs/m2_ablation_lora_r4_train_20260516_170116.out.log` 和 `.err.log`，确认 epoch 1 正常完成。  
2. 确认 `checkpoints/m2_ablation_lora_r4/last_model.pth` 与 `best_model.pth` 落盘后记录首轮 train/val 指标。  
3. `m2_lora_r4` 完成 50 epoch 后，再跑 val/test 评估；在此之前不要启动 `m2_lora_r16`。  

## 6) 冻结决策（防跑偏）
- 当前阶段先做 **LoRA 主线**；Boundary 分支后续再启用。  
- 2026-05-15 新增：数据集中存在两种 annotation 极性，最终实验必须统一为“裂缝=1、背景=0”；当前 `mask > 127` 训练出的 M2 指标只可作为问题定位记录，不可作为论文最终结果。  
- 所有评估口径统一：mIoU、Dice、Precision、Recall、Boundary-IoU、FPS。  
- M2 论文产物同时包含定量表与定性可视化：固定同一批 val 样本，对比 `GT`、`SAM-AMG`、`SAM-Box-Oracle`、`SAM+LoRA`，输出到 `paper/figures/` 与 `results/visualizations/`。  
- 每次执行任何任务前，必须先阅读 `AGENTS.md` 与 `PROJECT_STATE.md`，并先把本次执行计划写入 `plan.md`。  

## 7) 风险/阻塞
- Windows 下部分 pip 临时目录权限残留，可能影响临时安装流程。  
- Windows 后台训练使用多进程 DataLoader 时出现过进程静默退出；正式配置已固定 `training.num_workers: 0`。  
- `albumentations` 在线版本检查存在网络握手超时告警（不影响当前测试结论）。  
- 损坏样本 `data/train/images/226_01_01.png` 已在 DataLoader 中跳过；如后续补齐原图，可恢复为有效训练样本。  
- `data/train/images/940_05_01.png` 曾触发一次 libpng 解码失败，但复查可正常解码；当前通过运行时兜底避免偶发读取失败中断训练。  

## 8) 最新结果与代码锚点
- Baseline 代码：`baselines/sam_vanilla/`  
- 训练/评估/推理：`scripts/train.py`、`scripts/eval.py`、`scripts/predict.py`
- 当前执行计划：`plan.md`
- 最新测试记录：`analysis/test_reports/2026-05-12_smoke_and_unit.md`
- M2 论文可视化脚本：`scripts/visualize_m2_comparison.py`
- M2 定性图目标目录：`paper/figures/m2_qualitative/`
- M2 可视化 mask/index 目标目录：`results/visualizations/m2_qualitative/`
- SAM baseline 定性拼接图：`paper/figures/m2_baselines/`
- SAM baseline mask/叠图：`results/visualizations/m2_baselines/`
- LoRA probe 配置：`configs/m2_lora_probe_config.yaml`
- LoRA probe checkpoint：`checkpoints/m2_lora_probe/best_model.pth`
- LoRA probe 指标：`results/enhanced/sam_lora_probe_val_gpu.json`
- LoRA probe 对比表：`analysis/m2_probe_comparison_2026-05-13.md`、`paper/tables/m2_probe_comparison.csv`
- LoRA probe 定性图：`paper/figures/m2_qualitative_probe/`
- 正式 LoRA 训练日志：`logs/m2_lora_train_formal_detached_20260513_215158.out.log`、`logs/m2_lora_train_formal_detached_20260513_215158.err.log`
- 正式 LoRA 训练 PID/命令：`logs/m2_lora_train_formal_detached_20260513_215158.pid.txt`、`logs/m2_lora_train_formal_detached_20260513_215158.command.txt`
- 正式 LoRA 配置快照：`checkpoints/m2_lora/run_config_20260513_215204.json`
- 正式 LoRA 恢复训练日志：`logs/m2_lora_train_resume_decodefix_20260514_152808.out.log`、`logs/m2_lora_train_resume_decodefix_20260514_152808.err.log`
- 正式 LoRA 恢复训练 PID/命令：`logs/m2_lora_train_resume_decodefix_20260514_152808.pid.txt`、`logs/m2_lora_train_resume_decodefix_20260514_152808.command.txt`
- 正式 LoRA 恢复配置快照：`checkpoints/m2_lora/run_config_20260514_153121.json`
- 正式 LoRA val 指标：`results/enhanced/sam_lora_val_gpu.json`
- 正式 LoRA test 指标：`results/enhanced/sam_lora_test_gpu.json`
- 本次数据读取修复：`utils/data_loader.py`、`tests/test_data_loader.py`
- 本次修复文件：`tests/test_enhanced_sam.py`、`tests/test_lora_adapter.py`
- SAM 权重路径：`checkpoints/sam_vit_b_01ec64.pth`（SHA256: `EC2DF62732614E57411CDCF32A23FFDF28910380D03139EE0F4FCBE91EB8C912`）

## 2026-05-15 最新状态更新（M2 mask 极性修正）
- 已确认数据集中存在两种 annotation 极性：`mosaic_*` 多为黑底白裂缝，非 mosaic 多为白底黑裂缝。
- 已新增统一归一化工具：`utils/mask_utils.py`，规则为输出“裂缝=1、背景=0”。
- 已接入训练与评估数据读取：`utils/data_loader.py`。
- 已接入 SAM baseline GT 读取：`baselines/sam_vanilla/common.py`。
- 已接入 M2 定性图 GT/Box-Oracle 读取：`scripts/visualize_m2_comparison.py`。
- 已新增测试：`tests/test_mask_utils.py`。
- 已通过测试：`.\venv\Scripts\python.exe -m pytest tests\test_mask_utils.py tests\test_data_loader.py -v --basetemp analysis\pytest_tmp_mask_polarity`，结果 `6 passed`。
- 审计记录：`analysis/m2_mask_polarity_audit_2026-05-15.md`。
- 结论：旧 `checkpoints/m2_lora/best_model.pth`、`results/enhanced/sam_lora_*_gpu.json` 和旧论文图不能作为最终论文结果；必须基于修正后的 mask 极性重新训练与评估。

## 2026-05-15 最新状态更新（M2 polarityfix 训练前自检与启动）
- 已新增训练前预检脚本：`scripts/preflight_m2_polarityfix.py`，输出 `analysis/m2_polarityfix_preflight_2026-05-15.json`。
- 已新增正式训练配置：`configs/m2_lora_polarityfix_config.yaml`，保持 50 epoch、batch size 4、`num_workers=0`，输出目录由命令指定为 `checkpoints/m2_lora_polarityfix/`。
- 已新增真实 batch 冒烟脚本：`scripts/smoke_m2_polarityfix_batch.py`，输出 `analysis/m2_polarityfix_batch_smoke_2026-05-15.json`。
- 已通过关键测试：`.\venv\Scripts\python.exe -m pytest tests\test_mask_utils.py tests\test_data_loader.py tests\test_train_smoke.py -v --basetemp analysis\pytest_tmp_preflight`，结果 `7 passed`。
- 已通过数据预检：train/val/test 归一化后最大裂缝前景比例分别为 `0.096022`、`0.095681`、`0.095681`；仅 `data/train/images/226_01_01.png` 被既有坏图过滤逻辑跳过。
- 已通过真实 batch smoke：train_loss=`4.994488`，val_loss=`5.020304`；修正极性后初始 IoU 低属于未训练状态。
- 正式训练已启动并确认进入 Train 进度条：PID=`8768`（cmd 守护进程），Python 子进程运行中；日志为 `logs/m2_lora_polarityfix_train_20260515_092421.out.log`、`logs/m2_lora_polarityfix_train_20260515_092421.err.log`。
- 本次训练配置快照：`checkpoints/m2_lora_polarityfix/run_config_20260515_092707.json`。
## 2026-05-15 最新状态更新（M2 正式表格与定性图）
- 已完成正式 M2 val 对比表：`analysis/m2_formal_comparison_2026-05-15.md`、`paper/tables/m2_formal_comparison.csv`。
- 已完成正式定性对比图：`paper/figures/m2_qualitative/`；mask、overlay、index 位于 `results/visualizations/m2_qualitative/`。
- 已完成补充混合前景样本图：`paper/figures/m2_qualitative_mixed/`；mask、overlay、index 位于 `results/visualizations/m2_qualitative_mixed/`。
- 抽查统计：默认自动样本 GT 前景接近全图，补充样本覆盖约 0.1% 到 97% 前景比例；低前景 mosaic 样本中 `SAM+LoRA` 未出现全图前景输出。
- 当前进行中：人工抽查混合前景样本图，筛选可进入论文正文的 3-4 张代表图。
- 下一步：如人工抽查无错位、空 mask 或明显阈值伪影，进入 M2 正式结果归档与论文表述整理。
## 2026-05-16 Update: M2 polarityfix formal val/test evaluation completed
- Formal corrected evaluation used `checkpoints/m2_lora_polarityfix/best_model.pth` with config `configs/m2_lora_polarityfix_config.yaml`.
- Validation output: `results/enhanced/sam_lora_polarityfix_val_gpu.json`
  - `mIoU=0.708027`, `Dice=0.829059`, `Precision=0.772059`, `Recall=0.895146`, `Boundary-IoU=0.825317`, `FPS=50.014706`, `loss=0.791307`.
- Test output: `results/enhanced/sam_lora_polarityfix_test_gpu.json`
  - `mIoU=0.703368`, `Dice=0.825856`, `Precision=0.767555`, `Recall=0.893740`, `Boundary-IoU=0.822562`, `FPS=50.297236`, `loss=0.798718`.
- Generalization check: val/test metrics are close (`mIoU` gap about 0.004659; `Boundary-IoU` gap about 0.002755), so this corrected model does not show an obvious validation-only overfit signal.
- Current In Progress: rerun or rebuild corrected M2 comparison artifacts so all tables and qualitative figures use polarity-fixed masks and `checkpoints/m2_lora_polarityfix/best_model.pth`.

## 2026-05-16 Update: M2 polarityfix training completed
- Formal run completed normally: `logs/m2_lora_polarityfix_train_20260515_092421.out.log`.
- Checkpoints written under `checkpoints/m2_lora_polarityfix/`:
  - `best_model.pth` updated at 2026-05-15 22:08:50.
  - `last_model.pth` updated at 2026-05-15 23:48:59.
- Train log reached `[Epoch 050/50]`; best monitored `val_iou=0.6330`, reached at epoch 44.
- Final epoch metrics: `train_loss=0.6848`, `train_iou=0.6842`, `val_loss=0.7918`, `val_iou=0.6323`.
- Current In Progress: run corrected formal validation/test evaluation with `checkpoints/m2_lora_polarityfix/best_model.pth`, then regenerate comparison tables and qualitative figures using the corrected mask polarity pipeline.
- Important: older `checkpoints/m2_lora/` metrics and figures remain diagnostic only, because they were produced before the annotation polarity fix.

## 2026-05-16 Update: M2 polarityfix comparison artifacts rebuilt
- Resumed and completed the missing SAM-AMG polarity-fixed validation segment from index 1200 to the end of the 1591-sample validation split.
- Merged 13 SAM-AMG part files using raw TP/FP/FN/TN, Boundary-IoU sums, and inference time:
  - `results/baselines/sam_amg_polarityfix_val_gpu.json`
  - validation metrics: `mIoU=0.044840`, `Dice=0.085830`, `Precision=0.077845`, `Recall=0.095641`, `Boundary-IoU=0.029975`, `FPS=0.351242`.
- Rebuilt the corrected M2 comparison table:
  - `paper/tables/m2_formal_comparison.csv`
  - `analysis/m2_formal_comparison_2026-05-16.md`
- Regenerated corrected qualitative figures with `checkpoints/m2_lora_polarityfix/best_model.pth`:
  - `paper/figures/m2_qualitative/`
  - `results/visualizations/m2_qualitative/`
  - `paper/figures/m2_qualitative_mixed/`
  - `results/visualizations/m2_qualitative_mixed/`
- Current In Progress: manually inspect corrected qualitative figures and select 3-4 representative samples for the paper body.

## 2026-05-16 Update: M2 qualitative screening and wording completed
- Built a corrected contact sheet for visual screening:
  - `analysis/figures/m2_polarityfix_contact_sheet_2026-05-16.png`
- Selected four real-image qualitative panels for the paper body:
  - `214_01_01`
  - `1615_07_01`
  - `347_01_01`
  - `1220_04_01`
- Copied selected panels into:
  - `paper/figures/m2_selected/`
- Selection rationale and reserve samples recorded in:
  - `analysis/m2_qualitative_selection_2026-05-16.md`
- Drafted formal M2 result wording in:
  - `analysis/m2_result_wording_2026-05-16.md`
- Decision: keep `mosaic_*` panels as supplementary/audit evidence only; do not prioritize them for the main qualitative figure because visible tiling can distract from method comparison.
- Current In Progress: archive corrected M2 result materials and decide the next experiment branch before starting new runs.

## 2026-05-16 Update: M2 corrected archive and ablation plan completed
- Archived corrected M2 paper-facing materials under:
  - `paper/m2_corrected/`
- Added archive index:
  - `paper/m2_corrected/README.md`
- Added ablation plan:
  - `analysis/m2_ablation_plan_2026-05-16.md`
  - `paper/tables/m2_ablation_plan.csv`
- Planned execution order:
  - P0: LoRA rank ablations `m2_lora_r4`, `m2_lora_r16`
  - P1: loss weight ablations
  - P2: Boundary refinement branch
- Decision: do not start full ablation training until P0 config files and preflight/smoke checks pass.
- Current In Progress: create P0 ablation configs and run preflight only.

## 2026-05-16 Update: M2 P0 ablation configs and smoke preflight completed
- Added P0 LoRA-rank ablation configs:
  - `configs/ablations/m2_lora_r4_polarityfix.yaml`
  - `configs/ablations/m2_lora_r16_polarityfix.yaml`
- Verified both configs differ from `configs/m2_lora_polarityfix_config.yaml` only by `lora.rank` and `lora.alpha`.
- Preflight tests passed:
  - `.\venv\Scripts\python.exe -m pytest tests\test_mask_utils.py tests\test_data_loader.py tests\test_train_smoke.py -v --basetemp analysis\pytest_tmp_ablation_preflight`
  - result: `7 passed`, 1 known albumentations online-version warning.
- Real one-batch smoke passed for both P0 configs:
  - `analysis/m2_ablation_lora_r4_batch_smoke_2026-05-16.json`
  - `analysis/m2_ablation_lora_r16_batch_smoke_2026-05-16.json`
- Current In Progress: decide whether to start full P0 rank ablation training, beginning with `m2_lora_r4` only.

## 2026-05-16 Update: M2 P0 r4 full training launched
- Started full 50-epoch `m2_lora_r4` training with config:
  - `configs/ablations/m2_lora_r4_polarityfix.yaml`
- Output checkpoint directory:
  - `checkpoints/m2_ablation_lora_r4/`
- Launch metadata:
  - command: `logs/m2_ablation_lora_r4_train_20260516_170116.command.cmd`
  - pid/path record: `logs/m2_ablation_lora_r4_train_20260516_170116.pid.txt`
  - stdout: `logs/m2_ablation_lora_r4_train_20260516_170116.out.log`
  - stderr: `logs/m2_ablation_lora_r4_train_20260516_170116.err.log`
- Process check after launch:
  - cmd PID `12588` alive.
  - Python training process alive.
  - stdout contains `[INFO] 使用设备: cuda`.
  - stderr only shows the known `data/train/images/226_01_01.png` unreadable-image skip warning.
- Current In Progress: monitor r4 training until first checkpoint/epoch metrics are available; do not launch `m2_lora_r16` concurrently.

## 2026-05-16 Update: M2 P0 r4 first checkpoint confirmed
- Confirmed full 50-epoch `m2_lora_r4` training is still running.
- Process status:
  - cmd PID `12588` alive.
  - Python training process alive.
- First epoch completed and wrote checkpoints under:
  - `checkpoints/m2_ablation_lora_r4/best_model.pth`
  - `checkpoints/m2_ablation_lora_r4/last_model.pth`
  - `checkpoints/m2_ablation_lora_r4/run_config_20260516_170452.json`
- Epoch 1 metrics from stdout:
  - `train_loss=1.1755`
  - `train_iou=0.5381`
  - `val_loss=0.9691`
  - `val_iou=0.5715`
  - `best=0.5715`
- Stderr tail shows epoch 2 training progress and no fatal error; progress was about `29%` of epoch 2 at the latest check.
- Cleaned transient Python cache directories (`__pycache__`) generated by recent runs.
- No image artifacts were generated in this monitoring step, so no image2 inspection was required.
- Current In Progress: continue monitoring `m2_lora_r4` until full 50-epoch completion; do not launch `m2_lora_r16` until r4 completes and is evaluated.
