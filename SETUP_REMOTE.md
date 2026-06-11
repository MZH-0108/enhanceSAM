# SETUP_REMOTE.md — 远程 GPU 机器（Windows）训练环境配置清单

> 目标：在一台 **Windows + NVIDIA GPU** 的机器上，从零把本项目跑起来（训练 / 评估 / 对比实验），
> 并装好 Claude Code 以便接手后续调参与诊断。
> 所有命令默认在 **PowerShell** 中执行，项目根目录记为 `enhanceSAM\`。

---

## 0. 前提检查

```powershell
# 确认显卡和驱动（能看到 GPU 型号与 CUDA 驱动版本即可）
nvidia-smi

# 确认已装 Git
git --version
```

- 需要 NVIDIA 显卡（vit_b + LoRA 很轻量，**单卡 8GB 显存即可**，如 RTX 3060/4060 起步；4090 更宽裕）。
- 训练数据 `data\` 和 SAM 预训练权重**不在 Git 仓库里**（已被 `.gitignore` 忽略），需单独传输，见第 5、6 节。

---

## 1. 安装 Node.js 与 Claude Code

1. 安装 Node.js LTS（≥ 18）：https://nodejs.org/ 下载 Windows 安装包，或用 winget：
   ```powershell
   winget install OpenJS.NodeJS.LTS
   ```
2. 安装 Claude Code（**重开一个 PowerShell** 让 PATH 生效后）：
   ```powershell
   npm install -g @anthropic-ai/claude-code
   claude --version
   ```

---

## 2. 拉取代码仓库

任选一种（已在该机登录 `gh` 的话第一种最省事）：

```powershell
# 方式 A：gh CLI（推荐，已登录 GitHub）
gh repo clone MZH-0108/enhanceSAM
cd enhanceSAM

# 方式 B：HTTPS
git clone https://github.com/MZH-0108/enhanceSAM.git
cd enhanceSAM

# 方式 C：SSH（需在该机配置好 SSH key）
git clone git@github.com:MZH-0108/enhanceSAM.git
cd enhanceSAM
```

---

## 3. 创建 Python 虚拟环境

> ⚠️ 建议使用 **Python 3.10 或 3.11**。不要用 3.13——部分依赖（segment-anything、albumentations 等）
> 在 3.13 上可能缺少预编译 wheel。用 `py -3.11 --version` 确认本机有对应版本。

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1     # 激活后命令行前缀会出现 (venv)

# 若 PowerShell 报执行策略错误，先放开当前用户策略再激活：
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
```

---

## 4. 安装 PyTorch（GPU 版）+ 项目依赖

```powershell
# 先按 GPU 的 CUDA 版本装 PyTorch 2.6（与 nvidia-smi 显示的驱动匹配）
# —— CUDA 12.1（最常用）：
pip install torch==2.6.0+cu121 torchvision==0.21.0+cu121 --index-url https://download.pytorch.org/whl/cu121
# —— 或 CUDA 11.8：
# pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 再装其余依赖
pip install -r requirements.txt

# 验证 GPU 可用（必须打印 True）
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

**（可选，强 baseline）** 论文若需要 ImageNet 预训练的强 CNN 对比（比当前从零训练的 UNet/DeepLab 更能扛审稿）：
```powershell
pip install segmentation-models-pytorch
```

---

## 5. 准备训练数据

`data\` 被 Git 忽略，需从源机器（你的本机）传到该 GPU 机器，保持同样结构：

```
enhanceSAM\data\
  ├─ train\images\   train\annotations\
  ├─ val\images\     val\annotations\
  └─ test\images\    test\annotations\
```
- 标注为二值 PNG（裂缝 / 背景；数据加载时由 `utils\mask_utils.normalize_crack_mask` 统一成 裂缝=1、背景=0）。
- 传输方式：U 盘、`scp`、rsync、或网盘均可。传完用下面命令核对数量：
  ```powershell
  (Get-ChildItem data\val\annotations\*.png).Count   # 应为 1591
  ```

---

## 6. 下载 SAM 预训练权重

> ⚠️ README 提到的 `scripts\download_models.py` **当前不存在**，请用下面的官方直链手动下载。

```powershell
mkdir checkpoints\pretrained -Force
# ViT-B 权重（约 375MB，本项目使用）
curl.exe -L -o checkpoints\pretrained\sam_vit_b_01ec64.pth `
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
其他规格（如需）：`sam_vit_l_0b3195.pth` / `sam_vit_h_4b8939.pth`，同一目录前缀替换文件名即可。

---

## 7. 冒烟测试（确认全链路通）

```powershell
# 轻量假模型 + 小样本，1 epoch 跑通即说明环境 OK，不需要真实权重/GPU
pytest tests\test_train_smoke.py -v
```

---

## 8. 正式训练 / 评估命令

```powershell
# 训练（SAM + LoRA 主线，配置见 configs\train_config.yaml）
python scripts\train.py `
  --data_root data\ `
  --sam_checkpoint checkpoints\pretrained\sam_vit_b_01ec64.pth `
  --config configs\train_config.yaml `
  --output_dir checkpoints\m2_lora_polarityfix `
  --device cuda

# 评估（val / test）
python scripts\eval.py `
  --checkpoint checkpoints\m2_lora_polarityfix\best_model.pth `
  --sam_checkpoint checkpoints\pretrained\sam_vit_b_01ec64.pth `
  --split val `
  --output results\enhanced\sam_lora_val.json `
  --device cuda
```

### CNN baseline（论文对比，注意 Windows 专属环境变量）
```powershell
$env:TORCHDYNAMO_DISABLE="1"
$env:NO_ALBUMENTATIONS_UPDATE="1"

python scripts\train_cnn_baseline.py `
  --data_root data `
  --config configs\baselines\unet_polarityfix.yaml `
  --output_dir checkpoints\baselines\unet_polarityfix `
  --device cuda

python scripts\eval_cnn_baseline.py `
  --data_root data --split val `
  --config configs\baselines\unet_polarityfix.yaml `
  --checkpoint checkpoints\baselines\unet_polarityfix\best_model.pth `
  --output results\baselines\unet_polarityfix_val.json `
  --device cuda
```

---

## 9. Windows 专属注意事项（踩过的坑）

| 坑 | 说明与对策 |
|---|---|
| **DataLoader 多进程静默退出** | `configs\*.yaml` 中保持 `training.num_workers: 0`，否则后台训练时 worker 子进程可能无声崩溃。 |
| **torch.compile / dynamo 报错** | 运行前 `$env:TORCHDYNAMO_DISABLE="1"`。 |
| **albumentations 联网检查更新卡顿** | 运行前 `$env:NO_ALBUMENTATIONS_UPDATE="1"`。 |
| **中文/特殊路径读图失败** | 项目已用 `np.fromfile + cv2.imdecode` 规避，尽量仍把项目放在纯英文路径下。 |
| **长训练中断难恢复** | 训练支持断点恢复：`--resume checkpoints\...\last_model.pth`。 |
| **PowerShell 续行符** | 上面命令用反引号 `` ` `` 续行；若改用 cmd，请改成 `^` 或写成单行。 |

---

## 10. 启动 Claude Code 接手

```powershell
cd enhanceSAM
claude
```
启动后它会自动读取仓库根目录的 **`CLAUDE.md`**（项目架构、约定、已知坑都在里面），无需重新交代背景。

**建议第一句话告诉它当前任务**，例如：
> 先读 CLAUDE.md 和 analysis\ 下最新的实验记录。当前要解决的是“指标口径”问题：
> 给 utils\metrics.py 增量加上 macro（逐图平均）的前景 IoU/Dice，与现有 micro 指标并列输出；
> 并支持在原分辨率评估。然后用 m2_lora_polarityfix 的 checkpoint 在 val 上重新评估，
> 对比 micro 与 macro 的差距，定位效果图质量问题。

---

> 维护提示：本文件记录的是**远程训练机**的一次性环境搭建流程；
> 若依赖版本或权重获取方式变化，请同步更新本文件与 `requirements.txt`。
