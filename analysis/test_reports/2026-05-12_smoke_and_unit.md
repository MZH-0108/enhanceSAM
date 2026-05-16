# smoke 与单测执行记录（2026-05-12）

## 1) 执行环境
- 操作系统：Windows（项目工作目录 `D:\MZH\codex\enhanceSAM`）
- 虚拟环境：`venv`（本次全部命令均使用 `venv\Scripts\python.exe`）
- Python：`3.13.12`
- PyTorch：`2.11.0+cpu`
- pytest：`9.0.3`

## 2) 依赖安装
执行命令：

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

结果：安装成功（含 `torch`、`torchvision`、`pytest`、`pytest-cov` 等）。

## 3) smoke test
执行命令：

```powershell
venv\Scripts\python.exe -m pytest tests/test_train_smoke.py -v
```

结果摘要：
- `tests/test_train_smoke.py::test_train_main_smoke_one_epoch` 通过
- 统计：`1 passed`
- 耗时：约 `7.64s`（pytest 总耗时显示）

告警信息：
- `albumentations` 版本检查出现网络握手超时告警，不影响测试通过与训练主流程 smoke 验证结论。

## 4) 结论
- 当前“在具备 torch + pytest 环境执行 smoke test 并固化结果”已完成。  
- 下一步应执行全量 `pytest tests -v`，并处理已识别的测试与实现文案不一致风险（参数报告字符串断言）。

## 5) 全量单测（执行与失败定位）
首次执行命令：

```powershell
venv\Scripts\python.exe -m pytest tests -v
```

首次结果摘要：
- `35 passed, 13 failed, 1 warning`
- 失败集中在两类：
  1. `tests/test_enhanced_sam.py` 中 `MockSAM` 不是 `nn.Module`，导致 LoRA 注入阶段缺少 `named_modules` 接口。  
  2. `tests/test_lora_adapter.py` 的参数报告断言为英文文案，而实现已是中文文案。  

## 6) 修复动作
- 修改 `tests/test_enhanced_sam.py`
  - `MockSAM` 改为继承 `nn.Module`，并保持子模块挂载方式不变。
  - 参数报告断言改为与当前实现一致的中文字段（`EnhancedSAM 参数报告`、`总参数量`、`可训练参数量`）。
- 修改 `tests/test_lora_adapter.py`
  - 参数报告断言改为中文字段（`LoRA 参数效率报告`、`注入层数`、`总参数量`、`LoRA 参数量`）。

## 7) 回归结果
复测命令：

```powershell
venv\Scripts\python.exe -m pytest tests -v
```

复测结果：
- `48 passed, 1 warning`
- 告警仍为 `albumentations` 在线版本检查超时，不影响测试通过与当前工程结论。
