# M2 polarity-fixed result wording draft (2026-05-16)

## Suggested quantitative-result paragraph

在统一标注极性后，本文重新训练并评估了 SAM+LoRA 主线模型。验证集结果如表所示，SAM+LoRA 在 mIoU、Dice、Precision、Recall 与 Boundary-IoU 上均明显优于两条 SAM 基线。其中，SAM+LoRA 在验证集取得 mIoU 0.7080、Dice 0.8291、Boundary-IoU 0.8253，显著高于 SAM-AMG 的 mIoU 0.0448 和 SAM-Box-Oracle 的 mIoU 0.0708。由于隧道裂缝目标通常呈细长、低面积占比形态，Boundary-IoU 更能反映边界贴合质量；SAM+LoRA 的 Boundary-IoU 达到 0.8253，说明 LoRA 微调后模型能够更稳定地恢复细长裂缝结构。

## Suggested generalization paragraph

在独立测试集上，SAM+LoRA 获得 mIoU 0.7034、Dice 0.8259、Boundary-IoU 0.8226。与验证集相比，mIoU 差值约为 0.0047，Boundary-IoU 差值约为 0.0028，说明当前模型在验证集与测试集之间保持了较一致的性能表现，未观察到明显的验证集特异性过拟合现象。

## Suggested qualitative-result paragraph

定性结果建议使用 `214_01_01`、`1615_07_01`、`347_01_01` 与 `1220_04_01` 四个样本作为正文图。前两者展示较大或分叉裂缝场景，后两者展示低前景比例、低对比度细裂缝场景。视觉对比中，SAM-AMG 往往无法稳定提取连续裂缝，SAM-Box-Oracle 虽借助 GT 框提示但容易出现大面积误分割；SAM+LoRA 在多数样本中能够沿裂缝主体给出更连续的预测，尤其在边界贴合和细长结构保持方面更稳定。

## Important caveat

旧版 `checkpoints/m2_lora/` 及其 `results/enhanced/sam_lora_*_gpu.json` 是在标注极性未统一前产生的诊断结果，不应进入最终论文正文或最终对比表。最终 M2 主线应使用 `checkpoints/m2_lora_polarityfix/best_model.pth`、`paper/tables/m2_formal_comparison.csv` 和 `analysis/m2_formal_comparison_2026-05-16.md`。
