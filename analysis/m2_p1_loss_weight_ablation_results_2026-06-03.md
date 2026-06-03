# M2 P1 loss-weight ablation results (2026-06-03)

## Inputs
- r16 val/test: `results/ablations/m2_lora_r16_val.json`, `results/ablations/m2_lora_r16_test.json`
- pos5 val/test: `results/ablations/m2_loss_pos5_val.json`, `results/ablations/m2_loss_pos5_test.json`
- pos20 val/test: `results/ablations/m2_loss_pos20_val.json`, `results/ablations/m2_loss_pos20_test.json`
- Table: `paper/tables/m2_p1_loss_weight_ablation_results.csv`

## Summary
- `m2_loss_pos5` remains the best P1 loss-weight setting among the evaluated runs for `mIoU`, `Dice`, `Precision`, `Boundary-IoU`, and loss.
- `m2_loss_pos20` increases recall, but it lowers precision and degrades `mIoU`/`Boundary-IoU` on both val and test.
- The pos20 result suggests overly high positive weighting over-expands crack predictions rather than improving usable segmentation quality.

## Metric comparison against r16

### Validation
- pos5: mIoU `+0.003205`, Precision `+0.022133`, Recall `-0.022594`, Boundary-IoU `+0.002313`.
- pos20: mIoU `-0.013181`, Precision `-0.032138`, Recall `+0.023464`, Boundary-IoU `-0.007291`.

### Test
- pos5: mIoU `+0.002502`, Precision `+0.021426`, Recall `-0.022899`, Boundary-IoU `+0.002335`.
- pos20: mIoU `-0.013883`, Precision `-0.033366`, Recall `+0.024197`, Boundary-IoU `-0.008457`.

## Generalization check
- pos20 val/test gaps are small: mIoU gap `0.004384`, Boundary-IoU gap `0.001900`.
- The degradation is therefore consistent across val/test rather than a split-specific anomaly.

## Decision note
- Prefer `m2_loss_pos5` for the current LoRA-r16 loss-weight setting.
- Do not prioritize another higher `pos_weight` run unless a qualitative review specifically shows missed cracks are more harmful than over-expansion.
- The next planned P1 branch can evaluate `m2_loss_bound0` if boundary-loss contribution still needs isolation.
