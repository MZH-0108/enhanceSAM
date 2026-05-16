# M2 formal comparison (2026-05-15)

## Validation metrics

| Method | mIoU | Dice | Precision | Recall | Boundary-IoU | FPS | ms/image | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SAM-AMG | 0.012200 | 0.024107 | 0.480311 | 0.012364 | 0.017539 | 0.3170 | 3154.3141 | `results/baselines/sam_amg_val_gpu.json` |
| SAM-Box-Oracle | 0.890213 | 0.941918 | 0.988996 | 0.899119 | 0.132016 | 8.9097 | 112.2372 | `results/baselines/sam_box_oracle_val_gpu.json` |
| SAM+LoRA | 0.991246 | 0.995604 | 0.993153 | 0.998067 | 0.658638 | 50.2724 | 19.8916 | `results/enhanced/sam_lora_val_gpu.json` |

## Test-set sanity check

The formal SAM+LoRA checkpoint also keeps nearly identical performance on the held-out test split:

| Split | mIoU | Dice | Precision | Recall | Boundary-IoU | FPS | ms/image | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| test | 0.991093 | 0.995526 | 0.993046 | 0.998019 | 0.654690 | 49.8339 | 20.0667 | `results/enhanced/sam_lora_test_gpu.json` |

## Notes

- Evaluation uses the formal M2 validation split, CUDA inference, and the same metric definitions across all three methods.
- The formal SAM+LoRA checkpoint is `checkpoints/m2_lora/best_model.pth`, selected at epoch 44 with training monitor `val_iou=0.8455`.
- The training monitor IoU and `scripts/eval.py` mIoU are not directly interchangeable because they are computed by different loops/aggregation paths.
- Filename-level split leakage check found `train-val=0`, `train-test=0`, and `val-test=0` duplicate stems.
- The val/test consistency does not show an obvious overfitting signal, but the high absolute mIoU still needs qualitative visual inspection for empty-mask, alignment, and threshold artifacts.
