# M2 polarity-fixed formal comparison (2026-05-16)

## Validation metrics

| Method | mIoU | Dice | Precision | Recall | Boundary-IoU | FPS | ms/image | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SAM-AMG | 0.044840 | 0.085830 | 0.077845 | 0.095641 | 0.029975 | 0.3512 | 2847.0416 | `results/baselines/sam_amg_polarityfix_val_gpu.json` |
| SAM-Box-Oracle | 0.070797 | 0.132233 | 0.073671 | 0.644718 | 0.179169 | 11.1135 | 89.9808 | `results/baselines/sam_box_oracle_polarityfix_val_gpu.json` |
| SAM+LoRA | 0.708027 | 0.829059 | 0.772059 | 0.895146 | 0.825317 | 50.0147 | 19.9941 | `results/enhanced/sam_lora_polarityfix_val_gpu.json` |

## Test-set sanity check

| Split | mIoU | Dice | Precision | Recall | Boundary-IoU | FPS | ms/image | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| test | 0.703368 | 0.825856 | 0.767555 | 0.893740 | 0.822562 | 50.2972 | 19.8818 | `results/enhanced/sam_lora_polarityfix_test_gpu.json` |

## Notes

- This comparison uses polarity-fixed mask loading and the corrected checkpoint `checkpoints/m2_lora_polarityfix/best_model.pth`.
- The older `checkpoints/m2_lora/` results are diagnostic only because they were produced before the annotation polarity fix.
- SAM-AMG validation was resumed from split index 1200 and merged from 13 part files using raw TP/FP/FN/TN, boundary-IoU sums, and inference time. The merged output covers all 1591 validation samples.
- Validation and test metrics remain close for SAM+LoRA: mIoU gap is about 0.004659 and Boundary-IoU gap is about 0.002755.
