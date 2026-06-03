# M2 P0 LoRA rank ablation results (2026-05-18)

## Inputs

- Baseline r8 checkpoint: `checkpoints/m2_lora_polarityfix/best_model.pth`
- r4 checkpoint: `checkpoints/m2_ablation_lora_r4/best_model.pth`
- r16 checkpoint: `checkpoints/m2_ablation_lora_r16/best_model.pth`
- r4 evaluation outputs:
  - `results/ablations/m2_lora_r4_val.json`
  - `results/ablations/m2_lora_r4_test.json`
- r16 evaluation outputs:
  - `results/ablations/m2_lora_r16_val.json`
  - `results/ablations/m2_lora_r16_test.json`
- Summary table: `paper/tables/m2_p0_rank_ablation_results.csv`

## Results

| Experiment | Rank | Split | mIoU | Dice | Precision | Recall | Boundary-IoU | FPS | Delta mIoU vs r8 | Delta Boundary-IoU vs r8 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_lora_r4 | 4 | val | 0.701067 | 0.824267 | 0.763628 | 0.895368 | 0.820465 | 50.489337 | -0.006961 | -0.004852 |
| m2_lora_r4 | 4 | test | 0.697111 | 0.821527 | 0.760079 | 0.893784 | 0.818568 | 49.969437 | -0.006257 | -0.003994 |
| m2_lora_r8 | 8 | val | 0.708027 | 0.829059 | 0.772059 | 0.895146 | 0.825317 | 50.014706 | 0.000000 | 0.000000 |
| m2_lora_r8 | 8 | test | 0.703368 | 0.825856 | 0.767555 | 0.893740 | 0.822562 | 50.297236 | 0.000000 | 0.000000 |
| m2_lora_r16 | 16 | val | 0.714335 | 0.833367 | 0.780902 | 0.893390 | 0.829867 | 49.813337 | 0.006308 | 0.004550 |
| m2_lora_r16 | 16 | test | 0.710653 | 0.830856 | 0.777628 | 0.891905 | 0.829133 | 50.551171 | 0.007285 | 0.006571 |

## Interpretation

- r4 is consistently below the r8 corrected mainline on both validation and test. The drop is modest, but it affects both region overlap and boundary overlap.
- r16 is consistently above r8 on both validation and test. The improvement is modest but stable: validation mIoU improves by `0.006308`, test mIoU by `0.007285`, validation Boundary-IoU by `0.004550`, and test Boundary-IoU by `0.006571`.
- r16 does not show an obvious validation-only overfit signal. Its val/test mIoU gap is about `0.003682`, and its Boundary-IoU gap is about `0.000733`.
- Inference speed is effectively unchanged across ranks in this setup, staying around 50 FPS.

## Decision

For the P0 rank ablation, `m2_lora_r16` is the best-performing tested rank. It should be treated as the preferred LoRA-rank setting for any next-stage loss-weight or boundary-refinement experiments, unless parameter budget is prioritized over the small metric gain.

Recommended next step: create P1 loss-weight ablation configs from the r16 setting, then run preflight and one-batch smoke before launching any full P1 training.
