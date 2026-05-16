# M2 ablation plan after polarity-fixed mainline (2026-05-16)

## Baseline for all ablations

- Baseline checkpoint/result: `checkpoints/m2_lora_polarityfix/best_model.pth`
- Baseline config: `configs/m2_lora_polarityfix_config.yaml`
- Baseline validation metrics: mIoU `0.708027`, Dice `0.829059`, Boundary-IoU `0.825317`
- Baseline test metrics: mIoU `0.703368`, Dice `0.825856`, Boundary-IoU `0.822562`
- Common data/mask rule: use normalized masks where crack is `1` and background is `0`
- Common hardware/runtime rule: Windows training uses `training.num_workers: 0`

## Execution order

Run one ablation at a time. Do not start Boundary-refinement experiments until the LoRA-rank and loss-weight ablations have produced at least validation metrics.

| Phase | Experiment ID | Change vs baseline | Config path | Checkpoint dir | Val output | Test output | Purpose | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | `m2_lora_r4` | `lora.rank=4`, `lora.alpha=8.0` | `configs/ablations/m2_lora_r4_polarityfix.yaml` | `checkpoints/m2_ablation_lora_r4/` | `results/ablations/m2_lora_r4_val.json` | `results/ablations/m2_lora_r4_test.json` | Test smaller adapter capacity and parameter efficiency. | P0 |
| A | `m2_lora_r16` | `lora.rank=16`, `lora.alpha=32.0` | `configs/ablations/m2_lora_r16_polarityfix.yaml` | `checkpoints/m2_ablation_lora_r16/` | `results/ablations/m2_lora_r16_val.json` | `results/ablations/m2_lora_r16_test.json` | Test whether higher rank improves thin-crack recovery. | P0 |
| B | `m2_loss_pos5` | `loss.pos_weight=5.0` | `configs/ablations/m2_loss_pos5_polarityfix.yaml` | `checkpoints/m2_ablation_loss_pos5/` | `results/ablations/m2_loss_pos5_val.json` | `results/ablations/m2_loss_pos5_test.json` | Test whether current positive weighting over-expands cracks. | P1 |
| B | `m2_loss_pos20` | `loss.pos_weight=20.0` | `configs/ablations/m2_loss_pos20_polarityfix.yaml` | `checkpoints/m2_ablation_loss_pos20/` | `results/ablations/m2_loss_pos20_val.json` | `results/ablations/m2_loss_pos20_test.json` | Test stronger foreground weighting for recall-sensitive crack detection. | P1 |
| B | `m2_loss_bound0` | `loss.w_bound=0.0` | `configs/ablations/m2_loss_bound0_polarityfix.yaml` | `checkpoints/m2_ablation_loss_bound0/` | `results/ablations/m2_loss_bound0_val.json` | `results/ablations/m2_loss_bound0_test.json` | Verify contribution of boundary-weighted BCE while boundary module is off. | P1 |
| C | `m2_boundary_on` | `boundary.use_boundary=true` | `configs/ablations/m2_boundary_on_polarityfix.yaml` | `checkpoints/m2_ablation_boundary_on/` | `results/ablations/m2_boundary_on_val.json` | `results/ablations/m2_boundary_on_test.json` | Test explicit boundary refinement branch after LoRA mainline is stable. | P2 |

## Required preflight before each run

1. Copy from `configs/m2_lora_polarityfix_config.yaml`; change only the listed variable for that experiment.
2. Keep `training.num_workers: 0`, `training.epochs: 50`, `checkpoint.save_best: true`, and `checkpoint.monitor: "val_iou"` unless the plan is explicitly revised.
3. Run existing focused tests before the first ablation batch:
   `.\venv\Scripts\python.exe -m pytest tests\test_mask_utils.py tests\test_data_loader.py tests\test_train_smoke.py -v --basetemp analysis\pytest_tmp_ablation_preflight`
4. For each new config, run a one-batch smoke/probe or a one-epoch dry run before launching the full 50-epoch run.
5. Store command, PID, log paths, config snapshot, and checkpoint directory in `PROJECT_STATE.md`.

## Stop conditions

- Stop immediately if loss becomes NaN/Inf or validation IoU is NaN.
- Stop and inspect data/config if the first epoch cannot produce `best_model.pth` or `last_model.pth`.
- Stop low-priority phases if both P0 rank ablations are worse than baseline by more than `0.02` validation mIoU and do not improve Boundary-IoU.
- Do not use any result in final tables until both validation and test JSON files exist and point to the intended checkpoint/config.

## Expected cost

The polarity-fixed 50-epoch mainline run took roughly a working-day scale run on the current Windows/CUDA environment. Treat each full ablation as expensive. Run P0 first, then decide whether P1/P2 are worth the compute based on validation and test deltas.
