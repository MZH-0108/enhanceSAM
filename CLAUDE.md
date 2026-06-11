# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Enhanced SAM for **tunnel crack segmentation** — a research codebase (target: Chinese-core journal submission) that fine-tunes Meta's Segment Anything Model (SAM, ViT-B) with **LoRA** for parameter-efficient training (~1.69% of params trainable), plus an optional **boundary refinement** branch. End-to-end, prompt-free crack segmentation.

## Workflow conventions (read these before coding)

This repo is driven by two living documents that take precedence over inferring intent from code:

- **`PROJECT_STATE.md`** — the source of truth for what to work on. Before any coding task, read it and execute **only** the item under "当前进行中" (Current In Progress); do not parallelize or scope-drift. After finishing, write back: completed work, next 1–3 steps, and artifact paths. If requirements change mid-task, update "冻结决策" (Frozen Decisions) first, then implement.
- **`AGENTS.md`** — repository guidelines (style, testing, commits).

Other conventions:
- **Chinese comments are mandatory** on new/changed core code: explain what the code does, its inputs/outputs, and *why*; annotate tensor shape changes for complex transforms. The entire codebase is documented this way.
- **Commits**: `type(scope): subject` (e.g. `feat(models): add boundary loss weighting`). Keep commits small; run tests first.
- **Frozen decision (current)**: LoRA is the main line; the **boundary branch is deferred** — `configs/train_config.yaml` sets `boundary.use_boundary: false` even though `EnhancedSAMConfig` defaults it to `True`. Drive behavior from the YAML, not the dataclass defaults.
- **Unified metric reporting** across all methods: mIoU, Dice, Precision, Recall, Boundary-IoU, FPS.
- **Mask polarity**: a recurring correctness concern (see the many `*polarityfix*` configs/analyses). Masks are binary `0=background / 255=crack`; keep prediction/target polarity consistent when adding eval or loss code.

## Commands

Environment (Windows shell in `venv`, but this agent runs bash — use forward slashes):
```bash
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
python scripts/download_models.py              # fetch SAM pretrained weights
```

Tests / lint / format:
```bash
pytest tests -v                                # full suite
pytest tests -v --cov=models --cov=utils       # with coverage
pytest tests/test_lora_adapter.py -v           # single module
pytest tests/test_lora_adapter.py::test_name   # single test
black models tests && isort models tests && flake8 models tests
```
Tests use synthetic tensors / mock models and must **not** download external weights; assert shapes plus NaN/Inf safety for numerical code. `tests/test_train_smoke.py` is a 1-epoch end-to-end smoke test on a lightweight fake model.

Main pipeline:
```bash
python scripts/train.py   --data_root data/ --sam_checkpoint sam_vit_b_01ec64.pth --config configs/train_config.yaml --output_dir checkpoints/
python scripts/eval.py    --checkpoint checkpoints/best_model.pth --sam_checkpoint sam_vit_b_01ec64.pth --split val
python scripts/predict.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth --sam_checkpoint sam_vit_b_01ec64.pth --output out.png
```

Baselines (for paper comparison, same data split + metric definitions):
```bash
# SAM vanilla — AMG (real auto mode) and Box-Oracle (upper bound under GT-derived box prompts)
python baselines/sam_vanilla/eval_amg.py        --data_root data --split val --sam_checkpoint <pth> --output results/baselines/sam_amg_val.json
python baselines/sam_vanilla/eval_box_oracle.py --data_root data --split val --sam_checkpoint <pth> --output results/baselines/sam_box_oracle_val.json
# CNN baselines (UNet / DeepLabV3-like), configs in configs/baselines/
python scripts/train_cnn_baseline.py --config configs/baselines/unet_polarityfix.yaml --output_dir checkpoints/cnn_unet
python scripts/eval_cnn_baseline.py  --checkpoint <pth> --config <yaml> --split val
```

**Windows note**: keep `training.num_workers: 0` — multiprocess DataLoader workers can silently die during background training on Windows.

## Architecture

The model is assembled in layers; understanding one file in isolation won't show the full picture.

- **`models/sam_base.py`** — loads the official SAM via `sam_model_registry`. `patch_sam_for_img_size` interpolates ViT positional embeddings so SAM (default 1024) can run at the training resolution (512). Training always patches when `img_size != 1024`.
- **`models/lora_adapter.py`** — `LoRALinear` implements `y = Wx + (α/r)·B(Ax)` with frozen `W`, trainable `A` (Kaiming) and `B` (zero-init, so the LoRA path is a no-op at init). `LoRAAdapter` walks `named_modules()` and **replaces `nn.Linear` layers whose name contains any `target_modules` substring** (qkv, proj, q/k/v/out_proj, lin1, lin2) via `setattr` on the parent, then freezes everything except `lora_A`/`lora_B`. `merge_lora()` folds the low-rank update into `W` for zero-overhead inference. `LORA_CONFIGS` holds per-submodule presets; `apply_lora_to_sam(..., preset=...)` is the one-shot entry point.
- **`models/boundary_refinement.py`** — `BoundaryDetector` (Sobel-initialized) predicts a boundary map from image embeddings; `BoundaryRefineNet` iteratively refines the coarse mask (default 3 iterations); `BoundaryLoss` = weighted BCE + Dice + boundary term (`pos_weight` is large, ~10, because crack pixels are sparse). Currently inactive (see frozen decision).
- **`models/enhanced_sam.py`** — `EnhancedSAM` ties it together. `forward()`: image_encoder → prompt_encoder → mask_decoder produces coarse `masks` + `iou_pred`; if boundary is enabled it picks the highest-IoU mask and refines it. Returns a dict `{masks, iou_pred, [refined_mask, boundary_map]}`. `compute_loss()` prefers `refined_mask`, else gathers the top-`iou_pred` mask, and downsamples the target with **nearest** interpolation to preserve binary values. `EnhancedSAMConfig` + `build_enhanced_sam()` are the construction API. The package `__init__.py` re-exports these top-level symbols.

Output/tensor contract (keep explicit when editing): `masks` `(B, num_masks, H/4, W/4)` logits, `iou_pred` `(B, num_masks)`, `refined_mask` `(B, 1, H/4, W/4)` logits, `boundary_map` `(B, 1, H/4, W/4)`.

- **`utils/`** — `data_loader.py` (`build_dataloaders` reads `data/{train,val,test}/{images,annotations}`), `metrics.py` (`SegmentationMetricMeter`, `select_final_logits`, `to_binary`, `ensure_mask_shape`), `visualization.py`, `mask_utils.py`. `metrics`/`visualization` are the shared eval/predict backbone — reuse them rather than re-implementing IoU/overlay logic.
- **`configs/`** — `train_config.yaml` is the baseline; `ablations/` (LoRA rank r4/r16, loss-weight sweeps) and `baselines/` (UNet, DeepLabV3) are config-driven variants. Scripts dump a `run_config_<timestamp>.json` snapshot alongside checkpoints for reproducibility.
- **`analysis/`** — dated markdown + JSON experiment records (milestones M1, M2...). This is where ablation results and decisions are logged; mirror that format when recording new experiments.

## Data layout

```
data/{train,val,test}/images/        # RGB *.jpg/*.png
data/{train,val,test}/annotations/   # binary masks *.png (0=bg, 255=crack)
```
`data/`, `checkpoints/`, `logs/` are git-ignored runtime artifacts.
