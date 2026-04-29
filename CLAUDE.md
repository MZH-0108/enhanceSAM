# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tunnel crack segmentation system built on SAM (Segment Anything Model) with LoRA fine-tuning and boundary refinement. Target: SCI journal publication.

**Removed modules** (do NOT re-introduce): FSA (FreqSpatialAttention), DAP (DefectAwarePrompt) — both decreased IoU.

## Common Commands

```bash
# Install dependencies (install PyTorch with CUDA separately first)
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_lora_adapter.py -v

# Run a single test
pytest tests/test_enhanced_sam.py::test_enhanced_sam_forward -v

# Tests with coverage
pytest tests/ -v --cov=models --cov=utils --cov-report=html

# Lint / format
black --check .
isort --check .
flake8 .

# Type check
mypy models/
```

Training/eval scripts (`scripts/train.py`, `scripts/eval.py`, `scripts/predict.py`) are planned but not yet implemented (Phase 3).

## Architecture

```
Input Image (B, 3, 512, 512)
    |
SAM Image Encoder (ViT-B) + LoRA injected into attention/MLP layers
    | (B, 256, H/16, W/16)
    |--- SAM Mask Decoder + LoRA --> coarse masks + IoU predictions
    |--- BoundaryDetector (Sobel-init conv) --> boundary probability map
            |
    BoundaryRefineNet (3 residual iterations, guided by boundary map)
            |
    Refined mask (B, 1, H/4, W/4) logits
```

### Key modules in `models/`

- **`sam_base.py`** — Wraps `segment_anything` API. `SAMBase` handles image encoding and prompt-based forward pass. `patch_sam_for_img_size()` interpolates positional encodings for non-1024 input sizes.
- **`lora_adapter.py`** — `LoRALinear` replaces `nn.Linear` with a frozen weight + trainable low-rank A/B matrices. `LoRAAdapter` scans a model, replaces matching layers, and manages freeze/merge lifecycle. `apply_lora_to_sam()` is the one-liner entry point.
- **`boundary_refinement.py`** — `BoundaryDetector` (Sobel-init edge conv), `BoundaryRefineNet` (iterative mask refinement with residual blocks), `BoundaryLoss` (BCE + Dice + boundary-weighted BCE), `BoundaryMetrics` (IoU, Dice, boundary F-measure).
- **`enhanced_sam.py`** — `EnhancedSAM` integrates all three modules. `build_enhanced_sam(sam)` is the factory. `compute_loss()` handles target downsampling and loss computation. Training is **end-to-end without prompts**: `model(image=images)` with no points/boxes/masks.

### Data flow during training

```python
outputs = model(image=images)          # no prompts — end-to-end
loss_dict = model.compute_loss(outputs, masks)  # targets auto-downsampled
loss_dict['loss'].backward()
```

### Configuration

`EnhancedSAMConfig` (dataclass) controls everything: LoRA rank/alpha/targets, boundary channels/iterations, loss weights. `configs/train_config.yaml` has the default training hyperparameters. The `LORA_CONFIGS` dict in `lora_adapter.py` provides presets for different SAM submodules.

## Code Conventions

- PEP 8 with 88-char line width (Black)
- **Code comments in Chinese**
- Google-style docstrings with tensor shape annotations like `(B, C, H, W)`
- Type annotations on all public functions
- Import order: stdlib, third-party, local
- Configs as `@dataclass` classes
- Device management via `.to(device)`, never hardcode `.cuda()`
- Git commits: Conventional Commits format `<type>(<scope>): <subject>`

## Testing Notes

Tests use mock SAM classes (no real SAM checkpoint needed). All tests run on CPU. The mock classes are defined in each test file — `MockSAM`, `MockImageEncoder`, `MockPromptEncoder`, `MockMaskDecoder` produce random tensors of correct shapes.

## Key Design Decisions

- **Image size**: 512x512 (not SAM's default 1024), requires positional encoding interpolation
- **LoRA rank**: 8, alpha: 16.0 — injects into qkv, proj, q/k/v_proj, out_proj, lin1, lin2
- **Loss**: BCE (w=1.0) + Dice (w=2.0) + boundary-weighted BCE (w=2.0), pos_weight=10.0 for class imbalance
- **Boundary refinement**: 3 iterations, 64 mid-channels, Sobel-initialized edge detection
- **Output head zero-init**: BoundaryRefineNet's output conv starts at zero so initial refined mask equals coarse mask
