"""Enhanced SAM: Integration of SAM + LoRA + Boundary Refinement.

This module integrates:
1. SAM base model (image encoder + mask decoder)
2. LoRA adapter for parameter-efficient fine-tuning
3. Boundary refinement for precise edge detection

Main class: EnhancedSAM
Factory: build_enhanced_sam(sam, config)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_adapter import LoRAAdapter, LoRAConfig, apply_lora_to_sam
from .boundary_refinement import (
    BoundaryDetector,
    BoundaryRefineNet,
    BoundaryLoss,
)


@dataclass
class EnhancedSAMConfig:
    """Configuration for EnhancedSAM.

    Args:
        vit_embed_dim: ViT internal dimension (768/1024/1280 for vit_b/l/h)
        prompt_embed_dim: SAM neck output dimension (always 256)
        use_lora: Whether to use LoRA fine-tuning
        lora_rank: LoRA rank r
        lora_alpha: LoRA scaling coefficient
        lora_dropout: LoRA dropout probability
        lora_target_modules: Module names to inject LoRA
        use_boundary: Whether to use boundary refinement
        boundary_mid_channels: Boundary refinement intermediate channels
        boundary_num_iterations: Number of refinement iterations
        boundary_detector_mid: Boundary detector intermediate channels
        loss_w_bce: BCE loss weight
        loss_w_dice: Dice loss weight
        loss_w_bound: Boundary loss weight
        loss_pos_weight: Positive class weight for imbalance
        loss_boundary_radius: Boundary region radius

    Example:
        >>> config = EnhancedSAMConfig(
        ...     vit_embed_dim=768,
        ...     use_lora=True,
        ...     use_boundary=True
        ... )
    """

    # Model dimensions
    vit_embed_dim: int = 768  # 768(vit_b), 1024(vit_l), 1280(vit_h)
    prompt_embed_dim: int = 256  # Always 256 for SAM

    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "qkv",
            "proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "lin1",
            "lin2",
        ]
    )

    # Boundary refinement configuration
    use_boundary: bool = True
    boundary_mid_channels: int = 64
    boundary_num_iterations: int = 3
    boundary_detector_mid: int = 64

    # Loss configuration
    loss_w_bce: float = 1.0
    loss_w_dice: float = 2.0
    loss_w_bound: float = 2.0
    loss_pos_weight: float = 10.0
    loss_boundary_radius: int = 3


class EnhancedSAM(nn.Module):
    """Enhanced SAM for tunnel crack segmentation.

    Integrates SAM with LoRA fine-tuning and boundary refinement.

    Args:
        sam: Original SAM model from segment_anything
        config: EnhancedSAMConfig

    Forward inputs:
        image: Input image (B, 3, H, W), preprocessed
        points: Optional point prompts (coords, labels)
        boxes: Optional box prompts (B, 4)
        masks: Optional mask prompts (B, 1, H, W)
        multimask: Whether to output multiple masks

    Forward outputs:
        Dictionary with keys:
            - masks: Predicted masks (B, num_masks, H/4, W/4) logits
            - iou_pred: IoU predictions (B, num_masks)
            - refined_mask: Refined mask (B, 1, H/4, W/4) logits (if use_boundary)
            - boundary_map: Boundary probability (B, 1, H/4, W/4) (if use_boundary)

    Example:
        >>> from segment_anything import sam_model_registry
        >>> sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        >>> config = EnhancedSAMConfig()
        >>> model = EnhancedSAM(sam, config)
        >>> image = torch.randn(2, 3, 512, 512)
        >>> outputs = model(image=image)
    """

    def __init__(self, sam: nn.Module, config: EnhancedSAMConfig) -> None:
        super().__init__()
        self.config = config
        self.sam = sam

        # 1. Apply LoRA to SAM
        self.lora_adapter: Optional[LoRAAdapter] = None
        if config.use_lora:
            lora_cfg = LoRAConfig(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
            )
            self.lora_adapter = apply_lora_to_sam(sam, config=lora_cfg, freeze=True)

        # 2. Boundary refinement modules
        self.boundary_detector: Optional[BoundaryDetector] = None
        self.boundary_refiner: Optional[BoundaryRefineNet] = None

        if config.use_boundary:
            self.boundary_detector = BoundaryDetector(
                in_channels=config.prompt_embed_dim,
                mid_channels=config.boundary_detector_mid,
            )
            self.boundary_refiner = BoundaryRefineNet(
                feat_channels=config.prompt_embed_dim,
                mid_channels=config.boundary_mid_channels,
                num_iterations=config.boundary_num_iterations,
            )

        # 3. Loss function
        self.criterion = BoundaryLoss(
            w_bce=config.loss_w_bce,
            w_dice=config.loss_w_dice,
            w_bound=config.loss_w_bound,
            boundary_radius=config.loss_boundary_radius,
            pos_weight=config.loss_pos_weight,
        )

    def forward(
        self,
        image: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through EnhancedSAM.

        Args:
            image: Input image (B, 3, H, W)
            points: Optional (coords, labels) tuple
            boxes: Optional box prompts (B, 4)
            masks: Optional mask prompts (B, 1, H, W)
            multimask: Whether to output multiple masks

        Returns:
            Dictionary with masks, iou_pred, and optionally refined_mask, boundary_map
        """
        # Image encoding
        image_embeddings = self.sam.image_encoder(image)  # (B, 256, H/16, W/16)

        # Prompt encoding
        sparse_emb, dense_emb = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # Mask decoding
        low_res_masks, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=multimask,
        )

        out = {
            "masks": low_res_masks,  # (B, num_masks, H/4, W/4)
            "iou_pred": iou_pred,  # (B, num_masks)
        }

        # Boundary refinement
        if self.boundary_detector is not None and self.boundary_refiner is not None:
            # Detect boundaries
            boundary_map = self.boundary_detector(image_embeddings)

            # Select best mask for refinement
            best_idx = iou_pred.argmax(dim=1, keepdim=True)  # (B, 1)
            best_mask = low_res_masks.gather(
                1,
                best_idx.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, *low_res_masks.shape[-2:]),
            )  # (B, 1, H/4, W/4)

            # Refine mask
            refined = self.boundary_refiner(
                coarse_mask=best_mask,
                boundary_map=boundary_map,
                features=image_embeddings,
            )

            out["refined_mask"] = refined
            out["boundary_map"] = boundary_map

        return out

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for training.

        Args:
            outputs: Dictionary from forward()
            targets: Ground truth masks (B, 1, H, W) binary

        Returns:
            Dictionary with loss components
        """
        # Get prediction (use refined mask if available)
        if "refined_mask" in outputs:
            pred = outputs["refined_mask"]
        else:
            # Use best mask from multi-mask output
            best_idx = outputs["iou_pred"].argmax(dim=1, keepdim=True)
            H, W = outputs["masks"].shape[-2:]
            pred = outputs["masks"].gather(
                1, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            )

        # Resize target to match prediction
        H, W = pred.shape[-2:]
        tgt_resized = F.interpolate(targets.float(), size=(H, W), mode="nearest")

        # Compute loss
        return self.criterion(pred, tgt_resized)

    def trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_params())

    def total_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def param_report(self) -> str:
        """Generate parameter report.

        Returns:
            Formatted report string
        """
        total = self.total_count()
        trainable = self.trainable_count()
        pct = trainable / total * 100 if total > 0 else 0
        lora_n = self.lora_adapter.lora_count() if self.lora_adapter else 0

        modules = []
        if self.config.use_lora:
            modules.append("LoRA")
        if self.config.use_boundary:
            modules.append("Boundary")

        lines = [
            "=" * 60,
            "EnhancedSAM Parameter Report",
            "=" * 60,
            f"  Total params     : {total:,}",
            f"  Trainable params : {trainable:,}  ({pct:.3f}%)",
            f"  LoRA params      : {lora_n:,}",
            f"  Modules          : SAM + {' + '.join(modules)}",
            "=" * 60,
        ]
        return "\n".join(lines)


def build_enhanced_sam(
    sam: nn.Module,
    config: Optional[EnhancedSAMConfig] = None,
) -> EnhancedSAM:
    """Build EnhancedSAM from a SAM model.

    Args:
        sam: SAM model from segment_anything
        config: EnhancedSAMConfig (uses defaults if None)

    Returns:
        EnhancedSAM instance

    Example:
        >>> from segment_anything import sam_model_registry
        >>> sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        >>> model = build_enhanced_sam(sam)
        >>> print(model.param_report())
    """
    if config is None:
        config = EnhancedSAMConfig()

    return EnhancedSAM(sam, config)
