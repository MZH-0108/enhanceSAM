"""SAM base model wrapper for tunnel crack segmentation.

This module provides a clean interface to load and use SAM models with
custom image sizes and preprocessing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple
from segment_anything import sam_model_registry


def load_sam_model(
    model_type: str = "vit_b",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> nn.Module:
    """Load SAM pretrained model.

    Args:
        model_type: SAM model type, one of ['vit_b', 'vit_l', 'vit_h']
        checkpoint_path: Path to SAM checkpoint file
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded SAM model

    Raises:
        ValueError: If model_type is invalid
        FileNotFoundError: If checkpoint_path doesn't exist

    Example:
        >>> sam = load_sam_model('vit_b', 'sam_vit_b_01ec64.pth')
        >>> print(sam)
    """
    if model_type not in sam_model_registry:
        raise ValueError(
            f"Invalid model_type: {model_type}. "
            f"Choose from {list(sam_model_registry.keys())}"
        )

    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam = sam.to(device)
    sam.eval()

    return sam


def patch_sam_for_img_size(sam: nn.Module, img_size: int) -> None:
    """Patch SAM model to support custom image sizes.

    SAM is pretrained on 1024x1024 images. This function adapts the
    positional embeddings for different image sizes (e.g., 512x512).

    Args:
        sam: SAM model to patch
        img_size: Target image size (must be divisible by 16)

    Raises:
        ValueError: If img_size is not divisible by 16

    Example:
        >>> sam = load_sam_model('vit_b', 'sam_vit_b_01ec64.pth')
        >>> patch_sam_for_img_size(sam, 512)
    """
    if img_size % 16 != 0:
        raise ValueError(f"img_size must be divisible by 16, got {img_size}")

    # Update image encoder
    if hasattr(sam, "image_encoder"):
        sam.image_encoder.img_size = img_size

        # Interpolate positional embeddings
        if hasattr(sam.image_encoder, "pos_embed"):
            old_size = int(sam.image_encoder.pos_embed.shape[1] ** 0.5)
            new_size = img_size // 16

            if old_size != new_size:
                pos_embed = sam.image_encoder.pos_embed
                pos_embed = pos_embed.reshape(1, old_size, old_size, -1)
                pos_embed = pos_embed.permute(0, 3, 1, 2)

                pos_embed = torch.nn.functional.interpolate(
                    pos_embed,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )

                pos_embed = pos_embed.permute(0, 2, 3, 1)
                pos_embed = pos_embed.reshape(1, -1, pos_embed.shape[-1])
                sam.image_encoder.pos_embed = pos_embed


class SAMBase(nn.Module):
    """SAM base wrapper with custom image size support.

    This class wraps the original SAM model and provides:
    - Custom image size support (512x512, 1024x1024, etc.)
    - Simplified forward interface
    - Preprocessing utilities

    Args:
        sam_model: Loaded SAM model from segment_anything
        img_size: Target image size (default: 1024)

    Example:
        >>> sam = load_sam_model('vit_b', 'sam_vit_b_01ec64.pth')
        >>> sam_base = SAMBase(sam, img_size=512)
        >>> image = torch.randn(2, 3, 512, 512)
        >>> embeddings = sam_base.encode_image(image)
    """

    def __init__(self, sam_model: nn.Module, img_size: int = 1024) -> None:
        super().__init__()
        self.sam = sam_model
        self.img_size = img_size

        # Patch for custom image size
        if img_size != 1024:
            patch_sam_for_img_size(self.sam, img_size)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to feature embeddings.

        Args:
            image: Input image tensor (B, 3, H, W), normalized

        Returns:
            Image embeddings (B, 256, H/16, W/16)

        Raises:
            ValueError: If image size doesn't match expected size
        """
        if image.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected image size {self.img_size}x{self.img_size}, "
                f"got {image.shape[-2]}x{image.shape[-1]}"
            )

        return self.sam.image_encoder(image)

    def forward(
        self,
        image: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through SAM.

        Args:
            image: Input image (B, 3, H, W)
            points: Optional point prompts (coords, labels)
            boxes: Optional box prompts (B, 4)
            masks: Optional mask prompts (B, 1, H, W)
            multimask_output: Whether to output multiple masks

        Returns:
            masks: Predicted masks (B, num_masks, H/4, W/4)
            iou_predictions: IoU predictions (B, num_masks)
        """
        # Encode image
        image_embeddings = self.encode_image(image)

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )

        # Decode masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return low_res_masks, iou_predictions

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    def __repr__(self) -> str:
        return f"SAMBase(img_size={self.img_size}, device={self.device})"
