"""Boundary refinement module for precise crack edge detection.

This module implements:
- BoundaryDetector: Detects crack boundaries using Sobel-initialized convolutions
- BoundaryRefineNet: Iteratively refines masks guided by boundary maps
- BoundaryLoss: Boundary-aware loss function
- BoundaryMetrics: Evaluation metrics for boundary quality
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BoundaryDetector(nn.Module):
    """Lightweight boundary detection branch.

    Uses Sobel-initialized convolutions to detect edges in feature maps.

    Args:
        in_channels: Input feature channels (typically 256 for SAM)
        mid_channels: Intermediate channels (default: 64)

    Input:
        Feature map (B, C, H, W)

    Output:
        Boundary probability map (B, 1, H, W) in [0, 1]

    Example:
        >>> detector = BoundaryDetector(in_channels=256, mid_channels=64)
        >>> features = torch.randn(2, 256, 64, 64)
        >>> boundary_map = detector(features)
        >>> print(boundary_map.shape)  # (2, 1, 64, 64)
    """

    def __init__(self, in_channels: int, mid_channels: int = 64) -> None:
        super().__init__()

        # Edge detection convolution (Sobel-initialized)
        self.edge_conv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )

        # Detection head
        self.head = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, mid_channels // 2, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=1, bias=True),
        )

        # Initialize with Sobel filters
        self._init_edge_weights(in_channels, mid_channels)

    def _init_edge_weights(self, in_c: int, out_c: int) -> None:
        """Initialize edge convolution with Sobel filters.

        Args:
            in_c: Input channels
            out_c: Output channels
        """
        with torch.no_grad():
            w = self.edge_conv.weight.data

            # Kaiming initialization as base
            nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")

            # Add Sobel filters to first half (horizontal) and second half (vertical)
            half = out_c // 2

            # Sobel horizontal: [[-1,-2,-1], [0,0,0], [1,2,1]]
            sobel_h = (
                torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
                .view(1, 3, 3)
                / 4.0
            )

            # Sobel vertical: [[-1,0,1], [-2,0,2], [-1,0,1]]
            sobel_v = (
                torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                .view(1, 3, 3)
                / 4.0
            )

            # Add Sobel to weights
            for i in range(min(half, out_c)):
                w[i] = w[i] + sobel_h.expand(in_c, -1, -1)
            for i in range(half, out_c):
                w[i] = w[i] + sobel_v.expand(in_c, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, C, H, W)

        Returns:
            Boundary probability map (B, 1, H, W)
        """
        x = self.edge_conv(x)
        x = self.head(x)
        return torch.sigmoid(x)


class _RefineBlock(nn.Module):
    """Residual refinement block.

    Args:
        channels: Number of channels
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        return self.relu(x + self.conv(x))


class BoundaryRefineNet(nn.Module):
    """Iterative mask refinement guided by boundary maps.

    Args:
        feat_channels: Feature map channels (default: 256)
        mid_channels: Intermediate channels (default: 64)
        num_iterations: Number of refinement iterations (default: 3)

    Input:
        coarse_mask: Initial mask logits (B, 1, H, W)
        boundary_map: Boundary probability (B, 1, H, W)
        features: Encoder features (B, C, H', W')

    Output:
        refined_mask: Refined mask logits (B, 1, H, W)

    Example:
        >>> refiner = BoundaryRefineNet(feat_channels=256, mid_channels=64)
        >>> coarse = torch.randn(2, 1, 128, 128)
        >>> boundary = torch.randn(2, 1, 128, 128)
        >>> features = torch.randn(2, 256, 128, 128)
        >>> refined = refiner(coarse, boundary, features)
    """

    def __init__(
        self,
        feat_channels: int = 256,
        mid_channels: int = 64,
        num_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations

        # Input projection: concat(coarse_mask, boundary_map, features)
        in_c = 1 + 1 + feat_channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_c, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Refinement blocks
        self.refine_blocks = nn.ModuleList(
            [_RefineBlock(mid_channels) for _ in range(num_iterations)]
        )

        # Output head (residual connection)
        self.out_head = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)

        # Initialize output head to zeros (start as identity)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        coarse_mask: torch.Tensor,
        boundary_map: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with iterative refinement.

        Args:
            coarse_mask: Initial mask logits (B, 1, H, W)
            boundary_map: Boundary probability (B, 1, H, W)
            features: Encoder features (B, C, H', W')

        Returns:
            Refined mask logits (B, 1, H, W)
        """
        H, W = coarse_mask.shape[-2:]

        # Resize features and boundary to match coarse_mask
        if features.shape[-2:] != (H, W):
            features = F.interpolate(
                features, size=(H, W), mode="bilinear", align_corners=False
            )

        if boundary_map.shape[-2:] != (H, W):
            boundary_map = F.interpolate(
                boundary_map, size=(H, W), mode="bilinear", align_corners=False
            )

        # Concatenate inputs
        x = torch.cat([coarse_mask, boundary_map, features], dim=1)
        x = self.input_proj(x)

        # Iterative refinement
        for block in self.refine_blocks:
            x = block(x)

        # Residual connection with coarse mask
        return coarse_mask + self.out_head(x)


class BoundaryLoss(nn.Module):
    """Combined boundary-aware segmentation loss.

    Loss = w_bce * BCE + w_dice * Dice + w_bound * BoundaryBCE

    Args:
        w_bce: Weight for standard BCE loss (default: 1.0)
        w_dice: Weight for Dice loss (default: 2.0)
        w_bound: Weight for boundary-weighted BCE (default: 2.0)
        boundary_radius: Dilation radius for boundary region (default: 3)
        pos_weight: Positive class weight for imbalance (default: 10.0)

    Example:
        >>> criterion = BoundaryLoss(w_bce=1.0, w_dice=2.0, w_bound=2.0)
        >>> pred = torch.randn(2, 1, 128, 128)
        >>> target = torch.randint(0, 2, (2, 1, 128, 128)).float()
        >>> loss_dict = criterion(pred, target)
        >>> print(loss_dict['loss'])
    """

    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 2.0,
        w_bound: float = 2.0,
        boundary_radius: int = 3,
        pos_weight: float = 10.0,
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_bound = w_bound
        self.radius = boundary_radius
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    @staticmethod
    def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
            eps: Epsilon for numerical stability

        Returns:
            Dice loss (scalar)
        """
        pred_f = pred.flatten(1)
        tgt_f = target.flatten(1)
        inter = (pred_f * tgt_f).sum(dim=1)
        union = pred_f.sum(dim=1) + tgt_f.sum(dim=1)
        dice = (2.0 * inter + eps) / (union + eps)
        return (1.0 - dice).mean()

    @staticmethod
    def get_boundary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Extract boundary region via dilation.

        Args:
            mask: Binary mask (B, 1, H, W)
            radius: Dilation radius

        Returns:
            Boundary mask (B, 1, H, W)
        """
        k = 2 * radius + 1
        kernel = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype)
        dilated = F.conv2d(mask.float(), kernel, padding=radius)
        boundary = (dilated > 0).float() - mask.float()
        return boundary

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        boundary_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred_logits: Predicted logits (B, 1, H, W)
            target: Ground truth binary mask (B, 1, H, W) in {0, 1}
            boundary_gt: Optional precomputed boundary mask

        Returns:
            Dictionary with keys:
                - loss: Total loss
                - bce: BCE loss component
                - dice: Dice loss component
                - boundary: Boundary BCE component
        """
        pred_prob = torch.sigmoid(pred_logits)
        tgt = target.float()

        # Standard BCE loss
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, tgt, pos_weight=self.pos_weight
        )

        # Dice loss
        dice = self._dice_loss(pred_prob, tgt)

        # Boundary-weighted BCE
        if boundary_gt is None:
            boundary_gt = self.get_boundary_mask(target, self.radius)

        # Weight map: 1.0 for background, 5.0 for boundary
        boundary_weight = 1.0 + boundary_gt.float() * 4.0

        bnd = F.binary_cross_entropy_with_logits(
            pred_logits, tgt, weight=boundary_weight, pos_weight=self.pos_weight
        )

        # Combined loss
        total = self.w_bce * bce + self.w_dice * dice + self.w_bound * bnd

        return {
            "loss": total,
            "bce": bce.detach(),
            "dice": dice.detach(),
            "boundary": bnd.detach(),
        }


class BoundaryMetrics:
    """Evaluation metrics for boundary quality.

    Provides:
    - IoU (Intersection over Union)
    - Dice coefficient
    - Boundary F-measure
    """

    @staticmethod
    def iou(
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute IoU (Intersection over Union).

        Args:
            pred_mask: Predicted probabilities (B, 1, H, W)
            gt_mask: Ground truth (B, 1, H, W)
            threshold: Binarization threshold
            eps: Epsilon for numerical stability

        Returns:
            IoU per sample (B,)
        """
        pred = (pred_mask >= threshold).float()
        gt = gt_mask.float()

        inter = (pred * gt).sum(dim=(1, 2, 3))
        union = (pred + gt).clamp(max=1).sum(dim=(1, 2, 3))

        return inter / (union + eps)

    @staticmethod
    def dice(
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Dice coefficient.

        Args:
            pred_mask: Predicted probabilities (B, 1, H, W)
            gt_mask: Ground truth (B, 1, H, W)
            threshold: Binarization threshold
            eps: Epsilon for numerical stability

        Returns:
            Dice per sample (B,)
        """
        pred = (pred_mask >= threshold).float()
        gt = gt_mask.float()

        inter = (pred * gt).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))

        return (2.0 * inter + eps) / (denom + eps)

    @staticmethod
    def boundary_f_measure(
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        radius: int = 3,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Compute Boundary F-measure (BF-score).

        Args:
            pred_mask: Predicted probabilities (B, 1, H, W)
            gt_mask: Ground truth (B, 1, H, W)
            radius: Boundary tolerance radius
            threshold: Binarization threshold

        Returns:
            Boundary F-measure per sample (B,)
        """
        pred_bin = (pred_mask >= threshold).float()

        # Extract boundaries
        pred_bound = BoundaryLoss.get_boundary_mask(pred_bin, radius)
        gt_bound = BoundaryLoss.get_boundary_mask(gt_mask, radius)

        # Dilate boundaries for tolerance
        k = 2 * radius + 1
        kernel = torch.ones(1, 1, k, k, device=pred_mask.device, dtype=pred_mask.dtype)

        pred_dil = (F.conv2d(pred_bound, kernel, padding=radius) > 0).float()
        gt_dil = (F.conv2d(gt_bound, kernel, padding=radius) > 0).float()

        # Precision and recall
        tp_p = (pred_bound * gt_dil).sum(dim=(1, 2, 3))
        tp_r = (gt_bound * pred_dil).sum(dim=(1, 2, 3))

        prec = tp_p / (pred_bound.sum(dim=(1, 2, 3)) + 1e-6)
        rec = tp_r / (gt_bound.sum(dim=(1, 2, 3)) + 1e-6)

        # F-measure
        return 2 * prec * rec / (prec + rec + 1e-6)
