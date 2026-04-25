"""LoRA (Low-Rank Adaptation) adapter for parameter-efficient fine-tuning.

This module implements LoRA for SAM, enabling fine-tuning with only 1-2% of
trainable parameters. LoRA injects trainable low-rank matrices into linear layers.

Reference:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """LoRA hyperparameter configuration.

    Args:
        rank: Low-rank dimension r (typically 4, 8, or 16)
        alpha: Scaling coefficient; effective scale = alpha / rank
        dropout: Dropout probability on the LoRA path
        target_modules: Substrings to match module names for LoRA injection

    Example:
        >>> config = LoRAConfig(rank=8, alpha=16.0, dropout=0.1)
        >>> print(config.scale)  # 16.0 / 8 = 2.0
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: List[str] = field(
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

    @property
    def scale(self) -> float:
        """Compute effective scaling factor."""
        return self.alpha / self.rank


# Recommended LoRA configs for different SAM components
LORA_CONFIGS: Dict[str, LoRAConfig] = {
    "image_encoder": LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["qkv", "proj"],
    ),
    "prompt_encoder": LoRAConfig(
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=["proj"],
    ),
    "mask_decoder": LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "lin1", "lin2"],
    ),
    "full": LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
    ),
}


class LoRALinear(nn.Module):
    """Linear layer augmented with LoRA bypass.

    Forward computation:
        y = W @ x + scale * (B @ A @ x)

    Where:
        - W: Frozen original weight (out_features, in_features)
        - A: Trainable low-rank matrix (rank, in_features)
        - B: Trainable low-rank matrix (out_features, rank)
        - scale: alpha / rank

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank r
        alpha: Scaling coefficient
        dropout: Dropout on LoRA input path
        bias: Whether to include bias term

    Example:
        >>> layer = LoRALinear(256, 512, rank=8, alpha=16.0)
        >>> x = torch.randn(4, 256)
        >>> y = layer(x)
        >>> print(y.shape)  # (4, 512)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank
        self.merged = False

        # Frozen original weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )

        # Frozen bias (if exists)
        self.bias_param = (
            nn.Parameter(torch.zeros(out_features), requires_grad=False)
            if bias
            else None
        )

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize LoRA weights.

        - A: Kaiming uniform (same as nn.Linear)
        - B: Zeros (ensures LoRA starts as identity)
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Base linear transformation
        base = F.linear(x, self.weight, self.bias_param)

        # If merged, LoRA is already in weight
        if self.merged:
            return base

        # LoRA path: x -> dropout -> A -> B -> scale
        lora = self.lora_dropout(x)
        lora = F.linear(lora, self.lora_A)  # (*, rank)
        lora = F.linear(lora, self.lora_B)  # (*, out_features)

        return base + self.scale * lora

    def merge_lora(self) -> None:
        """Merge LoRA weights into base weight for inference.

        After merging:
            W_merged = W + scale * (B @ A)

        This eliminates LoRA overhead during inference.
        """
        if not self.merged:
            self.weight.data += self.scale * (self.lora_B @ self.lora_A)
            self.merged = True

    def unmerge_lora(self) -> None:
        """Unmerge LoRA weights for continued training.

        Restores:
            W = W_merged - scale * (B @ A)
        """
        if self.merged:
            self.weight.data -= self.scale * (self.lora_B @ self.lora_A)
            self.merged = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> "LoRALinear":
        """Create LoRALinear from existing nn.Linear layer.

        Args:
            linear: Source linear layer
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout

        Returns:
            LoRALinear layer with copied weights

        Example:
            >>> linear = nn.Linear(256, 512)
            >>> lora_linear = LoRALinear.from_linear(linear, rank=8)
        """
        has_bias = linear.bias is not None
        layer = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
        )

        # Copy weights from original layer
        layer.weight.data.copy_(linear.weight.data)
        if has_bias and linear.bias is not None:
            layer.bias_param.data.copy_(linear.bias.data)

        # Move to same device
        layer = layer.to(linear.weight.device)

        return layer

    def extra_repr(self) -> str:
        """Extra representation for print()."""
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.3f}, merged={self.merged}"
        )


class LoRAAdapter:
    """Manages LoRA injection and lifecycle for a model.

    This class:
    - Identifies target linear layers
    - Replaces them with LoRALinear layers
    - Manages freezing/unfreezing
    - Provides parameter statistics

    Args:
        model: Model to inject LoRA into
        config: LoRA configuration

    Example:
        >>> model = MyModel()
        >>> config = LoRAConfig(rank=8, alpha=16.0)
        >>> adapter = LoRAAdapter(model, config)
        >>> adapter.inject()
        >>> adapter.freeze_base()
        >>> print(adapter.param_report())
    """

    def __init__(self, model: nn.Module, config: LoRAConfig) -> None:
        self.model = model
        self.config = config
        self._layers: Dict[str, LoRALinear] = {}
        self._injected = False

    def _should_inject(self, name: str, module: nn.Module) -> bool:
        """Check if module should be replaced with LoRA.

        Args:
            name: Module name
            module: Module instance

        Returns:
            True if should inject LoRA
        """
        # Must be nn.Linear
        if not isinstance(module, nn.Linear):
            return False

        # If no target modules specified, inject all Linear layers
        if not self.config.target_modules:
            return True

        # Check if name matches any target module substring
        return any(target in name for target in self.config.target_modules)

    def inject(self) -> "LoRAAdapter":
        """Inject LoRA into target linear layers.

        Returns:
            Self for method chaining
        """
        if self._injected:
            return self

        replacements = []

        # Find all target modules
        for full_name, module in self.model.named_modules():
            if not self._should_inject(full_name, module):
                continue

            # Parse parent and child names
            parts = full_name.rsplit(".", 1)
            parent_name = parts[0] if len(parts) > 1 else ""
            child_name = parts[-1]

            # Get parent module
            parent = self.model
            if parent_name:
                for p in parent_name.split("."):
                    parent = getattr(parent, p)

            replacements.append((parent, child_name, module, full_name))

        # Replace with LoRALinear
        for parent, child_name, linear, full_name in replacements:
            lora_layer = LoRALinear.from_linear(
                linear,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
            )
            setattr(parent, child_name, lora_layer)
            self._layers[full_name] = lora_layer

        self._injected = True
        return self

    def freeze_base(self) -> "LoRAAdapter":
        """Freeze base model weights, only train LoRA parameters.

        Returns:
            Self for method chaining
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_A" in name or "lora_B" in name
        return self

    def unfreeze_all(self) -> "LoRAAdapter":
        """Unfreeze all parameters.

        Returns:
            Self for method chaining
        """
        for param in self.model.parameters():
            param.requires_grad = True
        return self

    def merge_all(self) -> "LoRAAdapter":
        """Merge all LoRA weights into base weights.

        Returns:
            Self for method chaining
        """
        for layer in self._layers.values():
            layer.merge_lora()
        return self

    def unmerge_all(self) -> "LoRAAdapter":
        """Unmerge all LoRA weights.

        Returns:
            Self for method chaining
        """
        for layer in self._layers.values():
            layer.unmerge_lora()
        return self

    def trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_params())

    def total_count(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def lora_count(self) -> int:
        """Count LoRA parameters (A + B matrices)."""
        return sum(
            layer.lora_A.numel() + layer.lora_B.numel()
            for layer in self._layers.values()
        )

    def injected_layers(self) -> Dict[str, LoRALinear]:
        """Get dictionary of injected LoRA layers."""
        return dict(self._layers)

    def param_report(self) -> str:
        """Generate parameter efficiency report.

        Returns:
            Formatted report string
        """
        total = self.total_count()
        lora_n = self.lora_count()
        trainable = self.trainable_count()
        pct = lora_n / total * 100 if total > 0 else 0

        lines = [
            "=" * 55,
            "LoRA Parameter Efficiency Report",
            "=" * 55,
            f"  Injected layers : {len(self._layers)}",
            f"  Total params    : {total:,}",
            f"  LoRA params     : {lora_n:,}  ({pct:.3f}% of total)",
            f"  Trainable params: {trainable:,}",
            f"  LoRA rank       : {self.config.rank}",
            f"  LoRA alpha      : {self.config.alpha}",
            f"  Scale factor    : {self.config.scale:.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)


def apply_lora_to_sam(
    sam_model: nn.Module,
    config: Optional[LoRAConfig] = None,
    preset: str = "full",
    freeze: bool = True,
) -> LoRAAdapter:
    """One-shot LoRA injection for SAM models.

    Args:
        sam_model: SAM model to inject LoRA into
        config: Custom LoRA config (overrides preset)
        preset: Preset config name ('full', 'image_encoder', etc.)
        freeze: Whether to freeze base weights after injection

    Returns:
        LoRAAdapter instance

    Example:
        >>> from segment_anything import sam_model_registry
        >>> sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        >>> adapter = apply_lora_to_sam(sam, preset='full', freeze=True)
        >>> print(adapter.param_report())
    """
    cfg = config or LORA_CONFIGS.get(preset, LORA_CONFIGS["full"])
    adapter = LoRAAdapter(sam_model, cfg)
    adapter.inject()

    if freeze:
        adapter.freeze_base()

    return adapter
