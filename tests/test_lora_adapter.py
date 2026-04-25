"""Unit tests for LoRA adapter."""
import pytest
import torch
import torch.nn as nn
from models.lora_adapter import (
    LoRAConfig,
    LoRALinear,
    LoRAAdapter,
    apply_lora_to_sam,
    LORA_CONFIGS,
)


def test_lora_config():
    """Test LoRAConfig dataclass."""
    config = LoRAConfig(rank=8, alpha=16.0, dropout=0.1)

    assert config.rank == 8
    assert config.alpha == 16.0
    assert config.scale == 2.0  # 16.0 / 8
    assert config.dropout == 0.1
    assert len(config.target_modules) > 0


def test_lora_config_presets():
    """Test preset configurations."""
    assert "full" in LORA_CONFIGS
    assert "image_encoder" in LORA_CONFIGS
    assert "mask_decoder" in LORA_CONFIGS

    full_config = LORA_CONFIGS["full"]
    assert full_config.rank == 8
    assert full_config.alpha == 16.0


def test_lora_linear_init():
    """Test LoRALinear initialization."""
    layer = LoRALinear(256, 512, rank=8, alpha=16.0, dropout=0.1)

    assert layer.in_features == 256
    assert layer.out_features == 512
    assert layer.rank == 8
    assert layer.scale == 2.0
    assert not layer.merged

    # Check parameter shapes
    assert layer.weight.shape == (512, 256)
    assert layer.lora_A.shape == (8, 256)
    assert layer.lora_B.shape == (512, 8)

    # Check requires_grad
    assert not layer.weight.requires_grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


def test_lora_linear_forward():
    """Test LoRALinear forward pass."""
    layer = LoRALinear(256, 512, rank=8, alpha=16.0, dropout=0.0)
    x = torch.randn(4, 256)

    y = layer(x)

    assert y.shape == (4, 512)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_lora_linear_merge():
    """Test LoRA weight merging."""
    layer = LoRALinear(256, 512, rank=8, alpha=16.0, dropout=0.0)
    x = torch.randn(4, 256)

    # Output before merge
    y_before = layer(x)

    # Merge
    layer.merge_lora()
    assert layer.merged

    # Output after merge should be same
    y_after = layer(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)


def test_lora_linear_unmerge():
    """Test LoRA weight unmerging."""
    layer = LoRALinear(256, 512, rank=8, alpha=16.0, dropout=0.0)
    x = torch.randn(4, 256)

    # Original output
    y_original = layer(x)

    # Merge and unmerge
    layer.merge_lora()
    layer.unmerge_lora()
    assert not layer.merged

    # Output should be same as original
    y_restored = layer(x)
    assert torch.allclose(y_original, y_restored, atol=1e-5)


def test_lora_linear_from_linear():
    """Test creating LoRALinear from nn.Linear."""
    linear = nn.Linear(256, 512)
    x = torch.randn(4, 256)

    # Get original output
    with torch.no_grad():
        y_original = linear(x)

    # Convert to LoRALinear
    lora_linear = LoRALinear.from_linear(linear, rank=8, alpha=16.0, dropout=0.0)

    # Initially, LoRA should be identity (B initialized to zeros)
    with torch.no_grad():
        y_lora = lora_linear(x)

    # Should be very close (LoRA starts as identity)
    assert torch.allclose(y_original, y_lora, atol=1e-5)


class SimpleModel(nn.Module):
    """Simple model for testing LoRAAdapter."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(256, 768)
        self.proj = nn.Linear(768, 256)
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        x = self.qkv(x)
        x = self.proj(x)
        x = self.fc(x)
        return x


def test_lora_adapter_inject():
    """Test LoRA injection."""
    model = SimpleModel()
    config = LoRAConfig(rank=8, alpha=16.0, target_modules=["qkv", "proj"])

    adapter = LoRAAdapter(model, config)
    adapter.inject()

    # Check that qkv and proj are replaced
    assert isinstance(model.qkv, LoRALinear)
    assert isinstance(model.proj, LoRALinear)

    # fc should not be replaced (not in target_modules)
    assert isinstance(model.fc, nn.Linear)
    assert not isinstance(model.fc, LoRALinear)

    # Check injected layers
    injected = adapter.injected_layers()
    assert "qkv" in injected
    assert "proj" in injected
    assert len(injected) == 2


def test_lora_adapter_freeze():
    """Test freezing base weights."""
    model = SimpleModel()
    config = LoRAConfig(rank=8, alpha=16.0, target_modules=["qkv", "proj"])

    adapter = LoRAAdapter(model, config)
    adapter.inject()
    adapter.freeze_base()

    # Check that only LoRA parameters are trainable
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


def test_lora_adapter_param_count():
    """Test parameter counting."""
    model = SimpleModel()
    config = LoRAConfig(rank=8, alpha=16.0, target_modules=["qkv", "proj"])

    total_before = sum(p.numel() for p in model.parameters())

    adapter = LoRAAdapter(model, config)
    adapter.inject()

    total_after = adapter.total_count()
    lora_params = adapter.lora_count()

    # Total should increase (LoRA params added)
    assert total_after > total_before

    # LoRA params should be reasonable
    # qkv: (8, 256) + (768, 8) = 2048 + 6144 = 8192
    # proj: (8, 768) + (256, 8) = 6144 + 2048 = 8192
    # Total: 16384
    assert lora_params == 16384


def test_lora_adapter_merge_all():
    """Test merging all LoRA layers."""
    model = SimpleModel()
    config = LoRAConfig(rank=8, alpha=16.0, target_modules=["qkv", "proj"])

    adapter = LoRAAdapter(model, config)
    adapter.inject()

    x = torch.randn(4, 256)
    y_before = model(x)

    # Merge all
    adapter.merge_all()

    y_after = model(x)

    # Output should be same
    assert torch.allclose(y_before, y_after, atol=1e-5)

    # Check that layers are merged
    assert model.qkv.merged
    assert model.proj.merged


def test_lora_adapter_param_report():
    """Test parameter report generation."""
    model = SimpleModel()
    config = LoRAConfig(rank=8, alpha=16.0, target_modules=["qkv"])

    adapter = LoRAAdapter(model, config)
    adapter.inject()

    report = adapter.param_report()

    assert "LoRA Parameter Efficiency Report" in report
    assert "Injected layers" in report
    assert "Total params" in report
    assert "LoRA params" in report


def test_apply_lora_to_sam():
    """Test one-shot LoRA application."""
    model = SimpleModel()

    adapter = apply_lora_to_sam(model, preset="full", freeze=True)

    # Check that LoRA is injected
    assert len(adapter.injected_layers()) > 0

    # Check that base is frozen
    trainable = adapter.trainable_count()
    lora_params = adapter.lora_count()

    # Only LoRA params should be trainable
    assert trainable == lora_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
