"""Unit tests for EnhancedSAM integration."""
import pytest
import torch
import torch.nn as nn
from models.enhanced_sam import (
    EnhancedSAMConfig,
    EnhancedSAM,
    build_enhanced_sam,
)


class MockSAM:
    """Mock SAM model for testing."""

    def __init__(self):
        self.image_encoder = MockImageEncoder()
        self.prompt_encoder = MockPromptEncoder()
        self.mask_decoder = MockMaskDecoder()

    def to(self, device):
        return self

    def eval(self):
        return self


class MockImageEncoder(nn.Module):
    """Mock image encoder."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 1)  # Dummy layer for parameters

    def forward(self, x):
        B = x.shape[0]
        H, W = x.shape[-2:]
        return torch.randn(B, 256, H // 16, W // 16)


class MockPromptEncoder(nn.Module):
    """Mock prompt encoder."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 256)  # Dummy layer

    def forward(self, points=None, boxes=None, masks=None):
        sparse = torch.randn(1, 2, 256)
        dense = torch.randn(1, 256, 64, 64)
        return sparse, dense

    def get_dense_pe(self):
        return torch.randn(1, 256, 64, 64)


class MockMaskDecoder(nn.Module):
    """Mock mask decoder."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(256, 256)  # Dummy layer

    def forward(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output=False,
    ):
        B = image_embeddings.shape[0]
        num_masks = 3 if multimask_output else 1
        H, W = image_embeddings.shape[-2:]
        masks = torch.randn(B, num_masks, H * 4, W * 4)
        iou = torch.randn(B, num_masks)
        return masks, iou


def test_enhanced_sam_config():
    """Test EnhancedSAMConfig."""
    config = EnhancedSAMConfig(
        vit_embed_dim=768,
        use_lora=True,
        lora_rank=8,
        use_boundary=True,
    )

    assert config.vit_embed_dim == 768
    assert config.use_lora is True
    assert config.lora_rank == 8
    assert config.use_boundary is True


def test_enhanced_sam_init():
    """Test EnhancedSAM initialization."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=True)

    model = EnhancedSAM(sam, config)

    assert model.config == config
    assert model.sam is sam
    assert model.lora_adapter is not None
    assert model.boundary_detector is not None
    assert model.boundary_refiner is not None


def test_enhanced_sam_init_no_lora():
    """Test EnhancedSAM without LoRA."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=False, use_boundary=True)

    model = EnhancedSAM(sam, config)

    assert model.lora_adapter is None
    assert model.boundary_detector is not None


def test_enhanced_sam_init_no_boundary():
    """Test EnhancedSAM without boundary refinement."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=False)

    model = EnhancedSAM(sam, config)

    assert model.lora_adapter is not None
    assert model.boundary_detector is None
    assert model.boundary_refiner is None


def test_enhanced_sam_forward():
    """Test EnhancedSAM forward pass."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=True)
    model = EnhancedSAM(sam, config)

    image = torch.randn(2, 3, 512, 512)
    outputs = model(image=image)

    assert "masks" in outputs
    assert "iou_pred" in outputs
    assert "refined_mask" in outputs
    assert "boundary_map" in outputs

    assert outputs["masks"].shape[0] == 2
    assert outputs["iou_pred"].shape[0] == 2
    assert outputs["refined_mask"].shape[0] == 2
    assert outputs["boundary_map"].shape[0] == 2


def test_enhanced_sam_forward_no_boundary():
    """Test forward pass without boundary refinement."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=False)
    model = EnhancedSAM(sam, config)

    image = torch.randn(2, 3, 512, 512)
    outputs = model(image=image)

    assert "masks" in outputs
    assert "iou_pred" in outputs
    assert "refined_mask" not in outputs
    assert "boundary_map" not in outputs


def test_enhanced_sam_forward_multimask():
    """Test forward pass with multiple masks."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=True)
    model = EnhancedSAM(sam, config)

    image = torch.randn(2, 3, 512, 512)
    outputs = model(image=image, multimask=True)

    assert outputs["masks"].shape[1] == 3  # 3 masks
    assert outputs["iou_pred"].shape[1] == 3


def test_enhanced_sam_compute_loss():
    """Test loss computation."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=True)
    model = EnhancedSAM(sam, config)

    image = torch.randn(2, 3, 512, 512)
    outputs = model(image=image)

    targets = torch.randint(0, 2, (2, 1, 512, 512)).float()
    loss_dict = model.compute_loss(outputs, targets)

    assert "loss" in loss_dict
    assert "bce" in loss_dict
    assert "dice" in loss_dict
    assert "boundary" in loss_dict

    assert loss_dict["loss"].item() > 0
    assert not torch.isnan(loss_dict["loss"])


def test_enhanced_sam_compute_loss_no_boundary():
    """Test loss computation without boundary refinement."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=False)
    model = EnhancedSAM(sam, config)

    image = torch.randn(2, 3, 512, 512)
    outputs = model(image=image)

    targets = torch.randint(0, 2, (2, 1, 512, 512)).float()
    loss_dict = model.compute_loss(outputs, targets)

    assert "loss" in loss_dict


def test_enhanced_sam_param_count():
    """Test parameter counting."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=True)
    model = EnhancedSAM(sam, config)

    total = model.total_count()
    trainable = model.trainable_count()

    assert total > 0
    assert trainable > 0
    assert trainable < total  # LoRA freezes base weights


def test_enhanced_sam_param_report():
    """Test parameter report generation."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=True)
    model = EnhancedSAM(sam, config)

    report = model.param_report()

    assert "EnhancedSAM Parameter Report" in report
    assert "Total params" in report
    assert "Trainable params" in report
    assert "LoRA" in report
    assert "Boundary" in report


def test_build_enhanced_sam():
    """Test build_enhanced_sam factory function."""
    sam = MockSAM()

    # With default config
    model = build_enhanced_sam(sam)
    assert isinstance(model, EnhancedSAM)
    assert model.config.use_lora is True
    assert model.config.use_boundary is True

    # With custom config
    config = EnhancedSAMConfig(use_lora=False, use_boundary=False)
    model = build_enhanced_sam(sam, config)
    assert model.lora_adapter is None
    assert model.boundary_detector is None


def test_enhanced_sam_trainable_params():
    """Test trainable parameters list."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=True)
    model = EnhancedSAM(sam, config)

    trainable = model.trainable_params()

    assert len(trainable) > 0
    assert all(p.requires_grad for p in trainable)


def test_enhanced_sam_with_prompts():
    """Test forward pass with various prompts."""
    sam = MockSAM()
    config = EnhancedSAMConfig(use_lora=True, use_boundary=True)
    model = EnhancedSAM(sam, config)

    image = torch.randn(2, 3, 512, 512)

    # With point prompts
    points = (torch.randn(2, 5, 2), torch.randint(0, 2, (2, 5)))
    outputs = model(image=image, points=points)
    assert "masks" in outputs

    # With box prompts
    boxes = torch.randn(2, 4)
    outputs = model(image=image, boxes=boxes)
    assert "masks" in outputs

    # With mask prompts
    masks = torch.randn(2, 1, 256, 256)
    outputs = model(image=image, masks=masks)
    assert "masks" in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
