"""Unit tests for SAM base wrapper."""
import pytest
import torch
from models.sam_base import load_sam_model, patch_sam_for_img_size, SAMBase


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


class MockImageEncoder:
    """Mock image encoder."""

    def __init__(self):
        self.img_size = 1024
        self.pos_embed = torch.randn(1, 64 * 64, 768)

    def __call__(self, x):
        B = x.shape[0]
        return torch.randn(B, 256, x.shape[-2] // 16, x.shape[-1] // 16)


class MockPromptEncoder:
    """Mock prompt encoder."""

    def __call__(self, points=None, boxes=None, masks=None):
        sparse = torch.randn(1, 2, 256)
        dense = torch.randn(1, 256, 64, 64)
        return sparse, dense

    def get_dense_pe(self):
        return torch.randn(1, 256, 64, 64)


class MockMaskDecoder:
    """Mock mask decoder."""

    def __call__(self, image_embeddings, image_pe,
                 sparse_prompt_embeddings, dense_prompt_embeddings,
                 multimask_output=False):
        B = image_embeddings.shape[0]
        num_masks = 3 if multimask_output else 1
        H, W = image_embeddings.shape[-2:]
        masks = torch.randn(B, num_masks, H * 4, W * 4)
        iou = torch.randn(B, num_masks)
        return masks, iou


def test_patch_sam_for_img_size():
    """Test SAM image size patching."""
    sam = MockSAM()

    # Test valid size
    patch_sam_for_img_size(sam, 512)
    assert sam.image_encoder.img_size == 512

    # Test invalid size
    with pytest.raises(ValueError):
        patch_sam_for_img_size(sam, 500)


def test_sam_base_init():
    """Test SAMBase initialization."""
    sam = MockSAM()
    sam_base = SAMBase(sam, img_size=512)

    assert sam_base.img_size == 512
    assert sam_base.sam is sam


def test_sam_base_encode_image():
    """Test image encoding."""
    sam = MockSAM()
    sam_base = SAMBase(sam, img_size=512)

    # Valid input
    image = torch.randn(2, 3, 512, 512)
    embeddings = sam_base.encode_image(image)

    assert embeddings.shape == (2, 256, 32, 32)

    # Invalid input size
    with pytest.raises(ValueError):
        image = torch.randn(2, 3, 1024, 1024)
        sam_base.encode_image(image)


def test_sam_base_forward():
    """Test forward pass."""
    sam = MockSAM()
    sam_base = SAMBase(sam, img_size=512)

    image = torch.randn(2, 3, 512, 512)
    masks, iou_pred = sam_base.forward(image)

    assert masks.shape[0] == 2
    assert iou_pred.shape[0] == 2
    assert not torch.isnan(masks).any()
    assert not torch.isnan(iou_pred).any()


def test_sam_base_forward_multimask():
    """Test forward pass with multiple masks."""
    sam = MockSAM()
    sam_base = SAMBase(sam, img_size=512)

    image = torch.randn(2, 3, 512, 512)
    masks, iou_pred = sam_base.forward(image, multimask_output=True)

    assert masks.shape[1] == 3  # 3 masks
    assert iou_pred.shape[1] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
