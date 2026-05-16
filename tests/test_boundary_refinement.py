"""Unit tests for boundary refinement module."""
import pytest
import torch
from models.boundary_refinement import (
    BoundaryDetector,
    BoundaryRefineNet,
    BoundaryLoss,
    BoundaryMetrics,
)


def test_boundary_detector_init():
    """Test BoundaryDetector initialization."""
    detector = BoundaryDetector(in_channels=256, mid_channels=64)

    assert detector.edge_conv.in_channels == 256
    assert detector.edge_conv.out_channels == 64


def test_boundary_detector_forward():
    """Test BoundaryDetector forward pass."""
    detector = BoundaryDetector(in_channels=256, mid_channels=64)
    features = torch.randn(2, 256, 64, 64)

    boundary_map = detector(features)

    assert boundary_map.shape == (2, 1, 64, 64)
    assert (boundary_map >= 0).all() and (boundary_map <= 1).all()
    assert not torch.isnan(boundary_map).any()


def test_boundary_refine_net_init():
    """Test BoundaryRefineNet initialization."""
    refiner = BoundaryRefineNet(feat_channels=256, mid_channels=64, num_iterations=3)

    assert refiner.num_iterations == 3
    assert len(refiner.refine_blocks) == 3


def test_boundary_refine_net_forward():
    """Test BoundaryRefineNet forward pass."""
    refiner = BoundaryRefineNet(feat_channels=256, mid_channels=64, num_iterations=3)

    coarse_mask = torch.randn(2, 1, 128, 128)
    boundary_map = torch.randn(2, 1, 128, 128)
    features = torch.randn(2, 256, 128, 128)

    refined_mask = refiner(coarse_mask, boundary_map, features)

    assert refined_mask.shape == (2, 1, 128, 128)
    assert not torch.isnan(refined_mask).any()


def test_boundary_refine_net_resize():
    """Test BoundaryRefineNet with different input sizes."""
    refiner = BoundaryRefineNet(feat_channels=256, mid_channels=64)

    coarse_mask = torch.randn(2, 1, 128, 128)
    boundary_map = torch.randn(2, 1, 64, 64)  # Different size
    features = torch.randn(2, 256, 32, 32)  # Different size

    refined_mask = refiner(coarse_mask, boundary_map, features)

    # Should resize to match coarse_mask
    assert refined_mask.shape == (2, 1, 128, 128)


def test_boundary_loss_init():
    """Test BoundaryLoss initialization."""
    criterion = BoundaryLoss(w_bce=1.0, w_dice=2.0, w_bound=2.0, pos_weight=10.0)

    assert criterion.w_bce == 1.0
    assert criterion.w_dice == 2.0
    assert criterion.w_bound == 2.0
    assert criterion.pos_weight == 10.0


def test_boundary_loss_forward():
    """Test BoundaryLoss forward pass."""
    criterion = BoundaryLoss()

    pred_logits = torch.randn(2, 1, 128, 128)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()

    loss_dict = criterion(pred_logits, target)

    assert "loss" in loss_dict
    assert "bce" in loss_dict
    assert "dice" in loss_dict
    assert "boundary" in loss_dict

    assert loss_dict["loss"].item() > 0
    assert not torch.isnan(loss_dict["loss"])


def test_boundary_loss_dice():
    """Test Dice loss computation."""
    criterion = BoundaryLoss()

    # Perfect prediction
    pred = torch.ones(2, 1, 128, 128)
    target = torch.ones(2, 1, 128, 128)

    dice_loss = criterion._dice_loss(pred, target)

    # Should be close to 0 for perfect prediction
    assert dice_loss.item() < 0.01


def test_boundary_loss_get_boundary_mask():
    """Test boundary mask extraction."""
    # Create a simple mask
    mask = torch.zeros(1, 1, 10, 10)
    mask[0, 0, 3:7, 3:7] = 1.0  # 4x4 square

    boundary = BoundaryLoss.get_boundary_mask(mask, radius=1)

    # Boundary should be around the square
    assert boundary.sum() > 0
    assert boundary[0, 0, 4, 4] == 0  # Inside should be 0
    assert boundary[0, 0, 2, 3] > 0  # Outside edge should be > 0


def test_boundary_metrics_iou():
    """Test IoU computation."""
    # Perfect prediction
    pred = torch.ones(2, 1, 128, 128)
    target = torch.ones(2, 1, 128, 128)

    iou = BoundaryMetrics.iou(pred, target, threshold=0.5)

    assert iou.shape == (2,)
    assert (iou == 1.0).all()  # Perfect IoU

    # No overlap
    pred = torch.ones(2, 1, 128, 128)
    target = torch.zeros(2, 1, 128, 128)

    iou = BoundaryMetrics.iou(pred, target, threshold=0.5)

    assert (iou == 0.0).all()  # Zero IoU


def test_boundary_metrics_dice():
    """Test Dice coefficient computation."""
    # Perfect prediction
    pred = torch.ones(2, 1, 128, 128)
    target = torch.ones(2, 1, 128, 128)

    dice = BoundaryMetrics.dice(pred, target, threshold=0.5)

    assert dice.shape == (2,)
    assert (dice == 1.0).all()  # Perfect Dice

    # No overlap
    pred = torch.ones(2, 1, 128, 128)
    target = torch.zeros(2, 1, 128, 128)

    dice = BoundaryMetrics.dice(pred, target, threshold=0.5)

    assert (dice < 0.01).all()  # Near-zero Dice


def test_boundary_metrics_boundary_f_measure():
    """Test Boundary F-measure computation."""
    # Create masks with boundaries
    pred = torch.zeros(2, 1, 128, 128)
    pred[:, :, 30:98, 30:98] = 1.0

    target = torch.zeros(2, 1, 128, 128)
    target[:, :, 32:96, 32:96] = 1.0  # Slightly different

    bf_score = BoundaryMetrics.boundary_f_measure(pred, target, radius=3)

    assert bf_score.shape == (2,)
    assert (bf_score > 0).all()
    assert (bf_score <= 1.0).all()


def test_boundary_loss_with_precomputed_boundary():
    """Test BoundaryLoss with precomputed boundary."""
    criterion = BoundaryLoss()

    pred_logits = torch.randn(2, 1, 128, 128)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    boundary_gt = BoundaryLoss.get_boundary_mask(target, radius=3)

    loss_dict = criterion(pred_logits, target, boundary_gt=boundary_gt)

    assert "loss" in loss_dict
    assert loss_dict["loss"].item() > 0


def test_boundary_detector_sobel_init():
    """Test that Sobel initialization doesn't cause NaN."""
    detector = BoundaryDetector(in_channels=256, mid_channels=64)

    # Check weights are not NaN
    assert not torch.isnan(detector.edge_conv.weight).any()

    # Forward pass should work
    features = torch.randn(2, 256, 64, 64)
    output = detector(features)

    assert not torch.isnan(output).any()


def test_refine_block():
    """Test _RefineBlock residual connection."""
    from models.boundary_refinement import _RefineBlock

    block = _RefineBlock(channels=64)
    x = torch.randn(2, 64, 32, 32)

    y = block(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
