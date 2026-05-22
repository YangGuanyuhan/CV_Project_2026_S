"""Tests for bounding box operations."""

import pytest


def test_box_iou():
    """Test IoU calculation between two boxes."""
    # TODO: Implement when box_ops is ready
    # from src.utils.box_ops import box_iou
    #
    # box1 = np.array([0, 0, 10, 10])
    # box2 = np.array([5, 5, 15, 15])
    # iou = box_iou(box1, box2)
    # expected_iou = 25 / 175  # intersection / union
    # assert abs(iou - expected_iou) < 1e-6
    pass


def test_box_iou_identical():
    """Test IoU of identical boxes should be 1.0."""
    # TODO: Implement when box_ops is ready
    # from src.utils.box_ops import box_iou
    #
    # box = np.array([0, 0, 10, 10])
    # iou = box_iou(box, box)
    # assert abs(iou - 1.0) < 1e-6
    pass


def test_box_iou_no_overlap():
    """Test IoU of non-overlapping boxes should be 0.0."""
    # TODO: Implement when box_ops is ready
    # from src.utils.box_ops import box_iou
    #
    # box1 = np.array([0, 0, 10, 10])
    # box2 = np.array([20, 20, 30, 30])
    # iou = box_iou(box1, box2)
    # assert abs(iou - 0.0) < 1e-6
    pass


def test_nms():
    """Test Non-Maximum Suppression."""
    # TODO: Implement when box_ops is ready
    # from src.utils.box_ops import nms
    #
    # boxes = np.array([
    #     [0, 0, 10, 10],
    #     [1, 1, 11, 11],
    #     [20, 20, 30, 30],
    # ])
    # scores = np.array([0.9, 0.8, 0.7])
    # keep = nms(boxes, scores, iou_threshold=0.5)
    # assert len(keep) == 2  # First and third boxes should be kept
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
