"""Tests for bounding box operations."""

import numpy as np
import pytest

from src.utils.box_ops import box_cxcywh_to_xyxy, box_iou, box_xyxy_to_cxcywh, nms


def test_box_iou():
    """Test IoU calculation between two boxes."""
    box1 = np.array([[0, 0, 10, 10]], dtype=np.float64)
    box2 = np.array([[5, 5, 15, 15]], dtype=np.float64)
    iou = box_iou(box1, box2)
    expected_iou = 25 / 175  # intersection / union
    assert abs(iou[0, 0] - expected_iou) < 1e-6


def test_box_iou_identical():
    """Test IoU of identical boxes should be 1.0."""
    box = np.array([[0, 0, 10, 10]], dtype=np.float64)
    iou = box_iou(box, box)
    assert abs(iou[0, 0] - 1.0) < 1e-6


def test_box_iou_no_overlap():
    """Test IoU of non-overlapping boxes should be 0.0."""
    box1 = np.array([[0, 0, 10, 10]], dtype=np.float64)
    box2 = np.array([[20, 20, 30, 30]], dtype=np.float64)
    iou = box_iou(box1, box2)
    assert abs(iou[0, 0] - 0.0) < 1e-6


def test_nms():
    """Test Non-Maximum Suppression."""
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [20, 20, 30, 30],
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 2  # First and third boxes should be kept


def test_box_cxcywh_to_xyxy():
    """Test coordinate conversion from cxcywh to xyxy."""
    boxes = np.array([[5, 5, 10, 10]], dtype=np.float64)
    result = box_cxcywh_to_xyxy(boxes)
    expected = np.array([[0, 0, 10, 10]], dtype=np.float64)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_box_xyxy_to_cxcywh():
    """Test coordinate conversion from xyxy to cxcywh."""
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float64)
    result = box_xyxy_to_cxcywh(boxes)
    expected = np.array([[5, 5, 10, 10]], dtype=np.float64)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_box_iou_batch():
    """Test IoU with multiple boxes."""
    boxes1 = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float64)
    boxes2 = np.array([[5, 5, 15, 15], [0, 0, 10, 10]], dtype=np.float64)
    iou = box_iou(boxes1, boxes2)
    assert iou.shape == (2, 2)
    # box1[0] vs box2[0]: intersection=25, union=175
    assert abs(iou[0, 0] - 25 / 175) < 1e-6
    # box1[0] vs box2[1]: identical boxes
    assert abs(iou[0, 1] - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
