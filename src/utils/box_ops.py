"""Bounding box operations: IoU, NMS, coordinate format conversions."""

import numpy as np


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        boxes: (N, 4) array in (center_x, center_y, width, height) format.

    Returns:
        (N, 4) array in (x1, y1, x2, y2) format.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def box_xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        boxes: (N, 4) array in (x1, y1, x2, y2) format.

    Returns:
        (N, 4) array in (center_x, center_y, width, height) format.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=-1)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes1: (M, 4) array of boxes.
        boxes2: (N, 4) array of boxes.

    Returns:
        (M, N) IoU matrix.
    """
    boxes1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection coordinates
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # (M, N, 2)
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (M, N, 2)

    wh = np.clip(rb - lt, 0, None)  # (M, N, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (M, N)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / np.clip(union, 1e-6, None)
    return iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list[int]:
    """Non-Maximum Suppression.

    Args:
        boxes: (N, 4) array in (x1, y1, x2, y2) format.
        scores: (N,) array of confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of kept indices.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).flatten()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.clip(union, 1e-6, None)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
