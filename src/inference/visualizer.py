"""Visualization utilities for detection results."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Color palette for different categories (BGR format for OpenCV)
COLORS = [
    (0, 0, 255),  # red
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 165, 255),  # orange
    (128, 0, 128),  # purple
    (0, 128, 0),  # dark green
    (128, 128, 0),  # teal
]


def get_color(index: int) -> tuple[int, int, int]:
    """Get a color from the palette by index."""
    return COLORS[index % len(COLORS)]


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray | list,
    scores: list[float],
    phrases: list[str],
    box_format: str = "xyxy",
    score_threshold: float = 0.0,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: Input image as numpy array (H, W, 3) in BGR format.
        boxes: Array of bounding boxes.
        scores: List of confidence scores.
        phrases: List of category/phrase labels.
        box_format: Format of boxes ('xyxy' or 'xywh').
        score_threshold: Only draw boxes above this score.
        thickness: Line thickness for boxes.
        font_scale: Font scale for labels.

    Returns:
        Annotated image (copy of input).
    """
    vis_image = image.copy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

    # Track unique phrases for color assignment
    phrase_colors: dict[str, tuple[int, int, int]] = {}
    color_idx = 0

    for _i, (box, score, phrase) in enumerate(zip(boxes, scores, phrases, strict=False)):
        if score < score_threshold:
            continue

        if box_format == "xyxy":
            x1, y1, x2, y2 = box
        elif box_format == "xywh":
            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            raise ValueError(f"Unknown box format: {box_format}")

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Assign consistent color per phrase
        if phrase not in phrase_colors:
            phrase_colors[phrase] = get_color(color_idx)
            color_idx += 1
        color = phrase_colors[phrase]

        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        label = f"{phrase}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(
            vis_image,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        # Draw label text
        cv2.putText(
            vis_image,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis_image


def save_annotated_image(
    image: np.ndarray,
    output_path: str,
    boxes: np.ndarray | list | None = None,
    scores: list[float] | None = None,
    phrases: list[str] | None = None,
    **draw_kwargs,
) -> str:
    """Save an annotated image to disk.

    Args:
        image: Input image as numpy array (H, W, 3) in BGR format.
        output_path: Path to save the annotated image.
        boxes: Optional bounding boxes to draw.
        scores: Optional confidence scores.
        phrases: Optional phrase labels.
        **draw_kwargs: Additional arguments passed to draw_boxes().

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vis_image = image.copy()

    if boxes is not None and len(boxes) > 0:
        vis_image = draw_boxes(
            vis_image,
            boxes=boxes,
            scores=scores or [],
            phrases=phrases or [],
            **draw_kwargs,
        )

    cv2.imwrite(str(output_path), vis_image)
    logger.info("Saved annotated image to %s", output_path)
    return str(output_path)


def draw_gt_and_predictions(
    image: np.ndarray,
    gt_boxes: np.ndarray | list | None = None,
    gt_labels: list[str] | None = None,
    pred_boxes: np.ndarray | list | None = None,
    pred_scores: list[float] | None = None,
    pred_labels: list[str] | None = None,
) -> np.ndarray:
    """Draw both ground truth and prediction boxes on an image.

    GT boxes are drawn in green (dashed), prediction boxes in red (solid).

    Args:
        image: Input image (H, W, 3) BGR.
        gt_boxes: Ground truth boxes (xyxy format).
        gt_labels: GT category labels.
        pred_boxes: Prediction boxes (xyxy format).
        pred_scores: Prediction confidence scores.
        pred_labels: Prediction category labels.

    Returns:
        Annotated image.
    """
    vis_image = image.copy()

    # Draw GT boxes in green dashed style
    if gt_boxes is not None:
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # Draw dashed rectangle
            _draw_dashed_rect(vis_image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if gt_labels and i < len(gt_labels):
                cv2.putText(
                    vis_image,
                    f"GT: {gt_labels[i]}",
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 0),
                    1,
                )

    # Draw prediction boxes in red solid style
    if pred_boxes is not None and len(pred_boxes) > 0:
        pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
        for _i, (box, score, label) in enumerate(
            zip(pred_boxes, pred_scores or [], pred_labels or [], strict=False)
        ):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 220), 2)
            cv2.putText(
                vis_image,
                f"{label}: {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 220),
                1,
            )

    return vis_image


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_len: int = 10,
) -> None:
    """Draw a dashed rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top edge
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_len)
    # Bottom edge
    _draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_len)
    # Left edge
    _draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_len)
    # Right edge
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_len)


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_len: int = 10,
) -> None:
    """Draw a dashed line on an image."""
    dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
    if dist == 0:
        return

    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    drawn = 0.0
    while drawn < dist:
        start_x = int(pt1[0] + dx * drawn)
        start_y = int(pt1[1] + dy * drawn)
        end = min(drawn + dash_len, dist)
        end_x = int(pt1[0] + dx * end)
        end_y = int(pt1[1] + dy * end)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
        drawn += dash_len * 2
