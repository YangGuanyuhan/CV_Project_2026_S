"""COCO-specific visualization for predictions and ground truth."""

import logging
from pathlib import Path

import cv2
import numpy as np

from src.inference.visualizer import draw_gt_and_predictions

logger = logging.getLogger(__name__)


def visualize_coco_predictions(
    image_dir: str,
    coco_results: list[dict],
    coco_gt,
    output_dir: str,
    n_success: int = 10,
    n_failure: int = 10,
    score_threshold: float = 0.3,
) -> dict:
    """Visualize COCO predictions, separating success and failure cases.

    A "success" case is one where the model's top prediction overlaps with GT
    (IoU > 0.5). A "failure" case is one with significant false positives or
    missed detections.

    Args:
        image_dir: Path to COCO image directory.
        coco_results: List of COCO-format prediction dicts.
        coco_gt: pycocotools COCO object with ground truth.
        output_dir: Directory to save visualizations.
        n_success: Number of success cases to save.
        n_failure: Number of failure cases to save.
        score_threshold: Minimum score for displaying predictions.

    Returns:
        Dictionary with counts of saved images.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    success_dir = output_dir / "success_cases"
    failure_dir = output_dir / "failure_cases"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    # Group predictions by image_id
    from collections import defaultdict

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for r in coco_results:
        if r["score"] >= score_threshold:
            preds_by_image[r["image_id"]].append(r)

    # Get image IDs that have predictions
    image_ids = sorted(preds_by_image.keys())

    stats = {"success": 0, "failure": 0, "skipped": 0}

    for img_id in image_ids:
        if stats["success"] >= n_success and stats["failure"] >= n_failure:
            break

        # Load image
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = image_dir / img_info["file_name"]
        if not img_path.exists():
            stats["skipped"] += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            stats["skipped"] += 1
            continue

        # Get GT annotations
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            cat = coco_gt.loadCats(ann["category_id"])[0]
            gt_labels.append(cat["name"])

        # Get predictions
        preds = preds_by_image[img_id]
        pred_boxes = [
            [r["bbox"][0], r["bbox"][1], r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3]]
            for r in preds
        ]
        pred_scores = [r["score"] for r in preds]

        # Get category names for predictions
        pred_labels = []
        for r in preds:
            cat = coco_gt.loadCats(r["category_id"])[0]
            pred_labels.append(cat["name"])

        # Draw both GT and predictions
        vis_image = draw_gt_and_predictions(
            image=image,
            gt_boxes=np.array(gt_boxes) if gt_boxes else None,
            gt_labels=gt_labels,
            pred_boxes=np.array(pred_boxes) if pred_boxes else None,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
        )

        # Classify as success or failure based on prediction quality
        is_success = _is_success_case(preds, gt_anns, coco_gt)

        if is_success and stats["success"] < n_success:
            out_path = success_dir / f"success_{img_id:012d}.jpg"
            cv2.imwrite(str(out_path), vis_image)
            stats["success"] += 1
        elif not is_success and stats["failure"] < n_failure:
            out_path = failure_dir / f"failure_{img_id:012d}.jpg"
            cv2.imwrite(str(out_path), vis_image)
            stats["failure"] += 1

    logger.info(
        "Visualization: %d success, %d failure, %d skipped",
        stats["success"],
        stats["failure"],
        stats["skipped"],
    )
    return stats


def _is_success_case(preds: list[dict], gt_anns: list[dict], coco_gt) -> bool:
    """Determine if a prediction set is a 'success' case.

    A case is successful if:
    - There are predictions (not completely missed)
    - At least one prediction overlaps well with GT
    - Not too many false positives relative to GT
    """
    if not preds or not gt_anns:
        # No GT + predictions = false positive; no predictions + GT = missed
        return False

    from src.utils.box_ops import box_iou

    # Convert GT to xyxy
    gt_boxes = []
    for ann in gt_anns:
        x, y, w, h = ann["bbox"]
        gt_boxes.append([x, y, x + w, y + h])

    pred_boxes = []
    for r in preds:
        x, y, w, h = r["bbox"]
        pred_boxes.append([x, y, x + w, y + h])

    if not gt_boxes or not pred_boxes:
        return False

    iou_matrix = box_iou(np.array(pred_boxes), np.array(gt_boxes))

    # Check if at least one prediction matches GT well
    max_iou_per_pred = iou_matrix.max(axis=1)
    has_good_match = (max_iou_per_pred > 0.5).any()

    # Not too many false positives
    fp_ratio = (max_iou_per_pred < 0.3).sum() / max(len(preds), 1)

    return has_good_match and fp_ratio < 0.7


def create_coco_prompt_file(output_path: str) -> None:
    """Create and save the COCO 80-class prompt file.

    Args:
        output_path: Path to save the prompt text file.
    """
    from src.datasets.coco_categories import build_coco_prompt

    prompt = build_coco_prompt()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    logger.info("Saved COCO prompt to %s", output_path)


def create_category_mapping_file(output_path: str) -> None:
    """Create and save the COCO category mapping JSON.

    Args:
        output_path: Path to save the mapping JSON file.
    """
    import json

    from src.datasets.coco_categories import (
        CATEGORY_ID_TO_NAME,
        COCO_80_CLASSES,
    )

    mapping = {
        "num_categories": len(COCO_80_CLASSES),
        "categories": [
            {"category_id": cat_id, "name": name, "prompt_index": i}
            for i, (cat_id, name) in enumerate(CATEGORY_ID_TO_NAME.items())
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved category mapping to %s", output_path)
