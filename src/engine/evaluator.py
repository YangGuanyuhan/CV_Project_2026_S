"""COCO evaluation engine for Grounding DINO.

Runs inference on COCO images, converts predictions to COCO format,
and computes standard COCO metrics using pycocotools.
"""

import copy
import json
import logging
import time
from pathlib import Path

import cv2

from src.datasets.coco_categories import (
    build_coco_prompt,
    build_phrase_to_category_mapping,
)
from src.models.grounding_dino import GroundingDINOModel

logger = logging.getLogger(__name__)


class COCOEvaluator:
    """Evaluator for running Grounding DINO on COCO and computing metrics.

    Handles the full pipeline: load images -> run inference -> convert to COCO
    format -> evaluate with pycocotools -> save results.

    Example:
        >>> evaluator = COCOEvaluator(
        ...     model=model,
        ...     annotation_file="data/coco/annotations/instances_val2017.json",
        ... )
        >>> results = evaluator.evaluate(
        ...     image_dir="data/coco/val2017",
        ...     output_dir="outputs/coco_eval/full_val2017",
        ... )
    """

    def __init__(
        self,
        model: GroundingDINOModel,
        annotation_file: str,
        text_prompt: str | None = None,
    ):
        """Initialize the evaluator.

        Args:
            model: Loaded Grounding DINO model.
            annotation_file: Path to COCO annotation JSON.
            text_prompt: Text prompt for detection. If None, uses default COCO 80-class prompt.
        """
        self.model = model
        self.annotation_file = str(Path(annotation_file).resolve())
        self.text_prompt = text_prompt or build_coco_prompt()

        # Load COCO GT
        try:
            from pycocotools.coco import COCO

            self._coco_gt = COCO(self.annotation_file)
        except ImportError as e:
            raise RuntimeError(
                "pycocotools is required for COCO evaluation. Install with: pip install pycocotools"
            ) from e

        logger.info("COCO Evaluator initialized")
        logger.info("  Annotation file: %s", self.annotation_file)
        logger.info("  Prompt: %s", self.text_prompt[:80] + "...")
        logger.info("  GT images: %d", len(self._coco_gt.getImgIds()))

    def run_inference(
        self,
        image_dir: str,
        image_ids: list[int] | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        max_images: int | None = None,
    ) -> tuple[list[dict], list[int]]:
        """Run inference on COCO images and collect raw predictions.

        Args:
            image_dir: Directory containing COCO images.
            image_ids: List of image IDs to process. If None, processes all.
            box_threshold: Box confidence threshold.
            text_threshold: Text-token matching threshold.
            max_images: Maximum number of images to process.

        Returns:
            Tuple of (raw_predictions, eval_image_ids).
        """
        image_dir = Path(image_dir)

        if image_ids is None:
            image_ids = sorted(self._coco_gt.getImgIds())

        if max_images is not None:
            image_ids = image_ids[:max_images]

        eval_image_ids = list(image_ids)
        logger.info("Running inference on %d images...", len(image_ids))

        # Collect all unique phrases for mapping
        all_phrases: set[str] = set()
        raw_predictions = []

        start_time = time.time()

        for i, img_id in enumerate(image_ids):
            if (i + 1) % 100 == 0 or i == 0:
                logger.info("  Processing image %d/%d (id=%d)...", i + 1, len(image_ids), img_id)

            # Load image
            img_info = self._coco_gt.loadImgs(img_id)[0]
            img_path = image_dir / img_info["file_name"]

            if not img_path.exists():
                logger.warning("Image not found: %s", img_path)
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Failed to read image: %s", img_path)
                continue

            # Run inference
            try:
                boxes, scores, phrases = self.model.predict(
                    image=image,
                    text_prompt=self.text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            except Exception as e:
                logger.error("Inference failed for image %d: %s", img_id, e)
                continue

            all_phrases.update(phrases)

            raw_predictions.append(
                {
                    "image_id": img_id,
                    "boxes": boxes,
                    "scores": scores,
                    "phrases": phrases,
                }
            )

        elapsed = time.time() - start_time
        logger.info(
            "Inference complete: %d images in %.1fs (%.3f s/image)",
            len(raw_predictions),
            elapsed,
            elapsed / max(len(raw_predictions), 1),
        )

        # Build phrase -> category mapping
        self._phrase_mapping = build_phrase_to_category_mapping(list(all_phrases))
        logger.info("Mapped %d unique phrases to COCO categories", len(self._phrase_mapping))
        unmapped = all_phrases - set(self._phrase_mapping.keys())
        if unmapped:
            logger.warning("Unmapped phrases: %s", list(unmapped)[:20])

        return raw_predictions, eval_image_ids

    def convert_to_coco_format(self, raw_predictions: list[dict]) -> list[dict]:
        """Convert raw predictions to COCO detection result format.

        Args:
            raw_predictions: List of raw prediction dicts from run_inference().

        Returns:
            List of COCO-format result dicts with keys:
                image_id, category_id, bbox, score
        """
        coco_results = []

        for pred in raw_predictions:
            image_id = pred["image_id"]
            boxes = pred["boxes"]
            scores = pred["scores"]
            phrases = pred["phrases"]

            for box, score, phrase in zip(boxes, scores, phrases, strict=False):
                # Map phrase to COCO category_id
                category_id = self._phrase_mapping.get(phrase)
                if category_id is None:
                    continue

                # Convert xyxy to COCO xywh format
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                # Skip degenerate boxes
                if w <= 0 or h <= 0:
                    continue

                coco_results.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score),
                    }
                )

        logger.info("Converted %d detections to COCO format", len(coco_results))
        return coco_results

    def evaluate(
        self,
        coco_results: list[dict] | None = None,
        raw_predictions: list[dict] | None = None,
        eval_image_ids: list[int] | None = None,
    ) -> dict:
        """Run COCO evaluation using pycocotools.

        Args:
            coco_results: Pre-converted COCO format results. If None, uses raw_predictions.
            raw_predictions: Raw predictions to convert and evaluate.
            eval_image_ids: List of image IDs to evaluate against. If None, uses all GT images.

        Returns:
            Dictionary with evaluation metrics.
        """
        if coco_results is None:
            if raw_predictions is None:
                raise ValueError("Must provide either coco_results or raw_predictions")
            coco_results = self.convert_to_coco_format(raw_predictions)

        if not coco_results:
            logger.warning("No predictions to evaluate!")
            num_requested = (
                len(eval_image_ids) if eval_image_ids else len(self._coco_gt.getImgIds())
            )
            return {
                "metrics": {},
                "summary": "No predictions",
                "num_predictions": 0,
                "num_requested_images": num_requested,
                "num_images_with_predictions": 0,
            }

        # Load results into COCO API (use deepcopy to avoid polluting coco_results)
        from pycocotools.cocoeval import COCOeval

        coco_dt = self._coco_gt.loadRes(copy.deepcopy(coco_results))

        # Run COCOeval for bbox
        coco_eval = COCOeval(self._coco_gt, coco_dt, "bbox")

        # Restrict evaluation to the requested image IDs
        if eval_image_ids is not None:
            coco_eval.params.imgIds = eval_image_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            "AP": float(coco_eval.stats[0]),
            "AP50": float(coco_eval.stats[1]),
            "AP75": float(coco_eval.stats[2]),
            "APS": float(coco_eval.stats[3]),
            "APM": float(coco_eval.stats[4]),
            "APL": float(coco_eval.stats[5]),
            "AR1": float(coco_eval.stats[6]),
            "AR10": float(coco_eval.stats[7]),
            "AR100": float(coco_eval.stats[8]),
            "ARS": float(coco_eval.stats[9]),
            "ARM": float(coco_eval.stats[10]),
            "ARL": float(coco_eval.stats[11]),
        }

        summary_lines = [
            f"AP (IoU=0.50:0.95): {metrics['AP']:.4f}",
            f"AP50 (IoU=0.50):    {metrics['AP50']:.4f}",
            f"AP75 (IoU=0.75):    {metrics['AP75']:.4f}",
            f"APS (small):        {metrics['APS']:.4f}",
            f"APM (medium):       {metrics['APM']:.4f}",
            f"APL (large):        {metrics['APL']:.4f}",
        ]

        images_with_preds = len(set(r["image_id"] for r in coco_results))

        return {
            "metrics": metrics,
            "summary": "\n".join(summary_lines),
            "num_predictions": len(coco_results),
            "num_requested_images": len(eval_image_ids)
            if eval_image_ids
            else len(self._coco_gt.getImgIds()),
            "num_images_with_predictions": images_with_preds,
        }

    def save_results(
        self,
        output_dir: str,
        coco_results: list[dict],
        eval_results: dict,
        config: dict | None = None,
    ) -> None:
        """Save all evaluation results to disk.

        Args:
            output_dir: Directory to save results.
            coco_results: COCO-format prediction results (clean, not polluted by loadRes).
            eval_results: Evaluation metrics from evaluate().
            config: Optional config dict to save as snapshot.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions JSON (clean detection results only)
        pred_path = output_dir / "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(coco_results, f, indent=2)
        logger.info("Saved predictions to %s", pred_path)

        # Save metrics JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        logger.info("Saved metrics to %s", metrics_path)

        # Save eval log
        log_path = output_dir / "eval.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("COCO Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Annotation file: {self.annotation_file}\n")
            f.write(f"Text prompt: {self.text_prompt}\n")
            f.write(f"Num predictions: {eval_results.get('num_predictions', 0)}\n")
            f.write(f"Num requested images: {eval_results.get('num_requested_images', 0)}\n")
            f.write(
                f"Num images with predictions: {eval_results.get('num_images_with_predictions', 0)}\n\n"
            )
            f.write("Metrics:\n")
            f.write(eval_results.get("summary", "N/A") + "\n")
        logger.info("Saved eval log to %s", log_path)

        # Save config snapshot
        if config is not None:
            config_path = output_dir / "config.yaml"
            try:
                from omegaconf import OmegaConf

                if hasattr(config, "to_yaml"):
                    with open(config_path, "w", encoding="utf-8") as f:
                        f.write(config.to_yaml())
                else:
                    config_dict = OmegaConf.to_container(config, resolve=True)
                    with open(config_path, "w", encoding="utf-8") as f:
                        OmegaConf.save(config_dict, f)
            except Exception:
                with open(config_path, "w", encoding="utf-8") as f:
                    f.write(str(config))
            logger.info("Saved config to %s", config_path)

        # Save phrase mapping
        mapping_path = output_dir / "phrase_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self._phrase_mapping, f, indent=2)
        logger.info("Saved phrase mapping to %s", mapping_path)
