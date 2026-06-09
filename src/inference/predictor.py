"""High-level predictor for Grounding DINO inference."""

import json
import logging
import time
from pathlib import Path

import cv2

from src.models.grounding_dino import GroundingDINOModel

logger = logging.getLogger(__name__)


class Predictor:
    """High-level predictor wrapping Grounding DINO for easy inference.

    Supports single image and batch inference with structured output.

    Example:
        >>> predictor = Predictor(
        ...     config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ...     checkpoint_path="checkpoints/groundingdino_swint_ogc.pth",
        ... )
        >>> result = predictor.predict("image.jpg", "person . car .")
        >>> predictor.save_results(result, "output_dir/")
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str | None = None,
    ):
        """Initialize the predictor.

        Args:
            config_path: Path to the Grounding DINO config file.
            checkpoint_path: Path to the model checkpoint.
            device: Device to run on ('cuda' or 'cpu').
        """
        self.model = GroundingDINOModel(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def predict(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> dict:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image.
            text_prompt: Text prompt for detection.
            box_threshold: Confidence threshold for detections.
            text_threshold: Threshold for text-token matching.

        Returns:
            Dictionary with keys:
                - image_path: str
                - image_size: (height, width)
                - boxes: list of [x1, y1, x2, y2]
                - scores: list of float
                - phrases: list of str
                - num_detections: int
                - inference_time: float (seconds)
        """
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        start_time = time.time()
        boxes, scores, phrases = self.model.predict(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        inference_time = time.time() - start_time

        result = {
            "image_path": image_path,
            "image_size": list(image.shape[:2]),  # (H, W)
            "boxes": boxes.tolist() if len(boxes) > 0 else [],
            "scores": scores,
            "phrases": phrases,
            "num_detections": len(boxes),
            "inference_time": round(inference_time, 4),
        }

        logger.info(
            "Predicted %d detections for %s in %.3fs",
            len(boxes),
            Path(image_path).name,
            inference_time,
        )
        return result

    def predict_batch(
        self,
        image_paths: list[str],
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """Run inference on a batch of images.

        Args:
            image_paths: List of paths to input images.
            text_prompt: Text prompt for detection.
            box_threshold: Confidence threshold for detections.
            text_threshold: Threshold for text-token matching.

        Returns:
            List of result dictionaries, one per image.
        """
        results = []
        total_start = time.time()

        for i, image_path in enumerate(image_paths):
            logger.info("Processing image %d/%d: %s", i + 1, len(image_paths), image_path)
            try:
                result = self.predict(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed to process %s: %s", image_path, e)
                results.append(
                    {
                        "image_path": image_path,
                        "error": str(e),
                        "num_detections": 0,
                    }
                )

        total_time = time.time() - total_start
        n_success = sum(1 for r in results if "error" not in r)
        logger.info(
            "Batch inference complete: %d/%d succeeded in %.1fs",
            n_success,
            len(image_paths),
            total_time,
        )
        return results

    def predict_directory(
        self,
        image_dir: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> list[dict]:
        """Run inference on all images in a directory.

        Args:
            image_dir: Path to directory containing images.
            text_prompt: Text prompt for detection.
            box_threshold: Confidence threshold for detections.
            text_threshold: Threshold for text-token matching.
            extensions: Tuple of valid image file extensions.

        Returns:
            List of result dictionaries.
        """
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {image_dir}")

        image_paths = sorted(str(p) for p in image_dir.iterdir() if p.suffix.lower() in extensions)

        if not image_paths:
            logger.warning("No images found in %s with extensions %s", image_dir, extensions)
            return []

        logger.info("Found %d images in %s", len(image_paths), image_dir)
        return self.predict_batch(
            image_paths=image_paths,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    @staticmethod
    def save_results(
        results: list[dict] | dict,
        output_dir: str,
        prefix: str = "",
    ) -> None:
        """Save prediction results to JSON files.

        Args:
            results: Single result dict or list of result dicts.
            output_dir: Directory to save output files.
            prefix: Optional prefix for output filenames.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(results, dict):
            results = [results]

        for result in results:
            image_name = Path(result.get("image_path", "unknown")).stem
            filename = (
                f"{prefix}{image_name}_predictions.json"
                if prefix
                else f"{image_name}_predictions.json"
            )
            output_path = output_dir / filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info("Saved predictions to %s", output_path)
