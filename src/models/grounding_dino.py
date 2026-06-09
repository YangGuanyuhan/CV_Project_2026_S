"""Grounding DINO model wrapper for project integration.

Wraps the official groundingdino-py package to provide a unified interface
compatible with our project's inference and evaluation pipelines.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class GroundingDINOModel:
    """Wrapper around official Grounding DINO for project integration.

    Uses the groundingdino-py package for model loading and inference,
    while providing a clean interface that matches our project conventions.

    Example:
        >>> model = GroundingDINOModel(
        ...     config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ...     checkpoint_path="checkpoints/groundingdino_swint_ogc.pth",
        ... )
        >>> boxes, scores, phrases = model.predict(image, "person . car . dog .")
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str | None = None,
    ):
        """Initialize the Grounding DINO model.

        Args:
            config_path: Path to the Grounding DINO config file (.py).
            checkpoint_path: Path to the model checkpoint (.pth).
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.

        Raises:
            FileNotFoundError: If config or checkpoint file does not exist.
            RuntimeError: If model loading fails.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        config_path = str(Path(config_path).resolve())
        checkpoint_path = str(Path(checkpoint_path).resolve())

        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        logger.info("Loading Grounding DINO model...")
        logger.info("  Config: %s", config_path)
        logger.info("  Checkpoint: %s", checkpoint_path)
        logger.info("  Device: %s", device)

        try:
            from groundingdino.util.inference import load_model

            self._model = load_model(config_path, checkpoint_path, device=device)
            self._model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Grounding DINO model: {e}") from e

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 25,
    ) -> tuple[np.ndarray, list[float], list[str]]:
        """Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, 3) in BGR format (OpenCV).
            text_prompt: Text prompt with categories separated by ' . '.
            box_threshold: Confidence threshold for box detection.
            text_threshold: Threshold for text-token matching.

        Returns:
            Tuple of:
                - boxes: (N, 4) array in (x1, y1, x2, y2) pixel coordinates.
                - scores: List of N confidence scores.
                - phrases: List of N detected phrase strings.
        """
        from groundingdino.util.inference import predict as gd_predict
        from groundingdino.util.inference import preprocess_caption

        # Convert BGR to RGB and to PIL-compatible format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Grounding DINO expects the image as a tensor
        # Use the official preprocessing
        image_pil = self._numpy_to_pil(image_rgb)

        # Preprocess image for the model
        transform = self._get_transform()
        image_tensor = transform(image_pil).to(self.device)

        # Preprocess caption
        caption = preprocess_caption(text_prompt)

        # Run inference
        boxes, logits, phrases = gd_predict(
            model=self._model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Convert normalized (cx, cy, w, h) to pixel (x1, y1, x2, y2)
        h, w = image.shape[:2]
        if len(boxes) > 0:
            boxes_np = boxes.cpu().numpy()
            # boxes are in (cx, cy, w, h) normalized [0, 1]
            boxes_np[:, 0] *= w
            boxes_np[:, 1] *= h
            boxes_np[:, 2] *= w
            boxes_np[:, 3] *= h
            # Convert to xyxy
            from src.utils.box_ops import box_cxcywh_to_xyxy

            boxes_np = box_cxcywh_to_xyxy(boxes_np)
            # Clip to image bounds
            boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, w)
            boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, h)
        else:
            boxes_np = np.zeros((0, 4), dtype=np.float64)

        scores_list = logits.cpu().numpy().tolist()
        phrases_list = list(phrases)

        return boxes_np, scores_list, phrases_list

    @staticmethod
    def _numpy_to_pil(image: np.ndarray):
        """Convert numpy array (RGB) to PIL Image."""
        from PIL import Image

        return Image.fromarray(image)

    @staticmethod
    def _get_transform():
        """Get the standard image transform for Grounding DINO."""
        from groundingdino.datasets import transforms as T

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return transform

    def to(self, device: str):
        """Move model to specified device."""
        self.device = device
        self._model = self._model.to(device)
        return self
