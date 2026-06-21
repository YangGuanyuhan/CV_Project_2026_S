"""Grounding DINO model wrapper for project integration.

This wrapper uses the MMDetection Grounding DINO inference path.  That keeps
frontend demo inference consistent with the COCO evaluation setup: predictions
are selected with ``test_cfg.max_per_img`` top-k instead of a fixed
``box_threshold`` filter before post-processing.
"""

import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MMDT_ROOT = PROJECT_ROOT.parent / "mmdetection"
DEFAULT_BERT_MODEL = PROJECT_ROOT / "models" / "bert-base-uncased"


class GroundingDINOModel:
    """Wrapper around MMDetection Grounding DINO for project integration.

    Uses MMDetection's model and post-processing implementation while providing
    the same clean interface used by the project's predictor, evaluator and
    frontend server.

    Example:
        >>> model = GroundingDINOModel(
        ...     checkpoint_path="checkpoints/groundingdino_swint_coco_epoch_5.pth",
        ... )
        >>> boxes, scores, phrases = model.predict(image, "person . car . dog .")
    """

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str = "checkpoints/groundingdino_swint_coco_epoch_5.pth",
        device: str | None = None,
        mmdet_root: str | None = None,
        bert_model_name: str | None = None,
    ):
        """Initialize the Grounding DINO model.

        Args:
            config_path: Optional path to an MMDetection config file. If not
                provided, the config embedded in the MMEngine checkpoint is used.
            checkpoint_path: Path to the MMDetection model checkpoint (.pth).
            device: Device to run on ('cuda' or 'cpu'). Auto-detected if None.
            mmdet_root: Optional path to an MMDetection source checkout.
            bert_model_name: Optional local path or HuggingFace name for BERT.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            RuntimeError: If model loading fails.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = str(Path(checkpoint_path).resolve())

        if config_path is not None:
            config_path = str(Path(config_path).resolve())
            if not Path(config_path).exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.bert_model_name = self._resolve_bert_model_name(bert_model_name)

        self._set_transformers_offline_env()
        self._ensure_mmdet_on_path(mmdet_root)

        logger.info("Loading MMDetection Grounding DINO model...")
        logger.info("  Config: %s", config_path)
        logger.info("  Checkpoint: %s", checkpoint_path)
        logger.info("  BERT: %s", self.bert_model_name)
        logger.info("  Device: %s", device)

        try:
            from mmengine.config import Config
            from mmdet.apis import DetInferencer

            model_cfg = Config.fromfile(config_path) if config_path else self._load_checkpoint_config(checkpoint_path)
            if "model" not in model_cfg:
                logger.warning(
                    "Config %s is not an MMDetection config; falling back to checkpoint metadata.",
                    config_path,
                )
                model_cfg = self._load_checkpoint_config(checkpoint_path)
            self._patch_mmdet_config(model_cfg)

            self._inferencer = DetInferencer(
                model=model_cfg,
                weights=checkpoint_path,
                device=device,
                show_progress=False,
            )
            self._model = self._inferencer.model
            self.max_per_img = int(self._model.cfg.model.test_cfg.get("max_per_img", 300))
            logger.info("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load MMDetection Grounding DINO model: {e}") from e

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> tuple[np.ndarray, list[float], list[str]]:
        """Run top-k inference on a single image.

        Args:
            image: Input image as numpy array (H, W, 3) in BGR format (OpenCV).
            text_prompt: Text prompt with categories separated by ' . '.
            box_threshold: Kept for API compatibility; not used for filtering.
            text_threshold: Kept for API compatibility; MMDetection builds the
                token-positive map from the prompt instead.

        Returns:
            Tuple of:
                - boxes: (N, 4) array in (x1, y1, x2, y2) pixel coordinates.
                - scores: List of N confidence scores.
                - phrases: List of N detected phrase strings.
        """
        del box_threshold, text_threshold

        entities = self._parse_prompt_entities(text_prompt)
        results = self._inferencer(
            image,
            texts=text_prompt,
            custom_entities=True,
            return_datasamples=True,
            no_save_vis=True,
            no_save_pred=True,
        )

        data_sample = results["predictions"][0]
        pred_instances = data_sample.pred_instances.cpu()

        boxes_np = pred_instances.bboxes.numpy().astype(np.float64)
        scores_list = pred_instances.scores.numpy().astype(float).tolist()
        labels = pred_instances.labels.numpy().astype(int).tolist()
        phrases_list = [self._label_to_phrase(label, entities) for label in labels]

        # MMDetection already rescales boxes to original image space. Clip to
        # guard against tiny numeric overshoots after resize inversion.
        h, w = image.shape[:2]
        if len(boxes_np) > 0:
            boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, w)
            boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, h)
        else:
            boxes_np = np.zeros((0, 4), dtype=np.float64)

        return boxes_np, scores_list, phrases_list

    @staticmethod
    def _ensure_mmdet_on_path(mmdet_root: str | None) -> None:
        """Add a local MMDetection checkout to sys.path when needed."""
        candidates = [
            mmdet_root,
            os.environ.get("MMDET_ROOT"),
            str(DEFAULT_MMDT_ROOT),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser().resolve()
            if (path / "mmdet").is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
                return

    @staticmethod
    def _resolve_bert_model_name(bert_model_name: str | None) -> str:
        """Resolve BERT path/name, preferring local project files when complete."""
        candidates = [
            bert_model_name,
            os.environ.get("GROUNDING_DINO_BERT"),
            str(DEFAULT_BERT_MODEL),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.is_dir() and (path / "config.json").exists():
                return str(path.resolve())
            if not path.exists():
                return candidate
        return "bert-base-uncased"

    @staticmethod
    def _set_transformers_offline_env() -> None:
        """Keep demo inference from falling back to remote Hugging Face calls."""
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    @staticmethod
    def _load_checkpoint_config(checkpoint_path: str):
        """Load the MMEngine config embedded in a checkpoint."""
        from mmengine.config import Config

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg_text = checkpoint.get("meta", {}).get("cfg")
        if not cfg_text:
            raise RuntimeError(
                "Checkpoint does not contain MMDetection config metadata; "
                "pass an MMDetection config_path explicitly."
            )
        return Config.fromstring(cfg_text, file_format=".py")

    def _patch_mmdet_config(self, cfg) -> None:
        """Patch runtime-only paths in a dumped MMDetection config."""
        cfg.model.language_model.name = self.bert_model_name
        if "lang_model_name" in cfg:
            cfg.lang_model_name = self.bert_model_name
        self._replace_bert_references(cfg)
        cfg.model.test_cfg.max_per_img = int(cfg.model.test_cfg.get("max_per_img", 300))

    def _replace_bert_references(self, node) -> None:
        """Replace nested BERT model/tokenizer names with the local model path."""
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"name", "lang_model_name", "tokenizer_name", "text_encoder_type"} and value == "bert-base-uncased":
                    node[key] = self.bert_model_name
                else:
                    self._replace_bert_references(value)
            return

        if isinstance(node, list):
            for item in node:
                self._replace_bert_references(item)

    @staticmethod
    def _parse_prompt_entities(text_prompt: str) -> list[str]:
        """Parse Grounding DINO dot-separated prompts into entity labels."""
        entities = [part.strip().lower() for part in re.split(r"\s*\.\s*", text_prompt) if part.strip()]
        if not entities:
            entities = [text_prompt.strip().lower()]
        return entities

    def _label_to_phrase(self, label: int, entities: list[str]) -> str:
        if 0 <= label < len(entities):
            return entities[label]
        classes = self._model.dataset_meta.get("classes", [])
        if 0 <= label < len(classes):
            return classes[label]
        return str(label)

    def to(self, device: str):
        """Move model to specified device."""
        self.device = device
        self._model = self._model.to(device)
        return self
