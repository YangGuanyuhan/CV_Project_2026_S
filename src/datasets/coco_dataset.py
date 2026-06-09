"""COCO dataset utilities for loading annotations and sanity checks."""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class COCODatasetHelper:
    """Helper for COCO dataset operations.

    Provides utilities for loading annotations, performing sanity checks,
    and managing category mappings.

    Example:
        >>> helper = COCODatasetHelper("data/coco/annotations/instances_val2017.json")
        >>> info = helper.get_dataset_info()
        >>> helper.sanity_check("data/coco/val2017", n=5)
    """

    def __init__(self, annotation_file: str):
        """Load COCO annotations.

        Args:
            annotation_file: Path to COCO annotation JSON file.

        Raises:
            FileNotFoundError: If annotation file does not exist.
        """
        self.annotation_file = str(Path(annotation_file).resolve())
        if not Path(self.annotation_file).exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        logger.info("Loading COCO annotations from %s", self.annotation_file)

        try:
            from pycocotools.coco import COCO

            self._coco = COCO(self.annotation_file)
        except ImportError:
            logger.warning("pycocotools not available, using manual JSON loading")
            self._coco = None
            with open(self.annotation_file, encoding="utf-8") as f:
                self._data = json.load(f)

    @property
    def coco(self):
        """Access the underlying pycocotools COCO object."""
        return self._coco

    def get_dataset_info(self) -> dict:
        """Get summary information about the dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        if self._coco is not None:
            img_ids = self._coco.getImgIds()
            cat_ids = self._coco.getCatIds()
            ann_ids = self._coco.getAnnIds()

            return {
                "num_images": len(img_ids),
                "num_categories": len(cat_ids),
                "num_annotations": len(ann_ids),
                "image_ids": sorted(img_ids),
                "category_ids": sorted(cat_ids),
                "categories": [
                    {"id": c["id"], "name": c["name"]} for c in self._coco.loadCats(cat_ids)
                ],
            }
        else:
            images = self._data.get("images", [])
            annotations = self._data.get("annotations", [])
            categories = self._data.get("categories", [])

            return {
                "num_images": len(images),
                "num_categories": len(categories),
                "num_annotations": len(annotations),
                "image_ids": sorted(img["id"] for img in images),
                "category_ids": sorted(cat["id"] for cat in categories),
                "categories": [{"id": c["id"], "name": c["name"]} for c in categories],
            }

    def sanity_check(self, image_dir: str, n: int = 5) -> dict:
        """Perform sanity checks on the dataset.

        Checks that annotation file is valid and sample images
        match their annotations.

        Args:
            image_dir: Path to the image directory.
            n: Number of random images to check.

        Returns:
            Dictionary with check results.
        """
        image_dir = Path(image_dir)
        info = self.get_dataset_info()
        results = {
            "annotation_loaded": True,
            "num_images": info["num_images"],
            "num_categories": info["num_categories"],
            "num_annotations": info["num_annotations"],
            "checks": [],
            "passed": True,
        }

        # Check random sample of images
        rng = np.random.default_rng(42)
        sample_ids = rng.choice(
            info["image_ids"], size=min(n, len(info["image_ids"])), replace=False
        )

        for img_id in sample_ids:
            check = {"image_id": int(img_id)}

            if self._coco is not None:
                img_info = self._coco.loadImgs(img_id)[0]
            else:
                img_info = next(img for img in self._data["images"] if img["id"] == img_id)

            check["file_name"] = img_info["file_name"]
            check["width"] = img_info["width"]
            check["height"] = img_info["height"]

            # Check if image file exists
            img_path = image_dir / img_info["file_name"]
            check["file_exists"] = img_path.exists()

            results["checks"].append(check)
            if not check["file_exists"]:
                results["passed"] = False

        n_passed = sum(1 for c in results["checks"] if c["file_exists"])
        results["images_checked"] = len(results["checks"])
        results["images_found"] = n_passed

        logger.info(
            "Sanity check: %d/%d images found in %s",
            n_passed,
            len(results["checks"]),
            image_dir,
        )

        return results

    def get_image_ids(self, subset_size: int | None = None, seed: int = 42) -> list[int]:
        """Get image IDs, optionally limited to a subset.

        Args:
            subset_size: If set, return only this many image IDs.
            seed: Random seed for reproducible subset selection.

        Returns:
            List of image IDs.
        """
        info = self.get_dataset_info()
        image_ids = info["image_ids"]

        if subset_size is not None and subset_size < len(image_ids):
            rng = np.random.default_rng(seed)
            image_ids = sorted(rng.choice(image_ids, size=subset_size, replace=False).tolist())

        return image_ids

    def load_annotations_for_image(self, image_id: int) -> list[dict]:
        """Load all annotations for a given image.

        Args:
            image_id: COCO image ID.

        Returns:
            List of annotation dicts with keys: category_id, bbox, area, etc.
        """
        if self._coco is not None:
            ann_ids = self._coco.getAnnIds(imgIds=image_id)
            return self._coco.loadAnns(ann_ids)
        else:
            return [ann for ann in self._data.get("annotations", []) if ann["image_id"] == image_id]
