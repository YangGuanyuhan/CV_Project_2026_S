"""Generate COCO evaluation visualizations (success/failure cases).

Usage:
    python scripts/generate_visualizations.py \
        --coco_image_dir data/coco/val2017 \
        --coco_ann_file data/coco/annotations/instances_val2017.json \
        --predictions outputs/coco_eval/subset_100/predictions.json \
        --output_dir outputs/visualizations/coco_eval \
        --n_success 10 --n_failure 10
"""

import argparse
import json
import logging

from pycocotools.coco import COCO

from src.inference.coco_visualizer import visualize_coco_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate COCO visualizations")
    parser.add_argument("--coco_image_dir", required=True, help="COCO image directory")
    parser.add_argument("--coco_ann_file", required=True, help="COCO annotation file")
    parser.add_argument("--predictions", required=True, help="predictions.json path")
    parser.add_argument("--output_dir", default="outputs/visualizations/coco_eval")
    parser.add_argument("--n_success", type=int, default=10)
    parser.add_argument("--n_failure", type=int, default=10)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    args = parser.parse_args()

    logger.info("Loading COCO annotations from %s", args.coco_ann_file)
    coco_gt = COCO(args.coco_ann_file)

    logger.info("Loading predictions from %s", args.predictions)
    with open(args.predictions, encoding="utf-8") as f:
        coco_results = json.load(f)
    logger.info("Loaded %d predictions", len(coco_results))

    stats = visualize_coco_predictions(
        image_dir=args.coco_image_dir,
        coco_results=coco_results,
        coco_gt=coco_gt,
        output_dir=args.output_dir,
        n_success=args.n_success,
        n_failure=args.n_failure,
        score_threshold=args.score_threshold,
    )

    logger.info("Done! Saved to %s", args.output_dir)
    logger.info(
        "Success: %d, Failure: %d, Skipped: %d",
        stats["success"],
        stats["failure"],
        stats["skipped"],
    )


if __name__ == "__main__":
    main()
