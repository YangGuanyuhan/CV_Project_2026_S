"""Evaluation script for Grounding DINO on COCO.

Runs zero-shot object detection evaluation on COCO val2017,
computing standard COCO metrics (AP, AP50, AP75, APS, APM, APL).

Usage:
    # Full COCO val2017 evaluation
    python scripts/eval.py \\
        --config configs/grounding_dino.yaml \\
        --checkpoint checkpoints/groundingdino_swint_coco_epoch_5.pth \\
        --coco_image_dir data/coco/val2017 \\
        --coco_ann_file data/coco/annotations/instances_val2017.json

    # Subset evaluation (100 images)
    python scripts/eval.py \\
        --config configs/grounding_dino.yaml \\
        --checkpoint checkpoints/groundingdino_swint_coco_epoch_5.pth \\
        --coco_image_dir data/coco/val2017 \\
        --coco_ann_file data/coco/annotations/instances_val2017.json \\
        --subset_size 100 \\
        --output_dir outputs/coco_eval/subset_100

    # Custom thresholds
    python scripts/eval.py \\
        --config configs/grounding_dino.yaml \\
        --checkpoint checkpoints/groundingdino_swint_coco_epoch_5.pth \\
        --coco_image_dir data/coco/val2017 \\
        --coco_ann_file data/coco/annotations/instances_val2017.json \\
        --box_threshold 0.3 --text_threshold 0.25
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Grounding DINO on COCO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grounding_dino.yaml",
        help="Path to config file",
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/groundingdino_swint_coco_epoch_5.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to MMDetection model config (default: use checkpoint metadata)",
    )

    # COCO data
    parser.add_argument(
        "--coco_image_dir",
        type=str,
        required=True,
        help="Path to COCO image directory (e.g., data/coco/val2017)",
    )
    parser.add_argument(
        "--coco_ann_file",
        type=str,
        required=True,
        help="Path to COCO annotation file",
    )

    # Evaluation options
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        help="Custom text prompt (default: COCO 80-class dot-separated)",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.35,
        help="Compatibility option; MMDetection inference keeps top-k predictions",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="Compatibility option; MMDetection builds token-positive maps from text",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Number of images to evaluate (None = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/coco_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto-detect)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # Load project config
    config = OmegaConf.load(args.config)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Grounding DINO COCO Evaluation")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("COCO images: %s", args.coco_image_dir)
    logger.info("COCO annotations: %s", args.coco_ann_file)
    logger.info("Box threshold: %.2f", args.box_threshold)
    logger.info("Text threshold: %.2f", args.text_threshold)
    logger.info("Subset size: %s", args.subset_size or "all")
    logger.info("Output directory: %s", output_dir)

    # Save config snapshot
    config_snapshot = {
        "config_file": args.config,
        "model_config": args.model_config,
        "checkpoint": args.checkpoint,
        "coco_image_dir": args.coco_image_dir,
        "coco_ann_file": args.coco_ann_file,
        "text_prompt": args.text_prompt or "default COCO 80-class",
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "device": args.device or "auto",
    }
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(OmegaConf.create(config_snapshot), f)

    # Import project modules
    from src.datasets.coco_dataset import COCODatasetHelper
    from src.engine.evaluator import COCOEvaluator
    from src.models.grounding_dino import GroundingDINOModel

    # Step 1: Dataset sanity check
    logger.info("\n--- Step 1: Dataset Sanity Check ---")
    try:
        dataset_helper = COCODatasetHelper(args.coco_ann_file)
        info = dataset_helper.get_dataset_info()
        logger.info(
            "Dataset: %d images, %d categories, %d annotations",
            info["num_images"],
            info["num_categories"],
            info["num_annotations"],
        )

        sanity = dataset_helper.sanity_check(args.coco_image_dir, n=5)
        if not sanity["passed"]:
            logger.warning("Sanity check failed: some images not found!")

        # Save data check log
        with open(output_dir / "data_check.log", "w", encoding="utf-8") as f:
            json.dump(sanity, f, indent=2)
    except Exception as e:
        logger.error("Dataset check failed: %s", e)
        sys.exit(1)

    # Step 2: Get image IDs
    logger.info("\n--- Step 2: Preparing Image List ---")
    image_ids = dataset_helper.get_image_ids(
        subset_size=args.subset_size,
        seed=args.seed,
    )
    logger.info("Evaluating on %d images", len(image_ids))

    # Step 3: Load model
    logger.info("\n--- Step 3: Loading Model ---")
    try:
        model = GroundingDINOModel(
            config_path=args.model_config,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    # Step 4: Run inference
    logger.info("\n--- Step 4: Running Inference ---")
    evaluator = COCOEvaluator(
        model=model,
        annotation_file=args.coco_ann_file,
        text_prompt=args.text_prompt,
    )

    raw_predictions, eval_image_ids = evaluator.run_inference(
        image_dir=args.coco_image_dir,
        image_ids=image_ids,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    # Step 5: Convert to COCO format
    logger.info("\n--- Step 5: Converting to COCO Format ---")
    coco_results = evaluator.convert_to_coco_format(raw_predictions)
    logger.info("Total COCO-format predictions: %d", len(coco_results))

    # Step 6: Evaluate
    logger.info("\n--- Step 6: Computing COCO Metrics ---")
    eval_results = evaluator.evaluate(coco_results=coco_results, eval_image_ids=eval_image_ids)

    # Step 7: Save results
    logger.info("\n--- Step 7: Saving Results ---")
    evaluator.save_results(
        output_dir=str(output_dir),
        coco_results=coco_results,
        eval_results=eval_results,
        config=config,
    )

    # Save prompt and category mapping
    from src.datasets.coco_categories import build_coco_prompt

    prompt_path = output_dir.parent / "coco_prompt.txt"
    if not prompt_path.exists():
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(build_coco_prompt())

    # Print summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info("\nResults:")
    logger.info(eval_results["summary"])
    logger.info("\nPredictions: %d", eval_results["num_predictions"])
    logger.info("Requested images: %d", eval_results["num_requested_images"])
    logger.info("Images with predictions: %d", eval_results["num_images_with_predictions"])
    logger.info("Total time: %.1fs", total_time)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
