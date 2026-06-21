"""Inference script for Grounding DINO.

Run single-image or batch inference with configurable prompts and thresholds.

Usage:
    # Single image
    python scripts/inference.py --image data/demo_images/test.jpg --text "person . car ."

    # Batch from directory
    python scripts/inference.py --image_dir data/demo_images --text "person . car . dog ."

    # With custom thresholds
    python scripts/inference.py --image test.jpg --text "cat ." \
        --box_threshold 0.3 --text_threshold 0.25
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grounding DINO Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to single input image")
    input_group.add_argument("--image_dir", type=str, help="Path to directory of images")

    # Model options
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help='Text prompt for detection (e.g., "person . car . dog .")',
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to MMDetection config file (default: use checkpoint metadata)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/groundingdino_swint_coco_epoch_5.pth",
        help="Path to model checkpoint",
    )

    # Threshold options
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

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference_demo",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        default=True,
        help="Save predictions as JSON (default: True)",
    )
    parser.add_argument(
        "--no_save_json",
        action="store_false",
        dest="save_json",
        help="Do not save predictions as JSON",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Grounding DINO Inference")
    logger.info("=" * 60)
    logger.info("Text prompt: %s", args.text)
    logger.info("Config: %s", args.config)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Box threshold: %.2f", args.box_threshold)
    logger.info("Text threshold: %.2f", args.text_threshold)
    logger.info("Output directory: %s", output_dir)

    # Import project modules
    from src.inference.predictor import Predictor
    from src.inference.visualizer import save_annotated_image

    # Initialize predictor
    logger.info("Loading model...")
    predictor = Predictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    logger.info("Model loaded successfully.")

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    else:
        image_dir = Path(args.image_dir)
        if not image_dir.is_dir():
            logger.error("Image directory not found: %s", image_dir)
            sys.exit(1)
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        image_paths = sorted(str(p) for p in image_dir.iterdir() if p.suffix.lower() in extensions)
        if not image_paths:
            logger.error("No images found in %s", image_dir)
            sys.exit(1)

    logger.info("Processing %d image(s)...", len(image_paths))

    # Run inference
    total_start = time.time()
    all_results = []

    for i, image_path in enumerate(image_paths):
        logger.info("[%d/%d] %s", i + 1, len(image_paths), Path(image_path).name)

        try:
            result = predictor.predict(
                image_path=image_path,
                text_prompt=args.text,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            all_results.append(result)

            # Save annotated image
            image = cv2.imread(image_path)
            image_name = Path(image_path).stem
            annotated_path = output_dir / f"{image_name}_annotated.jpg"

            import numpy as np

            save_annotated_image(
                image=image,
                output_path=str(annotated_path),
                boxes=np.array(result["boxes"]) if result["boxes"] else None,
                scores=result["scores"],
                phrases=result["phrases"],
            )

            # Save JSON
            if args.save_json:
                json_path = output_dir / f"{image_name}_predictions.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(
                "  -> %d detections in %.3fs",
                result["num_detections"],
                result["inference_time"],
            )

        except Exception as e:
            logger.error("  -> Failed: %s", e)
            all_results.append({"image_path": image_path, "error": str(e)})

    total_time = time.time() - total_start

    # Save run log
    log_data = {
        "text_prompt": args.text,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "num_images": len(image_paths),
        "total_time": round(total_time, 2),
        "avg_time_per_image": round(total_time / max(len(image_paths), 1), 4),
        "results_summary": [
            {
                "image": Path(r.get("image_path", "")).name,
                "detections": r.get("num_detections", 0),
                "time": r.get("inference_time", 0),
                "error": r.get("error"),
            }
            for r in all_results
        ],
    }

    log_path = output_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("=" * 60)
    logger.info("Inference complete!")
    logger.info("  Images processed: %d", len(image_paths))
    logger.info("  Total time: %.1fs", total_time)
    logger.info("  Average time per image: %.3fs", log_data["avg_time_per_image"])
    n_success = sum(1 for r in all_results if "error" not in r)
    logger.info("  Successful: %d/%d", n_success, len(image_paths))
    logger.info("  Output directory: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
