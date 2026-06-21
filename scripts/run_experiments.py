"""Run ablation experiments for Grounding DINO evaluation.

Supports threshold sensitivity and prompt format comparison experiments
on COCO subsets.

Usage:
    # Run all experiments
    python scripts/run_experiments.py \\
        --checkpoint checkpoints/groundingdino_swint_coco_epoch_5.pth \\
        --coco_image_dir data/coco/val2017 \\
        --coco_ann_file data/coco/annotations/instances_val2017.json \\
        --subset_size 500

    # Run only threshold experiments
    python scripts/run_experiments.py \\
        --checkpoint checkpoints/groundingdino_swint_coco_epoch_5.pth \\
        --coco_image_dir data/coco/val2017 \\
        --coco_ann_file data/coco/annotations/instances_val2017.json \\
        --subset_size 500 \\
        --experiments threshold

    # Run only prompt experiments
    python scripts/run_experiments.py \\
        --checkpoint checkpoints/groundingdino_swint_coco_epoch_5.pth \\
        --coco_image_dir data/coco/val2017 \\
        --coco_ann_file data/coco/annotations/instances_val2017.json \\
        --subset_size 500 \\
        --experiments prompt
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Experiment definitions
THRESHOLD_EXPERIMENTS = [
    {"name": "A1", "box_threshold": 0.25, "text_threshold": 0.20},
    {"name": "A2", "box_threshold": 0.35, "text_threshold": 0.25},
    {"name": "A3", "box_threshold": 0.45, "text_threshold": 0.30},
]

PROMPT_EXPERIMENTS = [
    {
        "name": "B1",
        "format": "dot-separated",
        "prompt": "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush .",
    },
    {
        "name": "B2",
        "format": "comma-separated",
        "prompt": "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush,",
    },
    {
        "name": "B3",
        "format": "sentence-style",
        "prompt": "There are person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush in the image.",
    },
]


def run_single_experiment(
    model,
    annotation_file: str,
    image_dir: str,
    image_ids: list[int],
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    output_dir: str,
) -> dict:
    """Run a single evaluation experiment.

    Returns:
        Dictionary with metrics and metadata.
    """
    from src.engine.evaluator import COCOEvaluator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = COCOEvaluator(
        model=model,
        annotation_file=annotation_file,
        text_prompt=text_prompt,
    )

    raw_predictions, eval_image_ids = evaluator.run_inference(
        image_dir=image_dir,
        image_ids=image_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    coco_results = evaluator.convert_to_coco_format(raw_predictions)
    eval_results = evaluator.evaluate(coco_results=coco_results, eval_image_ids=eval_image_ids)

    evaluator.save_results(
        output_dir=str(output_dir),
        coco_results=coco_results,
        eval_results=eval_results,
    )

    # Count average boxes per image (denominator = all evaluated images, not just those with predictions)
    avg_boxes = len(coco_results) / len(eval_image_ids) if eval_image_ids else 0

    return {
        "metrics": eval_results["metrics"],
        "num_predictions": len(coco_results),
        "avg_boxes_per_image": round(avg_boxes, 2),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/groundingdino_swint_coco_epoch_5.pth",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=None,
        help="Path to MMDetection config file (default: use checkpoint metadata)",
    )
    parser.add_argument("--coco_image_dir", type=str, required=True)
    parser.add_argument("--coco_ann_file", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/experiments")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["threshold", "prompt"],
        choices=["threshold", "prompt"],
        help="Which experiments to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model once
    from src.datasets.coco_dataset import COCODatasetHelper
    from src.models.grounding_dino import GroundingDINOModel

    logger.info("Loading model...")
    model = GroundingDINOModel(
        config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Get image IDs
    dataset_helper = COCODatasetHelper(args.coco_ann_file)
    image_ids = dataset_helper.get_image_ids(subset_size=args.subset_size, seed=args.seed)
    logger.info("Using %d images for experiments", len(image_ids))

    output_dir = Path(args.output_dir)

    # --- Threshold Sensitivity ---
    if "threshold" in args.experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Experiment A: Threshold Sensitivity")
        logger.info("=" * 60)

        exp_dir = output_dir / "threshold_sensitivity"
        threshold_results = []

        for exp in THRESHOLD_EXPERIMENTS:
            logger.info(
                "\n--- Run %s (box=%.2f, text=%.2f) ---",
                exp["name"],
                exp["box_threshold"],
                exp["text_threshold"],
            )

            result = run_single_experiment(
                model=model,
                annotation_file=args.coco_ann_file,
                image_dir=args.coco_image_dir,
                image_ids=image_ids,
                text_prompt=None,  # Use default COCO prompt
                box_threshold=exp["box_threshold"],
                text_threshold=exp["text_threshold"],
                output_dir=str(exp_dir / f"run_{exp['name'].lower()}"),
            )

            threshold_results.append(
                {
                    "run": exp["name"],
                    "box_threshold": exp["box_threshold"],
                    "text_threshold": exp["text_threshold"],
                    **result["metrics"],
                    "avg_boxes_per_image": result["avg_boxes_per_image"],
                }
            )

        # Save summary
        _save_summary_csv(threshold_results, str(exp_dir / "summary.csv"))
        _save_summary_md(threshold_results, str(exp_dir / "summary.md"), "Threshold Sensitivity")

    # --- Prompt Format Comparison ---
    if "prompt" in args.experiments:
        logger.info("\n" + "=" * 60)
        logger.info("Experiment B: Prompt Format Comparison")
        logger.info("=" * 60)

        exp_dir = output_dir / "prompt_comparison"
        prompt_results = []

        for exp in PROMPT_EXPERIMENTS:
            logger.info("\n--- Run %s (%s) ---", exp["name"], exp["format"])

            result = run_single_experiment(
                model=model,
                annotation_file=args.coco_ann_file,
                image_dir=args.coco_image_dir,
                image_ids=image_ids,
                text_prompt=exp["prompt"],
                box_threshold=0.35,
                text_threshold=0.25,
                output_dir=str(exp_dir / f"run_{exp['name'].lower()}"),
            )

            # Save prompt text
            prompt_path = exp_dir / f"run_{exp['name'].lower()}" / "prompt.txt"
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write(exp["prompt"])

            prompt_results.append(
                {
                    "run": exp["name"],
                    "format": exp["format"],
                    **result["metrics"],
                    "avg_boxes_per_image": result["avg_boxes_per_image"],
                }
            )

        _save_summary_csv(prompt_results, str(exp_dir / "summary.csv"))
        _save_summary_md(prompt_results, str(exp_dir / "summary.md"), "Prompt Format Comparison")

    logger.info("\nAll experiments complete! Results saved to %s", output_dir)


def _save_summary_csv(results: list[dict], path: str) -> None:
    """Save experiment results as CSV."""
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved summary CSV to %s", path)


def _save_summary_md(results: list[dict], path: str, title: str) -> None:
    """Save experiment results as Markdown table."""
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title} Results\n\n")

        # Table header
        headers = list(results[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        # Table rows
        for row in results:
            values = []
            for h in headers:
                v = row[h]
                if isinstance(v, float):
                    values.append(f"{v:.4f}")
                else:
                    values.append(str(v))
            f.write("| " + " | ".join(values) + " |\n")

        f.write("\nGenerated by `scripts/run_experiments.py`\n")

    logger.info("Saved summary Markdown to %s", path)


if __name__ == "__main__":
    main()
