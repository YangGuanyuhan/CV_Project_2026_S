"""Evaluation script for Grounding DINO."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Grounding DINO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grounding_dino.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="Output directory for results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluation configuration:")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {output_dir}")

    # TODO: Implement evaluation
    # 1. Load model from checkpoint
    # 2. Build validation dataset
    # 3. Run inference
    # 4. Compute metrics (mAP)
    # 5. Save results

    print("\n[TODO] Evaluation not yet implemented")
    print("See docs/project_timeline.md for implementation plan")


if __name__ == "__main__":
    main()
