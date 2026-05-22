"""Inference script for Grounding DINO."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding DINO Inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text prompt for detection",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/groundingdino_swint_ogc.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save visualization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Inference configuration:")
    print(f"  Image: {args.image}")
    print(f"  Text prompt: {args.text}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Threshold: {args.threshold}")

    # TODO: Implement inference
    # 1. Load model
    # 2. Load and preprocess image
    # 3. Run inference
    # 4. Visualize results

    print("\n[TODO] Inference not yet implemented")
    print("See docs/project_timeline.md for implementation plan")


if __name__ == "__main__":
    main()
