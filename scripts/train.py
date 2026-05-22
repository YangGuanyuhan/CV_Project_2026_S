"""Training script for Grounding DINO."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Train Grounding DINO")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/grounding_dino.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Override config with command line args
    if args.resume:
        config.training.resume = args.resume
    if args.output_dir:
        config.training.output_dir = args.output_dir

    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config, output_dir / "config.yaml")

    print("Training configuration:")
    print(OmegaConf.to_yaml(config))

    # TODO: Implement training loop
    # 1. Build model
    # 2. Build dataset and dataloader
    # 3. Build optimizer and scheduler
    # 4. Training loop
    # 5. Save checkpoints

    print("\n[TODO] Training not yet implemented")
    print("See docs/project_timeline.md for implementation plan")


if __name__ == "__main__":
    main()
