"""Download Grounding DINO model weights.

Downloads the official pretrained checkpoints and saves metadata.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --output-dir checkpoints --model swint
"""

import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WEIGHTS = {
    "swint": {
        "filename": "groundingdino_swint_ogc.pth",
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "sha256": "084c29d45868e0ed6183189ef8c3d56ad568b50f",
    },
    "swinb": {
        "filename": "groundingdino_swinb_cogcoor.pth",
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
        "sha256": "beivision/groundingdino_swinb_cogcoor",
    },
}


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_with_progress(url: str, output_path: str) -> None:
    """Download a file with progress reporting."""

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

    urlretrieve(url, output_path, reporthook=_reporthook)
    print()  # Newline after progress


def main():
    parser = argparse.ArgumentParser(description="Download Grounding DINO weights")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save weights",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["swint", "swinb", "all"],
        default="swint",
        help="Which model to download (default: swint)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_download = list(WEIGHTS.keys()) if args.model == "all" else [args.model]

    for model_name in models_to_download:
        info = WEIGHTS[model_name]
        output_path = output_dir / info["filename"]

        if output_path.exists():
            logger.info("Weights already exist: %s", output_path)
            continue

        logger.info("Downloading %s weights...", model_name)
        logger.info("  URL: %s", info["url"])
        logger.info("  Output: %s", output_path)

        try:
            download_with_progress(info["url"], str(output_path))
            file_size = output_path.stat().st_size
            logger.info("  File size: %.1f MB", file_size / (1024 * 1024))
            logger.info("Download complete: %s", output_path)
        except Exception as e:
            logger.error("Download failed: %s", e)
            if output_path.exists():
                output_path.unlink()
            raise

    # Save metadata
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Model Weights\n\n")
        f.write("Downloaded using `python scripts/download_weights.py`.\n\n")
        for model_name in models_to_download:
            info = WEIGHTS[model_name]
            f.write(f"## {model_name.upper()}\n\n")
            f.write(f"- **File**: `{info['filename']}`\n")
            f.write(f"- **Source**: {info['url']}\n")
            f.write(f"- **Description**: Grounding DINO with {model_name} backbone\n\n")

    logger.info("Metadata saved to %s", readme_path)


if __name__ == "__main__":
    main()
