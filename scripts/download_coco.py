"""Download COCO 2017 dataset for evaluation.

Downloads val2017 images and annotations for COCO evaluation.

Usage:
    python scripts/download_coco.py
    python scripts/download_coco.py --output-dir data/coco --split val2017
"""

import argparse
import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


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
    print()


def extract_zip(zip_path: str, output_dir: str) -> None:
    """Extract a zip file."""
    logger.info("Extracting %s to %s...", zip_path, output_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    logger.info("Extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/coco",
        help="Directory to save COCO data",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["val2017", "annotations"],
        choices=["val2017", "train2017", "annotations"],
        help="Which splits to download (default: val2017 annotations)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded zip files after extraction",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_dir = output_dir / "_downloads"
    download_dir.mkdir(exist_ok=True)

    for split in args.split:
        url = COCO_URLS[split]
        zip_name = url.split("/")[-1]
        zip_path = download_dir / zip_name
        extract_dir = str(output_dir)

        # Check if already extracted
        if split == "annotations":
            ann_file = output_dir / "annotations" / "instances_val2017.json"
            if ann_file.exists():
                logger.info("Annotations already exist: %s", ann_file)
                continue
        elif split in ("val2017", "train2017"):
            img_dir = output_dir / split
            if img_dir.exists() and any(img_dir.iterdir()):
                logger.info("Images already exist: %s", img_dir)
                continue

        # Download
        if not zip_path.exists():
            logger.info("Downloading %s...", split)
            logger.info("  URL: %s", url)
            download_with_progress(url, str(zip_path))
        else:
            logger.info("Zip already downloaded: %s", zip_path)

        # Extract
        extract_zip(str(zip_path), extract_dir)

        # Clean up
        if not args.keep_zip:
            zip_path.unlink()
            logger.info("Removed zip file: %s", zip_path)

    # Print summary
    logger.info("\nCOCO dataset download complete!")
    logger.info("Directory structure:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file() and p.suffix != ".zip":
            logger.info("  %s", p.relative_to(output_dir))


if __name__ == "__main__":
    main()
