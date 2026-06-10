"""Generate compact report figures from existing evaluation outputs.

This script does not run model inference. It only reads the saved metrics,
experiment summaries, and qualitative examples, then creates figure assets
that can be embedded in the final report or presentation.

Usage:
    python scripts/generate_report_highlights.py
"""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_METRICS = Path("outputs/coco_eval/full_val2017/metrics.json")
DEFAULT_THRESHOLD_CSV = Path("outputs/experiments/threshold_sensitivity/summary.csv")
DEFAULT_PROMPT_CSV = Path("outputs/experiments/prompt_comparison/summary.csv")
DEFAULT_QUALITATIVE_DIR = Path("outputs/visualizations/coco_eval")
DEFAULT_OUTPUT_DIR = Path("outputs/visualizations/report_highlights")

COLORS = {
    "ink": "#263238",
    "muted": "#607D8B",
    "grid": "#CFD8DC",
    "blue": "#1E88E5",
    "teal": "#00897B",
    "amber": "#F9A825",
    "red": "#E53935",
    "violet": "#6A4C93",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
        }
    )


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", offset: float = 0.01) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["ink"],
        )


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_overview(metrics_payload: dict, output_dir: Path) -> Path:
    metrics = metrics_payload["metrics"]
    names = ["AP", "AP50", "AP75", "APS", "APM", "APL"]
    labels = [
        "AP 0.50:0.95",
        "AP50",
        "AP75",
        "AP small",
        "AP medium",
        "AP large",
    ]
    values = [metrics[name] for name in names]
    bar_colors = [
        COLORS["blue"],
        COLORS["teal"],
        COLORS["teal"],
        COLORS["red"],
        COLORS["amber"],
        COLORS["blue"],
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    positions = list(range(len(labels)))
    bars = ax.barh(positions, values, color=bar_colors, alpha=0.95)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.65, max(values) + 0.08))
    ax.set_xlabel("COCO metric value")
    ax.set_title(
        "Grounding DINO zero-shot performance on COCO val2017\n"
        f"{metrics_payload['num_predictions']:,} predictions over "
        f"{metrics_payload['num_images_with_predictions']:,}/"
        f"{metrics_payload['num_requested_images']:,} images",
        pad=14,
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=9,
        )
    path = output_dir / "01_metrics_overview.png"
    save_fig(fig, path)
    return path


def plot_size_gap(metrics_payload: dict, output_dir: Path) -> Path:
    metrics = metrics_payload["metrics"]
    labels = ["Small", "Medium", "Large"]
    values = [metrics["APS"], metrics["APM"], metrics["APL"]]
    ratio = metrics["APL"] / max(metrics["APS"], 1e-9)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(labels, values, color=[COLORS["red"], COLORS["amber"], COLORS["blue"]])
    annotate_bars(ax, bars)
    ax.set_ylim(0, max(0.65, max(values) + 0.08))
    ax.set_ylabel("AP")
    ax.set_title("Object size strongly affects detection quality")
    ax.text(
        1.0,
        max(values) + 0.035,
        f"Large-object AP is {ratio:.1f}x small-object AP",
        ha="center",
        fontsize=10,
        color=COLORS["ink"],
    )
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    path = output_dir / "02_size_gap.png"
    save_fig(fig, path)
    return path


def plot_threshold_sensitivity(rows: list[dict], output_dir: Path) -> Path:
    rows = sorted(rows, key=lambda r: r["box_threshold"])
    x = [row["box_threshold"] for row in rows]
    labels = [
        f"{row['box_threshold']:.2f}\ntext {row['text_threshold']:.2f}" for row in rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    ax = axes[0]
    for metric, color in [
        ("AP", COLORS["blue"]),
        ("AP50", COLORS["teal"]),
        ("AP75", COLORS["violet"]),
    ]:
        y = [row[metric] for row in rows]
        ax.plot(x, y, marker="o", linewidth=2.2, label=metric, color=color)
        for xi, yi in zip(x, y, strict=True):
            ax.text(xi, yi + 0.009, f"{yi:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.34, 0.62)
    ax.set_xlabel("Threshold setting")
    ax.set_ylabel("COCO metric")
    ax.set_title("Lower thresholds improve recall/AP")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    avg_boxes = [row["avg_boxes_per_image"] for row in rows]
    bars = ax.bar(labels, avg_boxes, color=COLORS["amber"])
    annotate_bars(ax, bars, fmt="{:.2f}", offset=0.25)
    ax.set_ylim(0, max(avg_boxes) + 3.0)
    ax.set_xlabel("Threshold setting")
    ax.set_ylabel("Average detections per image")
    ax.set_title("Recall gain costs more proposals")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    path = output_dir / "03_threshold_sensitivity.png"
    save_fig(fig, path)
    return path


def plot_prompt_comparison(rows: list[dict], output_dir: Path) -> Path:
    labels = [str(row["format"]) for row in rows]
    x_positions = list(range(len(labels)))
    width = 0.34
    ap_values = [row["AP"] for row in rows]
    ap50_values = [row["AP50"] for row in rows]
    baseline = ap_values[0]

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    bars_ap = ax.bar(
        [x - width / 2 for x in x_positions],
        ap_values,
        width=width,
        label="AP",
        color=COLORS["blue"],
    )
    bars_ap50 = ax.bar(
        [x + width / 2 for x in x_positions],
        ap50_values,
        width=width,
        label="AP50",
        color=COLORS["teal"],
    )
    annotate_bars(ax, bars_ap, offset=0.012)
    annotate_bars(ax, bars_ap50, offset=0.012)
    for idx, value in enumerate(ap_values[1:], start=1):
        drop = (baseline - value) / baseline * 100
        ax.text(
            idx - width / 2,
            value + 0.075,
            f"{drop:.0f}% lower AP",
            ha="center",
            fontsize=9,
            color=COLORS["red"],
        )
    ax.set_xticks(x_positions, labels)
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("COCO metric")
    ax.set_title("Prompt formatting is a major performance lever")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    path = output_dir / "04_prompt_format_comparison.png"
    save_fig(fig, path)
    return path


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) // 2
    y = box[1] + (box[3] - box[1] - text_h) // 2
    draw.text((x, y), text, font=font, fill=fill)


def make_collage(
    image_paths: list[Path],
    title: str,
    output_path: Path,
    *,
    n_examples: int,
) -> Path | None:
    selected = image_paths[:n_examples]
    if not selected:
        return None

    cols = 3
    rows = math.ceil(len(selected) / cols)
    tile_w = 520
    image_h = 360
    title_h = 34
    header_h = 60
    pad = 18
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = header_h + rows * (image_h + title_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(28)
    label_font = load_font(18)

    draw.rectangle((0, 0, canvas_w, header_h), fill=COLORS["ink"])
    centered_text(draw, (0, 0, canvas_w, header_h), title, header_font, "white")

    for idx, image_path in enumerate(selected):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (tile_w + pad)
        y0 = header_h + pad + row * (image_h + title_h + pad)
        label_box = (x0, y0, x0 + tile_w, y0 + title_h)

        label = image_path.stem.replace("success_", "success ").replace("failure_", "failure ")
        centered_text(draw, label_box, label, label_font, COLORS["ink"])

        img = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
        img.thumbnail((tile_w, image_h), Image.Resampling.LANCZOS)
        img_x = x0 + (tile_w - img.width) // 2
        img_y = y0 + title_h + (image_h - img.height) // 2
        draw.rectangle(
            (x0, y0 + title_h, x0 + tile_w, y0 + title_h + image_h),
            outline=COLORS["grid"],
            width=2,
        )
        canvas.paste(img, (img_x, img_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path


def make_qualitative_grids(
    qualitative_dir: Path, output_dir: Path, *, n_examples: int
) -> list[Path]:
    success = sorted((qualitative_dir / "success_cases").glob("*.jpg"))
    failure = sorted((qualitative_dir / "failure_cases").glob("*.jpg"))
    paths: list[Path] = []
    success_path = make_collage(
        success,
        "Qualitative success cases",
        output_dir / "05_success_cases_grid.jpg",
        n_examples=n_examples,
    )
    failure_path = make_collage(
        failure,
        "Qualitative failure cases",
        output_dir / "06_failure_cases_grid.jpg",
        n_examples=n_examples,
    )
    if success_path is not None:
        paths.append(success_path)
    if failure_path is not None:
        paths.append(failure_path)
    return paths


def write_index(output_dir: Path, generated_paths: list[Path]) -> Path:
    index_path = output_dir / "README.md"
    rel_paths = [path.as_posix() for path in generated_paths]
    body = [
        "# Report Highlight Figures",
        "",
        "Generated by `scripts/generate_report_highlights.py` from saved COCO metrics,",
        "ablation CSV files, and qualitative detection examples.",
        "",
        "| Figure | Intended use |",
        "|---|---|",
    ]
    descriptions = {
        "01_metrics_overview.png": "Main full-val2017 result slide/report figure",
        "02_size_gap.png": "Failure-mode discussion for small objects",
        "03_threshold_sensitivity.png": "Ablation figure for threshold tradeoffs",
        "04_prompt_format_comparison.png": "Ablation figure for prompt formatting",
        "05_success_cases_grid.jpg": "Qualitative success examples",
        "06_failure_cases_grid.jpg": "Qualitative failure examples",
    }
    for rel_path in rel_paths:
        name = Path(rel_path).name
        body.append(f"| `{name}` | {descriptions.get(name, 'Report figure')} |")
    body.extend(
        [
            "",
            "Inputs:",
            "",
            f"- `{DEFAULT_METRICS.as_posix()}`",
            f"- `{DEFAULT_THRESHOLD_CSV.as_posix()}`",
            f"- `{DEFAULT_PROMPT_CSV.as_posix()}`",
            f"- `{DEFAULT_QUALITATIVE_DIR.as_posix()}`",
            "",
        ]
    )
    index_path.write_text("\n".join(body), encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-report highlight figures")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--threshold_csv", type=Path, default=DEFAULT_THRESHOLD_CSV)
    parser.add_argument("--prompt_csv", type=Path, default=DEFAULT_PROMPT_CSV)
    parser.add_argument("--qualitative_dir", type=Path, default=DEFAULT_QUALITATIVE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n_examples", type=int, default=6)
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = read_json(args.metrics)
    threshold_rows = read_rows(args.threshold_csv)
    prompt_rows = read_rows(args.prompt_csv)

    generated = [
        plot_metrics_overview(metrics_payload, args.output_dir),
        plot_size_gap(metrics_payload, args.output_dir),
        plot_threshold_sensitivity(threshold_rows, args.output_dir),
        plot_prompt_comparison(prompt_rows, args.output_dir),
    ]
    generated.extend(
        make_qualitative_grids(
            args.qualitative_dir,
            args.output_dir,
            n_examples=args.n_examples,
        )
    )
    index_path = write_index(args.output_dir, generated)

    print(f"Generated {len(generated)} figures in {args.output_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
