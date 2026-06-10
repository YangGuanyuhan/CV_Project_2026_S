# CV Project 2026 - Open-Vocabulary Object Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A course project for **CS 6476 Computer Vision** at Georgia Tech, focusing on **Open-Vocabulary Object Detection and Visual Grounding** using Grounding DINO.

## Overview

This project reproduces and evaluates the [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) model for open-vocabulary object detection. The model can detect objects using arbitrary text descriptions rather than a fixed category set.

### Key Features

- **Open-Vocabulary Detection**: Detect objects using free-form text prompts
- **Visual Grounding**: Localize image regions described by natural language expressions
- **COCO Evaluation**: Comprehensive evaluation on the COCO 2017 validation set
- **Ablation Studies**: Threshold sensitivity and prompt format analysis

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- PyTorch 2.0+

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/YangGuanyuhan/CV_Project_2026_S.git
cd CV_Project_2026_S

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Server Setup

```bash
# Automated setup script
bash scripts/setup_env.sh

# Or with virtual environment
bash scripts/setup_env.sh --venv
```

### Download Weights and Data

```bash
# Download Grounding DINO weights
python scripts/download_weights.py

# Download COCO 2017 val2017 + annotations
python scripts/download_coco.py
```

## Quick Start

### Single Image Inference

```bash
python scripts/inference.py \
    --image path/to/image.jpg \
    --text "person . car . dog ." \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth
```

### Batch Inference

```bash
python scripts/inference.py \
    --image_dir path/to/images/ \
    --text "person . bicycle . car ." \
    --output_dir outputs/inference_results
```

### COCO Evaluation

```bash
# Subset evaluation (100 images)
python scripts/eval.py \
    --config configs/grounding_dino.yaml \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json \
    --subset_size 100

# Full val2017 evaluation
python scripts/eval.py \
    --config configs/grounding_dino.yaml \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json
```

### Run Experiments

```bash
python scripts/run_experiments.py \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json \
    --subset_size 500
```

## Project Structure

```
CV_Project_2026_S/
├── src/                        # Source code
│   ├── models/                # Model wrapper (Grounding DINO)
│   │   └── grounding_dino.py  # Official model wrapper
│   ├── datasets/              # COCO dataset utilities
│   │   ├── coco_categories.py # 80 COCO classes, prompt builder
│   │   └── coco_dataset.py    # Dataset helper
│   ├── engine/                # Evaluation engine
│   │   └── evaluator.py       # COCOeval integration
│   ├── inference/             # Inference tools
│   │   ├── predictor.py       # High-level predictor
│   │   ├── visualizer.py      # Detection visualization
│   │   └── coco_visualizer.py # COCO-specific visualization
│   └── utils/                 # Utilities
│       └── box_ops.py         # IoU, NMS, coordinate conversion
├── configs/
│   └── grounding_dino.yaml    # Project configuration
├── scripts/
│   ├── inference.py           # Single/batch inference
│   ├── eval.py                # COCO evaluation
│   ├── run_experiments.py     # Ablation experiments
│   ├── download_weights.py    # Download model weights
│   ├── download_coco.py       # Download COCO dataset
│   ├── setup_env.sh           # Linux environment setup
│   └── setup_env.ps1          # Windows environment setup
├── tests/
│   └── test_box_ops.py        # Unit tests for box operations
├── data/                      # Datasets (not tracked)
├── checkpoints/               # Model weights (not tracked)
├── outputs/                   # Results and visualizations
└── docs/
    ├── plans/                 # Project plans
    ├── stage_reports/         # Stage-by-stage reports
    └── final_report.md        # Final project report
```

## Results

| Model | Backbone | AP<sub>50</sub> | AP | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|-------|----------|-----------------|-----|-----------------|-----------------|-----------------|
| Grounding DINO | Swin-T | 0.5317 | 0.4055 | 0.2587 | 0.4328 | 0.5510 |

*Zero-shot results on COCO val2017 (5000 images), box_threshold=0.35, text_threshold=0.25.*

### Report Highlights

Generated report-ready figures are available in `outputs/visualizations/report_highlights/`:

- `01_metrics_overview.png`: full COCO val2017 metric overview
- `02_size_gap.png`: small/medium/large object performance gap
- `03_threshold_sensitivity.png`: threshold ablation and detection-count tradeoff
- `04_prompt_format_comparison.png`: prompt-format ablation
- `05_success_cases_grid.jpg`: qualitative success examples
- `06_failure_cases_grid.jpg`: qualitative failure examples

Regenerate them with:

```bash
python scripts/generate_report_highlights.py
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Linting

```bash
ruff check .          # Check for issues
ruff check --fix .    # Auto-fix issues
ruff format .         # Format code
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Team

| Member | Role | Focus Area |
|--------|------|------------|
| A | Model Architecture | Backbone, Language Encoder, Fusion Modules |
| B | Data & Training | Dataset Loading, Training Pipeline, Data Augmentation |
| C | Evaluation & Docs | Evaluator, Visualization, Documentation, Testing |

## Documentation

- [Installation Guide](docs/installation.md)
- [Project Timeline](docs/project_timeline.md)
- [Project Plans](docs/plans/)
- [Stage Reports](docs/stage_reports/)
- [Final Report](docs/final_report.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research
- [COCO Dataset](https://cocodataset.org/)
- Georgia Tech CS 6476 Course Staff
