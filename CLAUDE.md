# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Georgia Tech CS 6476 Computer Vision course project reproducing **Grounding DINO** for open-vocabulary object detection and visual grounding on COCO 2017. The project is config-driven (OmegaConf YAML) and the package is `cv-project-2026`.

**Model Strategy**: Uses the official `groundingdino-py` pip package as the inference backend. Our code wraps it with project-specific interfaces for inference, evaluation, and visualization. We do NOT implement the model architecture from scratch.

## Common Commands

```bash
# Install (development mode)
pip install -r requirements.txt && pip install -e ".[dev]"

# Environment setup (server)
bash scripts/setup_env.sh            # Linux
.\scripts\setup_env.ps1              # Windows PowerShell

# Download weights and data
python scripts/download_weights.py   # Download Grounding DINO checkpoint
python scripts/download_coco.py      # Download COCO 2017 val2017 + annotations

# Lint & format (ruff)
ruff check .                        # lint
ruff format --check .               # format check
ruff format . && ruff check --fix . # auto-fix

# Tests
pytest tests/ -v                    # run all tests
pytest tests/test_box_ops.py -v     # run single test file

# Inference
python scripts/inference.py \
    --image path/to/image.jpg \
    --text "person . car . dog ." \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth

# Batch inference
python scripts/inference.py \
    --image_dir path/to/images/ \
    --text "person . bicycle . car ." \
    --output_dir outputs/inference_results

# COCO Evaluation (subset)
python scripts/eval.py \
    --config configs/grounding_dino.yaml \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json \
    --subset_size 100

# COCO Evaluation (full val2017)
python scripts/eval.py \
    --config configs/grounding_dino.yaml \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json

# Ablation experiments
python scripts/run_experiments.py \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json \
    --subset_size 500
```

## Architecture

```
src/
├── models/
│   └── grounding_dino.py    # Wrapper around official groundingdino-py
├── datasets/
│   ├── coco_categories.py   # 80 COCO classes, prompt builder, phrase mapping
│   └── coco_dataset.py      # Dataset helper, sanity check, subset selection
├── engine/
│   └── evaluator.py         # COCOeval integration (inference → COCO format → metrics)
├── inference/
│   ├── predictor.py         # High-level predictor (single/batch/directory)
│   ├── visualizer.py        # Detection box drawing, GT+pred comparison
│   └── coco_visualizer.py   # COCO success/failure case visualization
└── utils/
    └── box_ops.py           # IoU, NMS, cxcywh↔xyxy conversion

scripts/
├── inference.py             # CLI: single/batch image inference
├── eval.py                  # CLI: COCO evaluation (subset or full)
├── run_experiments.py       # CLI: threshold/prompt ablation experiments
├── download_weights.py      # Download Grounding DINO checkpoints
├── download_coco.py         # Download COCO 2017 dataset
├── setup_env.sh             # Linux environment setup
└── setup_env.ps1            # Windows environment setup

configs/
└── grounding_dino.yaml      # Model, dataset, training, eval, grounding_dino settings

tests/
└── test_box_ops.py          # 7 tests for box operations (all passing)

docs/
├── plans/                   # Detailed project plans (6 files)
├── stage_reports/           # Stage-by-stage report templates
└── final_report.md          # Final report template
```

## Key Interfaces

```python
# Model wrapper
from src.models import GroundingDINOModel
model = GroundingDINOModel(
    config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
    checkpoint_path="checkpoints/groundingdino_swint_ogc.pth",
)
boxes, scores, phrases = model.predict(image, "person . car .", box_threshold=0.35)

# High-level predictor
from src.inference import Predictor
predictor = Predictor(config_path=..., checkpoint_path=...)
result = predictor.predict("image.jpg", "person . car .")

# COCO evaluation
from src.engine import COCOEvaluator
evaluator = COCOEvaluator(model, "data/coco/annotations/instances_val2017.json")
raw = evaluator.run_inference("data/coco/val2017", image_ids=[...])
coco_results = evaluator.convert_to_coco_format(raw)
metrics = evaluator.evaluate(coco_results=coco_results)

# Box utilities
from src.utils.box_ops import box_iou, nms, box_cxcywh_to_xyxy
```

## Configuration System

All configs use OmegaConf YAML (`configs/grounding_dino.yaml`). Scripts load via `OmegaConf.load()` and accept CLI overrides. Key config sections:

- `model` — backbone, language encoder, fusion module settings
- `dataset` — COCO paths, transforms
- `training` — optimizer, scheduler, fp16
- `evaluation` — mAP settings
- `grounding_dino` — model_config, checkpoint, thresholds, prompt, COCO paths

## Grounding DINO Integration

The project uses `groundingdino-py` as the official inference backend:
- Config file: `groundingdino/config/GroundingDINO_SwinT_OGC.py` (from the pip package)
- Checkpoint: `checkpoints/groundingdino_swint_ogc.pth` (downloaded separately)
- Default thresholds: box_threshold=0.35, text_threshold=0.25
- Prompt format: dot-separated COCO categories, e.g., `person . bicycle . car .`
- Phrase-to-category mapping: substring matching in `src/datasets/coco_categories.py`

## COCO Evaluation Protocol

- Dataset: COCO 2017 val2017 (5,000 images)
- Annotation: `instances_val2017.json`
- Metric: pycocotools COCOeval (AP, AP50, AP75, APS, APM, APL)
- Target: Swin-T OGC zero-shot AP ≈ 48.4/48.5
- Result format: `[{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}]`

## Code Style

- **Ruff** for both linting and formatting (line length 100, target Python 3.10)
- Lint rules: E, W, F (pyflakes), I (isort), B (flake8-bugbear), UP (pyupgrade)
- CI runs `ruff check .` and `ruff format --check .` on push/PR to main/develop
- Pre-commit hooks: ruff, trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, debug-statements

## Git Conventions

- Branches: `feat/`, `fix/`, `docs/`, `refactor/`, `test/` prefixes
- Commits: Conventional Commits format — `feat(models): add Swin Transformer backbone`
- PRs target `develop` branch, require at least one approval

## Server Portability

All paths are configurable via `configs/grounding_dino.yaml` CLI overrides. No hardcoded absolute paths. When transferring to a server:

1. Clone repo and run `bash scripts/setup_env.sh`
2. Run `python scripts/download_weights.py` and `python scripts/download_coco.py`
3. All outputs go to `outputs/` directory
