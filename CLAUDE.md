# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Georgia Tech CS 6476 Computer Vision course project reproducing **Grounding DINO** for open-vocabulary object detection and visual grounding on COCO 2017. The project is config-driven (OmegaConf YAML) and the package is `cv-project-2026`.

## Common Commands

```bash
# Install (development mode)
pip install -r requirements.txt && pip install -e ".[dev]"

# Lint & format (ruff)
ruff check .                        # lint
ruff format --check .               # format check
ruff format . && ruff check --fix . # auto-fix

# Tests
pytest tests/ -v                    # run all tests
pytest tests/test_box_ops.py -v     # run single test file
pytest tests/ -v --cov=src --cov-report=html  # with coverage

# Training / Evaluation / Inference
python scripts/train.py --config configs/grounding_dino.yaml
python scripts/eval.py --config configs/grounding_dino.yaml --checkpoint checkpoints/best.pth
python scripts/inference.py --image path/to/image.jpg --text "your text prompt"

# Pre-commit hooks
pre-commit install                  # one-time setup
pre-commit run --all-files          # manual run

# Makefile shortcuts
make install-dev   # install + pre-commit
make lint          # ruff check + format check
make format        # auto-format
make test          # pytest
make train         # train with default config
make eval          # eval with default config
```

## Architecture

```
src/
├── models/          # Model definitions
│   ├── backbone/    # Visual backbone (Swin Transformer)
│   ├── language/    # Language encoders (BERT, CLIP text)
│   └── fusion/      # Cross-modal fusion (deformable attention)
├── datasets/        # COCO dataset loaders and transforms
├── engine/          # Training loop, evaluator (mAP)
├── inference/       # Predictor, visualization
└── utils/           # Box ops (IoU, NMS), misc utilities
configs/             # YAML configs (OmegaConf)
scripts/             # Entry points: train.py, eval.py, inference.py
tests/               # pytest tests
data/                # COCO images + annotations (not tracked)
checkpoints/         # Model weights (not tracked)
```

The model follows Grounding DINO's architecture: **Swin Transformer backbone** → multi-scale features, **BERT language encoder** → text embeddings, **deformable attention fusion** → cross-modal feature interaction, then detection head with learnable queries.

## Configuration System

All configs use OmegaConf YAML (`configs/grounding_dino.yaml`). Scripts load via `OmegaConf.load()` and accept CLI overrides. Key config sections: `model` (backbone, language, fusion), `dataset` (COCO paths, transforms), `training` (optimizer, scheduler, fp16), `evaluation` (mAP settings).

## Code Style

- **Ruff** for both linting and formatting (line length 100, target Python 3.10)
- Lint rules: E, W, F (pyflakes), I (isort), B (flake8-bugbear), UP (pyupgrade)
- CI runs `ruff check .` and `ruff format --check .` on push/PR to main/develop
- Pre-commit hooks: ruff, trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, debug-statements

## Git Conventions

- Branches: `feat/`, `fix/`, `docs/`, `refactor/`, `test/` prefixes
- Commits: Conventional Commits format — `feat(models): add Swin Transformer backbone`
- PRs target `develop` branch, require at least one approval
