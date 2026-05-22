# CV Project 2026 - Open-Vocabulary Object Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A course project for **CS 6476 Computer Vision** at Georgia Tech, focusing on **Open-Vocabulary Object Detection and Visual Grounding** using Grounding DINO.

## Overview

This project reproduces and evaluates the [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) model for open-vocabulary object detection. The model can detect objects using arbitrary text descriptions rather than a fixed category set.

### Key Features

- **Open-Vocabulary Detection**: Detect objects using free-form text prompts
- **Visual Grounding**: Localize image regions described by natural language expressions
- **COCO Evaluation**: Comprehensive evaluation on the COCO dataset

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- PyTorch 2.0+

### Setup

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

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from src.models import GroundingDINO
from src.inference import Predictor

# Load model
model = GroundingDINO.from_pretrained("checkpoints/groundingdino_swint_ogc.pth")

# Run inference
predictor = Predictor(model)
results = predictor.predict(
    image="path/to/image.jpg",
    text_prompt="the red car on the left"
)
```

## Project Structure

```
CV_Project_2026_S/
├── src/                    # Source code
│   ├── models/            # Model definitions
│   ├── datasets/          # Dataset loaders
│   ├── engine/            # Training/evaluation engine
│   ├── inference/         # Inference tools
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Entry point scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── data/                  # Datasets (not tracked)
├── checkpoints/           # Model weights (not tracked)
└── docs/                  # Documentation
```

## Usage

### Training

```bash
python scripts/train.py --config configs/grounding_dino.yaml
```

### Evaluation

```bash
python scripts/eval.py --config configs/grounding_dino.yaml --checkpoint checkpoints/best.pth
```

### Inference

```bash
python scripts/inference.py --image path/to/image.jpg --text "your text prompt"
```

## Results

| Model | Backbone | AP<sub>50</sub> | AP | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|-------|----------|-----------------|-----|-----------------|-----------------|-----------------|
| Grounding DINO | Swin-T | - | - | - | - | - |
| Grounding DINO | Swin-B | - | - | - | - | - |

*Results will be updated after evaluation.*

## Team

| Member | Role | Focus Area |
|--------|------|------------|
| A | Model Architecture | Backbone, Language Encoder, Fusion Modules |
| B | Data & Training | Dataset Loading, Training Pipeline, Data Augmentation |
| C | Evaluation & Docs | Evaluator, Visualization, Documentation, Testing |

## Timeline

See [docs/project_timeline.md](docs/project_timeline.md) for detailed project timeline and milestones.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research
- [COCO Dataset](https://cocodataset.org/)
- Georgia Tech CS 6476 Course Staff
