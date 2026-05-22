# Installation Guide

This guide provides detailed instructions for setting up the development environment.

## Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU training)
- **Git**: For version control

## Step 1: Clone the Repository

```bash
git clone https://github.com/YangGuanyuhan/CV_Project_2026_S.git
cd CV_Project_2026_S
```

## Step 2: Create Virtual Environment

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Using Conda

```bash
# Create conda environment
conda create -n cv_project python=3.10
conda activate cv_project
```

## Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Step 4: Install Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

## Step 5: Download Data

See [data/README.md](../data/README.md) for instructions on downloading the COCO dataset.

## Step 6: Download Pre-trained Weights

See [checkpoints/README.md](../checkpoints/README.md) for instructions on downloading model weights.

## Step 7: Verify Installation

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
- Reduce batch size in config
- Use gradient accumulation
- Use mixed precision training (fp16)

### Import Errors

Make sure you have installed the package in development mode:
```bash
pip install -e .
```

### Permission Errors on Windows

Run the terminal as administrator or use:
```bash
pip install --user -r requirements.txt
```

## IDE Setup

### VS Code

Install recommended extensions:
- Python
- Pylance
- Ruff
- GitLens

### PyCharm

1. Open project directory
2. Configure Python interpreter to use the virtual environment
3. Enable type checking
