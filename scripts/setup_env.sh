#!/bin/bash
# Environment setup script for Linux servers
# Usage: bash scripts/setup_env.sh

set -e

echo "=========================================="
echo "Grounding DINO Project - Environment Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment (optional)
if [ "$1" = "--venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install project in development mode
echo "Installing project..."
pip install -e ".[dev]"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
python -c "from groundingdino.util.inference import load_model; print('groundingdino-py: OK')"

# Save environment info
echo ""
echo "Saving environment info..."
mkdir -p outputs
cat > outputs/env_info.txt << EOF
Environment Info
================
Date: $(date)
Python: $(python --version)
PyTorch: $(python -c "import torch; print(torch.__version__)")
CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")
CUDA version: $(python -c "import torch; print(torch.version.cuda)")
GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')")
OS: $(uname -a)
EOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download model weights: python scripts/download_weights.py"
echo "  2. Download COCO data: python scripts/download_coco.py"
echo "  3. Run inference: python scripts/inference.py --image test.jpg --text 'person .'"
