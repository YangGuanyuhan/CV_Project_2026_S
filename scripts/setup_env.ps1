# Environment setup script for Windows (PowerShell)
# Usage: .\scripts\setup_env.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Grounding DINO Project - Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
python --version

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install project in development mode
Write-Host "Installing project..." -ForegroundColor Yellow
pip install -e ".[dev]"

# Verify installation
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Verifying installation..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
try {
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>$null
} catch {
    Write-Host "GPU: Not available" -ForegroundColor Yellow
}
try {
    python -c "from groundingdino.util.inference import load_model; print('groundingdino-py: OK')"
} catch {
    Write-Host "groundingdino-py: FAILED" -ForegroundColor Red
}

# Save environment info
Write-Host ""
Write-Host "Saving environment info..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path outputs | Out-Null

$envInfo = @"
Environment Info
================
Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Python: $(python --version 2>&1)
PyTorch: $(python -c "import torch; print(torch.__version__)" 2>&1)
CUDA available: $(python -c "import torch; print(torch.cuda.is_available())" 2>&1)
CUDA version: $(python -c "import torch; print(torch.version.cuda)" 2>&1)
GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1)
OS: Windows $(Get-WmiObject Win32_OperatingSystem | Select-Object -ExpandProperty Version)
"@

$envInfo | Out-File -FilePath outputs/env_info.txt -Encoding UTF8

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Download model weights: python scripts/download_weights.py"
Write-Host "  2. Download COCO data: python scripts/download_coco.py"
Write-Host "  3. Run inference: python scripts/inference.py --image test.jpg --text 'person .'"
