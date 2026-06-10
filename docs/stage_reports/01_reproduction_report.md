# Stage Report: Grounding DINO Reproduction

## 1. Actual Work Completed

> **Status**: ✅ Completed
>
> Successfully reproduced Grounding DINO inference pipeline with official pretrained weights.

### Environment Setup

- Python version: 3.10.20
- PyTorch version: 2.7.1+cu118
- CUDA version: 11.8
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- groundingdino-py version: 0.4.0

### Model Loading

- Config: `groundingdino/config/GroundingDINO_SwinT_OGC.py`
- Checkpoint: `checkpoints/groundingdino_swint_ogc.pth`
- Device: cuda

### Demo Inference

- Number of test images: 5
- Output directory: `outputs/inference_demo/`
- Results: All 5 images processed successfully, 9 total detections

## 2. Original Plan vs Actual Outcome

| Plan Item | Status | Notes |
|-----------|--------|-------|
| Environment setup | ✅ Done | conda env grounding_dino, Python 3.10, PyTorch 2.7.1+cu118 |
| Weight download | ✅ Done | groundingdino_swint_ogc.pth (662MB) |
| Single image inference | ✅ Done | Via Predictor class |
| Batch inference wrapper | ✅ Done | Via scripts/inference.py --image_dir |
| 3+ demo visualizations | ✅ Done | 5 annotated images saved |
| 3+ prediction JSONs | ✅ Done | 5 prediction JSONs saved |

## 3. Problem Definition

This stage solves the problem of reproducing Grounding DINO's inference pipeline:
- Load the official pretrained model
- Run inference with text prompts
- Obtain bounding boxes, confidence scores, and phrase labels

The goal is NOT to train a new model, but to verify the zero-shot inference pipeline works correctly.

## 4. Approach

1. Install `groundingdino-py` official package
2. Download Swin-T OGC checkpoint
3. Wrap the official API in `src/models/grounding_dino.py`
4. Create `Predictor` class for high-level inference
5. Run inference on 3+ test images with COCO category prompts
6. Save annotated images and prediction JSONs

### Key Commands

```bash
# Environment setup
pip install -r requirements.txt
pip install -e .

# Download weights
python scripts/download_weights.py

# Run inference
python scripts/inference.py --image data/demo_images/test.jpg --text "person . car ."
```

## 5. Simplifications And Assumptions

- Using official pretrained weights (not training from scratch)
- Zero-shot inference only (no fine-tuning)
- Using the official `groundingdino-py` package rather than reimplementing
- Default thresholds: box_threshold=0.35, text_threshold=0.25

## 6. Results And Evaluation

### Demo Images

| Image | Detections | Inference Time | Notes |
|-------|-----------|----------------|-------|
| 000000000139.jpg | 5 | 6.093s | First image includes model warmup |
| 000000000285.jpg | 1 | 1.089s | |
| 000000000632.jpg | 1 | 0.967s | |
| 000000000776.jpg | 0 | 0.977s | No objects matched the prompt |
| 000000000872.jpg | 2 | 0.990s | |
| **Average** | **1.8** | **2.087s** | Excluding warmup: ~1.0s/image |

### Visualizations

> Reference: `outputs/inference_demo/*.jpg`

Annotated images with bounding boxes and confidence scores saved for all 5 test images.

## 7. Limitations And Failure Cases

- First image inference is slow (~6s) due to CUDA warmup and model initialization
- Some images produce 0 detections when no objects match the text prompt (e.g., 000000000776.jpg)
- Text prompt design affects detection quality — dot-separated format is recommended
- Inference speed is ~1s/image on RTX 4060 Laptop GPU

## 8. Future Work

- Extend to COCO val2017 evaluation
- Test with different threshold settings
- Compare different prompt formats

## 9. References

- Grounding DINO: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" (Liu et al., 2023)
- GroundingDINO Official Repository: https://github.com/IDEA-Research/GroundingDINO
- groundingdino-py: https://pypi.org/project/groundingdino-py/
