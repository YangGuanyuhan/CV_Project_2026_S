# Stage Report: Grounding DINO Reproduction

## 1. Actual Work Completed

> **Status**: Pending implementation
>
> This report will be completed after running the reproduction pipeline. Fill in with actual results.

### Environment Setup

- Python version: TBD
- PyTorch version: TBD
- CUDA version: TBD
- GPU: TBD
- groundingdino-py version: TBD

### Model Loading

- Config: `groundingdino/config/GroundingDINO_SwinT_OGC.py`
- Checkpoint: `checkpoints/groundingdino_swint_ogc.pth`
- Device: TBD

### Demo Inference

- Number of test images: TBD
- Output directory: `outputs/inference_demo/`
- Results: TBD

## 2. Original Plan vs Actual Outcome

| Plan Item | Status | Notes |
|-----------|--------|-------|
| Environment setup | TBD | |
| Weight download | TBD | |
| Single image inference | TBD | |
| Batch inference wrapper | TBD | |
| 3+ demo visualizations | TBD | |
| 3+ prediction JSONs | TBD | |

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

> Fill in after running inference.

### Demo Images

| Image | Detections | Inference Time | Notes |
|-------|-----------|----------------|-------|
| TBD | TBD | TBD | |

### Visualizations

> Reference: `outputs/inference_demo/*.jpg`

## 7. Limitations And Failure Cases

- TBD: Cases where the model fails to detect objects
- TBD: Cases of incorrect phrase matching
- TBD: Performance on edge cases

## 8. Future Work

- Extend to COCO val2017 evaluation
- Test with different threshold settings
- Compare different prompt formats

## 9. References

- Grounding DINO: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" (Liu et al., 2023)
- GroundingDINO Official Repository: https://github.com/IDEA-Research/GroundingDINO
- groundingdino-py: https://pypi.org/project/groundingdino-py/
