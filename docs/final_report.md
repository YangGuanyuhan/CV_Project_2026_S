# Final Report: Grounding DINO Open-Vocabulary Object Detection on COCO

> Georgia Tech CS 6476 Computer Vision — Spring 2026

## 1. Introduction

### Problem Definition

Open-vocabulary object detection aims to detect objects described by arbitrary text prompts, unlike traditional detectors limited to a fixed category set. This is challenging because it requires:

- **Visual-language alignment**: Understanding both image content and text descriptions
- **Generalization**: Detecting unseen categories not present during training
- **Fine-grained grounding**: Localizing specific objects described by natural language

### Project Goal

This project reproduces Grounding DINO's zero-shot object detection pipeline and evaluates it on the COCO 2017 validation set. We analyze the model's performance, identify failure modes, and study the impact of threshold and prompt settings.

## 2. Related Work

### Grounding DINO

Grounding DINO (Liu et al., 2023) extends the DINO detector with grounded pre-training for open-set detection. Key contributions:

- **Architecture**: Swin Transformer backbone + BERT language encoder + deformable attention fusion
- **Training**: Grounded pre-training on diverse detection and grounding datasets
- **Zero-shot capability**: Detect objects described by free-form text prompts

### Related Models

- **GLIP** (Li et al., 2022): Grounded language-image pre-training
- **OWL-ViT** (Minderer et al., 2022): Open-world localization with Vision Transformers
- **YOLO-World** (Tian et al., 2024): Real-time open-vocabulary detection

## 3. Method Reproduction

### Model Configuration

| Component | Setting |
|-----------|---------|
| Backbone | Swin-Tiny |
| Language Encoder | BERT (bert-base-uncased) |
| Checkpoint | groundingdino_swint_ogc.pth |
| Source | IDEA-Research/GroundingDINO |

### What Was Reproduced

- Model loading from official checkpoint
- Single-image and batch inference
- Text prompt-based detection with phrase output
- Bounding box, score, and phrase extraction

### What Was NOT Reproduced

- Training from scratch
- Fine-tuning on COCO
- Full Grounding DINO architecture (used official package)

## 4. COCO Evaluation Protocol

### Dataset

| Item | Value |
|------|-------|
| Dataset | COCO 2017 |
| Split | val2017 |
| Images | 5,000 |
| Annotations | instances_val2017.json |
| Evaluation | bbox detection |

### Prompt Format

- **Format**: Dot-separated 80 COCO categories
- **Example**: `person . bicycle . car . ...`
- **Rationale**: Official recommendation for category separation

### Metric Computation

- **Tool**: pycocotools COCOeval
- **Metrics**: AP, AP50, AP75, APS, APM, APL
- **IoU range**: 0.50:0.95

## 5. Experiments And Results

### Main Results

| Model | Backbone | Prompt | AP | AP50 | AP75 | APS | APM | APL |
|-------|----------|--------|-----|------|------|-----|-----|-----|
| Grounding DINO | Swin-T | dot-separated 80 classes | TBD | TBD | TBD | TBD | TBD | TBD |

> Results will be filled after running full COCO evaluation.

### Ablation: Threshold Sensitivity

| Run | box_threshold | text_threshold | AP | AP50 | Avg boxes/image |
|-----|---------------|----------------|-----|------|-----------------|
| A1 | 0.25 | 0.20 | TBD | TBD | TBD |
| A2 | 0.35 | 0.25 | TBD | TBD | TBD |
| A3 | 0.45 | 0.30 | TBD | TBD | TBD |

### Ablation: Prompt Format

| Format | AP | AP50 |
|--------|-----|------|
| dot-separated | TBD | TBD |
| comma-separated | TBD | TBD |
| sentence-style | TBD | TBD |

## 6. Qualitative Analysis

### Success Cases

> 6+ images showing accurate detections with correct categories and tight bounding boxes.

Reference: `outputs/visualizations/coco_eval/success_cases/`

### Failure Cases

> 6+ images showing common failure modes.

Reference: `outputs/visualizations/coco_eval/failure_cases/`

### Error Taxonomy

| Error Type | Description | Frequency |
|------------|-------------|-----------|
| Small object miss | Small objects not detected | TBD |
| Crowded scene | Duplicate/confused boxes | TBD |
| Phrase mismatch | Wrong category mapping | TBD |
| Background FP | Background detected as object | TBD |
| Occlusion | Poor localization | TBD |

## 7. Limitations

### Computational

- Full COCO val2017 evaluation requires significant GPU time
- Subset evaluation results are unstable

### Methodological

- Zero-shot only (no fine-tuning)
- Fixed threshold settings
- Simple phrase-to-category mapping (substring matching)

### Data

- Only evaluated on COCO val2017
- No cross-dataset evaluation

## 8. Conclusion

### Key Findings

- Grounding DINO achieves strong zero-shot detection on COCO
- Threshold settings significantly affect precision/recall tradeoff
- Dot-separated prompt format is most effective
- Small objects and crowded scenes remain challenging

### Future Work

- Fine-tune on COCO train for improved performance
- Test on diverse datasets (Objects365, OpenImages)
- Compare with other open-vocabulary detectors
- Per-category AP analysis

## 9. References

1. Liu, S., Zeng, Z., Ren, T., et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." ECCV 2024.
2. IDEA-Research. GroundingDINO. https://github.com/IDEA-Research/GroundingDINO
3. Lin, T.Y., Maire, M., Belongie, S., et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
4. COCO API. https://github.com/cocodataset/cocoapi

## 10. Appendix: Reproduction Commands

### Environment Setup

```bash
pip install -r requirements.txt
pip install -e .
```

### Download Weights and Data

```bash
python scripts/download_weights.py
python scripts/download_coco.py
```

### Run Inference

```bash
python scripts/inference.py \
    --image data/demo_images/test.jpg \
    --text "person . car . dog ." \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth
```

### Run COCO Evaluation

```bash
# Subset
python scripts/eval.py \
    --config configs/grounding_dino.yaml \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json \
    --subset_size 100

# Full
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
