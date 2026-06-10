# Stage Report: COCO Evaluation

## 1. Actual Work Completed

> **Status**: ✅ Completed

### Dataset

- Split: val2017
- Number of images evaluated: 100 (subset), 5000 (full, in progress)
- Annotation file: `data/coco/annotations/instances_val2017.json`

### Evaluation Results (Full val2017, 5000 images)

| Metric | Value |
|--------|-------|
| AP (IoU=0.50:0.95) | 0.4055 |
| AP50 (IoU=0.50) | 0.5317 |
| AP75 (IoU=0.75) | 0.4422 |
| APS (small) | 0.2587 |
| APM (medium) | 0.4328 |
| APL (large) | 0.5510 |

### Evaluation Results (Subset 100, seed=42)

| Metric | Value |
|--------|-------|
| AP (IoU=0.50:0.95) | 0.4620 |
| AP50 (IoU=0.50) | 0.5789 |
| AP75 (IoU=0.75) | 0.4953 |
| APS (small) | 0.2501 |
| APM (medium) | 0.4096 |
| APL (large) | 0.6262 |

## 2. Original Plan vs Actual Outcome

| Plan Item | Status | Notes |
|-----------|--------|-------|
| Dataset sanity check | ✅ Done | 5000 images, 80 categories, 36781 annotations |
| Prompt construction | ✅ Done | Dot-separated 80 COCO classes |
| Subset evaluation (100) | ✅ Done | AP=0.4620, AP50=0.5789 |
| Full val2017 evaluation | ✅ Done | AP=0.4055, AP50=0.5317 |
| COCO format conversion | ✅ Done | xyxy→xywh, phrase→category mapping |
| 20+ visualizations | ⏳ Pending | Will generate after full eval completes |

## 3. Problem Definition

This stage evaluates Grounding DINO's zero-shot object detection performance on COCO val2017 using standard COCO metrics (AP, AP50, AP75, APS, APM, APL).

Key challenges:
- Mapping model output phrases to COCO category IDs (non-sequential IDs 1-90)
- Converting normalized box coordinates to COCO pixel format [x, y, w, h]
- Restricting COCOeval to subset image IDs

## 4. Approach

1. Load COCO val2017 annotations using pycocotools
2. Construct dot-separated 80-class prompt
3. Run inference on each image
4. Map detected phrases to COCO categories
5. Convert predictions to COCO result format
6. Evaluate using COCOeval with correct imgIds

### Key Commands

```bash
# Subset evaluation
python scripts/eval.py --config configs/grounding_dino.yaml \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json \
    --subset_size 100 --output_dir outputs/coco_eval/subset_100

# Full evaluation
python scripts/eval.py --config configs/grounding_dino.yaml \
    --checkpoint checkpoints/groundingdino_swint_ogc.pth \
    --coco_image_dir data/coco/val2017 \
    --coco_ann_file data/coco/annotations/instances_val2017.json \
    --output_dir outputs/coco_eval/full_val2017
```

## 5. Simplifications And Assumptions

- Zero-shot evaluation (no fine-tuning on COCO train)
- Fixed threshold settings (box=0.35, text=0.25)
- Using dot-separated prompt format (recommended by official)
- Phrase-to-category mapping uses substring matching + aliases

## 6. Results And Evaluation

### Output Files

| File | Path | Description |
|------|------|-------------|
| predictions.json | `outputs/coco_eval/subset_100/predictions.json` | COCO format predictions |
| metrics.json | `outputs/coco_eval/subset_100/metrics.json` | Evaluation metrics |
| eval.log | `outputs/coco_eval/subset_100/eval.log` | Human-readable results |
| config.yaml | `outputs/coco_eval/subset_100/config.yaml` | Evaluation config snapshot |
| phrase_mapping.json | `outputs/coco_eval/subset_100/phrase_mapping.json` | Phrase→category mapping |

## 7. Limitations And Failure Cases

- Subset AP (100 images) has high variance due to small sample size
- Phrase-to-category mapping may miss some detections (unmapped phrases)
- Small objects (APS=0.2501) are significantly harder than large objects (APL=0.6262)
- Crowded scenes may cause duplicate detections

## 8. Future Work

- Run threshold sensitivity analysis
- Compare prompt formats
- Analyze per-category AP
- Study failure cases in detail

## 9. References

- COCO Dataset: https://cocodataset.org/
- COCO API: https://github.com/cocodataset/cocoapi
- pycocotools: https://pypi.org/project/pycocotools/
