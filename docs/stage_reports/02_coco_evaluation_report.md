# Stage Report: COCO Evaluation

## 1. Actual Work Completed

> **Status**: Pending implementation
>
> This report will be completed after running COCO evaluation. Fill in with actual results.

### Dataset

- Split: val2017
- Number of images evaluated: TBD
- Annotation file: `data/coco/annotations/instances_val2017.json`

### Evaluation Results

| Metric | Value |
|--------|-------|
| AP (IoU=0.50:0.95) | TBD |
| AP50 (IoU=0.50) | TBD |
| AP75 (IoU=0.75) | TBD |
| APS (small) | TBD |
| APM (medium) | TBD |
| APL (large) | TBD |

## 2. Original Plan vs Actual Outcome

| Plan Item | Status | Notes |
|-----------|--------|-------|
| Dataset sanity check | TBD | |
| Prompt construction | TBD | |
| Subset evaluation (100) | TBD | |
| Full val2017 evaluation | TBD | |
| COCO format conversion | TBD | |
| 20+ visualizations | TBD | |

## 3. Problem Definition

This stage evaluates Grounding DINO's zero-shot object detection performance on COCO val2017 using standard COCO metrics (AP, AP50, AP75, APS, APM, APL).

Key challenges:
- Mapping model output phrases to COCO category IDs
- Converting normalized box coordinates to COCO pixel format [x, y, w, h]
- Handling the 80-class COCO prompt format

## 4. Approach

1. Load COCO val2017 annotations using pycocotools
2. Construct dot-separated 80-class prompt
3. Run inference on each image
4. Map detected phrases to COCO categories
5. Convert predictions to COCO result format
6. Evaluate using COCOeval

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
- Phrase-to-category mapping uses substring matching

## 6. Results And Evaluation

> Fill in after running evaluation.

### Output Files

| File | Path | Description |
|------|------|-------------|
| predictions.json | `outputs/coco_eval/*/predictions.json` | COCO format predictions |
| metrics.json | `outputs/coco_eval/*/metrics.json` | Evaluation metrics |
| eval.log | `outputs/coco_eval/*/eval.log` | Human-readable results |
| config.yaml | `outputs/coco_eval/*/config.yaml` | Evaluation config snapshot |

## 7. Limitations And Failure Cases

- Subset AP is unstable due to small sample size
- Phrase-to-category mapping may miss some detections
- Small objects may be frequently missed
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
