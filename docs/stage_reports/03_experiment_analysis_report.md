# Stage Report: Experiment Analysis

## 1. Actual Work Completed

> **Status**: Pending implementation
>
> This report will be completed after running experiments. Fill in with actual results.

### Experiments Run

| Experiment | Status | Notes |
|------------|--------|-------|
| A: Threshold Sensitivity | TBD | |
| B: Prompt Format Comparison | TBD | |
| C: Qualitative Error Analysis | TBD | |

## 2. Original Plan vs Actual Outcome

> Fill in with actual experiment outcomes.

## 3. Problem Definition

This stage analyzes Grounding DINO's behavior through controlled experiments:
- How do thresholds affect precision/recall?
- How does prompt format affect detection quality?
- What are the common failure modes?

## 4. Approach

### Experiment A: Threshold Sensitivity

| Run | box_threshold | text_threshold | Dataset |
|-----|---------------|----------------|---------|
| A1 | 0.25 | 0.20 | COCO subset 500 |
| A2 | 0.35 | 0.25 | COCO subset 500 |
| A3 | 0.45 | 0.30 | COCO subset 500 |

### Experiment B: Prompt Format Comparison

| Run | Format | Example |
|-----|--------|---------|
| B1 | dot-separated | `person . bicycle . car .` |
| B2 | comma-separated | `person, bicycle, car,` |
| B3 | sentence-style | `There are person, bicycle and car in the image.` |

### Experiment C: Qualitative Error Taxonomy

| Error Type | Description |
|------------|-------------|
| Small object miss | Small objects not detected |
| Crowded scene confusion | Duplicate boxes or category confusion |
| Phrase/category mismatch | Phrase doesn't map to correct COCO category |
| Background false positive | Background detected as object |
| Occlusion/truncation | Poor localization of occluded objects |

## 5. Simplifications And Assumptions

- Using COCO subset (500 images) for ablation
- Only 3 threshold settings tested
- Only 3 prompt formats compared
- Error classification is manual/qualitative

## 6. Results And Evaluation

> Fill in after running experiments.

### Ablation Tables

> Reference: `outputs/experiments/threshold_sensitivity/summary.csv`
> Reference: `outputs/experiments/prompt_comparison/summary.csv`

### Qualitative Analysis

> Reference: `outputs/experiments/qualitative_errors/`

## 7. Limitations And Failure Cases

- TBD: Which categories are most affected by threshold changes
- TBD: Which prompt format works best and why
- TBD: Most common failure modes

## 8. Future Work

- Per-category AP analysis
- More threshold combinations
- Full val2017 ablation
- Comparison with other models (OWL-ViT, YOLO-World)

## 9. References

- Grounding DINO paper: Liu et al., 2023
- COCO dataset: Lin et al., 2014
