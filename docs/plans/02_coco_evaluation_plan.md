# Plan 02: COCO Evaluation

## 1. Objective

在 COCO 2017 上评估 Grounding DINO 的开放词汇目标检测性能，输出 COCO API 标准指标，并保存可复现的配置、日志和结果文件。

## 2. Dataset Protocol

| 项 | 默认选择 |
|---|---|
| Dataset | COCO 2017 |
| Split | `val2017` |
| Images | 5,000 validation images |
| Annotation | `instances_val2017.json` |
| Evaluation type | bbox detection |
| Metric tool | `pycocotools.cocoeval.COCOeval` |

建议数据结构：

```text
data/coco/
  annotations/
    instances_val2017.json
  val2017/
    000000000139.jpg
    ...
```

## 3. Evaluation Pipeline

### Step 1: Dataset Sanity Check

执行内容：

- 检查 COCO annotation 文件是否可被 COCO API 加载。
- 检查 `val2017` 图片数量。
- 随机抽取 5 张图片，确认 image id、file name、宽高和实际图片一致。

验收标准：

- [ ] `instances_val2017.json` 可正常加载。
- [ ] 完整评估时 `val2017` 图片数量为 5,000；subset 评估时记录 subset 数量。
- [ ] 至少 5 张图片通过 annotation 与实际尺寸一致性检查。
- [ ] 数据检查日志保存到 `outputs/coco_eval/data_check.log`。

### Step 2: Prompt Construction

执行内容：

- 使用 COCO 80 个类别名构造统一 prompt。
- 推荐格式：`person . bicycle . car . ...`
- 固定类别名顺序，保持和 COCO category id 映射一致。
- 保存 prompt 文本和类别映射。

验收标准：

- [ ] `outputs/coco_eval/coco_prompt.txt` 存在。
- [ ] `outputs/coco_eval/category_mapping.json` 存在，包含 `category_id`, `name`, `prompt_index` 或等价字段。
- [ ] prompt 中包含 80 个 COCO 类别。
- [ ] 不同实验使用的 prompt 都有独立记录，不能只写在代码中。

### Step 3: Model Inference On COCO

执行内容：

- 对 subset 或完整 `val2017` 逐图运行 Grounding DINO。
- 保存原始模型输出。
- 将模型输出转换为 COCO detection result 格式。

COCO result JSON 每条记录建议格式：

```json
{
  "image_id": 139,
  "category_id": 1,
  "bbox": [413.0, 158.0, 53.0, 138.0],
  "score": 0.92
}
```

注意：

- `bbox` 必须是 COCO 格式 `[x, y, width, height]`，单位是像素。
- `category_id` 必须使用 COCO 原始 category id，不是 0-based index。
- 对 Grounding DINO 输出的 phrase/token，需要映射回 COCO 类别名。

验收标准：

- [ ] subset 至少完成 100 张图片推理；正式评估建议完成 5,000 张 `val2017`。
- [ ] result JSON 可以被 `COCO.loadRes()` 成功加载。
- [ ] 每条预测包含 `image_id`, `category_id`, `bbox`, `score`。
- [ ] bbox 坐标为像素级 `[x, y, w, h]`，且 `w > 0`, `h > 0`。
- [ ] 输出文件保存到 `outputs/coco_eval/predictions_*.json`。

### Step 4: COCOeval Metrics

执行内容：

- 使用 `pycocotools` 计算 bbox AP。
- 输出并保存 COCO 标准 summary。
- 同时保存机器可读 metrics JSON。

建议命令：

```powershell
python scripts/eval.py `
  --config configs/grounding_dino.yaml `
  --checkpoint checkpoints/groundingdino_swint_ogc.pth `
  --coco_image_dir data/coco/val2017 `
  --coco_ann_file data/coco/annotations/instances_val2017.json `
  --output_dir outputs/coco_eval/full_val2017 `
  --split val2017
```

验收标准：

- [ ] COCOeval 成功输出 `Average Precision` 和 `Average Recall` summary。
- [ ] 至少记录 `AP`, `AP50`, `AP75`, `APS`, `APM`, `APL`。
- [ ] 指标保存到 `outputs/coco_eval/full_val2017/metrics.json`。
- [ ] 运行日志保存到 `outputs/coco_eval/full_val2017/eval.log`。
- [ ] 配置快照保存到 `outputs/coco_eval/full_val2017/config.yaml`。

## 4. Result Targets

| 设置 | 期望结果 | 验收方式 |
|---|---:|---|
| Official Swin-T OGC, full COCO val2017, zero-shot | AP 接近 `48.4/48.5` | 若差距超过 3 AP，需要排查 prompt、阈值、box 格式、类别映射 |
| 100-image subset | 不设固定 AP 目标 | 只验收 pipeline 正确性和日志完整性 |
| 500-image subset | 不设固定 AP 目标 | 用于快速比较阈值和 prompt |

说明：subset AP 波动会很大，不能直接和官方完整 COCO val 指标比较。

## 5. Debug Checklist

如果 AP 明显异常，按以下顺序排查：

- [ ] annotation 文件和图片 split 是否匹配。
- [ ] `image_id` 是否使用 COCO 原始 id。
- [ ] `category_id` 是否使用 COCO 原始 category id。
- [ ] bbox 是否为 `[x, y, w, h]` 像素坐标，而不是归一化坐标。
- [ ] 是否错误交换了 `x/y` 或 `w/h`。
- [ ] prompt 是否包含所有 80 类，且类别名拼写正确。
- [ ] phrase 到 category 的映射是否稳定。
- [ ] `box_threshold` 是否过高导致大量漏检。
- [ ] `text_threshold` 是否过高导致类别短语缺失。

## 6. Evaluation Acceptance Checklist

本阶段完成定义：

- [ ] COCO 数据检查通过。
- [ ] subset evaluation 跑通，并保存预测 JSON、metrics、日志。
- [ ] full val2017 evaluation 跑通，或有明确算力限制说明和替代 subset 评估。
- [ ] 指标表能放入最终报告。
- [ ] 至少保存 20 张 COCO 可视化样例，其中包含成功和失败案例。
- [ ] 所有结果都能追溯到具体 config、checkpoint、prompt 和 threshold。
- [ ] 完成 `docs/stage_reports/02_coco_evaluation_report.md`，详细说明实际评估范围、数据 split、prompt 和类别映射、评估指标、简化假设、异常结果、局限性和引用。

## 7. Expected Evidence

```text
outputs/coco_eval/
  coco_prompt.txt
  category_mapping.json
  data_check.log
  subset_100/
    predictions.json
    metrics.json
    eval.log
    config.yaml
  full_val2017/
    predictions.json
    metrics.json
    eval.log
    config.yaml
outputs/visualizations/coco_eval/
  success_*.jpg
  failure_*.jpg
docs/stage_reports/
  02_coco_evaluation_report.md
```

## 8. Required Stage Documentation

`docs/stage_reports/02_coco_evaluation_report.md` 必须至少包含：

- **Actual Work**：实际完成的是 COCO subset、full `val2017`，还是两者都有。
- **Problem Definition**：本阶段要解决的问题是用 COCO API 量化评估 Grounding DINO 的 bbox detection 性能。
- **Approach**：数据准备、prompt 构造、模型推理、phrase/category 映射、bbox 格式转换和 COCOeval 调用流程。
- **Simplifications And Assumptions**：例如只跑 100/500 张 subset、固定阈值、不做 test-dev、不做 fine-tuning。
- **Evaluation Evidence**：`predictions.json`、`metrics.json`、`eval.log`、关键 AP 表和可视化样例。
- **Limitations**：subset 指标不稳定、类别映射错误、阈值敏感、小目标漏检等问题。
- **Failure Cases**：至少列出 3 个评估中发现的失败场景，并解释可能原因。
- **Next Steps**：继续工作时如何扩展到 full val、更多阈值搜索、类别级分析或更稳定的 mapping。
- **References**：COCO dataset、COCO API/pycocotools、GroundingDINO 官方 evaluation 说明。
