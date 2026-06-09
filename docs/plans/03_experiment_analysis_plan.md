# Plan 03: Experiments And Analysis

## 1. Objective

在基础 COCO evaluation 跑通后，通过少量可控实验分析 Grounding DINO 的性能变化、成功场景和失败模式，为最终报告提供定量和定性证据。

## 2. Required Experiments

至少完成以下 3 类实验中的 2 类；如果时间允许，建议全部完成。

### Experiment A: Threshold Sensitivity

目的：分析 `box_threshold` 和 `text_threshold` 对 AP、漏检、误检的影响。

建议设置：

| Run | box_threshold | text_threshold | Dataset |
|---|---:|---:|---|
| A1 | 0.25 | 0.20 | COCO subset 500 |
| A2 | 0.35 | 0.25 | COCO subset 500 |
| A3 | 0.45 | 0.30 | COCO subset 500 |

验收标准：

- [ ] 每个 run 都有独立 config、predictions、metrics、log。
- [ ] 输出一张表，包含 `AP`, `AP50`, `AP75`, 平均每图预测框数量。
- [ ] 结论中说明阈值升高后 precision/recall 的变化趋势。

### Experiment B: Prompt Format Comparison

目的：分析类别 prompt 写法对开放词汇检测结果的影响。

建议设置：

| Run | Prompt format | Example |
|---|---|---|
| B1 | Dot-separated categories | `person . bicycle . car .` |
| B2 | Comma-separated categories | `person, bicycle, car,` |
| B3 | Sentence-style prompt | `There are person, bicycle and car in the image.` |

验收标准：

- [ ] 至少比较 2 种 prompt 格式。
- [ ] 每种 prompt 的完整文本保存到输出目录。
- [ ] 指标表包含 AP 和 AP50。
- [ ] 报告中解释为什么 Grounding DINO 官方建议类别名用 `.` 分隔。

### Experiment C: Qualitative Error Taxonomy

目的：总结模型在 COCO 场景中的典型失败案例。

建议失败类型：

| 类型 | 说明 |
|---|---|
| Small object miss | 小目标漏检 |
| Crowded scene confusion | 拥挤场景中重复框或类别混淆 |
| Phrase/category mismatch | phrase 与 COCO 类别映射不稳定 |
| Background false positive | 背景区域被误检为目标 |
| Occlusion/truncation | 遮挡或截断目标定位不准 |

验收标准：

- [ ] 至少选择 12 张 COCO 图片进行可视化分析。
- [ ] 至少包含 6 个成功案例和 6 个失败案例。
- [ ] 每张失败案例图配一句错误说明。
- [ ] 最终整理出不少于 3 类主要失败原因。

## 3. Optional Experiments

如果进度充足，可以补充：

- Swin-T 和 Swin-B checkpoint 对比。
- CPU 与 GPU 推理耗时对比。
- 不同 subset size 的指标稳定性对比，例如 100、500、1000 张。
- 针对 COCO 小目标类别的单独分析。
- 对高频类别和低频类别分别统计 AP 或 AP50。

验收标准：

- [ ] 可选实验必须有明确问题，不做无解释的结果堆叠。
- [ ] 每个可选实验至少包含 1 个定量表或 1 组定性图。
- [ ] 报告中说明该实验对项目结论的贡献。

## 4. Analysis Outputs

建议输出结构：

```text
outputs/experiments/
  threshold_sensitivity/
    run_a1/
    run_a2/
    run_a3/
    summary.csv
    summary.md
  prompt_comparison/
    run_b1/
    run_b2/
    summary.csv
    summary.md
  qualitative_errors/
    success_cases/
    failure_cases/
    error_taxonomy.md
```

## 5. Tables For Final Report

### Main Result Table

| Model | Checkpoint | Dataset | Prompt | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Grounding DINO Swin-T | OGC | COCO val2017 | dot-separated 80 classes | TBD | TBD | TBD | TBD | TBD | TBD |

### Ablation Table

| Experiment | box_threshold | text_threshold | Prompt format | AP | AP50 | Avg boxes/image |
|---|---:|---:|---|---:|---:|---:|
| Baseline | 0.35 | 0.25 | dot-separated | TBD | TBD | TBD |

## 6. Analysis Acceptance Checklist

本阶段完成定义：

- [ ] 至少完成 2 类实验，其中必须包含 threshold 或 prompt 对比之一。
- [ ] 所有实验都能追溯到具体配置和输出目录。
- [ ] 至少有 1 张主结果表和 1 张 ablation 表。
- [ ] 至少有 12 张定性分析图。
- [ ] 失败案例总结不少于 3 类，并解释可能原因。
- [ ] 最终结论不是只复述指标，而是能说明 Grounding DINO 在 COCO 上擅长什么、不擅长什么。
- [ ] 完成 `docs/stage_reports/03_experiment_analysis_report.md`，详细说明实际完成的实验、原计划未完成部分、实验假设、指标解释、失败案例、局限性、未来工作和引用。

## 7. Required Stage Documentation

`docs/stage_reports/03_experiment_analysis_report.md` 必须至少包含：

- **Actual Work**：实际完成了哪些 threshold、prompt 或 qualitative error 实验。
- **Original Plan vs Actual Outcome**：哪些实验没有完成，原因是时间、算力、实现限制还是结果无效。
- **Problem Definition**：本阶段要解决的问题是理解模型表现，而不是单纯堆叠结果。
- **Approach**：每个实验的变量、固定条件、数据规模、运行命令和输出路径。
- **Simplifications And Assumptions**：例如只用 subset 做 ablation、只比较少量阈值、只做人工错误分类。
- **Results And Evaluation**：主结果表、ablation 表、可视化图和对趋势的解释。
- **Limitations And Failure Cases**：至少 3 类失败场景，说明什么时候不工作以及可能原因。
- **Future Work**：继续做时如何扩展实验，例如类别级 AP、更多 prompt、full val 对比。
- **References**：实验设计或分析中参考的论文、官方仓库、COCO API 或其他工具。

## 8. Interpretation Guide

写分析时优先回答这些问题：

- AP 和 AP50 的差异是否说明定位精度不足？
- 小目标 AP 是否明显低于中/大目标？
- 降低 threshold 是提升 recall，还是带来过多 false positives？
- dot-separated prompt 是否比自然句 prompt 更稳定？
- 哪些类别更容易被 prompt 或 phrase mapping 影响？
- 失败案例主要来自视觉困难，还是语言类别映射困难？
