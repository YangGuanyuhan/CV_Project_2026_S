# Plan 04: Report And Submission

## 1. Objective

整理最终项目报告、演示材料和仓库交付内容，确保老师或助教能够按照文档复现核心流程，并理解实验结果和分析结论。

最终报告必须诚实描述项目实际完成的内容。不要把未完成的目标写成已完成工作；如果实际结果较小，也要通过清晰的问题定义、方法说明、评估证据和局限性分析体现价值。

## 2. Final Report Structure

建议报告结构：

```text
1. Introduction
2. Related Work
3. Method Reproduction
4. COCO Evaluation Protocol
5. Experiments And Results
6. Qualitative Analysis
7. Limitations
8. Conclusion
9. References
10. Appendix: Reproduction Commands
```

每一节最低要求：

| 章节 | 必须回答的问题 |
|---|---|
| Introduction | 为什么开放词汇检测比固定类别检测更有挑战？本项目目标是什么？ |
| Related Work | Grounding DINO 与传统 detector / DETR-style detector / vision-language model 的关系是什么？ |
| Method Reproduction | 使用了哪个 checkpoint、config、prompt、threshold？实际复现了哪些组件？哪些没有复现？ |
| COCO Evaluation Protocol | 数据 split、类别映射、result JSON 格式、指标计算方式和简化假设是什么？ |
| Experiments And Results | 主结果和 ablation 结果是什么？和官方指标是否接近？如果不接近，可能原因是什么？ |
| Qualitative Analysis | 成功案例和失败案例分别说明了什么？方法什么时候不工作？ |
| Limitations | 算力、数据、prompt、类别映射、模型版本有哪些限制？这些限制如何影响结论？ |
| Conclusion | 最终学到了什么？如果继续做，下一步会改进什么？ |

## 3. Required Figures And Tables

| 类型 | 数量 | 说明 |
|---|---:|---|
| 模型结构图 | 1 | 可引用 Grounding DINO 论文/官方结构图，并说明模块作用 |
| 主结果表 | 1 | COCO AP/AP50/AP75/APS/APM/APL |
| ablation 表 | 1 | threshold 或 prompt 对比 |
| 成功案例图 | >= 6 | 检测框准确、类别正确 |
| 失败案例图 | >= 6 | 漏检、误检、定位不准或类别映射错误 |
| pipeline 图 | 1 | image + prompt -> model -> boxes/phrases -> COCO JSON -> COCOeval |

验收标准：

- [ ] 每个表格都有标题和简短解释。
- [ ] 每张图都有 caption，说明输入 prompt 和观察结论。
- [ ] 报告中的核心指标能对应到 `outputs/` 中的 metrics 文件。
- [ ] 没有只展示可视化但不解释现象的图片。

## 4. Reproducibility Section

报告或 README 中必须包含：

- 环境版本：Python、PyTorch、CUDA、GPU、GroundingDINO commit 或 package version。
- 权重信息：checkpoint 名称、下载链接、hash 或文件大小。
- 数据信息：COCO split、annotation 文件、图片数量。
- 推理命令：单图 demo 和 COCO evaluation 命令。
- 配置路径：使用的 YAML/Python config。
- 输出路径：metrics、predictions、visualizations。

验收标准：

- [ ] 新成员可以根据 README 在不读源码的情况下运行 demo。
- [ ] 新成员可以根据 README 启动 COCO subset evaluation。
- [ ] 所有绝对本地路径都被替换为相对路径或占位符。
- [ ] 文档中明确说明完整 COCO evaluation 的预计耗时和硬件。

## 5. Repository Submission Checklist

提交前检查：

- [ ] `README.md` 更新项目目标、安装、推理、评估命令。
- [ ] `docs/plans/` 中的计划文件存在且状态最新。
- [ ] `docs/stage_reports/01_reproduction_report.md` 已完成，且描述实际复现结果。
- [ ] `docs/stage_reports/02_coco_evaluation_report.md` 已完成，且描述实际评估范围和指标。
- [ ] `docs/stage_reports/03_experiment_analysis_report.md` 已完成，且描述实验、失败案例和局限性。
- [ ] `docs/final_report.md` 或最终 PDF 已完成，且包含 actual work、assumptions、limitations、future work、references。
- [ ] `configs/grounding_dino.yaml` 中默认路径合理，不包含个人机器专属路径。
- [ ] `scripts/inference.py --help` 能运行。
- [ ] `scripts/eval.py --help` 能运行。
- [ ] 大文件没有误提交到 Git，例如 COCO 图片、完整 checkpoint、巨大输出文件。
- [ ] `.gitignore` 覆盖数据集、权重、缓存、日志和临时输出。
- [ ] `outputs/` 中保留轻量示例或通过文档说明如何生成，不提交过大的结果。

## 6. Presentation Plan

建议 8-10 页 slides：

| 页 | 内容 |
|---:|---|
| 1 | 项目题目、成员、目标 |
| 2 | 问题定义：开放词汇检测和 visual grounding |
| 3 | Grounding DINO 方法概览 |
| 4 | 复现 pipeline |
| 5 | COCO evaluation protocol |
| 6 | 主结果表 |
| 7 | threshold/prompt ablation |
| 8 | 成功和失败案例 |
| 9 | 局限性与改进方向 |
| 10 | 总结 |

验收标准：

- [ ] slides 中至少包含 1 张 pipeline 图。
- [ ] slides 中至少包含 1 张结果表。
- [ ] slides 中至少包含 4 张可视化案例。
- [ ] 演示时能在 2 分钟内说明如何复现 demo。

## 7. Final Acceptance Criteria

最终提交必须满足：

- [ ] 复现 Grounding DINO 推理流程。
- [ ] 完成 COCO subset evaluation。
- [ ] 完成 COCO full val2017 evaluation，或给出算力不足说明和可复现 subset 结果。
- [ ] 输出标准 COCO detection metrics。
- [ ] 至少完成 1 组 ablation。
- [ ] 至少完成 12 张 qualitative visualization。
- [ ] 每个阶段都有详细文档说明，且能区分原计划、实际完成内容和未完成内容。
- [ ] 最终报告准确描述实际完成的项目，不夸大结果。
- [ ] 最终报告包含方法、协议、结果、分析、简化假设、局限性、失败场景、未来工作和引用。
- [ ] 仓库文档足以让他人复现核心结果。

## 8. Final Report Documentation Checklist

最终报告提交前逐项检查：

- [ ] 是否准确描述了实际解决的问题，而不是泛泛介绍开放词汇检测。
- [ ] 是否详细解释了 approach，包括模型、数据、prompt、阈值、类别映射和评估方式。
- [ ] 是否明确列出 simplifications and assumptions，例如 zero-shot、官方权重、subset、固定 threshold。
- [ ] 是否用结果表、日志或可视化评估了每个核心结论。
- [ ] 是否展示并解释了方法不工作的场景。
- [ ] 是否说明了为什么会失败，例如小目标、遮挡、拥挤场景、phrase/category mismatch。
- [ ] 是否写出继续工作时会做的具体步骤，而不是泛泛写“提高精度”。
- [ ] 是否引用所有实际使用或参考的论文、仓库、数据集和工具。

## 9. References To Cite

报告至少引用：

- Grounding DINO paper: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection".
- GroundingDINO official repository.
- COCO dataset paper or official website.
- COCO API / pycocotools repository.

引用格式可以使用 BibTeX、IEEE 或课程指定格式，但需要保持全文一致。
