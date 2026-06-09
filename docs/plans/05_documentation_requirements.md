# Plan 05: Documentation Requirements

## 1. Purpose

本文件规定项目每个阶段必须完成的详细文档说明。它的核心原则是：准确描述和评估实际完成的工作，而不是只描述原本希望完成的目标。一个小但被认真验证、清楚解释其限制的结果，比一个宏大但没有任何部分做扎实的结果更有价值。

## 2. Mandatory Rule

每个阶段都必须提交阶段文档。没有阶段文档的阶段不能标记为完成，即使代码或实验已经运行。

阶段文档必须回答：

- 本阶段实际做了什么？
- 本阶段原计划做什么，但最终没有完成什么？
- 试图解决的具体问题是什么？
- 方法细节是什么，包括模型、数据、命令、配置和输出？
- 做了哪些简化、假设或替代方案？
- 结果如何评估？证据在哪里？
- 方法什么时候不工作？为什么不工作？
- 如果继续做，下一步会做什么？
- 本阶段参考、复用或比较了哪些相关工作？

## 3. Required Files

| 阶段 | 文件路径 | 完成条件 |
|---|---|---|
| Grounding DINO 复现 | `docs/stage_reports/01_reproduction_report.md` | demo 推理结果被描述和评估 |
| COCO 评估 | `docs/stage_reports/02_coco_evaluation_report.md` | 数据、协议、指标和异常结果被解释 |
| 实验分析 | `docs/stage_reports/03_experiment_analysis_report.md` | ablation、可视化和失败案例被分析 |
| 最终报告 | `docs/final_report.md` 或 PDF | 全项目问题、方法、结果、局限性和引用完整 |

## 4. Required Stage Report Template

每个阶段报告建议使用以下结构：

```markdown
# Stage Report: <阶段名称>

## 1. Actual Work Completed

说明本阶段实际完成了哪些内容。必须具体到模型、数据、脚本、配置、命令和输出路径。

## 2. Original Plan vs Actual Outcome

说明原计划是什么，实际完成了什么，哪些内容没有完成，以及原因。

## 3. Problem Definition

准确描述本阶段试图解决的问题。不要把问题写得比实际工作更大。

## 4. Approach

详细说明方法流程，包括输入、模型、处理步骤、输出格式和评估方式。

## 5. Simplifications And Assumptions

列出所有简化和假设，例如使用官方权重、只跑 subset、固定阈值、只评估 bbox、不做训练。

## 6. Results And Evaluation

用指标、表格、图片、日志或人工检查说明结果。每个主要结果都要指向可检查文件。

## 7. Limitations And Failure Cases

说明方法什么时候不工作、失败案例是什么、可能原因是什么。

## 8. Future Work

说明如果继续做，会优先改进哪些部分，以及为什么。

## 9. References

列出本阶段阅读、使用、复用或对比的论文、仓库、数据集和工具。
```

## 5. Evidence Requirements

阶段文档中的每个核心结论都需要对应证据：

| 结论类型 | 可接受证据 |
|---|---|
| 环境已搭建 | 环境版本、安装日志、导入测试输出 |
| demo 已复现 | 推理命令、输出图片、预测 JSON、运行日志 |
| COCO 评估已完成 | `predictions.json`、`metrics.json`、COCOeval summary |
| ablation 有结论 | 多个 run 的配置、指标表、差异解释 |
| 存在失败案例 | 可视化图片、错误说明、可能原因 |
| 存在限制 | 具体触发条件、现象、原因分析 |

## 6. Quality Checklist

每份阶段文档提交前必须检查：

- [ ] 是否只声称已经实际完成并验证过的内容。
- [ ] 是否清楚区分 original plan 和 actual outcome。
- [ ] 是否说明所有重要简化和假设。
- [ ] 是否为每个指标或图像给出解释。
- [ ] 是否说明至少一个限制或失败场景。
- [ ] 是否写出下一步工作，而不是只说“效果还有待提升”。
- [ ] 是否列出相关论文、官方仓库、数据集和工具引用。
- [ ] 是否给出输出文件路径，方便他人检查。

## 7. References That Must Be Considered

最终报告和相关阶段报告至少考虑引用：

- Grounding DINO paper: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection".
- GroundingDINO official repository.
- COCO dataset and COCO official website.
- COCO API or `pycocotools`.

如果项目中实际使用了其他实现、教程、模型权重、数据处理脚本或评估工具，也必须在对应阶段文档中引用。
