# Grounding DINO + COCO Project Plans

本目录用于管理本项目的阶段性计划与验收标准。项目目标已经固定为：

- 复现 Grounding DINO 开放词汇目标检测流程。
- 在 COCO 2017 数据集上完成评估与分析。

## Plan Files

| 文件 | 作用 | 主要验收对象 |
|---|---|---|
| [00_project_plan.md](00_project_plan.md) | 总体路线、范围、里程碑、交付物 | 项目边界、阶段完成定义、最终交付清单 |
| [01_reproduction_plan.md](01_reproduction_plan.md) | Grounding DINO 环境搭建与推理复现 | 可运行环境、官方权重、单图/批量推理输出 |
| [02_coco_evaluation_plan.md](02_coco_evaluation_plan.md) | COCO 数据准备、推理、格式转换、指标评估 | COCO result JSON、COCOeval 指标、可复现实验日志 |
| [03_experiment_analysis_plan.md](03_experiment_analysis_plan.md) | 阈值、prompt、模型配置等实验与误差分析 | 实验表格、可视化案例、失败类型总结 |
| [04_report_submission_plan.md](04_report_submission_plan.md) | 最终报告、演示材料与仓库交付检查 | 报告结构、结果复现说明、提交前 checklist |
| [05_documentation_requirements.md](05_documentation_requirements.md) | 每个阶段必须完成的详细文档规范 | 阶段报告、假设说明、局限性、引用 |
| [06_single_gpu_finetune_plan.md](06_single_gpu_finetune_plan.md) | 单卡微调实施方案（命令、耗时、排查） | 单卡可运行训练、短程微调结果、可复现实验记录 |

## Recommended Execution Order

1. 完成 [00_project_plan.md](00_project_plan.md) 中的项目范围确认。
2. 按 [01_reproduction_plan.md](01_reproduction_plan.md) 复现 Grounding DINO 推理流程。
3. 按 [02_coco_evaluation_plan.md](02_coco_evaluation_plan.md) 完成 COCO 子集和完整验证集评估。
4. 按 [03_experiment_analysis_plan.md](03_experiment_analysis_plan.md) 做定量和定性分析。
5. 按 [04_report_submission_plan.md](04_report_submission_plan.md) 准备最终报告、演示和代码仓库。

## Common Acceptance Rule

每个阶段都需要同时满足以下三类证据：

- **可运行证据**：命令、配置文件、日志或 notebook 能证明流程实际跑通。
- **可检查产物**：输出文件保存在 `outputs/`、`docs/` 或 `notebooks/` 中，路径明确。
- **可解释结论**：每个结果表格或图像至少配有一句说明，解释它验证了什么、还存在什么限制。
- **详细文档说明**：每个阶段都必须写明实际完成了什么、没有完成什么、采取了哪些简化或假设、结果如何评估、方法什么时候不工作、为什么不工作、继续做会怎么改进，并列出本阶段参考或使用的相关工作。

## Required Stage Documentation

每个阶段结束时必须提交一份阶段说明文档。文档质量优先于结果规模：一个小结果如果被准确描述、认真评估，并清楚说明限制，会比一个没有验证清楚的宏大结果更有价值。

| 阶段 | 必须文档 | 最低内容 |
|---|---|---|
| 复现阶段 | `docs/stage_reports/01_reproduction_report.md` | 实际跑通的模型、环境、命令、demo 结果、失败问题、引用 |
| COCO 评估阶段 | `docs/stage_reports/02_coco_evaluation_report.md` | 数据、prompt、类别映射、评估协议、指标、异常结果解释 |
| 实验分析阶段 | `docs/stage_reports/03_experiment_analysis_report.md` | 实验设置、结果表、可视化、失败类型、局限性 |
| 最终提交阶段 | `docs/final_report.md` 或 PDF | 完整问题定义、方法、简化假设、评估、局限性、未来工作、参考文献 |

所有阶段文档必须遵守 [05_documentation_requirements.md](05_documentation_requirements.md)。
