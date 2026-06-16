# Plan 00: Project Roadmap

## 1. Project Goal

本项目复现 Grounding DINO 的开放词汇目标检测流程，并在 COCO 2017 上完成检测性能评估、可视化和误差分析。项目重点不是从零训练大模型，而是完成一个可复现、可评估、可解释的 pipeline。

## 2. Scope

### In Scope

- 使用 Grounding DINO 官方开源实现或等价 HuggingFace/本地封装实现推理。
- 使用官方预训练权重，优先选择 `GroundingDINO-T / Swin-T / OGC` checkpoint。
- 在 COCO 2017 `val2017` 上做 zero-shot object detection evaluation。
- 支持小规模 COCO subset 快速验证，以及完整 `val2017` 正式评估。
- 输出 COCO API 可读取的 detection result JSON。
- 使用 `pycocotools` 或 COCO API 计算 `AP`, `AP50`, `AP75`, `APS`, `APM`, `APL`。
- 完成至少 1 组阈值或 prompt 设置对比实验。
- 给出定性可视化、失败案例和原因分析。

### Out of Scope

- 不要求从零复现 Grounding DINO 的完整预训练过程。
- 不要求在 COCO `train2017` 上完整 fine-tune。
- 不要求和 GLIP、OWL-ViT、YOLO-World、Detic 做完整横向对比。
- 不要求实现新的模型结构。

## 3. Milestones

| 阶段 | 时间建议 | 目标 | 主要产物 |
|---|---:|---|---|
| M1 | Week 1 | 环境搭建、权重下载、官方 demo 跑通 | demo 输出图、环境记录 |
| M2 | Week 2 | 封装本项目推理入口 | `scripts/inference.py` 可运行、JSON/图像输出 |
| M3 | Week 3 | COCO 数据准备和 subset evaluation 跑通 | subset result JSON、COCOeval 输出 |
| M4 | Week 4 | 完整 COCO val2017 评估 | full val 指标表、日志、配置快照 |
| M5 | Week 5 | 实验分析和可视化 | ablation 表格、失败案例图 |
| M6 | Week 6 | 报告、演示、仓库整理 | final report、README 更新、提交清单 |

## 4. Repository Deliverables

| 交付物 | 建议路径 | 说明 |
|---|---|---|
| 环境说明 | `docs/installation.md` 或 `docs/reproduction.md` | Python/CUDA/PyTorch/GroundingDINO 版本 |
| 推理脚本 | `scripts/inference.py` | 单图或批量图片推理 |
| COCO 评估脚本 | `scripts/eval.py` | 生成 COCO result JSON 并调用 COCOeval |
| 配置文件 | `configs/grounding_dino.yaml` | 权重、阈值、prompt、数据路径 |
| 评估输出 | `outputs/coco_eval/` | result JSON、metrics、logs |
| 可视化输出 | `outputs/visualizations/` | 检测框图像、失败案例 |
| 阶段说明文档 | `docs/stage_reports/` | 每个阶段实际完成内容、假设、评估、局限性、未来工作 |
| 最终报告 | `docs/final_report.md` 或 PDF | 方法、实验、分析、结论 |

## 5. Global Acceptance Criteria

项目完成时必须满足以下验收标准：

- [ ] 可以从干净环境按照文档完成依赖安装，且 `python -c "import torch"`、`python -c "import groundingdino"` 或等价导入成功。
- [ ] 官方或本项目封装的 Grounding DINO demo 能对至少 3 张自定义图片输出检测框、类别短语和置信度。
- [ ] COCO `val2017` 数据路径、annotation 路径、类别 prompt 生成逻辑在配置或文档中明确记录。
- [ ] 至少完成一次 COCO subset evaluation，subset 大小、图片 ID 列表或随机种子可追溯。
- [ ] 至少完成一次 COCO `val2017` 完整评估，或在算力不足时给出明确原因并提供可复现 subset 结果。
- [ ] 评估输出包含 COCO API 标准指标：`AP`, `AP50`, `AP75`, `APS`, `APM`, `APL`。
- [ ] 若使用官方 Swin-T OGC 权重和完整 COCO val2017，结果应接近官方 zero-shot AP 约 `48.4/48.5`；若差距明显，报告中必须解释可能原因。
- [ ] 至少包含 1 张定量结果表、6 张定性可视化图、3 类失败案例分析。
- [ ] 最终报告能说明：模型原理、复现步骤、COCO 评估协议、实验结果、误差分析和局限性。

## 6. Mandatory Documentation Policy

本项目强制执行阶段文档制度。每个阶段只有同时完成代码/实验产物和阶段说明文档，才算完成。

阶段文档必须体现以下原则：

- 准确描述实际完成的工作；如果某个原计划没有完成，必须明确说明原因。
- 详细解释本阶段试图解决的问题，而不是只列命令或贴结果。
- 说明采取了哪些简化、假设或替代方案，例如只评估 COCO subset、只使用官方权重、不做 fine-tuning。
- 用具体证据评估结果，包括指标、日志、可视化或失败案例。
- 主动说明方法的限制：什么时候不工作、为什么不工作。
- 写出如果继续做，下一步会如何改进。
- 引用本阶段阅读、复用或对比过的论文、仓库、数据集和工具。

建议文档路径：

```text
docs/stage_reports/
  01_reproduction_report.md
  02_coco_evaluation_report.md
  03_experiment_analysis_report.md
docs/final_report.md
```

## 7. Key Risks And Mitigation

| 风险 | 影响 | 应对方式 | 验收方式 |
|---|---|---|---|
| CUDA 或编译失败 | 无法运行官方扩展 | 记录 CUDA_HOME、PyTorch/CUDA 版本；必要时使用 CPU-only 或 Colab | 安装日志和推理 demo 成功 |
| COCO 完整评估耗时长 | 无法按时完成完整 val | 先跑 100/500 张 subset，再安排完整评估 | subset 指标和 full-run 计划都有记录 |
| prompt 格式不一致 | 指标波动大 | 固定 COCO 类别 prompt，类别名用 `.` 分隔 | 配置文件和日志保存 prompt |
| box 格式转换错误 | COCOeval 指标无效 | 添加小样本人工检查 | result JSON 通过 COCO API 加载 |
| 阈值设置不合理 | 漏检或误检严重 | 记录 `box_threshold` 和 `text_threshold`，做小规模搜索 | ablation 表格可复现 |
| 文档只写理想目标 | 评分时无法判断实际贡献 | 每阶段强制写 actual work、assumptions、limitations、future work | 阶段报告通过 checklist |

## 8. Definition Of Done

本项目的最终完成定义：

- 代码能跑，数据能评，结果能复现，结论能解释。
- 仓库中存在完整的命令、配置、日志和结果文件路径。
- 报告中的每个核心数值都能追溯到具体运行命令和输出文件。
- 每个阶段都有详细说明文档，能清楚区分原计划、实际完成内容和未完成内容。
