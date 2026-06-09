# Plan 01: Grounding DINO Reproduction

## 1. Objective

完成 Grounding DINO 推理流程复现：安装依赖、加载官方配置和权重、运行单图/批量图片推理，并将结果保存为可视化图片和结构化 JSON。

## 2. Recommended Baseline

| 项 | 选择 |
|---|---|
| Model | Grounding DINO |
| Backbone | Swin-T |
| Checkpoint | `groundingdino_swint_ogc.pth` |
| Config | `GroundingDINO_SwinT_OGC.py` 或本项目 `configs/grounding_dino.yaml` 对应配置 |
| Default text prompt | COCO 80 类别名，用 `. ` 分隔 |
| Default box threshold | `0.35` |
| Default text threshold | `0.25` |

说明：官方 README 建议不同类别或短语用 `.` 分隔，并提供了 Swin-T OGC 权重的单图 demo 与 COCO zero-shot evaluation 示例。

## 3. Tasks

### Task 1: Environment Setup

执行内容：

- 创建 Python 环境，建议 Python 3.10。
- 安装 PyTorch、torchvision、transformers、opencv、pycocotools 等依赖。
- 安装 GroundingDINO 官方包或将官方实现作为本项目依赖。
- 记录 CUDA、PyTorch、GPU 型号和系统信息。

建议命令：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

验收标准：

- [ ] `python --version` 输出 Python 3.10 或更高版本。
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 能正常运行。
- [ ] Grounding DINO 相关模块可以导入；如果使用官方源码，`pip install -e .` 无报错。
- [ ] 环境信息保存到 `outputs/env_info.txt` 或文档中。

### Task 2: Checkpoint And Config Preparation

执行内容：

- 下载官方 `groundingdino_swint_ogc.pth` 权重到 `checkpoints/` 或 `weights/`。
- 明确配置文件路径。
- 记录权重来源、文件大小、下载日期和 hash。

建议路径：

```text
checkpoints/
  groundingdino_swint_ogc.pth
configs/
  grounding_dino.yaml
```

验收标准：

- [ ] 权重文件存在，且路径写入配置文件。
- [ ] 配置文件中明确包含 backbone、checkpoint、box threshold、text threshold。
- [ ] 权重来源 URL 和 hash 记录在 `checkpoints/README.md` 或实验日志中。

### Task 3: Single Image Inference

执行内容：

- 选择至少 3 张测试图片，覆盖不同物体数量和场景。
- 使用自然语言 prompt 或 COCO 类别 prompt 运行推理。
- 保存带检测框图片、原始预测 JSON、运行日志。

建议输出：

```text
outputs/inference/
  image_001_annotated.jpg
  image_001_predictions.json
  image_002_annotated.jpg
  image_002_predictions.json
  image_003_annotated.jpg
  image_003_predictions.json
```

验收标准：

- [ ] 至少 3 张图片完成推理。
- [ ] 每张图片都有 annotated image 和 predictions JSON。
- [ ] JSON 中至少包含 `boxes`, `scores`, `phrases` 或等价字段。
- [ ] 检测框坐标和图像尺寸一致，没有明显越界。
- [ ] 可视化结果能肉眼看到主要目标被定位。

### Task 4: Batch Inference Wrapper

执行内容：

- 将单图推理整理为项目入口，例如 `scripts/inference.py`。
- 支持输入图片文件夹、prompt、checkpoint、config、threshold、输出目录。
- 支持保存 JSON Lines 或每图一个 JSON。

建议命令：

```powershell
python scripts/inference.py `
  --image_dir data/demo_images `
  --text "person . bicycle . car . dog . cat ." `
  --checkpoint checkpoints/groundingdino_swint_ogc.pth `
  --output_dir outputs/inference_demo `
  --box_threshold 0.35 `
  --text_threshold 0.25
```

验收标准：

- [ ] 命令行参数有 `--help` 说明。
- [ ] 输入单张图片和图片目录都能运行，或文档中明确只支持其中一种。
- [ ] 输出目录自动创建。
- [ ] 日志记录图片数量、平均推理时间、阈值、prompt。
- [ ] 异常输入有清晰报错，例如图片路径不存在、权重路径不存在。

## 4. Reproduction Acceptance Checklist

本阶段完成定义：

- [ ] 可以用一条命令完成 demo 推理。
- [ ] 可以复现官方 demo 的基本行为：输入 `(image, text)`，输出检测框、文本短语、置信度。
- [ ] 项目中保留至少 3 个可视化样例。
- [ ] 项目中保留至少 3 个结构化预测文件。
- [ ] README 或文档中记录了完整复现命令。
- [ ] 复现过程没有依赖本地隐藏路径；换机器后只需修改数据和权重路径。
- [ ] 完成 `docs/stage_reports/01_reproduction_report.md`，详细说明实际复现了什么、哪些官方能力没有复现、环境/算力假设、demo 结果、失败问题和参考资料。

## 5. Expected Evidence

| 证据 | 建议路径 |
|---|---|
| 环境记录 | `outputs/env_info.txt` |
| demo 图片 | `outputs/inference_demo/*.jpg` |
| demo 预测 | `outputs/inference_demo/*.json` |
| 运行日志 | `outputs/inference_demo/run.log` |
| 复现说明 | `docs/reproduction.md` 或 README 的 Quick Start |
| 阶段报告 | `docs/stage_reports/01_reproduction_report.md` |

## 6. Required Stage Documentation

`docs/stage_reports/01_reproduction_report.md` 必须至少包含：

- **Actual Work**：本阶段实际跑通的模型、权重、配置、命令和输出。
- **Problem Definition**：本阶段要解决的问题是复现 Grounding DINO 推理流程，而不是训练新模型。
- **Approach**：如何安装、加载 checkpoint、构造 prompt、执行推理和保存结果。
- **Simplifications And Assumptions**：例如使用官方预训练权重、只做 zero-shot inference、不做 fine-tuning。
- **Evaluation Evidence**：至少 3 张 demo 图、对应 JSON、日志和简短评价。
- **Limitations**：哪些图片或 prompt 效果不好，可能原因是什么。
- **Next Steps**：如果继续推进，如何把 demo 推理扩展到 COCO 评估。
- **References**：GroundingDINO 论文、官方仓库、使用的依赖文档。

## 7. Notes For Report

报告中需要回答：

- Grounding DINO 如何同时利用图像特征和文本 prompt？
- 本项目使用的是 zero-shot 推理还是 fine-tuning？
- 为什么 COCO 类别 prompt 要固定格式？
- box threshold 和 text threshold 分别影响什么？
