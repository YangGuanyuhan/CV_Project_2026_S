# 单卡微调方案（Grounding DINO）

## 这份文档写了什么

本文档给出一套面向单卡 GPU（优先按 RTX 4060 8GB 估计）的可执行微调方案，包含：

- 为什么当前仓库不能直接微调（训练脚本现状）。
- 推荐的可落地路线（MMDetection 训练链路）。
- 单卡稳妥参数配置（避免 OOM 的默认值）。
- 三档训练计划（烟雾测试 / 短程微调 / 完整 12 epoch）与时间估算。
- 可直接复制执行的命令模板。
- 常见报错与排查建议。

---

## 1. 当前仓库训练现状

当前项目的 `scripts/train.py` 仍是占位版本，尚未实现训练循环，因此不能直接在本仓库内完成 Grounding DINO 的端到端 fine-tune。  
建议采用 **MMDetection 的 Grounding DINO 配置**进行训练，再把微调后权重拿回本项目做推理与评估展示。

---

## 2. 目标与策略

### 2.1 目标

- 在单卡环境下完成一次可复现的 COCO 微调流程。
- 以较小成本获得“相对 zero-shot 的可见提升”。
- 产出可写入报告的训练日志、评估指标和 checkpoint。

### 2.2 推荐策略

- 先跑 **1 epoch 烟雾测试**，确认环境与数据路径正确。
- 再跑 **3 epoch 短程微调**，优先拿可汇报结果。
- 时间充足再跑 **12 epoch 完整训练**。

---

## 3. 单卡资源与耗时估算（RTX 4060 8GB）

以下为经验估算，实际取决于数据读取速度、CPU、磁盘、驱动版本和 CUDA 环境：

| 训练档位 | 数据规模 | epoch | 预计耗时 |
|---|---|---:|---:|
| 烟雾测试 | COCO 全量 | 1 | 8-20 小时 |
| 短程微调（推荐） | COCO 全量 | 3 | 1-2.5 天 |
| 完整训练（课程不推荐硬冲） | COCO 全量 | 12 | 4-10 天 |

如果只为课程展示结果，建议优先“短程微调 + 对比实验图表”，投入产出更高。

---

## 4. 单卡稳妥配置（默认值）

为降低 OOM 风险，建议起步参数如下：

- `batch_size=1`
- `num_workers=2~4`
- 开启 `AMP`（混合精度）
- 输入分辨率先按默认配置，不要先上更大尺寸
- 先不做复杂数据增强改动

如果显存仍紧张：

1. 先把 `num_workers` 降到 2。  
2. 关闭可选增强。  
3. 最后再考虑进一步缩放输入或改更轻配置。

---

## 5. 可执行命令（MMDetection 路线）

> 说明：下面命令是“新建独立微调环境”的方案，不污染你当前项目的推理环境。

### 5.1 环境准备

```bash
conda create -n gdino_ft python=3.10 -y
conda activate gdino_ft

pip install -U pip
pip install -U openmim
mim install "mmengine>=0.10.0" "mmcv>=2.0.0" "mmdet>=3.3.0"
```

```bash
git clone https://github.com/open-mmlab/mmdetection.git /home/elfdr/cvp/mmdetection
pip install -v -e /home/elfdr/cvp/mmdetection
```

### 5.2 硬路径约定（已按你当前机器确认）

以下路径是当前机器可用的固定路径，不再依赖环境变量：

- COCO 根目录：`/home/elfdr/cvp/CV_Project_2026_S/data/coco`
- 预训练权重：`/home/elfdr/cvp/CV_Project_2026_S/checkpoints/groundingdino_swint_ogc.pth`
- MMDetection 目标安装目录：`/home/elfdr/cvp/mmdetection`（当前不存在，下面命令会创建）

当前机器已经包含：

- `val2017/`
- `annotations/instances_train2017.json`
- `annotations/instances_val2017.json`

当前机器尚未包含 `train2017/` 图像目录。因此：

- 烟雾测试命令临时使用 `val2017/` 和 `instances_val2017.json` 跑通训练链路。
- 真正短程微调前必须先下载 `train2017/` 图像。

### 5.3 1 epoch 烟雾测试

> 该命令只验证训练链路是否能跑通，不代表正式训练设置。因为当前机器没有 `train2017/` 图像，烟雾测试临时使用 `val2017/` 作为训练集。

```bash
python /home/elfdr/cvp/mmdetection/tools/train.py \
  /home/elfdr/cvp/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py \
  --work-dir /home/elfdr/cvp/mmdetection/work_dirs/gdino_smoke \
  --amp \
  --cfg-options \
    data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    load_from=/home/elfdr/cvp/CV_Project_2026_S/checkpoints/groundingdino_swint_ogc.pth \
    train_cfg.max_epochs=1 \
    train_dataloader.batch_size=1 \
    train_dataloader.num_workers=2 \
    train_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    train_dataloader.dataset.ann_file=annotations/instances_val2017.json \
    train_dataloader.dataset.data_prefix.img=val2017/ \
    val_dataloader.num_workers=2 \
    val_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    val_dataloader.dataset.ann_file=annotations/instances_val2017.json \
    val_dataloader.dataset.data_prefix.img=val2017/ \
    val_evaluator.ann_file=/home/elfdr/cvp/CV_Project_2026_S/data/coco/annotations/instances_val2017.json \
    test_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    test_dataloader.dataset.ann_file=annotations/instances_val2017.json \
    test_dataloader.dataset.data_prefix.img=val2017/ \
    test_evaluator.ann_file=/home/elfdr/cvp/CV_Project_2026_S/data/coco/annotations/instances_val2017.json \
    default_hooks.checkpoint.interval=1
```

### 5.4 3 epoch 短程微调（推荐）

正式微调需要先补齐 `train2017/` 图像：

```bash
cd /home/elfdr/cvp/CV_Project_2026_S
python scripts/download_coco.py --output-dir /home/elfdr/cvp/CV_Project_2026_S/data/coco --split train2017
```

```bash
python /home/elfdr/cvp/mmdetection/tools/train.py \
  /home/elfdr/cvp/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py \
  --work-dir /home/elfdr/cvp/mmdetection/work_dirs/gdino_ft_short \
  --amp \
  --cfg-options \
    data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    load_from=/home/elfdr/cvp/CV_Project_2026_S/checkpoints/groundingdino_swint_ogc.pth \
    train_cfg.max_epochs=3 \
    train_dataloader.batch_size=1 \
    train_dataloader.num_workers=4 \
    train_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    train_dataloader.dataset.ann_file=annotations/instances_train2017.json \
    train_dataloader.dataset.data_prefix.img=train2017/ \
    val_dataloader.num_workers=2 \
    val_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    val_dataloader.dataset.ann_file=annotations/instances_val2017.json \
    val_dataloader.dataset.data_prefix.img=val2017/ \
    val_evaluator.ann_file=/home/elfdr/cvp/CV_Project_2026_S/data/coco/annotations/instances_val2017.json \
    test_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    test_dataloader.dataset.ann_file=annotations/instances_val2017.json \
    test_dataloader.dataset.data_prefix.img=val2017/ \
    test_evaluator.ann_file=/home/elfdr/cvp/CV_Project_2026_S/data/coco/annotations/instances_val2017.json \
    default_hooks.checkpoint.interval=1
```

### 5.5 评估短程微调模型

```bash
python /home/elfdr/cvp/mmdetection/tools/test.py \
  /home/elfdr/cvp/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py \
  /home/elfdr/cvp/mmdetection/work_dirs/gdino_ft_short/epoch_3.pth \
  --work-dir /home/elfdr/cvp/mmdetection/work_dirs/gdino_ft_short_eval \
  --cfg-options \
    data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    test_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    test_dataloader.dataset.ann_file=annotations/instances_val2017.json \
    test_dataloader.dataset.data_prefix.img=val2017/ \
    test_evaluator.ann_file=/home/elfdr/cvp/CV_Project_2026_S/data/coco/annotations/instances_val2017.json
```

### 5.6 恢复训练（断点续训）

```bash
python /home/elfdr/cvp/mmdetection/tools/train.py \
  /home/elfdr/cvp/mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py \
  --work-dir /home/elfdr/cvp/mmdetection/work_dirs/gdino_ft_short \
  --amp \
  --resume auto \
  --cfg-options \
    data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    load_from=/home/elfdr/cvp/CV_Project_2026_S/checkpoints/groundingdino_swint_ogc.pth \
    train_cfg.max_epochs=3 \
    train_dataloader.batch_size=1 \
    train_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    train_dataloader.dataset.ann_file=annotations/instances_train2017.json \
    train_dataloader.dataset.data_prefix.img=train2017/ \
    val_dataloader.dataset.data_root=/home/elfdr/cvp/CV_Project_2026_S/data/coco/ \
    val_dataloader.dataset.ann_file=annotations/instances_val2017.json \
    val_dataloader.dataset.data_prefix.img=val2017/ \
    val_evaluator.ann_file=/home/elfdr/cvp/CV_Project_2026_S/data/coco/annotations/instances_val2017.json
```

---

## 6. 结果对比建议（写报告时可直接用）

建议至少保留以下对比：

- Zero-shot 基线：你当前项目已有 AP 结果。
- 1 epoch 微调：验证流程正确性与方向。
- 3 epoch 微调：展示“短程投入的收益”。

可汇报指标：

- AP / AP50 / AP75
- APS / APM / APL
- 推理时延（可选）

---

## 7. 常见问题与排查

### 7.1 OOM（显存不足）

- 确认 `batch_size=1`。
- 降低 `num_workers` 到 2。
- 确保开启 `--amp`。

### 7.2 数据路径报错

- 检查 `/home/elfdr/cvp/CV_Project_2026_S/data/coco` 是否包含 `train2017`、`val2017` 和 `annotations`。
- 检查 JSON 文件名是否与配置一致。

### 7.3 训练速度太慢

- 优先先跑 1 epoch + 3 epoch，不直接冲 12 epoch。
- 保持固定实验设置，先拿到可比较结果，再做参数优化。

---

## 8. 与本项目衔接方式

微调完成后，建议在本项目新增一个“fine-tuned checkpoint 评估”小节，内容包括：

1. 微调 checkpoint 路径与训练配置摘要。  
2. 用本项目现有评估脚本对比 zero-shot 与 fine-tuned 指标。  
3. 补充 5-10 张可视化案例（成功 + 失败）。  

这样可以在不重构现有代码的前提下，快速补齐“微调结果”这一块。
