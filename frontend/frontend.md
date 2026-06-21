# Frontend 使用说明

这个前端现在只展示本次在 `aaa` 上得到的训练结果，不再读取或展示今天之前的 COCO 复现结果、阈值实验、prompt 实验、旧样例图或旧报告图。

当前结果数据源只有：

```text
outputs\experiments\aaa_training_metrics.json
```

前端仍保留实时上传图片并调用本地模型推理的功能。

## 启动方式

在 PowerShell 中进入项目目录：

```powershell
cd E:\CS308-Computer-Vision\proj\CV_Project_2026_S
```

启动交互式前端服务：

```powershell
.\.venv\Scripts\python.exe frontend\server.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

如果页面左下角显示 `CUDA ready`，说明已经检测到 GPU。GPU 名称会从后端 `/api/status` 动态读取，不再写死某一张显卡型号。

## 页面区域

### 1. Training Result

页面顶部展示本次训练结果：

- 最佳 `mAP`
- 最佳 `AP50`
- 最佳 `AP75`
- 最佳 epoch
- 最新 epoch 的 `mAP`
- 数据来源 `aaa`

当前记录的最佳结果是 epoch 5：

```text
mAP 0.559
AP50 0.735
AP75 0.615
AP_s 0.400
AP_m 0.593
AP_l 0.712
```

页面还会展示：

- `mAP / AP50 / AP75` 随 epoch 变化的折线图
- `AP_s / AP_m / AP_l` 随 epoch 变化的分组柱状图
- 每个 epoch 的 mAP 条形图
- 最佳 epoch 的 AP_s / AP_m / AP_l 等详细指标
- 6 个 epoch 的完整表格，且每个指标列的最大值会加粗高亮
- 当前最佳 checkpoint：`checkpoints\groundingdino_swint_coco_epoch_5.pth`

### 2. Live Inference

这是实时检测区域。

使用步骤：

1. 点击 `Choose an image` 上传图片
2. 在 `Text prompt` 中输入要检测的对象
3. 根据需要调整两个阈值
4. 点击 `Run detection`
5. 查看右侧标注后的图片和检测列表

推荐 prompt 格式：

```text
person . car . dog . chair .
```

Grounding DINO 更适合使用英文词或短语，并用英文句点分隔不同类别。

阈值说明：

- `Box threshold`：控制检测框置信度，越高结果越少但更严格
- `Text threshold`：控制文字和图像区域的匹配强度，越高匹配越严格

常用默认值：

```text
Box threshold: 0.35
Text threshold: 0.25
```

如果检测不到目标，可以尝试：

- 降低 `Box threshold`
- 降低 `Text threshold`
- 使用更简单的英文类别词
- 给 prompt 加上更多可能类别

### 3. Auto run

勾选 `Auto run` 后，上传图片、修改 prompt 或调整阈值时，前端会自动重新推理。

注意：自动推理会频繁调用 GPU。如果只是调文字或阈值，建议等模型加载完成后再打开。

## 输出文件

实时上传推理产生的文件会保存到：

```text
outputs\frontend_live\uploads
outputs\frontend_live\annotated
```

其中：

- `uploads` 保存上传原图
- `annotated` 保存带检测框的结果图

## 常见问题

### 页面能打开，但不能上传推理

请确认你启动的是：

```powershell
.\.venv\Scripts\python.exe frontend\server.py
```

而不是：

```powershell
python -m http.server 8000
```

`http.server` 只能展示静态页面，不能调用模型。

### 第一次推理很慢

这是正常现象。第一次推理会加载 Grounding DINO 模型和相关文本编码器。加载完成后，页面左下角会显示 `model loaded`，之后推理会明显变快。

### 提示 checkpoint 不存在

请确认权重文件存在：

```text
checkpoints\groundingdino_swint_ogc.pth
```

## 演示建议

课堂展示时可以按这个顺序：

1. 展示本次 aaa 训练的 6 个 epoch 指标
2. 说明最佳结果出现在 epoch 5
3. 展示不同尺寸目标的 AP_s / AP_m / AP_l
4. 上传一张图片
5. 输入 `person . chair . car . dog .`
6. 点击 `Run detection`
7. 展示检测框、置信度和文字类别
