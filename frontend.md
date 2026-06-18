# Frontend 使用说明

这个前端用于演示 Grounding DINO 开放词汇目标检测项目。它包含两部分功能：

- 展示本机 RTX 4060 上完成的 COCO val2017 5000 张图片复现结果
- 上传任意图片，输入文字 prompt，并实时调用本地模型进行目标检测

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

如果页面左下角显示 `CUDA ready`，说明已经检测到 GPU。第一次点击推理时模型会加载到显存中，可能会慢一些；之后再次推理会快很多。

## 页面区域

### 1. Full Result

页面顶部展示完整 COCO val2017 复现结果：

- `AP`
- `AP50`
- `AP75`
- 预测框数量
- 有预测结果的图片数量
- 完整评估耗时

这些数据来自：

```text
outputs\coco_eval\my_full_val2017\metrics.json
result.txt
```

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

### 4. Saved Samples

这一部分展示项目中已经保存好的推理样例，可以用来快速讲解检测效果。

拖动 `Score threshold` 可以筛选显示的检测结果，但它只影响前端显示，不会重新运行模型。

### 5. Experiments

展示阈值实验和 prompt 格式实验结果。

可以切换：

- `Threshold`
- `Prompt`

### 6. Report Figures

展示项目报告中使用的可视化图片，包括：

- COCO 指标总览
- 不同尺寸目标的检测差异
- 阈值敏感性分析
- prompt 格式对比
- 成功案例
- 失败案例

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

### GPU 没有被使用

检查页面左下角是否显示 `CUDA ready`。也可以在 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果输出 `True` 和显卡名称，则说明 CUDA 可用。

## 演示建议

课堂展示时可以按这个顺序：

1. 先展示顶部 5000 张 COCO 完整复现指标
2. 说明 Grounding DINO 可以通过文字 prompt 检测开放类别
3. 上传一张图片
4. 输入 `person . chair . car . dog .`
5. 点击 `Run detection`
6. 展示检测框、置信度和文字类别
7. 调高或调低阈值，展示检测数量变化

