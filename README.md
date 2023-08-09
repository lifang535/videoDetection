# videoDetection
This is a test of video detection.

## 代码逻辑

通过 `run()` 提取文件 `"../input_videos/video03.mp4`，将视频分割成视频帧分别用 `process()` 处理，最后汇总车辆及人员情况，并输出经过处理的视频 `"../output_videos/processed_video03s.mp4"`

## 问题

### 1. 代码运行速度慢

改用 gpu 处理，试图使用两个 gpu，但是效果不明显
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
# model = torch.nn.DataParallel(model).to(device)
```

不进行梯度计算
```
with torch.no_grad():
    outputs = model(**inputs)
```

使用 `torch.compile()`，但是效果不明显

试图将视频分割 n 等分，分别开进程处理，但是显示 cuda 内存不足

### 2. 多模块划分不明显

试图分 input、detection、output 三个模块，用 grpc 传输视频帧等消息，但是代码一直有 bug，分模块可以考虑开多个进程多个 worker 处理视频

不太理解论文 Rim 的图中 Car Detection 和 Person Detection 为什么分成两部分，用一个模块的模型就可以识别出来单个视频帧的人和车的结果

Rim：https://dl.acm.org/doi/pdf/10.1145/3450268.3453521

加了后续两个模块，但是统计车辆及行人数量在函数内部已经做了，考虑用别的模型在后续模块获得更具体的信息

```
if model.config.id2label[label.item()] == "car":
    car_detection(frame, box)
if model.config.id2label[label.item()] == "person":
    person_detection(frame, box)
```

### 3. Traffic 汇总

采用对视频帧的采样，每 1 秒记录一次车辆信息，每 5 秒记录一次行人信息，防止对车辆和行人的重复计算
