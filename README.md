# videoDetection
This is a test of video detection.

## 代码逻辑

通过 `run()` 提取文件 `"../input_videos/video03.mp4`，将视频分割成视频帧分别用 `process()` 处理，最后汇总车辆及人员情况，并输出经过处理的视频 `"../output_videos/processed_video03s.mp4"`

## 问题

### 1. 代码运行速度慢

改用 gpu 处理
```
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
```

不进行梯度计算
```
with torch.no_grad():
    outputs = model(**inputs)
```

使用 `torch.compile()`，但是效果不明显

试图将视频分割 n 等分，分别开进程处理，但是 cuda 内存不足

### 2. 多模块划分不明显

试图分 input、detection、output 三个模块，用 grpc 传输视频帧等消息，但是代码一直有 bug，分模块可以考虑开多个进程多个 worker 处理视频

