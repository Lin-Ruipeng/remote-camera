# Python 接口与数据约定

[返回 README](../README.md) · [架构文档](ARCHITECTURE.md)

以下示例在完成 `uv sync --locked` 的项目环境中运行。图像始终为 OpenCV 的 **BGR** 数组，直接交给 PoseRunner；向 Qt 或其他 RGB 显示接口传递时再转换颜色通道。

## 只读取手机视频

先在另一终端运行 `uv run remote-camera camera`，确认实际虚拟设备路径。完整采集示例已提供：

```bash
uv run python examples/read_camera.py --source /dev/video10 --seconds 10
```

嵌入自己程序时：

```python
from remote_camera.capture import FrameSource

camera = FrameSource("/dev/video10", rotation=90, mirror=True)
camera.start()
sequence = 0
try:
    while True:
        sequence, packet = camera.frames.get(sequence, timeout=1)
        if camera.error:
            raise RuntimeError(camera.error)
        if packet is None:
            continue
        print(packet.frame_id, packet.image_bgr.shape, packet.captured_at)
        # 在此处理 packet.image_bgr；消费者慢时旧帧自动被跳过。
except KeyboardInterrupt:
    pass
finally:
    camera.stop()
```

| `FramePacket` 字段 | 类型与含义 |
| --- | --- |
| `image_bgr` | 连续内存的 `numpy.ndarray`，`uint8`，形状 `(H,W,3)`；已应用配置的旋转 / 镜像 |
| `frame_id` | 本次 FrameSource 的采集序号，从 1 递增；跨会话重新计数 |
| `captured_at` | `time.monotonic()` 秒数，本机 `VideoCapture.read()` 返回后的时刻、图像变换前 |

`captured_at` 不是 Unix 时间，也不是手机曝光时间。它可以与同一台电脑上同一时钟的 `time.monotonic()` 相减测量处理耗时，不能直接用作视频的手机采集时间或跨机器时钟同步。

`FrameSource.start()` 立即启动后台线程，打开设备失败会通过 `error` 返回。`stop()` 请求停止并等待释放；异常断开时可能需要等待 V4L2 读取超时，最多等待 12 秒，超时抛出 `CameraError`。每个实例仅用于一次启动；重新运行请创建新实例。

## 最新值缓冲的用法

`camera.frames` 和 `pipeline.results` 都是单槽 `LatestValue`：

```python
sequence, value = mailbox.get(after=sequence, timeout=1)
```

- `sequence` 是缓冲发布序号，不应当替代 `FramePacket.frame_id`。
- 只返回比 `after` 更新的值；超时或关闭后没有更新时，返回原序号和 `None`。
- 返回值可能跨过多个发布序号，这是主动跳过旧帧的行为。
- `None` 不能单独证明设备故障，应结合 `FrameSource.error` 或 `Pipeline.error / alive` 判断。
- 数据引用只读使用；需要修改或长期保留时，调用 `packet.image_bgr.copy()`。`frozen=True` 限制字段赋值，不会让 NumPy 内容变成不可写。

这样即使推理比采集慢，应用也不会累积待处理帧。该保证针对应用的缓冲，不消除手机、编码器、USB 和驱动自身的缓冲。

## 对单帧进行 CUDA Pose 推理

以下示例使用空白图验证接口，不需要手机；需要可用的 CUDA 和内置模型缓存 / 下载网络。保存为项目内的 Python 文件后通过 `uv run python 文件名.py` 运行：

```python
import threading
import time

import numpy as np

from remote_camera.capture import FramePacket
from remote_camera.pose import PoseRunner
from remote_camera.render import annotate

runner = PoseRunner(model="yolo26n-pose.pt", confidence=0.35)
runner.load(threading.Event())  # CUDA 检查、权重校验、模型预热

packet = FramePacket(
    image_bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
    frame_id=1,
    captured_at=time.monotonic(),
)
result = runner.infer(packet)
print(runner.gpu_name, result.boxes.shape, result.keypoints.shape, result.inference_ms)
annotated_bgr = annotate(result)  # 原图的标注副本，无显示 / 保存副作用
```

`PoseRunner.load(cancel_event)` 在 CUDA 不可用时抛出 `CameraError`，不回退 CPU。取消事件用于在启动 / 下载过程中请求停止；取消后不要继续调用 `infer`。加载内置模型时自动下载并校验固定 SHA-256；本地自定义模型可以传入绝对路径，必须提供 COCO 17 个关键点和关键点置信度。

`infer(packet)` 在调用线程同步执行，返回时相关结果已拷回 CPU。固定配置是 `cuda:0`、FP16、`imgsz=640`。GUI 和 Pipeline 会把调用放到后台线程；自行使用 PoseRunner 时，也不要在 GUI 主线程中长时间推理。一个 runner 应由一个推理线程串行使用。

### `PoseResult` 字段

| 字段 | 类型与含义 |
| --- | --- |
| `frame` | 与本次结果对应的原始 `FramePacket`；取这个字段绘图，避免使用另一个新帧 |
| `boxes` | `float32` 数组 `(N,5)`，每行为 `[x1,y1,x2,y2,confidence]` |
| `keypoints` | `float32` 数组 `(N,17,3)`，每个点为 `[x,y,confidence]` |
| `inference_ms` | 毫秒；包含预处理、GPU 推理和结果回传 CPU |
| `finished_at` | 同一 `time.monotonic()` 时钟下推理完成的秒数 |

第 `i` 个框对应第 `i` 组关键点。坐标是 `result.frame.image_bgr` 上的像素坐标，已经恢复到输入分辨率，不是模型内部 640 尺寸的坐标。旋转和镜像由采集阶段统一执行，因此输出与变换后的画面对齐。人体关键点的 left / right 表示人体自身的解剖侧，不能简单用屏幕左右列替代。

无人时 `N=0`；先检查数量再访问 `boxes[0]`。被遮挡或低置信度的点不应直接用来计算角度，先检查点的第三列置信度。`annotate` 默认只显示置信度 ≥0.5 的点，并要求骨架两端均可见。

## 直接使用完整后台 Pipeline

Pipeline 管理 CUDA 预热、自己的 scrcpy、采集和推理。以下程序运行 30 秒，不打开 GUI，不保存帧或关键点：

```python
import time

from remote_camera.backend import StreamConfig
from remote_camera.pipeline import Pipeline, PipelineConfig

session = Pipeline(
    PipelineConfig(
        stream=StreamConfig(facing="back", size="1280x720"),
        model="yolo26n-pose.pt",
        confidence=0.35,
        rotation=0,
        mirror=False,
    )
)
sequence = 0
deadline = time.monotonic() + 30  # 这个示例的时长包含模型启动阶段
session.start()
try:
    while session.alive and time.monotonic() < deadline:
        sequence, result = session.results.get(sequence, timeout=0.5)
        if result is not None:
            print(
                result.frame.frame_id,
                len(result.boxes),
                result.inference_ms,
                session.detection_fps,
            )
except KeyboardInterrupt:
    pass
finally:
    session.request_stop()
    session.join()
if session.error:
    raise RuntimeError(session.error)
```

接入外部已运行的生产者时，将配置改为：

```python
PipelineConfig(
    stream=StreamConfig(source="/dev/video10"),
    existing=True,
    model="yolo26n-pose.pt",
)
```

已有流模式不需要由 Pipeline 选择 USB 手机，停止时只释放自己的采集线程，不停止外部生产者。旋转 / 镜像和模型仍在本地应用。

| Pipeline 接口 | 用途 |
| --- | --- |
| `start()` | 启动一次后台会话 |
| `results.get(after, timeout)` | 读取最新对应帧的 PoseResult |
| `set_confidence(value)` | 更新之后推理所用的人体置信度 |
| `status` / `error` | 状态文字 / 异常原因 |
| `gpu_name` / `source` | 启动成功后的 GPU 名与实际源路径 |
| `capture` | 启动后的 FrameSource；模型加载阶段可能为 `None` |
| `detection_fps` / `inference_count` | 平均完成推理帧率 / 完成次数 |
| `alive` | 后台工作线程是否存活 |
| `request_stop()` | 发出非阻塞停止请求 |
| `join(timeout=30)` | 等待清理，超时抛出 `CameraError` |

将 `request_stop()` 用于 GUI 回调，再由定时器检查 `alive`。`join()` 会阻塞，适合命令行或窗口关闭后的外部等待，避免直接在正常 GUI 交互中阻塞调用。配置对象是不可变 dataclass；改变镜头、模型或方向时，停止旧会话并创建新 Pipeline。

## COCO 17 点索引

`remote_camera.pose.COCO_KEYPOINTS` 提供以下零基索引；骨架连接在同模块的 `SKELETON` 中。

| 索引 | 英文名称 | 中文 |
| --- | --- | --- |
| 0 | nose | 鼻子 |
| 1 / 2 | left_eye / right_eye | 左 / 右眼 |
| 3 / 4 | left_ear / right_ear | 左 / 右耳 |
| 5 / 6 | left_shoulder / right_shoulder | 左 / 右肩 |
| 7 / 8 | left_elbow / right_elbow | 左 / 右肘 |
| 9 / 10 | left_wrist / right_wrist | 左 / 右腕 |
| 11 / 12 | left_hip / right_hip | 左 / 右髋 |
| 13 / 14 | left_knee / right_knee | 左 / 右膝 |
| 15 / 16 | left_ankle / right_ankle | 左 / 右踝 |

例如只读取可靠的肩点：

```python
for person in result.keypoints:
    left_shoulder = person[5]
    right_shoulder = person[6]
    if min(left_shoulder[2], right_shoulder[2]) >= 0.5:
        print(left_shoulder[:2], right_shoulder[:2])
```

结果中的 `Person 1` 等编号按单帧排列，不是追踪 ID。动作识别、角度计算或跨帧追踪需要在这些接口上自行实现。
