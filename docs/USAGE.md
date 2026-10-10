# GUI 与命令行使用

[返回 README](../README.md) · [Python API](PYTHON_API.md)

## GUI 自己启动摄像头

```bash
uv run remote-camera gui
```

1. 刷新并选择 USB 手机；只有一台时可以自动选择。
2. 选择后置 / 前置、720p / 1080p 和 n / s 模型，检查虚拟摄像头路径。
3. 点击「开始检测」，等待模型下载、CUDA 预热和摄像头连接。
4. 点击「停止」释放资源；关闭窗口也会等待清理完成。

| 控件 / 指标 | 含义 |
| --- | --- |
| 手机与刷新 | 枚举 USB ADB 设备，显示模型和状态 |
| 镜头 | 后置或前置，由 scrcpy 请求对应摄像头 |
| 分辨率 | `1280x720` 或 `1920x1080`，默认 720p；采集帧率固定请求 30 FPS |
| 模型 | `yolo26n-pose.pt` 或 `yolo26s-pose.pt`；默认 n |
| 旋转 | 0 / 90 / 180 / 270 度；90 度表示顺时针 |
| 镜像 | 在旋转之后执行水平翻转，变换发生在推理前 |
| 置信度 | 人体框阈值，默认 0.35，可以运行中调整 |
| 使用已有摄像头流 | 只读取指定流，不启动或停止外部生产者 |
| 人数 | 当前结果的人体框数量，不是跨帧去重的人数 |
| 采集 / 检测 FPS | 本轮运行的平均读取帧率 / 完成推理帧率 |
| 推理耗时 | 对应帧的预处理、模型推理及结果拷回 CPU 的耗时 |
| 本机处理耗时 | 从该帧在电脑取出到 GUI 处理结果的耗时，不含手机与 USB 阶段 |
| GPU 与显存 | CUDA 设备名称与 PyTorch 显存池统计，不是整个进程的总显存 |

模型、镜头、分辨率、旋转和镜像需要停止后修改。低于 0.5 置信度的关键点不绘制；骨架连线仅在两端关键点都达到该阈值时绘制。画面按比例缩放，关键点坐标仍以变换后的原始图像像素为基准。

异常弹窗和状态提示提供原因；scrcpy 的诊断日志位于 `.state/scrcpy-*.log`。出现断线或模型失败后可修复环境并重新启动。

## 独立生产者与已有流

在终端 A 启动：

```bash
uv run remote-camera camera --facing back --size 1280x720
```

终端会输出实际设备路径。终端 B 接入：

```bash
uv run remote-camera gui --existing --source /dev/video10
```

已有流模式中的手机、镜头和分辨率由外部生产者控制；GUI 的模型、置信度、旋转和镜像仍应用于本地采集与推理。退出 GUI 保留终端 A 的摄像头，按终端 A 的 Ctrl+C 才会停止生产者。

同时连接多台手机时，先查看 `adb devices -l`，再用实际序列号替换占位内容：

```bash
uv run remote-camera camera --serial YOUR_DEVICE_SERIAL --source /dev/video10 --facing front
```

`camera` 仅提供视频，运行本身不加载 YOLO 或使用 CUDA。`doctor` 和完整检测功能会检查 CUDA。

## CLI 参数速查

所有命令均支持 `--help`，例如 `uv run remote-camera benchmark --help`。

### `doctor`

```bash
uv run remote-camera doctor
```

没有额外参数。检查 Python、三个系统可执行文件、单台 USB 手机及 Android API、标记为 Android Camera 的可读写虚拟设备，以及实际 CUDA 矩阵运算。退出码 0 表示全部通过，1 表示存在失败。

### `camera`

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--serial` | 自动选择唯一 USB 手机 | ADB 序列号 |
| `--source` | 自动查找 Android Camera | 要写入的虚拟设备路径；必须是相应标签的设备 |
| `--facing` | `back` | `back` / `front` |
| `--size` | `1280x720` | `1280x720` / `1920x1080` |

该命令请求 H.264、4 Mbit/s 和 30 FPS，关闭音频、手机控制及 scrcpy 独立播放窗口。正常 Ctrl+C / SIGTERM 会停止自己启动的 scrcpy 并释放写入端；错误时返回非零退出码。

### `gui`

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--source` | 界面中自动查找 | 预填虚拟设备路径 |
| `--existing` | 关闭 | 预选只读取已运行的流；需指定源路径 |

其他参数在窗口中设置，不是 GUI CLI 参数。

### `benchmark`

支持 `camera` 的所有参数，此外：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--seconds` | `600` | 首个推理结果后的测量时长，必须大于零 |
| `--existing` | 关闭 | 使用已运行的流，需要 `--source` |
| `--model` | `yolo26n-pose.pt` | 内置 n / s 名称，或本地模型路径 |
| `--rotation` | `0` | 0 / 90 / 180 / 270 |
| `--mirror` | 关闭 | 旋转后水平镜像 |
| `--report` | 无 | 将性能统计 JSON 写到指定文件；父目录自动创建 |

```bash
# 默认 10 分钟验收
uv run remote-camera benchmark --seconds 600 --report .state/benchmark.json

# 已有流，旋转并镜像
uv run remote-camera benchmark --existing --source /dev/video10 \
  --rotation 90 --mirror --seconds 60 --report .state/benchmark-rotated.json

# 本地自定义 COCO 17 点 Pose 权重
uv run remote-camera benchmark --model /absolute/path/custom-pose.pt --seconds 60
```

启动及模型预热不计入测量区间，每约 30 秒打印进度。报告包含持续时间、是否完整运行、硬件 / 软件版本、模型与源设备、采集 / 推理 FPS、平均 / P95 推理时间、本机处理 P95、PyTorch 显存峰值与观测人数集合。报告不包含视频、单帧关键点或 ADB 序列号。

完整运行且检测 FPS ≥20 时退出码为 0；提前中断、低于目标或发生错误时为非零。20 FPS 是本项目验收目标，修改 `--seconds` 不会改变此阈值。benchmark 不打开 GUI，因此其刷新率和处理耗时不包含实际桌面绘制。

## Python、OBS 与 FFmpeg

先运行独立 `camera` 生产者，再选择一种消费者：

```bash
uv run python examples/read_camera.py --source /dev/video10 --seconds 10
```

示例也可不传 `--source`，自动发现 Android Camera。示例持续读取 BGR 帧，不显示窗口、不写入视频，Ctrl+C 时释放采集线程。

OBS 中添加「视频采集设备（V4L2）」，选择 `Android Camera`。应在生产者启动后创建或刷新输入设备。如果使用 FFmpeg 工具（需自行安装 ffmpeg），可以预览或仅验证读取：

```bash
ffplay -f v4l2 -i /dev/video10
ffmpeg -f v4l2 -i /dev/video10 -t 5 -f null -
```

在已验证的本机驱动配置中，多消费者同时采集会出现 `Device or resource busy`。先停止 GUI / Python / OBS 的旧输入，再切换消费者。摄像头生产者占用与消费者占用是两种情况，排查见 [设备占用](TROUBLESHOOTING.md#摄像头占用设备忙或无法打开设备)。
