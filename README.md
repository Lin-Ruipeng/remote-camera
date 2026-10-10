# Android Pose：USB 安卓摄像头与 CUDA 实时人体姿态

手机后置摄像头 → scrcpy → V4L2 虚拟摄像头 → OpenCV → CUDA YOLO Pose → PySide6。

默认 1280×720 / 30 FPS、`yolo26n-pose.pt`、CUDA:0 / FP16、640 像素推理尺寸，
识别多人及 COCO 17 个人体关键点。实时视频和关键点默认仅在内存中处理。

GUI 支持设备选择、前后镜头、720p/1080p、n/s 模型、旋转和镜像，显示人体框、
关键点、骨架、人数、采集/检测 FPS、耗时和 GPU 信息。也可以独立启动手机摄像头，
通过 Python 或其他 V4L2 应用读取。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [安装与环境](docs/INSTALLATION.md) | uv、系统依赖、手机授权、CUDA、模型缓存、重启与重新连接 |
| [GUI 与命令行](docs/USAGE.md) | 全部 CLI 参数、GUI 控件、已有流、OBS / FFmpeg 接入、性能测试 |
| [Python API](docs/PYTHON_API.md) | BGR 采集、时间戳、PoseRunner、Pipeline、17 点索引、结果标注 |
| [架构与生命周期](docs/ARCHITECTURE.md) | 模块职责、线程与缓冲、帧对应关系、资源释放、文件目录 |
| [故障排查](docs/TROUBLESHOOTING.md) | ADB、权限、设备占用、CUDA、模型下载、DKMS、Qt 与性能 |
| [真机验证记录](VALIDATION.md) | 10 分钟性能、CUDA / FP16、功能与生命周期验证、已知限制 |
| [开发与贡献](CONTRIBUTING.md) | 开发环境、测试、格式检查、硬件验证与问题反馈 |

## 安装与运行

要求 Linux、Android 12 以上、支持所固定 cu126 wheels 的 NVIDIA 显卡与驱动。
安装脚本面向使用 apt 的 Ubuntu / Debian，下载的 scrcpy 包为 Linux x86_64。
当前已在 Ubuntu 26.04、
Android 15 和 RTX 4060 Laptop 上接入。开发者模式内还需打开 **USB 调试**，
通过支持数据传输的 USB 线连接，解锁手机并允许这台电脑的 RSA 调试授权。

先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv。
使用普通用户，从本机桌面终端运行：

```bash
git clone https://github.com/Lin-Ruipeng/remote-camera.git
cd remote-camera

# 一次性系统安装，只有系统安装和模块准备步骤使用 sudo
bash scripts/setup_system.sh

# 使用 Python 3.12.15，安装锁定的 CUDA 和 GUI 依赖
uv sync --locked
uv run remote-camera doctor
uv run remote-camera gui
```

Python 限定为 3.12，`.python-version` 固定 3.12.15，`uv.lock` 固定实际依赖版本。
首次安装包含 PyTorch、CUDA 用户态库与 Qt，下载量和磁盘占用较大；空间建议见
[安装文档](docs/INSTALLATION.md)。无需另装完整 CUDA Toolkit，系统脚本保留现有 NVIDIA 驱动。

GUI 中点击「开始检测」。首次启动会从 Ultralytics 官方下载模型至 `weights/`，
并完成 CUDA 预热。后续启动使用本地缓存；运行中的视频处理无需网络。
CUDA 不可用时会明确报告错误，需要先修复环境。

GUI 提供前后镜头、720p/1080p、n/s 模型、90°旋转、镜像与人体置信度设置。
运行中可调整置信度；切换镜头、模型或画面方向时，先停止，再修改并重新启动。
骨架只显示置信度 ≥0.5 的关键点。标注和关键点坐标对应旋转/镜像后的原始输入帧。

停止按钮和关闭窗口会等待采集线程释放设备并终止应用启动的 scrcpy。
异常详情保存在 `.state/scrcpy-*.log`，这些日志不包含视频数据。

## 在自己的 Python 程序中读取视频

终端 A 启动独立摄像头，Ctrl+C 停止：

```bash
uv run remote-camera camera
# 多台手机时通过 adb devices -l 找到序列号
uv run remote-camera camera --serial 手机序列号 --facing front
```

终端 B 读取帧，或让 GUI 接入已有流：

```bash
uv run python examples/read_camera.py --source /dev/video10 --seconds 10
uv run remote-camera gui --existing --source /dev/video10
```

`FrameSource` 在后台持续读取，`frames.get()` 返回最新帧，处理慢时自动跳过旧帧：

```python
from remote_camera.capture import FrameSource

camera = FrameSource("/dev/video10", rotation=0, mirror=False)
camera.start()
sequence = 0
try:
    while True:
        sequence, packet = camera.frames.get(sequence, timeout=1)
        if camera.error:
            raise RuntimeError(camera.error)
        if packet is not None:
            # image_bgr: uint8 NumPy (H, W, 3)
            # frame_id: 本次采集的帧序号
            # captured_at: 本机读取帧后的 time.monotonic() 时间
            image = packet.image_bgr
            # 在这里接入自己的处理代码
finally:
    camera.stop()
```

`PoseRunner.load(cancel_event)` 加载并预热 CUDA 模型，`infer(packet)` 返回 `PoseResult`：
人体框为 `(N,5)` 的 `[x1,y1,x2,y2,confidence]`，关键点为 `(N,17,3)` 的 `[x,y,confidence]`。
坐标以该结果中的 `frame.image_bgr` 像素为基准。关键点名称与连接关系见 `remote_camera.pose`。
预训练模型只提供二维人体关键点；这里没有动作分类或跨帧身份追踪。

其他 V4L2 应用也可读取 `Android Camera`，例如 OBS 或 `ffplay -f v4l2 -i /dev/video10`。
在已验证的本机 v4l2loopback 0.15.3 配置中，同时打开多个采集客户端会出现设备占用；GUI、Python 示例、
OBS/ffplay 请选择其中一个读取，切换前先停止之前的采集。独立 `camera` 命令是生产者，
可以与上述任意一个采集客户端同时运行。
默认优先使用空闲的 `/dev/video10`；如被占用，准备脚本会选择下一空闲编号。
通过 `v4l2-ctl --list-devices` 查询实际设备。使用已有流模式退出 GUI 时保留外部视频生产者。

## 检查与测试

```bash
uv run remote-camera doctor
uv run pytest
uv run ruff check .
uv run ruff format --check .

# 600 秒测试只保存统计，不保存视频或关键点
uv run remote-camera benchmark --seconds 600 --report .state/benchmark.json
```

测试报告包含采集/推理 FPS、推理均值和 P95、本机处理 P95、GPU 显存峰值及观测人数。
默认验收目标是持续 600 秒且检测刷新率 ≥20 FPS；未达到时 benchmark 返回非零。
GUI 的「本机处理」与报告中的处理耗时从电脑取帧时开始计时，不包括手机曝光、
编码与 USB 传输。采集和推理分离，各缓冲只保留一帧，GUI 约每 33 ms 读取最新结果。

RTX 4060 Laptop（8 GB）真机验证以默认配置持续运行 **600.01 秒**，检测刷新率
**29.92 FPS**，平均推理 **5.78 ms**、P95 **6.33 ms**。这是指定软硬件组合的实测，
其他设备需要独立测量；完整验证与统计口径见 [VALIDATION.md](VALIDATION.md)。

## 重启与故障排查

- **电脑重启后没有虚拟摄像头**：运行 `bash scripts/prepare_camera.sh`，重新创建设备并授予当前用户读写权限。该步骤需要 sudo；日常 GUI 使用普通用户。
- **手机是 unauthorized/offline**：解锁并确认 USB 调试授权，重新插拔数据线，运行 `adb devices -l`。无需启用无线 ADB。
- **摄像头占用或没有帧**：关闭手机上的相机/视频通话应用；停止已有 `camera` 命令，或选择 GUI「使用已有摄像头流」。
- **改变镜头后画面方向不合适**：停止检测，设置旋转与镜像，再启动。
- **CUDA 报错**：先检查 `nvidia-smi` 与 `remote-camera doctor`。项目固定 PyTorch 2.13.0 / torchvision 0.28.0 的官方 cu126 wheels，安装由 uv 的专用索引管理。
- **模型下载失败**：检查 GitHub 网络访问后重试，未完成的下载不会成为正式权重；也可通过 `PoseRunner` 或 benchmark 的 `--model` 使用本地 COCO 17 点 Pose 权重。
- **DKMS 编译失败**：确认当前内核对应的 headers 已安装，查看 `/var/lib/dkms/v4l2loopback/` 的编译日志。
- **Qt 无法打开窗口**：从本机桌面终端启动。依赖只安装 headless OpenCV，窗口由 PySide6 提供；系统脚本安装了常用 Qt 平台依赖。

## 官方参考

- [scrcpy 摄像头采集](https://github.com/Genymobile/scrcpy/blob/master/doc/camera.md)
- [scrcpy V4L2 输出](https://github.com/Genymobile/scrcpy/blob/master/doc/v4l2.md)
- [Ultralytics Pose](https://docs.ultralytics.com/tasks/pose)
- [uv 与 PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)

项目工具在 `.tools/`，模型在 `weights/`，运行日志在 `.state/`。系统脚本不会改动现有 NVIDIA 驱动。

## 范围与许可

当前实现针对 Linux V4L2 和 USB 安卓设备，不支持 Windows / macOS、无线 ADB 或模拟器。
预训练模型提供本地二维人体姿态；动作分类、跨帧追踪和三维重建需自行扩展。
`exclusive_caps=1` 的设备在生产者启动后才提供采集能力，已有流模式应先启动生产者。

本项目使用 [GNU AGPL-3.0-only](LICENSE)。第三方依赖、scrcpy 发行包和下载的模型保留
各自许可；它们的二进制文件与权重不包含在本仓库中。Ultralytics 的许可说明见其
[官方许可页面](https://www.ultralytics.com/license)。

`.venv/`、`.tools/`、`weights/`、`.state/` 和模型文件均被 Git 忽略。日常运行不会录制或上传
视频与关键点；日志可能包含手机标识，提交问题前请脱敏。
