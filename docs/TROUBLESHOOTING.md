# 故障排查

[返回 README](../README.md) · [安装步骤](INSTALLATION.md)

优先执行 `uv run remote-camera doctor`，按失败项排查。GUI 的异常弹窗可展开详细原因，scrcpy 日志在 `.state/scrcpy-*.log`。分享输出前删除手机序列号和私人本机路径。

## 没有手机或状态为 unauthorized / offline

```bash
adb devices -l
```

- 没有设备：确认使用数据线而非仅充电线，更换 USB 接口，打开 USB 调试，解锁手机。手机厂商的 USB 模式或调试策略也可能影响连接。
- `unauthorized`：在手机 RSA 授权弹窗中允许当前电脑；若无弹窗，可在开发者选项撤销 USB 调试授权后重新插入并授权。
- `offline`：先重新插拔。仍失败时，在没有其他 ADB 工作的情况下执行 `adb kill-server`、`adb start-server`，再检查；此操作会影响本机其他 ADB 会话。
- `no permissions`：重新运行系统安装脚本安装 Android udev 规则，重新插拔设备，在普通用户下检查。
- 多台手机：GUI 手动选择，或 `camera --serial YOUR_DEVICE_SERIAL`；doctor 的自动选择会要求唯一设备。

应用只处理 USB 安卓设备，过滤无线 ADB 地址和模拟器。检查 Android 最低版本：

```bash
adb -s YOUR_DEVICE_SERIAL shell getprop ro.build.version.sdk
```

需要 API 31 以上。低于该版本不能使用此直接摄像头方案。

## 没有 Android Camera 或 Permission denied

```bash
v4l2-ctl --list-devices
bash scripts/prepare_camera.sh
```

电脑重启后手动创建设备和 ACL 会消失，重新运行准备脚本即可。确认输出的实际路径，后续 `--source` 使用该路径。需要时检查权限：

```bash
getfacl /dev/video10
```

不要直接把任意电脑内置摄像头作为 `camera --source` 的目标；生产者要求设备标签是 `Android Camera`。已有流模式可以读取明确指定、已有采集能力的 V4L2 设备。

`exclusive_caps=1` 让空闲设备只呈现输出能力，生产者运行后才呈现采集能力。因此 OBS 暂时不列出输入时，先启动 `camera` 并刷新 OBS 设备列表。

## 摄像头占用、设备忙或无法打开设备

区分三种占用：

| 现象 | 原因与处理 |
| --- | --- |
| 手机端无法打开镜头 | 关闭手机相机、视频通话或扫码应用，再启动 |
| 提示虚拟设备已有生产者 | 停止独立 camera / 旧 GUI，或改用 `gui --existing --source ...` |
| OpenCV / V4L2 提示资源忙 | 停止正在采集同一设备的 GUI、Python、OBS、ffplay 等旧消费者 |

本机验证中，多消费者同时采集同一虚拟设备出现 `Device or resource busy`，不能据此保证所有驱动组合都支持或不支持并发。推荐保持一个生产者、一个消费者，切换消费者前先停止旧的。

可在本机使用 `fuser /dev/video10` 或 `lsof /dev/video10`（需已安装相应工具）定位占用。先在所属程序中正常停止，避免直接结束全部 adb 或 scrcpy。

## scrcpy 启动失败、黑屏或没有帧

查看最新 scrcpy 日志，检查镜头是否支持请求的分辨率和帧率：

```bash
.tools/scrcpy-linux-x86_64-v5.0.1/scrcpy --list-cameras
.tools/scrcpy-linux-x86_64-v5.0.1/scrcpy --list-camera-sizes
```

多台手机时添加 `--serial YOUR_DEVICE_SERIAL`。先使用后置 720p / 30 FPS，确保手机镜头没被其他应用占用。应用等待 scrcpy 就绪最多 20 秒；运行中 5 秒没有新帧会报告错误。

断开手机时，OpenCV 读取可能等待驱动超时。等待清理完成再插入、刷新并重新开始，不要同时启动多个会话抢占设备。已有流模式需先启动外部生产者；确认源路径没有写错。

## CUDA 不可用或 PyTorch 版本不对

```bash
nvidia-smi
uv sync --locked
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
uv run remote-camera doctor
```

项目固定 PyTorch 2.13.0 和 torchvision 0.28.0，绑定官方 cu126 wheel 索引。不要在同一环境手动安装 CPU 版 torch，或从其他索引替换锁定版本。确保运行的是项目的 uv 环境，而不是系统 Python 或其他虚拟环境。

`nvidia-smi` 正常不等于实际 PyTorch CUDA 运算一定成功；doctor 会分配 GPU 张量、执行矩阵乘法并把结果传回。驱动不可用或不兼容时先修复主机驱动，本项目安装脚本不负责驱动安装。

CUDA 不可用会明确失败，不回退 CPU。GUI 不启动独立摄像头前就会完成模型检查，因此模型加载失败时通常不需要额外清理摄像头生产者。

## 模型下载、校验或形状失败

- 网络失败：确认能访问 GitHub 的官方 release 下载地址，检查系统代理和证书，修复后重新开始。
- SHA-256 失败：按错误提示删除那一个损坏的内置权重文件，重新下载；不要修改源码校验值来绕过失败。
- 需要离线运行：提前准备已校验的内置模型缓存；独立实时处理不需要网络。
- 本地自定义模型失败：确认是 Pose 模型，关键点定义为 `(17,3)`，而不是普通 detect 权重或不同动物骨架。

自定义路径可用于 Python `PoseRunner` / `PipelineConfig.model` 或 `benchmark --model`，GUI 模型选择仅包含两个内置模型。下载中途取消会清理临时 `.part` 文件，不使用未完成的缓存。

## DKMS 或内核模块加载失败

```bash
uname -r
dkms status
lsmod | rg v4l2loopback
```

确认当前启动内核对应的 `linux-headers-$(uname -r)` 已安装。升级内核后，重启进入与 headers / DKMS 匹配的内核，再运行准备脚本。编译详情在 `/var/lib/dkms/v4l2loopback/` 对应版本和内核的日志中。

若出现签名或 `Key was rejected by service`，检查 Secure Boot 的模块签名 / MOK 注册流程，按照发行版提示处理。不要只因为摄像头不可用就盲目更换 NVIDIA 驱动或关闭安全设置。

若 apt 安装提示找不到包，先更新软件源索引；某个 `.bak` 文件被 apt 忽略的提示通常不是安装失败，检查最终退出状态和真正的错误信息。

## Qt 无法打开窗口

GUI 需要本机桌面图形会话；从普通桌面终端执行 `uv run remote-camera gui`。不要以 root 启动 GUI。`offscreen` 用于测试，不是正常实时窗口的运行方式。

系统脚本安装 `libxcb-cursor0` 和 `libxkbcommon0`。仍有平台插件问题时，检查详细 Qt 输出，并确认没有把普通 OpenCV 的 Qt 插件混入当前环境：

```bash
uv sync --locked
uv run python -c 'import cv2, PySide6; print(cv2.__version__, PySide6.__version__)'
```

项目只需要 headless OpenCV，显示由 PySide6 提供。远程 SSH / 容器环境需要自行准备图形会话、设备和 GPU 访问，这不是安装脚本自动配置的范围。

## 骨架错位、左右或方向不合适

停止后调整旋转和镜像，再开始。90 度表示顺时针旋转，镜像在旋转后应用。姿态输出必须叠加到 `result.frame.image_bgr`，不要叠加到采集线程刚拿到的另一张新帧。

人体左 / 右关键点指人体的解剖侧，不等于图像左右。遮挡点置信度较低时默认不绘制，不能据此认为推理没有运行。查看人数、框和关键点置信度以区分情况。

## 帧率低、显存高或延迟大

先恢复 n 模型、720p 和默认配置，关闭不需要的 GPU 程序，检查 `nvidia-smi` 与手机 USB 连接，再运行 benchmark。笔记本供电、散热和功耗模式会影响实测；1080p、较大模型或不同手机也可能降低速度。

采集 FPS 低说明输入侧已较慢；检测 FPS 低但采集正常时，重点看模型推理和 GPU；两者都正常但显示延迟大时，检查桌面绘制、驱动缓冲与手机链路。应用单槽缓冲只防止自己的积压，不消除其他阶段延迟。

PyTorch 显存池在运行后保留部分显存是正常行为，GUI / benchmark 的数值不等于 `nvidia-smi` 中整个进程占用。默认验收目标是完整 600 秒、检测 ≥20 FPS，具体统计字段见 [使用文档](USAGE.md)。

## 提交问题时提供什么

提供发行版与内核、Android 版本、GPU 和驱动版本、`uv --version`、doctor 的脱敏输出、执行命令、完整错误原因与最小复现步骤。性能问题附上 benchmark 统计及配置。不要上传真实视频、私人截图、手机序列号、凭据或整个 `.state/`。
