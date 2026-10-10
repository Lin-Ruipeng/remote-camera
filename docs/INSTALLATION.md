# 安装与环境准备

[返回 README](../README.md) · [故障排查](TROUBLESHOOTING.md)

## 1. 确认硬件与平台

需要 Linux x86_64 电脑、NVIDIA GPU、Android 12 / API 31 以上的手机，以及支持数据传输的 USB 线。电脑的 USB-A 接口可以连接手机，不需要无线网络传输视频。

系统安装脚本面向 Ubuntu / Debian 的 apt 包管理器，真机验证系统为 Ubuntu 26.04.1。其他发行版需自行安装同等系统依赖；本仓库下载器使用官方 Linux x86_64 scrcpy 包，不能直接用于 ARM 电脑。

先确认现有驱动工作正常：

```bash
uname -m
nvidia-smi
```

安装脚本不安装或升级 NVIDIA 驱动。Python 的 PyTorch wheel 自带推理所需 CUDA 用户态库；本项目不要求额外安装完整 CUDA Toolkit。驱动必须支持固定的 cu126 wheel，最终以 `doctor` 中的实际 CUDA 运算为准。

## 2. 安装 uv 并克隆项目

已有 uv 时直接执行 `uv --version`。未安装时，可按 [uv 官方说明](https://docs.astral.sh/uv/getting-started/installation/) 使用安装器：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

按照安装器提示重新打开终端或加载其 PATH 配置，随后：

```bash
git clone https://github.com/Lin-Ruipeng/remote-camera.git
cd remote-camera
uv --version
```

`.python-version` 固定为 3.12.15，`pyproject.toml` 限定 Python 3.12。`uv sync` 可准备该 Python；也可以提前执行 `uv python install 3.12.15`。不需要手动激活 `.venv`，后续命令统一使用 `uv run`。

## 3. 一次性系统安装

以普通用户运行，不要在脚本前加 sudo：

```bash
bash scripts/setup_system.sh
```

脚本仅在需要系统权限的步骤调用 sudo，会执行以下操作：

1. 安装 adb、Android 平台 udev 规则、v4l2loopback DKMS / 工具、V4L2 工具、ACL、当前内核 headers，以及常用 Qt 平台库。
2. 下载官方 scrcpy 5.0.1 Linux x86_64 发行包，校验固定 SHA-256，再解压到 `.tools/`。已有该版本程序时跳过下载。
3. 创建标记为 `Android Camera` 的虚拟摄像头，使用 `exclusive_caps=1`。优先 `/dev/video10`，已占用时选择下一空闲编号。
4. 使用 ACL 授予当前用户该设备的读写权限。

脚本不会自动执行 `apt-get update`。如果 apt 找不到包或软件源索引过期，先执行 `sudo apt-get update`，再重试。内核模块无法加载时，查看 [DKMS / Secure Boot 排查](TROUBLESHOOTING.md#dkms-或内核模块加载失败)。

检查结果：

```bash
v4l2-ctl --list-devices
ls -l /dev/video10
.tools/scrcpy-linux-x86_64-v5.0.1/scrcpy --version
```

实际设备号可能不是 10。摄像头未启动时，`exclusive_caps=1` 的设备只有输出能力；此时其他应用暂时看不到采集输入是正常现象。

## 4. 手机 USB 调试与授权

在手机设置中开启开发者选项和 **USB 调试**。通过 USB 数据线连接电脑，解锁手机，在授权弹窗中允许这台电脑。开发者选项的具体位置取决于手机厂商。

```bash
adb devices -l
```

设备状态应为 `device`。`unauthorized` 表示尚未完成授权；`offline` 表示连接异常；完全没有设备时先检查数据线和端口。遇到授权问题见 [ADB 排查](TROUBLESHOOTING.md#没有手机或状态为-unauthorized--offline)。

可检查系统和镜头能力：

```bash
# 多台手机时给 adb 和 scrcpy 添加 -s / --serial 对应参数
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
.tools/scrcpy-linux-x86_64-v5.0.1/scrcpy --list-cameras
.tools/scrcpy-linux-x86_64-v5.0.1/scrcpy --list-camera-sizes
```

应用在启动时校验 Android API ≥31。720p/30 FPS 是默认请求值，具体摄像头必须支持所请求的组合。关闭手机上正在占用摄像头的相机、视频通话或扫码应用。

## 5. 安装 Python 依赖

```bash
uv sync --locked
uv run python --version
uv run remote-camera doctor
```

`uv.lock` 固定实际解析版本。torch 和 torchvision 通过 uv 的显式源绑定从 `https://download.pytorch.org/whl/cu126` 安装；其他依赖不使用该专用源。默认包含开发组，因而 pytest 和 Ruff 也可用。

首次同步会下载 PyTorch、CUDA 用户态库和 PySide6 等较大的包。已验证环境的 `.venv` 约 7.1 GB，安装过程中还需要下载与解包缓存空间；请预留约 15–20 GB 空闲空间，实际占用随平台和缓存复用情况变化。不要另装普通 `opencv-python` 或普通 `ultralytics` 来覆盖此环境；项目使用官方 headless 变体，GUI 由 PySide6 提供。

`doctor` 成功时应包含 Python、adb、scrcpy、USB 安卓手机、虚拟摄像头、V4L2 设备和 CUDA 运算的 `[OK]`。任一检查失败时退出码为 1。多台手机同时连接时，doctor 的自动选择会报告需要明确设备；在 GUI 中手动选择手机，或让独立 `camera` 使用 `--serial`。

## 6. 首次启动与模型缓存

```bash
uv run remote-camera gui
```

点击「开始检测」后先检查 CUDA、加载模型并预热，再启动手机流。初次使用内置模型时会从 Ultralytics 官方 GitHub release 下载权重并验证 SHA-256：

| 文件 | 用途 | 缓存 |
| --- | --- | --- |
| `yolo26n-pose.pt` | 默认轻量模型 | `weights/yolo26n-pose.pt` |
| `yolo26s-pose.pt` | 可选较大模型 | `weights/yolo26s-pose.pt` |

下载不完整或校验失败的临时文件不会成为正式权重。内置缓存再次使用前也会校验。缓存准备好后，实时视频处理无需联网；电脑仍需通过 USB 连接手机。

离线环境可事先运行一次检测准备缓存，或把与源码中固定 SHA-256 一致的内置权重放到 `weights/`。自定义本地权重可通过 Python API 或 benchmark 的 `--model` 指定，必须是提供 COCO 17 点与关键点置信度的人体 Pose 模型。自定义 `.pt` 模型请只使用可信来源。

## 7. 重启与重新连接

虚拟设备和 ACL 是本次系统运行期间准备的，脚本不配置开机自动加载。电脑重启后：

```bash
cd remote-camera
bash scripts/prepare_camera.sh
uv run remote-camera doctor
uv run remote-camera gui
```

手机断开后，等待应用报告错误并释放资源。重新插入 USB、解锁并授权，在 GUI 中刷新设备，再点击开始。更换 USB 手机、多台设备或实际视频编号改变时，重新选择设备和源路径。

## 8. 本地目录与清理

| 目录 | 内容 | Git 状态 |
| --- | --- | --- |
| `.venv/` | Python、CUDA 与 GUI 依赖 | 忽略 |
| `.tools/` | 官方 scrcpy 发行包 | 忽略 |
| `weights/` | 模型缓存 | 忽略 |
| `.state/` | scrcpy 日志、Ultralytics 配置、显式生成的测试统计 | 忽略 |

停止应用后，可以删除不需要的 `.state/` 资料或模型缓存；删掉内置模型后下次会重新下载。删除 `.venv/` 后通过 `uv sync --locked` 重建。卸载本项目文件不会卸载 apt 安装的系统包，也不会修改 NVIDIA 驱动。

脚本总是在仓库目录准备工具；正常开发请从克隆目录使用 uv。高级嵌入场景可在导入包前设置 `REMOTE_CAMERA_HOME` 到已有、可写的根目录，运行工具与缓存将按该目录查找，详见 [架构文档](ARCHITECTURE.md)。
