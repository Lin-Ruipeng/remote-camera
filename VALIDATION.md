# 本机验证结果

验证日期：2026-10-10。环境：Ubuntu 26.04.1、Android 15、RTX 4060 Laptop GPU（8 GB）、
Python 3.12.15、scrcpy 5.0.1、PyTorch 2.13.0+cu126、torchvision 0.28.0+cu126。

## 10 分钟真机运行

后置摄像头，1280×720 / 30 FPS，`yolo26n-pose.pt`，CUDA:0 / FP16。

| 指标 | 实测 |
| --- | ---: |
| 持续时间 | 600.01 秒 |
| 采集帧率 | 29.93 FPS |
| 检测帧率 | 29.92 FPS |
| 推理平均耗时 | 5.78 ms |
| 推理 P95 | 6.33 ms |
| 本机处理 P95 | 6.45 ms |
| PyTorch 峰值分配显存 | 62.4 MB |
| PyTorch 峰值预留显存 | 80.0 MB |

达到持续 600 秒及检测刷新 ≥20 FPS 的验收目标。显存数值来自 PyTorch 分配器，
本机处理计时不包含手机曝光、编码或 USB 传输。以上是指定硬件与软件组合的一次实测，
不保证其他设备达到相同帧率。机器可读报告在本机 `.state/benchmark.json`。

## 功能验证

- `doctor` 的 Python、ADB、USB 授权、V4L2 与实际 CUDA 矩阵运算均通过。
- 验证实际推理后端的参数位于 `cuda:0` 且类型为 `torch.float16`。
- 空白输入、官方多人图片、单人裁剪和遮挡输入均完成推理，关键点形状为 `(N,17,3)`。
- Wayland 桌面 GUI 后置实时显示、前置镜头、90°旋转、镜像、两轮启停及关闭清理通过。
- 独立 `camera` 命令供已有流模式使用；GUI 推理停止后，外部生产者保持运行。
- 用文档中的 uv 命令运行 Python 示例，1 秒读取 32 帧；独立 FFmpeg 客户端读取通过。
- 对 `camera` 发送 SIGTERM 后，命令与其 scrcpy 子进程正常退出，释放虚拟摄像头。
- 主动终止推理会话的 scrcpy 生产者，错误被报告，采集线程与视频写入端均释放。
- 28 项 pytest 检查通过，Ruff 检查与格式检查通过。

详细功能统计在 `.state/pose-validation.json` 与 `.state/lifecycle-validation.json`。
GUI 验证截图在本机 `.state/gui-live.png`。这些是显式验证产生的本地资料，未纳入公开仓库；
克隆仓库后需自己运行验证生成报告。日常运行不会自动保存视频或关键点。

## 已确认的系统限制

在本次 v4l2loopback 0.15.3 与本机内核配置下，同时启动多个采集客户端会出现设备占用。
GUI、Python 示例、OBS/FFmpeg 可分别读取；切换前先停止前一个客户端。
独立 `camera` 是视频生产者，可与任一采集客户端并行运行。
