# 开发与贡献

[返回 README](README.md) · [架构](docs/ARCHITECTURE.md)

## 开发环境与检查

使用 Python 3.12 和 uv，在克隆目录执行：

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Ruff 以 Python 3.12 为目标，行宽 100。必要时使用 `uv run ruff format .` 调整格式。依赖修改应同步 `pyproject.toml` 和 `uv.lock`，保留 torch / torchvision 的显式 cu126 源绑定，避免混入带 GUI 的 OpenCV。

单元测试通过替身检查关键逻辑，无需手机与实际 CUDA 运算；依赖安装仍包含较大的 GPU / Qt 包。它们覆盖设备选择与子进程生命周期、最新帧缓冲、图像变换、模型下载校验、帧与结果对应、骨架标注及 GUI 非阻塞关闭。单元测试通过不能替代真机验证。

## 修改后如何验证

根据修改范围选择必要检查：

| 修改范围 | 建议验证 |
| --- | --- |
| 文档、参数说明 | 对照 CLI `--help` 与源码，检查本地链接和 Python 片段语法 |
| 采集、线程或进程管理 | 真机读取、重复启停、生产者停止 / USB 断开、关闭资源释放 |
| 旋转、镜像或渲染 | 对应帧标注、四个旋转方向、镜像后的关键点对齐 |
| 模型或依赖 | CUDA / FP16 实际后端、无人 / 单人 / 多人 / 遮挡输入 |
| 性能相关 | 记录配置与硬件，执行 600 秒 benchmark，并比较同等条件下的统计 |

```bash
uv run remote-camera doctor
uv run remote-camera benchmark --seconds 600 --report .state/benchmark.json
```

共享更新后的验证结论时明确环境、时长和统计口径，避免把本机测试结果写成通用保证。报告保存在 `.state/`，默认不提交；必要的脱敏统计可以整理到 Markdown。

## 设计约定

- 保持 GUI 主线程只处理交互与绘图，不把下载、摄像头读取或推理放进去。
- 缓冲只保留最新值，结果携带对应输入帧，旋转 / 镜像在推理前统一完成。
- 一个会话只停止自己拥有的生产者；已有流模式保留外部生产者。
- 采集线程自己释放视频设备，避免与阻塞读取并发调用 release。
- 日常运行不自动保存或上传视频和关键点；新增持久化功能应有明确用户控制。
- 重新开始使用新 Pipeline / FrameSource，不重复启动已经使用过的线程实例。

## 问题与 Pull Request

Issue 请提供最小复现、脱敏 doctor 输出和软硬件版本；性能问题附配置与 benchmark 统计。PR 描述说明具体问题、修改后的行为和完成的验证。与当前改动无关的驱动、内核或系统设置修改应单独说明。

不要提交 `.venv/`、`.tools/`、模型、`.state/`、录像、私人截图或凭据。公开仓库采用 [GNU AGPL-3.0-only](LICENSE)；第三方依赖及模型保留其自身许可。
