"""Own one capture/inference session; the Qt thread only polls results."""

import threading
import time
from dataclasses import dataclass, field

from .backend import CameraError, ScrcpyStream, StreamConfig, capture_capable
from .capture import FrameSource, LatestValue
from .pose import BUILTIN_MODELS, PoseResult, PoseRunner


@dataclass(frozen=True)
class PipelineConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    existing: bool = False
    model: str = BUILTIN_MODELS[0]
    confidence: float = 0.35
    rotation: int = 0
    mirror: bool = False


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results: LatestValue[PoseResult] = LatestValue()
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._run, name="pose-inference", daemon=True)
        self.status = "等待启动"
        self.error: str | None = None
        self.gpu_name = ""
        self.source = ""
        self.capture: FrameSource | None = None
        self.inference_count = 0
        self.running_since = 0.0
        self._confidence = config.confidence

    @property
    def detection_fps(self) -> float:
        if not self.running_since:
            return 0.0
        return self.inference_count / max(time.monotonic() - self.running_since, 0.001)

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    def start(self) -> None:
        self.thread.start()

    def request_stop(self) -> None:
        self.cancel.set()
        self.status = "正在停止并释放资源…"

    def set_confidence(self, value: float) -> None:
        self._confidence = value

    def join(self, timeout: float = 30) -> None:
        self.thread.join(timeout)
        if self.alive:
            raise CameraError("后台清理超时，请检查摄像头或模型下载连接。")

    def _run(self) -> None:
        stream: ScrcpyStream | None = None
        try:
            self.status = "检查 CUDA、下载/加载模型并预热…"
            runner = PoseRunner(self.config.model, self.config.confidence)
            runner.load(self.cancel)
            self.gpu_name = runner.gpu_name
            if self.cancel.is_set():
                return
            self.status = "连接摄像头…"
            if self.config.existing:
                self.source = self.config.stream.source or ""
                if not self.source or not capture_capable(self.source):
                    raise CameraError("已有流未就绪，请先启动 camera 命令并指定视频设备。")
            else:
                stream = ScrcpyStream(self.config.stream)
                self.source = stream.start(self.cancel)
            if self.cancel.is_set():
                return
            self.capture = FrameSource(self.source, self.config.rotation, self.config.mirror)
            self.capture.start()
            self.running_since = time.monotonic()
            self.status = f"运行中 · {self.source}"
            sequence = 0
            last_frame = time.monotonic()
            capability_check = time.monotonic()
            while not self.cancel.is_set():
                if stream:
                    stream.check()
                if self.capture.error:
                    raise CameraError(self.capture.error)
                if time.monotonic() - capability_check > 1:
                    capability_check = time.monotonic()
                    if not capture_capable(self.source):
                        raise CameraError("视频生产者已停止，请重新连接手机并启动视频流。")
                sequence, frame = self.capture.frames.get(sequence, timeout=0.1)
                if frame is None:
                    if time.monotonic() - last_frame > 5:
                        raise CameraError("超过 5 秒没有新视频帧，请检查手机和摄像头。")
                    continue
                last_frame = time.monotonic()
                runner.confidence = self._confidence
                result = runner.infer(frame)
                if not self.cancel.is_set():
                    self.inference_count += 1
                    self.results.publish(result)
        except Exception as exc:
            if not self.cancel.is_set():
                self.error = str(exc)
        finally:
            # Stop readers while the writer is still alive to unblock the last read.
            try:
                if self.capture:
                    self.capture.stop()
            except Exception as exc:
                self.error = self.error or str(exc)
            finally:
                if stream:
                    try:
                        stream.stop()
                    except Exception as exc:
                        self.error = self.error or str(exc)
                self.results.close()
                self.status = "运行失败" if self.error else "已停止"
