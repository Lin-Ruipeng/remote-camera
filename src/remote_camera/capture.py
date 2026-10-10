"""Continuous V4L2 reads with a single latest-frame slot."""

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .backend import CameraError


class LatestValue[T]:
    def __init__(self):
        self._condition = threading.Condition()
        self._value: T | None = None
        self._sequence = 0
        self._closed = False

    def publish(self, value: T) -> None:
        with self._condition:
            if not self._closed:
                self._value = value
                self._sequence += 1
                self._condition.notify_all()

    def get(self, after: int = 0, timeout: float = 0) -> tuple[int, T | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after or self._closed, timeout=timeout
            )
            if self._sequence <= after:
                return after, None
            return self._sequence, self._value

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


@dataclass(frozen=True)
class FramePacket:
    image_bgr: np.ndarray
    frame_id: int
    captured_at: float  # time.monotonic(), after this host's V4L2 read


def transform_frame(frame: np.ndarray, rotation: int, mirror: bool) -> np.ndarray:
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation 必须为 0、90、180 或 270")
    if rotation:
        frame = cv2.rotate(frame, rotations[rotation])
    if mirror:
        frame = cv2.flip(frame, 1)
    return np.ascontiguousarray(frame)


class FrameSource:
    def __init__(self, source: str, rotation: int = 0, mirror: bool = False):
        self.source = source
        self.rotation = rotation
        self.mirror = mirror
        self.frames: LatestValue[FramePacket] = LatestValue()
        self.cancel = threading.Event()
        self.error: str | None = None
        self.count = 0
        self.started_at = 0.0
        self.thread = threading.Thread(target=self._capture, name="v4l2-capture", daemon=True)

    @property
    def fps(self) -> float:
        return self.count / max(time.monotonic() - self.started_at, 0.001)

    def start(self) -> None:
        self.started_at = time.monotonic()
        self.thread.start()

    def _capture(self) -> None:
        cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise CameraError(
                    f"无法打开 {self.source}；请检查视频流、设备权限，或停止正在读取它的其他程序。"
                )
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.cancel.is_set():
                ok, frame = cap.read()
                captured_at = time.monotonic()
                if self.cancel.is_set():
                    break
                if not ok or frame is None:
                    raise CameraError("摄像头读取失败；请检查手机连接和 scrcpy 状态。")
                self.count += 1
                image = transform_frame(frame, self.rotation, self.mirror)
                self.frames.publish(FramePacket(image, self.count, captured_at))
        except Exception as exc:
            self.error = str(exc)
        finally:
            # Release in the capture thread; never race release() against read().
            cap.release()
            self.frames.close()

    def stop(self) -> None:
        self.cancel.set()
        self.frames.close()
        if self.thread.ident:
            self.thread.join(timeout=12)
            if self.thread.is_alive():
                raise CameraError("V4L2 读取未能在 12 秒内退出；请检查虚拟摄像头驱动。")
