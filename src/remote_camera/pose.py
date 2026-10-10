"""CUDA-only COCO pose inference, independent of the GUI."""

import hashlib
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .backend import CameraError
from .capture import FramePacket
from .paths import WEIGHTS, configure_runtime

BUILTIN_MODELS = ("yolo26n-pose.pt", "yolo26s-pose.pt")
MODEL_SHA256 = {
    "yolo26n-pose.pt": "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
    "yolo26s-pose.pt": "a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646",
}
COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
# Zero-based COCO keypoint indices.
SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True)
class PoseResult:
    frame: FramePacket
    boxes: np.ndarray  # (N, 5): x1, y1, x2, y2, confidence, original image pixels
    keypoints: np.ndarray  # (N, 17, 3): x, y, confidence, same coordinate space
    inference_ms: float  # includes preprocessing, GPU inference and CPU result transfer
    finished_at: float


def model_file(name: str, cancel: threading.Event) -> Path:
    configure_runtime()
    if name not in BUILTIN_MODELS:
        custom = Path(name).expanduser()
        if custom.is_file():
            return custom
        raise CameraError(f"模型文件不存在: {name}")
    destination = WEIGHTS / name
    if destination.is_file():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == MODEL_SHA256[name]:
            return destination
        raise CameraError(f"模型校验失败，请删除 {destination} 后重新下载。")
    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
    partial: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            expected = int(response.headers.get("Content-Length", "0"))
            received = 0
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=WEIGHTS, suffix=".part", delete=False) as file:
                partial = Path(file.name)
                while chunk := response.read(256 * 1024):
                    if cancel.is_set():
                        raise CameraError("模型下载已取消")
                    file.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
            if expected and received != expected:
                raise CameraError("模型下载不完整，请重新启动检测。")
            if digest.hexdigest() != MODEL_SHA256[name]:
                raise CameraError("模型 SHA-256 校验失败，请重新启动检测。")
        if cancel.is_set():
            raise CameraError("模型下载已取消")
        partial.replace(destination)
        return destination
    except Exception as exc:
        raise CameraError(f"下载 {name} 失败: {exc}") from exc
    finally:
        if partial:
            partial.unlink(missing_ok=True)


class PoseRunner:
    def __init__(self, model: str = BUILTIN_MODELS[0], confidence: float = 0.35):
        self.model_name = model
        self.confidence = confidence
        self.model = None
        self.gpu_name = ""

    def load(self, cancel: threading.Event) -> None:
        if cancel.is_set():
            return
        import torch

        if not torch.cuda.is_available():
            raise CameraError("CUDA 不可用，请运行 remote-camera doctor 检查驱动和 PyTorch。")
        self.gpu_name = torch.cuda.get_device_name(0)
        weights = model_file(self.model_name, cancel)
        if cancel.is_set():
            return
        from ultralytics import YOLO

        try:
            self.model = YOLO(str(weights), task="pose").to("cuda:0")
            if self.model.task != "pose" or tuple(self.model.model.kpt_shape) != (17, 3):
                raise CameraError("请使用具有 COCO 17 个关键点和关键点置信度的人体 Pose 模型。")
            self.infer(FramePacket(np.zeros((720, 1280, 3), dtype=np.uint8), 0, time.monotonic()))
            backend = self.model.predictor.model
            if backend.device.type != "cuda" or not backend.fp16:
                raise CameraError("推理后端未使用 CUDA/FP16")
        except Exception as exc:
            raise CameraError(f"加载或预热姿态模型失败: {exc}") from exc

    def infer(self, frame: FramePacket) -> PoseResult:
        if self.model is None:
            raise CameraError("模型尚未加载")
        started = time.monotonic()
        prediction = self.model.predict(
            frame.image_bgr,
            device=0,
            quantize="fp16",
            imgsz=640,
            conf=self.confidence,
            verbose=False,
            save=False,
            show=False,
        )[0]
        boxes = np.concatenate(
            (prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy()[:, None]),
            axis=1,
        ).astype(np.float32, copy=False)
        keypoints = prediction.keypoints.data.cpu().numpy().astype(np.float32, copy=False)
        finished = time.monotonic()  # .cpu() above waits for the corresponding GPU work
        return PoseResult(frame, boxes, keypoints, (finished - started) * 1000, finished)
