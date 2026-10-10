import threading
import time
from types import SimpleNamespace

import numpy as np

import remote_camera.pipeline as module
from remote_camera.backend import StreamConfig
from remote_camera.capture import FramePacket, LatestValue
from remote_camera.pipeline import Pipeline, PipelineConfig
from remote_camera.pose import PoseResult


def test_slow_inference_skips_frames_and_external_writer_is_never_stopped(monkeypatch):
    inferred = []

    class FakeCapture:
        def __init__(self, *_):
            self.frames = LatestValue()
            self.error = None
            self.count = 0
            self.fps = 100
            self.cancel = threading.Event()
            self.thread = threading.Thread(target=self.produce)

        def produce(self):
            while not self.cancel.is_set():
                self.count += 1
                image = np.full((2, 2, 3), self.count % 255, dtype=np.uint8)
                self.frames.publish(FramePacket(image, self.count, time.monotonic()))
                self.cancel.wait(0.01)

        def start(self):
            self.thread.start()

        def stop(self):
            self.cancel.set()
            self.thread.join(1)
            self.frames.close()

    class FakeRunner:
        def __init__(self, *_):
            self.gpu_name = "test GPU"

        def load(self, *_):
            pass

        def infer(self, frame):
            time.sleep(0.08)
            inferred.append(frame.frame_id)
            assert frame.image_bgr[0, 0, 0] == frame.frame_id % 255
            return PoseResult(frame, np.empty((0, 5)), np.empty((0, 17, 3)), 80, time.monotonic())

    def forbidden_stream(*_):
        raise AssertionError("Existing streams must not create/stop a scrcpy process")

    monkeypatch.setattr(module, "FrameSource", FakeCapture)
    monkeypatch.setattr(module, "PoseRunner", FakeRunner)
    monkeypatch.setattr(module, "ScrcpyStream", forbidden_stream)
    monkeypatch.setattr(module, "capture_capable", lambda _: True)
    session = Pipeline(PipelineConfig(stream=StreamConfig(source="/dev/test"), existing=True))
    session.start()
    sequence = 0
    seen = []
    deadline = time.monotonic() + 3
    try:
        while len(seen) < 3 and time.monotonic() < deadline:
            sequence, result = session.results.get(sequence, timeout=0.2)
            if result:
                seen.append(result.frame.frame_id)
    finally:
        session.request_stop()
        session.join(2)
    assert session.error is None
    assert len(seen) >= 3
    assert seen[1] - seen[0] > 1
    assert not session.capture.thread.is_alive()


def test_cuda_failure_is_reported_without_starting_camera(monkeypatch):
    class FailedRunner:
        def __init__(self, *_):
            pass

        def load(self, *_):
            raise RuntimeError("CUDA 不可用")

    monkeypatch.setattr(module, "PoseRunner", FailedRunner)
    session = Pipeline(PipelineConfig())
    session.start()
    session.join(2)
    assert session.error == "CUDA 不可用"
    assert session.capture is None
    assert session.status == "运行失败"


def test_disconnect_still_stops_owned_stream(monkeypatch):
    stopped = []

    class FakeStream:
        def __init__(self, *_):
            pass

        def start(self, *_):
            return "/dev/test"

        def check(self):
            raise RuntimeError("手机已断开")

        def stop(self):
            stopped.append("writer")

    class FakeCapture:
        error = None

        def __init__(self, *_):
            pass

        def start(self):
            pass

        def stop(self):
            stopped.append("reader")

    monkeypatch.setattr(module, "ScrcpyStream", FakeStream)
    monkeypatch.setattr(module, "FrameSource", FakeCapture)
    monkeypatch.setattr(
        module, "PoseRunner", lambda *_: SimpleNamespace(load=lambda *_: None, gpu_name="test")
    )
    session = Pipeline(PipelineConfig())
    session.start()
    session.join(2)
    assert session.error == "手机已断开"
    assert stopped == ["reader", "writer"]
