import io
import sys
import threading
from types import SimpleNamespace

import pytest

import remote_camera.pose as module
from remote_camera.backend import CameraError
from remote_camera.pose import PoseRunner


def test_runner_requires_cuda_before_loading_weights(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    runner = PoseRunner()
    with pytest.raises(CameraError, match="CUDA 不可用"):
        runner.load(threading.Event())
    assert runner.model is None


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_or_cancelled_download_never_leaves_a_cached_model(monkeypatch, tmp_path, cancelled):
    class Response(io.BytesIO):
        headers = {"Content-Length": "100"}

    monkeypatch.setattr(module, "configure_runtime", lambda: None)
    monkeypatch.setattr(module, "WEIGHTS", tmp_path)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response(b"incomplete"))
    cancel = threading.Event()
    if cancelled:
        cancel.set()
    with pytest.raises(CameraError, match="下载.*失败"):
        module.model_file("yolo26n-pose.pt", cancel)
    assert list(tmp_path.iterdir()) == []


def test_corrupt_cached_builtin_weights_are_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "configure_runtime", lambda: None)
    monkeypatch.setattr(module, "WEIGHTS", tmp_path)
    (tmp_path / "yolo26n-pose.pt").write_bytes(b"broken")
    with pytest.raises(CameraError, match="模型校验失败"):
        module.model_file("yolo26n-pose.pt", threading.Event())
