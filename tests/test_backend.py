from pathlib import Path

import pytest

import remote_camera.backend as module
from remote_camera.backend import AndroidDevice, CameraError, ScrcpyStream, StreamConfig


@pytest.mark.parametrize("state", ["unauthorized", "offline", "no"])
def test_unauthorized_or_offline_phone_requires_action(monkeypatch, state):
    monkeypatch.setattr(module, "list_android_devices", lambda: [AndroidDevice("USB", state, "x")])
    with pytest.raises(CameraError, match="允许 USB 调试"):
        module.select_android()


def test_multiple_phones_require_explicit_selection(monkeypatch):
    monkeypatch.setattr(
        module,
        "list_android_devices",
        lambda: [AndroidDevice("a", "device", "x"), AndroidDevice("b", "device", "y")],
    )
    with pytest.raises(CameraError, match="多台手机"):
        module.select_android()


def test_producer_startup_failure_keeps_log_and_reaps_process(monkeypatch, tmp_path):
    program = tmp_path / "fake-scrcpy"
    program.write_text("#!/bin/sh\necho 'camera is occupied'\nexit 1\n")
    program.chmod(0o755)
    monkeypatch.setattr(module, "STATE", tmp_path)
    monkeypatch.setattr(module, "select_android", lambda _: AndroidDevice("a", "device", "x"))
    monkeypatch.setattr(module, "loopback_device", lambda _: "/dev/test")
    monkeypatch.setattr(module, "capture_capable", lambda _: False)
    monkeypatch.setattr(module, "executable", lambda _: str(program))
    stream = ScrcpyStream(StreamConfig())
    with pytest.raises(CameraError, match="camera is occupied"):
        stream.start()
    assert stream.process is None
    assert Path(stream.log_path).is_file()
    assert "camera is occupied" in stream.log_tail()


def test_existing_producer_is_not_overwritten(monkeypatch):
    monkeypatch.setattr(module, "select_android", lambda _: AndroidDevice("a", "device", "x"))
    monkeypatch.setattr(module, "loopback_device", lambda _: "/dev/test")
    monkeypatch.setattr(module, "capture_capable", lambda _: True)
    with pytest.raises(CameraError, match="已有视频生产者"):
        ScrcpyStream(StreamConfig()).start()
