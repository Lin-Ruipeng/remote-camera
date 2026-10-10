"""ADB selection, V4L2 discovery and the owned scrcpy process."""

import fcntl
import os
import shutil
import signal
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from .paths import ROOT, STATE, TOOLS


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class AndroidDevice:
    serial: str
    state: str
    model: str


def executable(name: str) -> str:
    if name == "scrcpy":
        portable = TOOLS / "scrcpy-linux-x86_64-v5.0.1" / "scrcpy"
        if portable.is_file():
            return str(portable)
    path = shutil.which(name)
    if not path:
        raise CameraError(f"缺少 {name}，请运行 bash scripts/setup_system.sh")
    return path


def command(args: list[str], timeout: float = 12) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CameraError(f"执行 {args[0]} 失败: {exc}") from exc
    if result.returncode:
        raise CameraError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def list_android_devices() -> list[AndroidDevice]:
    output = command([executable("adb"), "devices", "-l"])
    devices = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2 or parts[0].startswith("*"):
            continue
        # Wireless transports and emulators are outside this USB application's scope.
        if parts[0].startswith("emulator-") or ":" in parts[0]:
            continue
        model = next((p[6:] for p in parts[2:] if p.startswith("model:")), parts[0])
        devices.append(AndroidDevice(parts[0], parts[1], model))
    return devices


def select_android(serial: str | None = None) -> AndroidDevice:
    devices = list_android_devices()
    if serial:
        devices = [device for device in devices if device.serial == serial]
    if not devices:
        raise CameraError("未发现 USB 安卓手机；请检查数据线、USB 调试和连接状态。")
    if len(devices) != 1:
        raise CameraError("发现多台手机，请在 GUI 选择设备或使用 --serial 指定。")
    device = devices[0]
    if device.state != "device":
        raise CameraError(
            f"手机状态为 {device.state}；请解锁手机并允许 USB 调试授权，然后刷新设备。"
        )
    sdk = command(
        [executable("adb"), "-s", device.serial, "shell", "getprop", "ro.build.version.sdk"]
    )
    if not sdk.isdigit() or int(sdk) < 31:
        raise CameraError(f"直接摄像头采集需要 Android 12/API 31 以上；当前 API: {sdk}")
    return device


def video_devices() -> dict[str, str]:
    devices = {}
    for name in sorted(Path("/sys/class/video4linux").glob("video*/name")):
        devices[f"/dev/{name.parent.name}"] = name.read_text().strip()
    return devices


def capture_capable(path: str) -> bool:
    """exclusive_caps loopback switches from OUTPUT to CAPTURE when a writer starts."""
    try:
        with open(path, "rb", buffering=0) as device:
            data = bytearray(104)
            fcntl.ioctl(device, 0x80685600, data)  # VIDIOC_QUERYCAP
            capabilities, device_caps = struct.unpack_from("II", data, 84)
            caps = device_caps if capabilities & 0x80000000 else capabilities
            return bool(caps & 0x00000001)
    except OSError:
        return False


def loopback_device(requested: str | None = None) -> str:
    devices = video_devices()
    if requested:
        if devices.get(requested) != "Android Camera":
            raise CameraError(f"{requested} 不是标记为 Android Camera 的虚拟摄像头。")
        path = requested
    else:
        matches = [path for path, label in devices.items() if label == "Android Camera"]
        if not matches:
            raise CameraError("没有 Android Camera；请运行 bash scripts/prepare_camera.sh")
        path = matches[0]
    if not os.access(path, os.R_OK | os.W_OK):
        raise CameraError(f"无法读写 {path}；请运行 bash scripts/prepare_camera.sh")
    return path


@dataclass(frozen=True)
class StreamConfig:
    serial: str | None = None
    source: str | None = None
    facing: str = "back"
    size: str = "1280x720"
    fps: int = 30
    bitrate: str = "4M"


class ScrcpyStream:
    def __init__(self, config: StreamConfig):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.source = ""
        self.log_path: Path | None = None

    def start(self, cancel: Event | None = None) -> str:
        device = select_android(self.config.serial)
        self.source = loopback_device(self.config.source)
        if capture_capable(self.source):
            raise CameraError(
                f"{self.source} 已有视频生产者；请选择‘使用已有流’，或先停止独立 camera 命令。"
            )
        args = [
            executable("scrcpy"),
            "--serial",
            device.serial,
            "--video-source=camera",
            f"--camera-facing={self.config.facing}",
            f"--camera-size={self.config.size}",
            f"--camera-fps={self.config.fps}",
            "--video-codec=h264",
            f"--video-bit-rate={self.config.bitrate}",
            "--no-audio",
            "--no-control",
            "--no-video-playback",
            f"--v4l2-sink={self.source}",
        ]
        STATE.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="scrcpy-", suffix=".log", dir=STATE, delete=False
        ) as log:
            # Keep the log for startup/disconnection diagnostics.
            self.log_path = Path(log.name)
            env = os.environ.copy()
            env["ADB"] = executable("adb")
            self.process = subprocess.Popen(
                args,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
                cwd=ROOT,
            )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if cancel and cancel.is_set():
                    raise CameraError("启动已取消")
                self.check()
                if capture_capable(self.source):
                    return self.source
                time.sleep(0.1)
            raise CameraError(f"scrcpy 启动超时。\n{self.log_tail()}")
        except BaseException:
            self.stop()
            raise

    def log_tail(self) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        with self.log_path.open("rb") as log:
            log.seek(max(0, self.log_path.stat().st_size - 4000))
            return log.read().decode(errors="replace")

    def check(self) -> None:
        if self.process and self.process.poll() is not None:
            raise CameraError(f"scrcpy 已退出；请检查手机连接或摄像头占用。\n{self.log_tail()}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=3)
        self.process = None
