import argparse
import json
import platform
import signal
import sys
import time
from pathlib import Path

from .backend import (
    CameraError,
    ScrcpyStream,
    StreamConfig,
    executable,
    loopback_device,
    select_android,
    video_devices,
)
from .paths import configure_runtime


def doctor() -> int:
    failures = 0

    def check(label, callback):
        nonlocal failures
        try:
            print(f"[OK] {label}: {callback()}", flush=True)
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {label}: {exc}", flush=True)

    check("Python", lambda: platform.python_version())
    for name in ("adb", "scrcpy", "v4l2-ctl"):
        check(name, lambda name=name: executable(name))
    check("USB 安卓手机", lambda: select_android())
    check("虚拟摄像头", lambda: loopback_device())
    check("V4L2 设备", video_devices)

    def cuda():
        import torch

        if not torch.cuda.is_available():
            raise CameraError("CUDA 不可用")
        # Exercise the driver, allocation, kernels and host transfer.
        tensor = torch.ones((32, 32), device="cuda:0")
        value = (tensor @ tensor).mean().item()
        assert value == 32
        return (
            f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}"
        )

    check("CUDA 运算", cuda)
    return int(failures > 0)


def stream_config(args) -> StreamConfig:
    return StreamConfig(args.serial, args.source, args.facing, args.size, 30)


def run_camera(args) -> int:
    stream = ScrcpyStream(stream_config(args))

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        source = stream.start()
        print(f"手机摄像头已启动: {source}；按 Ctrl+C 停止。", flush=True)
        while True:
            stream.check()
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        stream.stop()
        signal.signal(signal.SIGTERM, previous)


def benchmark(args) -> int:
    import numpy as np
    import torch

    from .pipeline import Pipeline, PipelineConfig

    session = Pipeline(
        PipelineConfig(
            stream=stream_config(args),
            existing=args.existing,
            model=args.model,
            rotation=args.rotation,
            mirror=args.mirror,
        )
    )
    sequence = 0
    timings = []
    host_timings = []
    people = set()
    beginning = None
    report_at = 0.0
    first_count = 0
    first_capture_count = 0
    interrupted = False
    session.start()
    try:
        while session.alive:
            sequence, result = session.results.get(sequence, timeout=0.1)
            now = time.monotonic()
            if result is not None:
                if beginning is None:
                    beginning = now
                    first_count = session.inference_count
                    first_capture_count = session.capture.count
                    torch.cuda.reset_peak_memory_stats(0)
                    print(f"预热完成，开始 {args.seconds:g} 秒真机测试。", flush=True)
                timings.append(result.inference_ms)
                host_timings.append((now - result.frame.captured_at) * 1000)
                people.add(len(result.boxes))
                if now - report_at >= 30:
                    report_at = now
                    print(
                        f"{now - beginning:.0f}s  采集 {session.capture.fps:.1f} FPS  "
                        f"检测 {session.detection_fps:.1f} FPS  推理 {result.inference_ms:.1f} ms",
                        flush=True,
                    )
                if now - beginning >= args.seconds:
                    break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        session.request_stop()
        session.join()
    if session.error:
        raise CameraError(session.error)
    if not timings:
        raise CameraError("测试未获得任何检测帧")
    elapsed = now - beginning
    report = {
        "duration_seconds": round(elapsed, 2),
        "completed": not interrupted and elapsed >= args.seconds,
        "gpu": session.gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": args.model,
        "source": session.source,
        "capture_fps": round((session.capture.count - first_capture_count) / elapsed, 2),
        "detection_fps": round((session.inference_count - first_count) / elapsed, 2),
        "inference_ms_mean": round(float(np.mean(timings)), 2),
        "inference_ms_p95": round(float(np.percentile(timings, 95)), 2),
        "host_processing_ms_p95": round(float(np.percentile(host_timings, 95)), 2),
        "peak_allocated_vram_mb": round(torch.cuda.max_memory_allocated(0) / 1024**2, 1),
        "peak_reserved_vram_mb": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
        "people_counts_observed": sorted(people),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return int(not report["completed"] or report["detection_fps"] < 20)


def main() -> int:
    configure_runtime()
    parser = argparse.ArgumentParser(description="USB 安卓摄像头与 CUDA 实时人体姿态检测")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="检查环境、USB 手机与 CUDA 运算")
    for name in ("camera", "benchmark"):
        sub = commands.add_parser(
            name, help="启动独立摄像头流" if name == "camera" else "真机性能测试"
        )
        sub.add_argument("--serial")
        sub.add_argument("--source", help="Android Camera 虚拟设备路径")
        sub.add_argument("--facing", choices=("back", "front"), default="back")
        sub.add_argument("--size", choices=("1280x720", "1920x1080"), default="1280x720")
        if name == "benchmark":
            sub.add_argument("--seconds", type=float, default=600)
            sub.add_argument("--existing", action="store_true")
            sub.add_argument("--model", default="yolo26n-pose.pt")
            sub.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
            sub.add_argument("--mirror", action="store_true")
            sub.add_argument("--report", help="仅保存性能统计 JSON，不保存视频或关键点")
    gui = commands.add_parser("gui", help="启动 PySide6 桌面检测窗口")
    gui.add_argument("--source")
    gui.add_argument("--existing", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "camera":
            return run_camera(args)
        if args.command == "benchmark":
            if args.seconds <= 0:
                parser.error("--seconds 必须大于零")
            return benchmark(args)
        from .gui import launch

        return launch(args.source, args.existing)
    except (CameraError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
