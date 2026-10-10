"""Run camera in another terminal, then uv run python examples/read_camera.py."""

import argparse
import time

from remote_camera.backend import loopback_device
from remote_camera.capture import FrameSource

parser = argparse.ArgumentParser()
parser.add_argument("--source")
parser.add_argument("--seconds", type=float, default=10)
args = parser.parse_args()
source = FrameSource(args.source or loopback_device())
source.start()
sequence = 0
deadline = time.monotonic() + args.seconds
try:
    while time.monotonic() < deadline:
        sequence, packet = source.frames.get(sequence, timeout=1)
        if source.error:
            raise RuntimeError(source.error)
        if packet is not None:
            # packet.image_bgr is a NumPy BGR array. Forward it to your own pipeline.
            print(
                f"frame={packet.frame_id} shape={packet.image_bgr.shape} "
                f"captured_at={packet.captured_at:.6f} fps={source.fps:.1f}",
                flush=True,
            )
except KeyboardInterrupt:
    pass
finally:
    source.stop()
