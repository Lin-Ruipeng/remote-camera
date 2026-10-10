import os
from pathlib import Path

ROOT = Path(os.environ.get("REMOTE_CAMERA_HOME", Path(__file__).resolve().parents[2]))
TOOLS = ROOT / ".tools"
STATE = ROOT / ".state"
WEIGHTS = ROOT / "weights"


def configure_runtime() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    config = STATE / "ultralytics"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
