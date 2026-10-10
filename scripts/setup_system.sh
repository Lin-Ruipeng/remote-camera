#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ${EUID} -eq 0 ]]; then
    echo "请以普通用户运行：bash scripts/setup_system.sh" >&2
    exit 1
fi
sudo apt-get install -y adb android-sdk-platform-tools-common \
    v4l2loopback-dkms v4l2loopback-utils v4l-utils acl \
    "linux-headers-$(uname -r)" libxcb-cursor0 libxkbcommon0
python3 "$project_dir/scripts/download_scrcpy.py"
bash "$project_dir/scripts/prepare_camera.sh"
echo "系统接入工具已准备。解锁手机并允许 USB 调试授权，然后运行 uv sync。"
