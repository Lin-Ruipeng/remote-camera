#!/usr/bin/env bash
set -euo pipefail

# A reboot removes a manually created loopback device. Re-run this script then.
if [[ ${EUID} -eq 0 ]]; then
    echo "请以普通用户运行本脚本；需要提权的步骤会调用 sudo。" >&2
    exit 1
fi
for name in /sys/class/video4linux/video*/name; do
    if [[ -f "$name" ]] && [[ $(cat "$name") == "Android Camera" ]]; then
        device="/dev/$(basename "$(dirname "$name")")"
        sudo setfacl -m "u:$(id -un):rw" "$device"
        echo "已准备摄像头: $device"
        exit 0
    fi
done

number=10
while [[ -e /dev/video${number} ]]; do ((number+=1)); done
device="/dev/video${number}"
if [[ -d /sys/module/v4l2loopback ]]; then
    sudo v4l2loopback-ctl add -n "Android Camera" -x 1 "$device"
else
    sudo modprobe v4l2loopback devices=1 video_nr="$number" \
        card_label="Android Camera" exclusive_caps=1
fi
sudo udevadm settle
sudo setfacl -m "u:$(id -un):rw" "$device"
echo "已准备摄像头: $device"
