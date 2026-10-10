#!/usr/bin/env python3
"""Fetch the verified official portable Linux build; no root required."""

import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

VERSION = "5.0.1"
SHA256 = "9f969d30cc574816077edecda65719c1c70b0aee1df1ba5fa00779ac07b8fd6e"
ROOT = Path(__file__).resolve().parents[1]
URL = (
    f"https://github.com/Genymobile/scrcpy/releases/download/v{VERSION}/"
    f"scrcpy-linux-x86_64-v{VERSION}.tar.gz"
)


def main() -> None:
    destination = ROOT / ".tools"
    binary = destination / f"scrcpy-linux-x86_64-v{VERSION}" / "scrcpy"
    if binary.exists():
        print(f"已安装: {binary}")
        return
    print(f"下载官方 scrcpy {VERSION}…", flush=True)
    with urllib.request.urlopen(URL, timeout=90) as response:
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"scrcpy SHA-256 不匹配: {digest}")
    destination.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    if not binary.is_file():
        raise RuntimeError("发行包未包含预期的 scrcpy 程序")
    print(f"已校验并安装: {binary}")


if __name__ == "__main__":
    main()
