import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

import remote_camera.gui as module


def test_gui_mode_controls_and_close_request_are_nonblocking(monkeypatch):
    monkeypatch.setattr(module, "list_android_devices", lambda: [])
    monkeypatch.setattr(module, "loopback_device", lambda: "/dev/test")
    app = QApplication.instance() or QApplication([])
    window = module.MainWindow()
    window.existing.setChecked(True)
    assert not window.facing.isEnabled()
    assert window.source.isEnabled()
    window.existing.setChecked(False)
    assert window.facing.isEnabled()

    class FakePipeline:
        def __init__(self):
            import threading

            self.cancel = threading.Event()

        def request_stop(self):
            self.cancel.set()

    fake = FakePipeline()
    window.pipeline = fake
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    assert fake.cancel.is_set()
    assert not window.start_button.isEnabled()
    window.pipeline = None
    window.close()
    app.processEvents()
