"""PySide6 desktop viewer. All device/model work stays off the Qt event loop."""

import queue
import signal
import threading
import time

import cv2
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .backend import StreamConfig, list_android_devices, loopback_device
from .pipeline import Pipeline, PipelineConfig
from .pose import BUILTIN_MODELS
from .render import annotate


class VideoCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.setMinimumSize(640, 360)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self.image.isNull():
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "连接手机后，点击「开始检测」"
            )
            return
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRectF(
            (self.width() - size.width()) / 2,
            (self.height() - size.height()) / 2,
            size.width(),
            size.height(),
        )
        painter.drawImage(target, self.image)


class MainWindow(QMainWindow):
    def __init__(self, source: str | None = None, existing: bool = False):
        super().__init__()
        self.setWindowTitle("Android Pose · CUDA 实时人体姿态")
        self.resize(1200, 850)
        self.pipeline: Pipeline | None = None
        self.sequence = 0
        self.closing = False
        self.discovery: queue.SimpleQueue = queue.SimpleQueue()
        self.discovering = False
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 16, 20, 16)
        title = QLabel("Android Pose")
        title.setStyleSheet("font-size: 26px; font-weight: 650; color: #eef6ff")
        layout.addWidget(title)
        layout.addWidget(QLabel("USB 摄像头  /  CUDA 人体姿态  /  COCO 17 关键点"))

        settings = QGroupBox("连接与检测设置")
        grid = QGridLayout(settings)
        for column in range(4):
            grid.setColumnStretch(column, 1)
        self.devices = QComboBox()
        self.devices.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.devices.setMinimumContentsLength(18)
        self.devices.addItem("自动选择唯一的 USB 手机", None)
        self.refresh = QPushButton("刷新设备")
        self.refresh.clicked.connect(self.refresh_devices)
        self.facing = QComboBox()
        self.facing.addItem("后置摄像头", "back")
        self.facing.addItem("前置摄像头", "front")
        self.resolution = QComboBox()
        self.resolution.addItems(["1280x720", "1920x1080"])
        self.model = QComboBox()
        self.model.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.model.setMinimumContentsLength(18)
        self.model.addItems(list(BUILTIN_MODELS))
        self.rotation = QComboBox()
        for angle in (0, 90, 180, 270):
            self.rotation.addItem(f"{angle}°", angle)
        self.mirror = QCheckBox("水平镜像")
        self.existing = QCheckBox("使用已有摄像头流")
        self.existing.setChecked(existing)
        try:
            default_source = loopback_device()
        except Exception:
            default_source = "/dev/video10"
        self.source = QLineEdit(source or default_source)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.05, 0.95)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.35)
        self.confidence.valueChanged.connect(self.change_confidence)
        fields = [
            ("USB 设备", self.devices),
            ("镜头", self.facing),
            ("分辨率 / 30 FPS", self.resolution),
            ("姿态模型", self.model),
            ("视频设备", self.source),
            ("画面旋转", self.rotation),
            ("人体置信度", self.confidence),
        ]
        for index, (label, control) in enumerate(fields):
            row, column = divmod(index, 4)
            grid.addWidget(QLabel(label), row * 2, column)
            grid.addWidget(control, row * 2 + 1, column)
        grid.addWidget(self.refresh, 3, 3)
        grid.addWidget(self.mirror, 4, 1)
        grid.addWidget(self.existing, 4, 0)
        self.existing.toggled.connect(self.update_controls)
        layout.addWidget(settings)
        self.canvas = VideoCanvas()
        layout.addWidget(self.canvas, 1)

        self.metrics = QLabel("人数 —   |   采集 — FPS   |   检测 — FPS   |   推理 — ms")
        self.metrics.setStyleSheet("font-size: 15px; color: #7ee4c3; padding: 8px 0;")
        layout.addWidget(self.metrics)
        self.gpu = QLabel("GPU：启动时检查 CUDA")
        layout.addWidget(self.gpu)
        controls = QHBoxLayout()
        self.start_button = QPushButton("开始检测")
        self.stop_button = QPushButton("停止")
        self.start_button.clicked.connect(self.start_session)
        self.stop_button.clicked.connect(self.stop_session)
        self.status = QLabel("等待启动")
        self.status.setWordWrap(True)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.status, 1)
        layout.addLayout(controls)
        self.locked_controls = [
            self.devices,
            self.facing,
            self.resolution,
            self.model,
            self.rotation,
            self.mirror,
            self.existing,
            self.source,
        ]
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #111a26; color: #c8d6e5; }
            QGroupBox { border: 1px solid #304054; border-radius: 8px;
                        margin-top: 14px; padding: 14px 10px 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; }
            QComboBox, QLineEdit, QDoubleSpinBox { background: #1e2b3b;
                border: 1px solid #3a4d63; border-radius: 4px; padding: 6px; }
            QPushButton { background: #245b81; border: none;
                border-radius: 5px; padding: 9px 16px; }
            QPushButton:hover { background: #3179a4; }
            QPushButton:disabled { background: #263443; color: #718194; }
        """)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.update_controls()
        self.refresh_devices()

    def refresh_devices(self) -> None:
        if self.discovering:
            return
        self.discovering = True
        self.refresh.setEnabled(False)

        def discover():
            try:
                self.discovery.put((list_android_devices(), None))
            except Exception as exc:
                self.discovery.put(([], str(exc)))

        threading.Thread(target=discover, name="adb-discovery", daemon=True).start()

    def update_controls(self) -> None:
        running = self.pipeline is not None
        for control in self.locked_controls:
            control.setEnabled(not running)
        if not running and self.existing.isChecked():
            for control in (self.devices, self.facing, self.resolution):
                control.setEnabled(False)
        self.start_button.setEnabled(not running and not self.closing)
        self.stop_button.setEnabled(running and not self.pipeline.cancel.is_set())
        self.refresh.setEnabled(not running and not self.discovering)

    def start_session(self) -> None:
        if self.pipeline:
            return
        source = self.source.text().strip() or None
        config = PipelineConfig(
            stream=StreamConfig(
                serial=self.devices.currentData(),
                source=source,
                facing=self.facing.currentData(),
                size=self.resolution.currentText(),
            ),
            existing=self.existing.isChecked(),
            model=self.model.currentText(),
            confidence=self.confidence.value(),
            rotation=self.rotation.currentData(),
            mirror=self.mirror.isChecked(),
        )
        self.sequence = 0
        self.pipeline = Pipeline(config)
        self.pipeline.start()
        self.update_controls()

    def change_confidence(self, value: float) -> None:
        if self.pipeline:
            self.pipeline.set_confidence(value)

    def stop_session(self) -> None:
        if self.pipeline:
            self.pipeline.request_stop()
        self.update_controls()

    def tick(self) -> None:
        if not self.discovery.empty():
            devices, error = self.discovery.get()
            selected = self.devices.currentData()
            self.devices.clear()
            self.devices.addItem("自动选择唯一的 USB 手机", None)
            for device in devices:
                self.devices.addItem(
                    f"{device.model} · {device.serial} · {device.state}", device.serial
                )
            index = self.devices.findData(selected)
            self.devices.setCurrentIndex(max(0, index))
            self.discovering = False
            if error and not self.pipeline:
                self.status.setText(error)
            self.update_controls()
        session = self.pipeline
        if session:
            self.status.setText(session.status)
            self.sequence, result = session.results.get(self.sequence)
            if result is not None:
                rgb = cv2.cvtColor(annotate(result), cv2.COLOR_BGR2RGB)
                self.canvas.image = QImage(
                    rgb.data,
                    rgb.shape[1],
                    rgb.shape[0],
                    rgb.strides[0],
                    QImage.Format.Format_RGB888,
                ).copy()
                self.canvas.update()
                host_ms = (time.monotonic() - result.frame.captured_at) * 1000
                self.metrics.setText(
                    f"人数 {len(result.boxes)}   |   采集 {session.capture.fps:.1f} FPS   |   "
                    f"检测 {session.detection_fps:.1f} FPS   |   推理 {result.inference_ms:.1f} ms"
                    f"   |   本机处理 {host_ms:.1f} ms"
                )
                import torch

                memory = torch.cuda.memory_reserved(0) / (1024**2)
                self.gpu.setText(
                    f"{session.gpu_name}  ·  CUDA:0 / FP16  ·  预留显存 {memory:.0f} MB"
                )
            if not session.alive:
                error = session.error
                self.pipeline = None
                self.status.setText(error.splitlines()[0] if error else "已停止，视频资源已释放")
                self.status.setToolTip(error or "")
                self.update_controls()
                if error and not self.closing:
                    message = QMessageBox(self)
                    message.setIcon(QMessageBox.Icon.Warning)
                    message.setWindowTitle("检测未能继续")
                    message.setText(error.splitlines()[0])
                    message.setDetailedText(error)
                    message.exec()
        if self.closing and self.pipeline is None:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.pipeline:
            self.closing = True
            self.stop_session()
            event.ignore()
        else:
            self.timer.stop()
            event.accept()


def launch(source: str | None = None, existing: bool = False) -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    window = MainWindow(source, existing)
    window.show()
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    return app.exec()
