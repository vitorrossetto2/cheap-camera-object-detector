from __future__ import annotations

import logging
import os
import queue
import threading
from base64 import b64encode
from dataclasses import dataclass
from tkinter import BooleanVar, END, DISABLED, NORMAL, PhotoImage, StringVar, Text, Tk
from tkinter import ttk
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from cheap_camera_object_detector.monitoring import (
    MonitorConfig,
    MonitorError,
    monitor_camera,
)
from cheap_camera_object_detector.notifications import (
    AlertEvent,
    AlertNotifier,
    CompositeNotifier,
    CooldownNotifier,
    WindowsBeepNotifier,
)
from cheap_camera_object_detector.protocols import VideoFrame
from cheap_camera_object_detector.rtsp_capture import CaptureError, capture_rtsp_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UiSettings:
    source: str
    target: str
    detection_enabled: bool
    model_name: str
    confidence: float
    transport: str
    image_size: int
    frame_interval_seconds: float
    alert_cooldown_seconds: float
    beep_frequency_hz: int
    beep_duration_ms: int
    open_timeout_ms: int
    read_timeout_ms: int
    warmup_frames: int
    frame_read_attempts: int


@dataclass(frozen=True)
class CaptureSettings:
    source: str
    attempts: int
    warmup_frames: int
    open_timeout_ms: int
    read_timeout_ms: int
    prefer_tcp: bool


@dataclass(frozen=True)
class UiLogEvent:
    message: str


@dataclass(frozen=True)
class UiFrameEvent:
    frame: VideoFrame


@dataclass(frozen=True)
class UiPreviewPlaceholderEvent:
    message: str


UiEvent: TypeAlias = UiLogEvent | UiFrameEvent | UiPreviewPlaceholderEvent


class ImageEncoderProtocol(Protocol):
    def imencode(self, ext: str, img: VideoFrame) -> tuple[bool, NDArray[np.uint8]]:
        ...


def default_ui_settings() -> UiSettings:
    return UiSettings(
        source=os.getenv("RTSP_URL", "rtsp://user:password@camera.local:554/stream1"),
        target=os.getenv("DETECTION_TARGET", "dog"),
        detection_enabled=_env_bool("DETECTION_ENABLED", True),
        model_name=os.getenv("DETECTION_MODEL", "yolo11n.pt"),
        confidence=_env_float("DETECTION_CONFIDENCE", 0.35),
        transport=os.getenv("RTSP_TRANSPORT", "udp").lower(),
        image_size=_env_int("DETECTION_IMAGE_SIZE", 640),
        frame_interval_seconds=_env_float("DETECTION_FRAME_INTERVAL_SECONDS", 0.5),
        alert_cooldown_seconds=_env_float("ALERT_COOLDOWN_SECONDS", 5.0),
        beep_frequency_hz=_env_int("BEEP_FREQUENCY_HZ", 1200),
        beep_duration_ms=_env_int("BEEP_DURATION_MS", 2000),
        open_timeout_ms=_env_int("CAPTURE_OPEN_TIMEOUT_MS", 10_000),
        read_timeout_ms=_env_int("CAPTURE_READ_TIMEOUT_MS", 10_000),
        warmup_frames=_env_int("CAPTURE_WARMUP_FRAMES", 3),
        frame_read_attempts=_env_int(
            "MONITOR_FRAME_READ_ATTEMPTS", _env_int("CAPTURE_ATTEMPTS", 8)
        ),
    )


def build_monitor_config(settings: UiSettings) -> MonitorConfig:
    _validate_transport(settings.transport)
    return MonitorConfig(
        source=settings.source,
        target=settings.target,
        detection_enabled=settings.detection_enabled,
        model_name=settings.model_name,
        confidence=settings.confidence,
        image_size=settings.image_size,
        frame_interval_seconds=settings.frame_interval_seconds,
        open_timeout_ms=settings.open_timeout_ms,
        read_timeout_ms=settings.read_timeout_ms,
        warmup_frames=settings.warmup_frames,
        frame_read_attempts=settings.frame_read_attempts,
        prefer_tcp=settings.transport == "tcp",
    )


def build_capture_settings(settings: UiSettings) -> CaptureSettings:
    _validate_transport(settings.transport)
    return CaptureSettings(
        source=settings.source,
        attempts=settings.frame_read_attempts,
        warmup_frames=settings.warmup_frames,
        open_timeout_ms=settings.open_timeout_ms,
        read_timeout_ms=settings.read_timeout_ms,
        prefer_tcp=settings.transport == "tcp",
    )


def run_desktop_ui() -> None:
    root = Tk()
    root.title("Image Behaviour Alerts")
    DesktopUi(root, default_ui_settings())
    root.mainloop()


class DesktopUi:
    def __init__(self, root: Tk, defaults: UiSettings) -> None:
        self._root = root
        self._events: queue.Queue[UiEvent] = queue.Queue()
        self._latest_frame = LatestFrameSlot()
        self._latest_labels = LatestLabelsSlot()
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None
        self._preview_image: PhotoImage | None = None

        self._source = StringVar(value=defaults.source)
        self._target = StringVar(value=defaults.target)
        self._detection_enabled = BooleanVar(value=defaults.detection_enabled)
        self._model_name = StringVar(value=defaults.model_name)
        self._confidence = StringVar(value=str(defaults.confidence))
        self._transport = StringVar(value=defaults.transport)
        self._available_labels_text = StringVar(
            value="Waiting for detector results."
        )

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame = ttk.Frame(root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(7, weight=1)
        frame.rowconfigure(9, weight=1)

        self._add_field(frame, "Camera source", self._source, 0)
        self._add_field(frame, "Target object", self._target, 1)
        self._add_field(frame, "Model", self._model_name, 2)
        self._add_field(frame, "Confidence", self._confidence, 3)
        self._add_transport(frame, 4)
        self._add_detection_toggle(frame, 5)
        self._add_available_labels(frame, 6)

        self._preview = ttk.Label(frame, text="No image captured yet.", anchor="center")
        self._preview.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(8, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 8))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)

        self._start_button = ttk.Button(
            actions, text="Start monitor", command=self._start_monitor
        )
        self._stop_button = ttk.Button(
            actions, text="Stop monitor", command=self._stop_monitor, state=DISABLED
        )
        self._capture_button = ttk.Button(
            actions, text="Capture image", command=self._capture_image
        )
        self._start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._stop_button.grid(row=0, column=1, sticky="ew", padx=6)
        self._capture_button.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        self._log = tk_text(frame)
        self._log.grid(row=9, column=0, columnspan=2, sticky="nsew")

        self._append_log("Ready.")
        self._root.after(100, self._drain_events)

    def _add_field(
        self,
        parent: ttk.Frame,
        label: str,
        variable: StringVar,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=2
        )

    def _add_transport(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, text="Transport").grid(row=row, column=0, sticky="w", pady=2)
        container = ttk.Frame(parent)
        container.grid(row=row, column=1, sticky="w", pady=2)
        ttk.Radiobutton(
            container, text="UDP", variable=self._transport, value="udp"
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            container, text="TCP", variable=self._transport, value="tcp"
        ).grid(row=0, column=1, sticky="w", padx=(12, 0))

    def _add_detection_toggle(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, text="Detection").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Checkbutton(
            parent,
            text="Enable object detection and alerts",
            variable=self._detection_enabled,
        ).grid(row=row, column=1, sticky="w", pady=2)

    def _add_available_labels(self, parent: ttk.Frame, row: int) -> None:
        ttk.Label(parent, text="Available labels").grid(
            row=row, column=0, sticky="w", pady=2
        )
        ttk.Label(
            parent,
            textvariable=self._available_labels_text,
            wraplength=720,
        ).grid(row=row, column=1, sticky="ew", pady=2)

    def _start_monitor(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        try:
            settings = self._read_settings()
            config = build_monitor_config(settings)
        except ValueError as exc:
            self._append_log(f"Invalid settings: {exc}")
            return

        self._stop_event.clear()
        self._available_labels_text.set(
            "Detection disabled."
            if not config.detection_enabled
            else "Waiting for detector results."
        )
        notifier = CooldownNotifier(
            CompositeNotifier(
                [
                    UiNotifier(self._events),
                    WindowsBeepNotifier(
                        frequency_hz=settings.beep_frequency_hz,
                        duration_ms=settings.beep_duration_ms,
                    ),
                ]
            ),
            cooldown_seconds=settings.alert_cooldown_seconds,
        )
        self._monitor_thread = threading.Thread(
            target=self._run_monitor,
            args=(config, notifier),
            daemon=True,
        )
        self._monitor_thread.start()
        self._set_monitor_running(True)
        self._append_log(f"Monitoring {config.target} on {_mask_source(config.source)}.")

    def _run_monitor(
        self,
        config: MonitorConfig,
        notifier: AlertNotifier,
    ) -> None:
        try:
            monitor_camera(
                config,
                notifier,
                stop_requested=self._stop_event.is_set,
                frame_observer=self._publish_latest_monitor_frame,
                detection_observer=self._publish_latest_available_labels,
                error_observer=self._queue_monitor_error,
            )
        except (MonitorError, ValueError) as exc:
            logger.exception("ui_monitor_failed error=%s", exc)
            self._events.put(UiPreviewPlaceholderEvent("Unable to read camera frame."))
            self._events.put(UiLogEvent(f"Monitor error: {exc}"))
        finally:
            self._events.put(UiLogEvent("Monitor stopped."))

    def _stop_monitor(self) -> None:
        self._stop_event.set()
        self._append_log("Stopping monitor after the current frame.")

    def _capture_image(self) -> None:
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        try:
            settings = build_capture_settings(self._read_settings())
        except ValueError as exc:
            self._append_log(f"Invalid settings: {exc}")
            return
        self._capture_button.configure(state=DISABLED)
        self._capture_thread = threading.Thread(
            target=self._run_capture,
            args=(settings,),
            daemon=True,
        )
        self._capture_thread.start()
        self._append_log(f"Capturing from {_mask_source(settings.source)}.")

    def _run_capture(self, settings: CaptureSettings) -> None:
        try:
            result = capture_rtsp_frame(
                settings.source,
                attempts=settings.attempts,
                warmup_frames=settings.warmup_frames,
                open_timeout_ms=settings.open_timeout_ms,
                read_timeout_ms=settings.read_timeout_ms,
                prefer_tcp=settings.prefer_tcp,
            )
        except (CaptureError, ValueError) as exc:
            logger.exception("ui_capture_failed error=%s", exc)
            self._events.put(UiPreviewPlaceholderEvent("Unable to read camera frame."))
            self._events.put(UiLogEvent(f"Capture error: {exc}"))
        else:
            self._events.put(UiFrameEvent(_copy_frame(result.frame)))
            self._events.put(UiLogEvent("Image captured."))
        finally:
            self._events.put(UiLogEvent("Capture finished."))

    def _read_settings(self) -> UiSettings:
        defaults = default_ui_settings()
        return UiSettings(
            source=self._source.get().strip(),
            target=self._target.get().strip(),
            detection_enabled=self._detection_enabled.get(),
            model_name=self._model_name.get().strip(),
            confidence=_parse_float("confidence", self._confidence.get()),
            transport=self._transport.get().strip().lower(),
            image_size=defaults.image_size,
            frame_interval_seconds=defaults.frame_interval_seconds,
            alert_cooldown_seconds=defaults.alert_cooldown_seconds,
            beep_frequency_hz=defaults.beep_frequency_hz,
            beep_duration_ms=defaults.beep_duration_ms,
            open_timeout_ms=defaults.open_timeout_ms,
            read_timeout_ms=defaults.read_timeout_ms,
            warmup_frames=defaults.warmup_frames,
            frame_read_attempts=defaults.frame_read_attempts,
        )

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if isinstance(event, UiFrameEvent):
                self._latest_frame.set(event.frame)
            elif isinstance(event, UiPreviewPlaceholderEvent):
                self._show_placeholder(event.message)
            else:
                message = event.message
                self._append_log(message)
                if message == "Monitor stopped.":
                    self._set_monitor_running(False)
                if message == "Capture finished.":
                    self._capture_button.configure(state=NORMAL)

        latest_frame = self._latest_frame.pop()
        if latest_frame is not None:
            self._show_frame(latest_frame)
        latest_labels = self._latest_labels.pop()
        if latest_labels is not None:
            self._available_labels_text.set(_format_available_labels(latest_labels))
        self._root.after(33, self._drain_events)

    def _publish_latest_monitor_frame(self, frame: VideoFrame) -> None:
        self._latest_frame.set(_copy_frame(frame))

    def _queue_monitor_error(self, exc: Exception) -> None:
        self._events.put(UiPreviewPlaceholderEvent("Unable to read camera frame."))
        self._events.put(UiLogEvent(f"Monitor error: {exc}"))

    def _publish_latest_available_labels(self, labels: tuple[str, ...]) -> None:
        self._latest_labels.set(labels)

    def _show_frame(self, frame: VideoFrame) -> None:
        image = _frame_to_photo_image(frame)
        self._preview_image = image
        self._preview.configure(image=image, text="")

    def _show_placeholder(self, message: str) -> None:
        self._preview_image = None
        self._preview.configure(image="", text=message)

    def _set_monitor_running(self, running: bool) -> None:
        self._start_button.configure(state=DISABLED if running else NORMAL)
        self._stop_button.configure(state=NORMAL if running else DISABLED)

    def _append_log(self, message: str) -> None:
        self._log.configure(state=NORMAL)
        self._log.insert(END, f"{message}\n")
        self._log.see(END)
        self._log.configure(state=DISABLED)


class UiNotifier:
    def __init__(self, events: queue.Queue[UiEvent]) -> None:
        self._events = events

    def notify(self, event: AlertEvent) -> None:
        best = max(event.detections, key=lambda detection: detection.confidence)
        self._events.put(
            UiLogEvent(
                f"Alert: {event.target} detected on {_mask_source(event.source)} "
                f"({best.confidence:.0%})"
            )
        )


class LatestFrameSlot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: VideoFrame | None = None

    def set(self, frame: VideoFrame) -> None:
        with self._lock:
            self._frame = frame

    def pop(self) -> VideoFrame | None:
        with self._lock:
            frame = self._frame
            self._frame = None
        return frame


class LatestLabelsSlot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._labels: tuple[str, ...] | None = None

    def set(self, labels: tuple[str, ...]) -> None:
        with self._lock:
            self._labels = labels

    def pop(self) -> tuple[str, ...] | None:
        with self._lock:
            labels = self._labels
            self._labels = None
        return labels


def tk_text(parent: ttk.Frame) -> Text:
    return Text(parent, height=12, wrap="word", state=DISABLED)


def _copy_frame(frame: VideoFrame) -> VideoFrame:
    return np.copy(frame)


def _frame_to_photo_image(frame: VideoFrame) -> PhotoImage:
    image = _fit_frame_for_preview(frame, max_width=720, max_height=405)
    ok, encoded = _cv2().imencode(".png", np.ascontiguousarray(image))
    if not ok:
        raise ValueError("nao foi possivel preparar a imagem para exibicao")
    data = b64encode(encoded.tobytes()).decode("ascii")
    return PhotoImage(data=data, format="PNG")


def _cv2() -> ImageEncoderProtocol:
    import cv2

    return cast(ImageEncoderProtocol, cv2)


def _fit_frame_for_preview(
    frame: VideoFrame,
    *,
    max_width: int,
    max_height: int,
) -> VideoFrame:
    height = frame.shape[0]
    width = frame.shape[1]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    row_step = max(1, (height + max_height - 1) // max_height)
    column_step = max(1, (width + max_width - 1) // max_width)
    return frame[::row_step, ::column_step]


def _format_available_labels(labels: tuple[str, ...]) -> str:
    if not labels:
        return "No objects detected."
    return ", ".join(labels)


def _parse_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} precisa ser um numero") from exc


def _env_int(name: str, fallback: int) -> int:
    value = os.getenv(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} precisa ser um numero inteiro") from exc


def _env_float(name: str, fallback: float) -> float:
    value = os.getenv(name)
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} precisa ser um numero") from exc


def _env_bool(name: str, fallback: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return fallback
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} precisa ser true ou false")


def _validate_transport(transport: str) -> None:
    if transport not in {"tcp", "udp"}:
        raise ValueError("RTSP_TRANSPORT precisa ser 'udp' ou 'tcp'")


def _mask_source(source: str) -> str:
    if "@" not in source or ":" not in source.split("@", 1)[0]:
        return source
    scheme, rest = source.split("://", 1)
    userinfo, host = rest.split("@", 1)
    username = userinfo.split(":", 1)[0]
    return f"{scheme}://{username}:***@{host}"
