from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import urlsplit, urlunsplit

from cheap_camera_object_detector.detection import (
    Detection,
    YoloObjectDetector,
    matching_detections,
)
from cheap_camera_object_detector.notifications import AlertEvent, AlertNotifier, now_timestamp
from cheap_camera_object_detector.rtsp_capture import (
    CaptureError,
    FrameReadResult,
    open_rtsp_capture,
    read_first_frame,
)
from cheap_camera_object_detector.protocols import VideoCaptureProtocol, VideoFrame


logger = logging.getLogger(__name__)
class MonitorError(RuntimeError):
    """Raised when the camera monitor cannot continue."""


@dataclass(frozen=True)
class MonitorConfig:
    source: str
    target: str
    detection_enabled: bool = True
    model_name: str = "yolo11n.pt"
    confidence: float = 0.35
    image_size: int = 640
    frame_interval_seconds: float = 0.5
    open_timeout_ms: int = 10_000
    read_timeout_ms: int = 10_000
    warmup_frames: int = 3
    frame_read_attempts: int = 8
    prefer_tcp: bool = True


class DetectionLogPayload(TypedDict):
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


def monitor_camera(
    config: MonitorConfig,
    notifier: AlertNotifier,
    *,
    max_frames: int = 0,
    stop_requested: Callable[[], bool] | None = None,
    frame_observer: Callable[[VideoFrame], None] | None = None,
    detection_observer: Callable[[tuple[str, ...]], None] | None = None,
    error_observer: Callable[[Exception], None] | None = None,
) -> None:
    _validate_config(config)
    if max_frames < 0:
        logger.error(
            "monitor_validation_failed parameter=max_frames value=%s reason=negative",
            max_frames,
        )
        raise ValueError("max_frames nao pode ser negativo")

    logger.info(
        "monitor_start source=%s target=%s detection_enabled=%s model=%s confidence=%s image_size=%s "
        "frame_interval_seconds=%s open_timeout_ms=%s read_timeout_ms=%s "
        "warmup_frames=%s frame_read_attempts=%s prefer_tcp=%s max_frames=%s",
        _mask_source_password(config.source),
        config.target,
        config.detection_enabled,
        config.model_name,
        config.confidence,
        config.image_size,
        config.frame_interval_seconds,
        config.open_timeout_ms,
        config.read_timeout_ms,
        config.warmup_frames,
        config.frame_read_attempts,
        config.prefer_tcp,
        max_frames,
    )
    detector = (
        YoloObjectDetector(
            config.model_name,
            confidence=config.confidence,
            image_size=config.image_size,
        )
        if config.detection_enabled
        else None
    )

    frame_count = 0
    capture_session = _MonitorCaptureSession(config)
    detection_runner = (
        _AsyncDetectionRunner(
            config,
            detector,
            notifier,
            detection_observer,
            error_observer,
        )
        if detector is not None
        else None
    )

    try:
        while max_frames == 0 or frame_count < max_frames:
            if stop_requested is not None and stop_requested():
                logger.info("monitor_stop_requested frame=%s", frame_count)
                break
            frame_count += 1
            try:
                frame = capture_session.read_frame(frame_count)
                if frame_observer is not None:
                    frame_observer(frame)
                if detection_runner is not None:
                    detection_runner.start_if_idle(frame, frame_count)
            except Exception as exc:
                logger.exception(
                    "monitor_iteration_failed frame=%s error=%s",
                    frame_count,
                    exc,
                )
                if error_observer is not None:
                    try:
                        error_observer(exc)
                    except Exception:
                        logger.exception(
                            "monitor_error_observer_failed frame=%s",
                            frame_count,
                        )

            if config.frame_interval_seconds:
                logger.debug(
                    "monitor_sleep seconds=%s frame=%s",
                    config.frame_interval_seconds,
                    frame_count,
                )
                time.sleep(config.frame_interval_seconds)
    finally:
        capture_session.close()
        if detection_runner is not None:
            detection_runner.close()
        logger.info(
            "monitor_stop source=%s frames_processed=%s",
            _mask_source_password(config.source),
            frame_count if "frame_count" in locals() else 0,
        )


class _MonitorCaptureSession:
    def __init__(self, config: MonitorConfig) -> None:
        self._config = config
        self._capture: VideoCaptureProtocol | None = None
        self._transport = ""

    def read_frame(self, frame_count: int) -> VideoFrame:
        try:
            result = self._read_frame_result(frame_count)
        except CaptureError as exc:
            logger.error(
                "frame_read_failed source=%s frame=%s error=%s",
                _mask_source_password(self._config.source),
                frame_count,
                exc,
            )
            self.close()
            raise MonitorError(str(exc)) from exc

        logger.info(
            "monitor_frame_captured frame=%s attempts_used=%s reads_used=%s "
            "transport=%s",
            frame_count,
            result.attempts_used,
            result.reads_used,
            self._transport,
        )
        return result.frame

    def close(self) -> None:
        if self._capture is None:
            return
        logger.info(
            "monitor_capture_release source=%s",
            _mask_source_password(self._config.source),
        )
        self._capture.release()
        self._capture = None
        self._transport = ""

    def _read_frame_result(self, frame_count: int) -> FrameReadResult:
        if self._capture is None:
            self._capture, self._transport = open_rtsp_capture(
                self._config.source,
                self._config.open_timeout_ms,
                self._config.read_timeout_ms,
                prefer_tcp=self._config.prefer_tcp,
            )

        result = read_first_frame(
            self._capture,
            log_context="monitor_capture",
        )
        return result


class _AsyncDetectionRunner:
    def __init__(
        self,
        config: MonitorConfig,
        detector: YoloObjectDetector,
        notifier: AlertNotifier,
        detection_observer: Callable[[tuple[str, ...]], None] | None,
        error_observer: Callable[[Exception], None] | None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._notifier = notifier
        self._detection_observer = detection_observer
        self._error_observer = error_observer
        self._thread: threading.Thread | None = None

    def start_if_idle(self, frame: VideoFrame, frame_count: int) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.info(
                "monitor_detection_skipped frame=%s reason=detection_in_progress",
                frame_count,
            )
            return

        self._thread = threading.Thread(
            target=self._run_detection,
            args=(frame, frame_count),
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _run_detection(self, frame: VideoFrame, frame_count: int) -> None:
        try:
            detections = self._detector.detect(frame)
            _log_detections(detections, frame_count, self._config.source)
            available_labels = _available_labels(detections)
            if self._detection_observer is not None:
                try:
                    self._detection_observer(available_labels)
                except Exception:
                    logger.exception(
                        "monitor_detection_observer_failed frame=%s",
                        frame_count,
                    )

            matches = matching_detections(detections, self._config.target)
            _log_match_evaluation(
                matches,
                detections,
                frame_count,
                self._config.target,
            )
            if matches:
                self._notifier.notify(
                    AlertEvent(
                        target=self._config.target,
                        detections=tuple(matches),
                        source=self._config.source,
                        timestamp=now_timestamp(),
                    )
                )
        except Exception as exc:
            logger.exception(
                "monitor_detection_failed frame=%s error=%s",
                frame_count,
                exc,
            )
            if self._error_observer is not None:
                try:
                    self._error_observer(exc)
                except Exception:
                    logger.exception(
                        "monitor_error_observer_failed frame=%s",
                        frame_count,
                    )


def _log_detections(
    detections: list[Detection],
    frame_count: int,
    source: str,
) -> None:
    logger.info(
        "object_detections frame=%s source=%s count=%s detections=%r",
        frame_count,
        _mask_source_password(source),
        len(detections),
        [_detection_log_payload(detection) for detection in detections],
    )


def _detection_log_payload(detection: Detection) -> DetectionLogPayload:
    return {
        "label": detection.label,
        "confidence": round(detection.confidence, 4),
        "box_xyxy": _rounded_box_xyxy(detection.box_xyxy),
    }


def _rounded_box_xyxy(
    values: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        round(values[0], 2),
        round(values[1], 2),
        round(values[2], 2),
        round(values[3], 2),
    )


def _log_match_evaluation(
    matches: list[Detection],
    detections: list[Detection],
    frame_count: int,
    target: str,
) -> None:
    logger.info(
        "target_rule_evaluation frame=%s target=%s passed=%s matches=%s "
        "available_labels=%r",
        frame_count,
        target,
        bool(matches),
        len(matches),
        [detection.label for detection in detections],
    )


def _available_labels(detections: list[Detection]) -> tuple[str, ...]:
    labels: list[str] = []
    for detection in detections:
        if detection.label not in labels:
            labels.append(detection.label)
    return tuple(labels)


def _mask_source_password(source: str) -> str:
    parsed = urlsplit(source)
    if parsed.password is None or parsed.hostname is None:
        return source

    username = parsed.username or ""
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    netloc = f"{username}:***@{host}" if username else host
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _validate_config(config: MonitorConfig) -> None:
    if not config.source.strip():
        logger.error("monitor_validation_failed parameter=source reason=blank")
        raise ValueError("source precisa ser informado")
    if config.detection_enabled and not config.target.strip():
        logger.error("monitor_validation_failed parameter=target reason=blank")
        raise ValueError("target precisa ser informado")
    if config.confidence <= 0 or config.confidence > 1:
        logger.error(
            "monitor_validation_failed parameter=confidence value=%s reason=out_of_range",
            config.confidence,
        )
        raise ValueError("confidence precisa estar entre 0 e 1")
    if config.image_size <= 0:
        logger.error(
            "monitor_validation_failed parameter=image_size value=%s reason=not_positive",
            config.image_size,
        )
        raise ValueError("image_size precisa ser maior que zero")
    if config.frame_interval_seconds < 0:
        logger.error(
            "monitor_validation_failed parameter=frame_interval_seconds value=%s reason=negative",
            config.frame_interval_seconds,
        )
        raise ValueError("frame_interval_seconds nao pode ser negativo")
    if config.open_timeout_ms <= 0:
        logger.error(
            "monitor_validation_failed parameter=open_timeout_ms value=%s reason=not_positive",
            config.open_timeout_ms,
        )
        raise ValueError("open_timeout_ms precisa ser maior que zero")
    if config.read_timeout_ms <= 0:
        logger.error(
            "monitor_validation_failed parameter=read_timeout_ms value=%s reason=not_positive",
            config.read_timeout_ms,
        )
        raise ValueError("read_timeout_ms precisa ser maior que zero")
    if config.warmup_frames < 0:
        logger.error(
            "monitor_validation_failed parameter=warmup_frames value=%s reason=negative",
            config.warmup_frames,
        )
        raise ValueError("warmup_frames nao pode ser negativo")
    if config.frame_read_attempts <= 0:
        logger.error(
            "monitor_validation_failed parameter=frame_read_attempts value=%s "
            "reason=not_positive",
            config.frame_read_attempts,
        )
        raise ValueError("frame_read_attempts precisa ser maior que zero")
    logger.info(
        "monitor_validation_passed source=%s target=%s detection_enabled=%s",
        _mask_source_password(config.source),
        config.target,
        config.detection_enabled,
    )
