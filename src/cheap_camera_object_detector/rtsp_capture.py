from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from cheap_camera_object_detector.protocols import (
    Cv2Protocol,
    VideoCaptureProtocol,
    VideoFrame,
)

logger = logging.getLogger(__name__)
LOW_LATENCY_FFMPEG_OPTIONS = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "max_delay": "0",
}


class CaptureError(RuntimeError):
    """Raised when an RTSP frame cannot be captured or written."""


@dataclass(frozen=True)
class CaptureResult:
    output_path: Path
    source_url: str
    attempts: int


@dataclass(frozen=True)
class FrameCaptureResult:
    frame: VideoFrame
    source_url: str
    attempts: int


@dataclass(frozen=True)
class FrameReadResult:
    frame: VideoFrame
    reads_used: int
    attempts_used: int


def capture_rtsp_frame(
    source_url: str,
    *,
    attempts: int = 8,
    warmup_frames: int = 3,
    open_timeout_ms: int = 10_000,
    read_timeout_ms: int = 10_000,
    prefer_tcp: bool = True,
) -> FrameCaptureResult:
    """Capture one JPEG-compatible frame from an RTSP stream."""
    _validate_rtsp_url(source_url)
    _validate_positive("attempts", attempts)
    _validate_non_negative("warmup_frames", warmup_frames)
    _validate_positive("open_timeout_ms", open_timeout_ms)
    _validate_positive("read_timeout_ms", read_timeout_ms)

    logger.info(
        "rtsp_frame_capture_start source=%s attempts=%s warmup_frames=%s "
        "open_timeout_ms=%s read_timeout_ms=%s prefer_tcp=%s",
        _mask_rtsp_password(source_url),
        attempts,
        warmup_frames,
        open_timeout_ms,
        read_timeout_ms,
        prefer_tcp,
    )

    capture, _transport = open_rtsp_capture(
        source_url,
        open_timeout_ms,
        read_timeout_ms,
        prefer_tcp=prefer_tcp,
    )

    try:
        frame_result = read_first_frame(
            capture,
            log_context="rtsp_capture",
        )
    finally:
        logger.info("rtsp_capture_release source=%s", _mask_rtsp_password(source_url))
        capture.release()

    result = FrameCaptureResult(
        frame=frame_result.frame,
        source_url=source_url,
        attempts=frame_result.attempts_used,
    )
    logger.info(
        "rtsp_frame_capture_succeeded attempts_used=%s transport=%s",
        result.attempts,
        _transport,
    )
    return result


def capture_rtsp_image(
    source_url: str,
    output_path: str | Path,
    *,
    attempts: int = 8,
    warmup_frames: int = 3,
    open_timeout_ms: int = 10_000,
    read_timeout_ms: int = 10_000,
    prefer_tcp: bool = True,
) -> CaptureResult:
    """Capture one JPEG-compatible frame from an RTSP stream and save it."""
    destination = Path(output_path)
    logger.info(
        "rtsp_capture_start source=%s output=%s attempts=%s warmup_frames=%s "
        "open_timeout_ms=%s read_timeout_ms=%s prefer_tcp=%s",
        _mask_rtsp_password(source_url),
        destination,
        attempts,
        warmup_frames,
        open_timeout_ms,
        read_timeout_ms,
        prefer_tcp,
    )
    frame_result = capture_rtsp_frame(
        source_url,
        attempts=attempts,
        warmup_frames=warmup_frames,
        open_timeout_ms=open_timeout_ms,
        read_timeout_ms=read_timeout_ms,
        prefer_tcp=prefer_tcp,
    )
    write_frame(frame_result.frame, destination)
    result = CaptureResult(
        output_path=destination.resolve(),
        source_url=source_url,
        attempts=frame_result.attempts,
    )
    logger.info(
        "rtsp_capture_succeeded output=%s attempts_used=%s",
        result.output_path,
        result.attempts,
    )
    return result


def open_rtsp_capture(
    source_url: str,
    open_timeout_ms: int,
    read_timeout_ms: int,
    *,
    prefer_tcp: bool,
) -> tuple[VideoCaptureProtocol, str]:
    attempted_transports: list[str] = []
    for transport in _transport_attempt_order(prefer_tcp):
        attempted_transports.append(transport)
        logger.info(
            "rtsp_capture_open_start source=%s transport=%s open_timeout_ms=%s "
            "read_timeout_ms=%s",
            _mask_rtsp_password(source_url),
            transport,
            open_timeout_ms,
            read_timeout_ms,
        )
        _configure_ffmpeg_transport(transport)
        capture = _open_capture(source_url, open_timeout_ms, read_timeout_ms)
        if capture.isOpened():
            logger.info(
                "rtsp_capture_open_succeeded source=%s transport=%s",
                _mask_rtsp_password(source_url),
                transport,
            )
            return capture, transport
        capture.release()
        logger.error(
            "rtsp_capture_open_failed source=%s transport=%s",
            _mask_rtsp_password(source_url),
            transport,
        )

    transports = ", ".join(transport.upper() for transport in attempted_transports)
    raise CaptureError(f"nao foi possivel abrir o stream RTSP usando {transports}")


def read_valid_frame(
    capture: VideoCaptureProtocol,
    *,
    attempts: int,
    warmup_frames: int,
    log_context: str,
    minimum_frame_stddev: float = 0.0,
) -> FrameReadResult:
    return read_first_frame(capture, log_context=log_context)


def read_first_frame(
    capture: VideoCaptureProtocol,
    *,
    log_context: str,
) -> FrameReadResult:
    ok, candidate = capture.read()
    _log_frame_read(
        log_context,
        1,
        ok=ok,
        candidate=candidate,
    )
    if not ok or candidate is None or candidate.size == 0:
        logger.error(
            "%s_frame_read_failed reason=invalid_first_frame reads_used=1",
            log_context,
        )
        raise CaptureError("primeiro frame RTSP nao foi recebido ou esta vazio")

    return FrameReadResult(
        frame=candidate,
        reads_used=1,
        attempts_used=1,
    )


def read_latest_valid_frame(
    capture: VideoCaptureProtocol,
    *,
    attempts: int,
    warmup_frames: int,
    log_context: str,
    minimum_frame_stddev: float = 0.0,
    drain_reads: int = 0,
) -> FrameReadResult:
    return read_first_frame(capture, log_context=log_context)


def write_frame(frame: VideoFrame, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _cv2().imwrite(str(destination), frame):
        logger.error("frame_write_failed output=%s", destination)
        raise CaptureError(f"nao foi possivel salvar a imagem em {destination}")
    return destination


def _configure_ffmpeg_transport(transport: str) -> None:
    options = _merge_ffmpeg_capture_options(
        os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", ""),
        transport,
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    logger.info(
        "opencv_ffmpeg_capture_options_set transport=%s options=%s",
        transport,
        options,
    )


def _transport_attempt_order(prefer_tcp: bool) -> tuple[str, str]:
    if prefer_tcp:
        return ("tcp", "udp")
    return ("udp", "tcp")


def _open_capture(
    source_url: str,
    open_timeout_ms: int,
    read_timeout_ms: int,
) -> VideoCaptureProtocol:
    cv2 = _cv2()
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        open_timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        read_timeout_ms,
    ]
    return cv2.VideoCapture(source_url, cv2.CAP_FFMPEG, params)


def _cv2() -> Cv2Protocol:
    import cv2

    return cast(Cv2Protocol, cv2)


def _validate_rtsp_url(source_url: str) -> None:
    parsed = urlsplit(source_url)
    if parsed.scheme.lower() != "rtsp":
        logger.error(
            "rtsp_capture_validation_failed parameter=source_url reason=invalid_scheme "
            "scheme=%s",
            parsed.scheme,
        )
        raise ValueError("a URL precisa usar o protocolo rtsp://")
    if not parsed.hostname:
        logger.error(
            "rtsp_capture_validation_failed parameter=source_url reason=missing_host"
        )
        raise ValueError("a URL RTSP precisa conter um host")
    logger.info("rtsp_capture_validation_passed source=%s", _mask_rtsp_password(source_url))


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        logger.error(
            "rtsp_capture_validation_failed parameter=%s value=%s reason=not_positive",
            name,
            value,
        )
        raise ValueError(f"{name} precisa ser maior que zero")


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        logger.error(
            "rtsp_capture_validation_failed parameter=%s value=%s reason=negative",
            name,
            value,
        )
        raise ValueError(f"{name} nao pode ser negativo")


def _log_frame_read(
    log_context: str,
    reads_used: int,
    *,
    ok: bool,
    candidate: VideoFrame | None,
) -> None:
    logger.info(
        "%s_frame_read index=%s ok=%s frame_is_none=%s "
        "frame_size=%s frame_stddev=%s",
        log_context,
        reads_used,
        ok,
        candidate is None,
        getattr(candidate, "size", None),
        _frame_stddev(candidate),
    )


def _frame_stddev(frame: VideoFrame | None) -> float | None:
    if frame is None:
        return None
    if frame.size == 0:
        return 0.0

    try:
        return float(frame.std())
    except (TypeError, ValueError):
        return None


def _merge_ffmpeg_capture_options(existing: str, transport: str) -> str:
    merged = _parse_ffmpeg_capture_options(existing)
    merged["rtsp_transport"] = transport
    for key, value in LOW_LATENCY_FFMPEG_OPTIONS.items():
        if key in {"fflags", "flags"}:
            merged[key] = _merge_ffmpeg_flag(merged.get(key, ""), value)
        elif key not in merged:
            merged[key] = value
    return "|".join(f"{key};{value}" for key, value in merged.items())


def _merge_ffmpeg_flag(existing: str, flag: str) -> str:
    if not existing:
        return flag
    values = existing.split("+")
    if flag in values:
        return existing
    return f"{existing}+{flag}"


def _parse_ffmpeg_capture_options(value: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for item in value.split("|"):
        if ";" not in item:
            continue
        key, option_value = item.split(";", 1)
        key = key.strip()
        option_value = option_value.strip()
        if key:
            options[key] = option_value
    return options


def _mask_rtsp_password(source_url: str) -> str:
    parsed = urlsplit(source_url)
    if parsed.password is None or parsed.hostname is None:
        return source_url

    username = parsed.username or ""
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    netloc = f"{username}:***@{host}" if username else host
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )
