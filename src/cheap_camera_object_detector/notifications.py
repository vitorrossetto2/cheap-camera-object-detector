from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from cheap_camera_object_detector.detection import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertEvent:
    target: str
    detections: tuple[Detection, ...]
    source: str
    timestamp: float


class AlertNotifier(Protocol):
    def notify(self, event: AlertEvent) -> None:
        """Publish an alert event."""


class CompositeNotifier:
    def __init__(self, notifiers: list[AlertNotifier]) -> None:
        self._notifiers = tuple(notifiers)
        logger.info(
            "composite_notifier_configured notifier_count=%s notifier_types=%r",
            len(self._notifiers),
            [type(notifier).__name__ for notifier in self._notifiers],
        )

    def notify(self, event: AlertEvent) -> None:
        logger.info(
            "composite_notifier_dispatch target=%s notifier_count=%s detections=%s",
            event.target,
            len(self._notifiers),
            len(event.detections),
        )
        for notifier in self._notifiers:
            logger.info(
                "composite_notifier_dispatch_one notifier=%s target=%s",
                type(notifier).__name__,
                event.target,
            )
            notifier.notify(event)


class ConsoleNotifier:
    def notify(self, event: AlertEvent) -> None:
        best = max(event.detections, key=lambda detection: detection.confidence)
        logger.info(
            "console_alert target=%s source=%s detections=%s best_label=%s "
            "best_confidence=%s",
            event.target,
            _mask_source_password(event.source),
            len(event.detections),
            best.label,
            round(best.confidence, 4),
        )
        print(
            f"Alerta: {event.target} detectado em {_mask_source_password(event.source)} "
            f"({best.confidence:.0%})"
        )


class WindowsBeepNotifier:
    def __init__(self, *, frequency_hz: int = 1200, duration_ms: int = 2000) -> None:
        if frequency_hz < 37 or frequency_hz > 32767:
            logger.error(
                "beep_notifier_validation_failed parameter=frequency_hz value=%s "
                "reason=out_of_range",
                frequency_hz,
            )
            raise ValueError("beep_frequency_hz precisa estar entre 37 e 32767")
        if duration_ms <= 0:
            logger.error(
                "beep_notifier_validation_failed parameter=duration_ms value=%s "
                "reason=not_positive",
                duration_ms,
            )
            raise ValueError("beep_duration_ms precisa ser maior que zero")
        self.frequency_hz = frequency_hz
        self.duration_ms = duration_ms
        logger.info(
            "beep_notifier_configured frequency_hz=%s duration_ms=%s",
            self.frequency_hz,
            self.duration_ms,
        )

    def notify(self, event: AlertEvent) -> None:
        if sys.platform != "win32":
            logger.info(
                "beep_notifier_skipped reason=unsupported_platform platform=%s",
                sys.platform,
            )
            print("Beep Windows ignorado: winsound so esta disponivel no Windows")
            return

        import winsound

        logger.info(
            "beep_notifier_triggered target=%s frequency_hz=%s duration_ms=%s",
            event.target,
            self.frequency_hz,
            self.duration_ms,
        )
        winsound.Beep(self.frequency_hz, self.duration_ms)


class CooldownNotifier:
    def __init__(self, notifier: AlertNotifier, *, cooldown_seconds: float) -> None:
        if cooldown_seconds < 0:
            logger.error(
                "cooldown_validation_failed cooldown_seconds=%s reason=negative",
                cooldown_seconds,
            )
            raise ValueError("cooldown_seconds nao pode ser negativo")
        self._notifier = notifier
        self._cooldown_seconds = cooldown_seconds
        self._last_notification = 0.0
        logger.info("cooldown_notifier_configured cooldown_seconds=%s", cooldown_seconds)

    def notify(self, event: AlertEvent) -> None:
        elapsed_seconds = event.timestamp - self._last_notification
        passed = elapsed_seconds >= self._cooldown_seconds
        logger.info(
            "cooldown_rule_evaluation target=%s passed=%s elapsed_seconds=%.3f "
            "cooldown_seconds=%s",
            event.target,
            passed,
            elapsed_seconds,
            self._cooldown_seconds,
        )
        if not passed:
            return
        self._last_notification = event.timestamp
        self._notifier.notify(event)


def now_timestamp() -> float:
    return time.time()


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
