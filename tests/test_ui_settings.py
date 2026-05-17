from __future__ import annotations

import os
import queue
import unittest
from unittest.mock import patch

import numpy as np

from cheap_camera_object_detector.ui import (
    DesktopUi,
    LatestFrameSlot,
    LatestLabelsSlot,
    UiFrameEvent,
    UiEvent,
    UiSettings,
    build_capture_settings,
    build_monitor_config,
    default_ui_settings,
)


class UiSettingsTests(unittest.TestCase):
    def test_default_ui_settings_read_existing_environment_values(self) -> None:
        env = {
            "RTSP_URL": "rtsp://user:secret@camera.local:554/stream1",
            "RTSP_TRANSPORT": "tcp",
            "DETECTION_TARGET": "person",
            "DETECTION_ENABLED": "false",
            "DETECTION_MODEL": "yolo11n.pt",
            "DETECTION_CONFIDENCE": "0.45",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = default_ui_settings()

        self.assertEqual("rtsp://user:secret@camera.local:554/stream1", settings.source)
        self.assertEqual("tcp", settings.transport)
        self.assertEqual("person", settings.target)
        self.assertFalse(settings.detection_enabled)
        self.assertEqual("yolo11n.pt", settings.model_name)
        self.assertEqual(0.45, settings.confidence)

    def test_build_monitor_config_maps_ui_settings(self) -> None:
        settings = ui_settings(transport="tcp")

        config = build_monitor_config(settings)

        self.assertEqual(settings.source, config.source)
        self.assertEqual(settings.target, config.target)
        self.assertEqual(settings.detection_enabled, config.detection_enabled)
        self.assertEqual(settings.model_name, config.model_name)
        self.assertEqual(settings.confidence, config.confidence)
        self.assertTrue(config.prefer_tcp)

    def test_build_capture_settings_maps_ui_settings(self) -> None:
        settings = ui_settings(transport="udp")

        capture = build_capture_settings(settings)

        self.assertEqual(settings.source, capture.source)
        self.assertEqual(settings.frame_read_attempts, capture.attempts)
        self.assertFalse(capture.prefer_tcp)

    def test_invalid_transport_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "RTSP_TRANSPORT"):
            build_monitor_config(ui_settings(transport="http"))

    def test_latest_frame_slot_keeps_only_newest_frame(self) -> None:
        first_frame = np.array([[[0, 0, 0]]], dtype=np.uint8)
        second_frame = np.array([[[255, 255, 255]]], dtype=np.uint8)
        slot = LatestFrameSlot()

        slot.set(first_frame)
        slot.set(second_frame)

        self.assertIs(second_frame, slot.pop())
        self.assertIsNone(slot.pop())

    def test_monitor_frame_callback_uses_latest_frame_slot_not_event_queue(self) -> None:
        frame = np.array([[[255, 255, 255]]], dtype=np.uint8)
        events: queue.Queue[UiEvent] = queue.Queue()
        ui = DesktopUi.__new__(DesktopUi)
        ui._events = events  # pyright: ignore[reportPrivateUsage]
        ui._latest_frame = LatestFrameSlot()  # pyright: ignore[reportPrivateUsage]

        ui._publish_latest_monitor_frame(frame)  # pyright: ignore[reportPrivateUsage]

        self.assertTrue(events.empty())
        self.assertIsNone(_pop_frame_event(events))
        latest_frame = ui._latest_frame.pop()  # pyright: ignore[reportPrivateUsage]
        self.assertIsNotNone(latest_frame)
        self.assertIsNot(frame, latest_frame)
        np.testing.assert_array_equal(frame, latest_frame)

    def test_latest_labels_slot_keeps_only_newest_labels(self) -> None:
        slot = LatestLabelsSlot()

        slot.set(("person",))
        slot.set(("cell phone", "dog"))

        self.assertEqual(("cell phone", "dog"), slot.pop())
        self.assertIsNone(slot.pop())

    def test_available_labels_callback_uses_latest_labels_slot_not_event_queue(self) -> None:
        events: queue.Queue[UiEvent] = queue.Queue()
        ui = DesktopUi.__new__(DesktopUi)
        ui._events = events  # pyright: ignore[reportPrivateUsage]
        ui._latest_labels = LatestLabelsSlot()  # pyright: ignore[reportPrivateUsage]

        ui._publish_latest_available_labels(("person",))  # pyright: ignore[reportPrivateUsage]
        ui._publish_latest_available_labels(("cell phone", "dog"))  # pyright: ignore[reportPrivateUsage]

        self.assertTrue(events.empty())
        latest_labels = ui._latest_labels.pop()  # pyright: ignore[reportPrivateUsage]
        self.assertEqual(("cell phone", "dog"), latest_labels)


def ui_settings(*, transport: str) -> UiSettings:
    return UiSettings(
        source="rtsp://user:secret@camera.local:554/stream1",
        target="dog",
        detection_enabled=True,
        model_name="yolo11n.pt",
        confidence=0.35,
        transport=transport,
        image_size=640,
        frame_interval_seconds=0.2,
        alert_cooldown_seconds=5,
        beep_frequency_hz=1200,
        beep_duration_ms=2000,
        open_timeout_ms=10_000,
        read_timeout_ms=10_000,
        warmup_frames=3,
        frame_read_attempts=8,
    )


def _pop_frame_event(events: queue.Queue[UiEvent]) -> UiFrameEvent | None:
    while not events.empty():
        event = events.get_nowait()
        if isinstance(event, UiFrameEvent):
            return event
    return None


if __name__ == "__main__":
    unittest.main()
