from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from cheap_camera_object_detector.ui import (
    LatestFrameSlot,
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
            "DETECTION_MODEL": "yolo11n.pt",
            "DETECTION_CONFIDENCE": "0.45",
        }

        with patch.dict(os.environ, env, clear=True):
            settings = default_ui_settings()

        self.assertEqual("rtsp://user:secret@camera.local:554/stream1", settings.source)
        self.assertEqual("tcp", settings.transport)
        self.assertEqual("person", settings.target)
        self.assertEqual("yolo11n.pt", settings.model_name)
        self.assertEqual(0.45, settings.confidence)

    def test_build_monitor_config_maps_ui_settings(self) -> None:
        settings = ui_settings(transport="tcp")

        config = build_monitor_config(settings)

        self.assertEqual(settings.source, config.source)
        self.assertEqual(settings.target, config.target)
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


def ui_settings(*, transport: str) -> UiSettings:
    return UiSettings(
        source="rtsp://user:secret@camera.local:554/stream1",
        target="dog",
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


if __name__ == "__main__":
    unittest.main()
