from __future__ import annotations

import unittest
import threading
from unittest.mock import patch

import numpy as np

from cheap_camera_object_detector.detection import Detection
from cheap_camera_object_detector.monitoring import MonitorConfig, monitor_camera
from cheap_camera_object_detector.notifications import AlertEvent
from cheap_camera_object_detector.protocols import VideoFrame
from cheap_camera_object_detector.rtsp_capture import CaptureError, FrameReadResult


class MonitoringObservabilityTests(unittest.TestCase):
    def test_monitor_logs_model_detections(self) -> None:
        FakeDetector.results = [
            [
                Detection("person", 0.81234, (1.234, 2.345, 30.456, 40.567)),
                Detection("cell phone", 0.65432, (5, 6, 7, 8)),
            ]
        ]
        observed_frames: list[VideoFrame] = []
        fake_cv2 = FakeCv2(read_frames=[fake_frame()])

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch("cheap_camera_object_detector.monitoring.YoloObjectDetector", FakeDetector),
            self.assertLogs("cheap_camera_object_detector.monitoring", level="INFO") as logs,
        ):
            monitor_camera(
                MonitorConfig(
                    source="rtsp://user:secret@camera.local:554/stream1",
                    target="person",
                    frame_interval_seconds=0,
                    warmup_frames=0,
                ),
                NoopNotifier(),
                max_frames=1,
                frame_observer=observed_frames.append,
            )

        detection_logs = [
            line for line in logs.output if "object_detections frame=" in line
        ]
        rule_logs = [
            line for line in logs.output if "target_rule_evaluation frame=" in line
        ]

        self.assertEqual(1, len(detection_logs))
        self.assertIn("object_detections frame=1", detection_logs[0])
        self.assertIn(
            "source=rtsp://user:***@camera.local:554/stream1", detection_logs[0]
        )
        self.assertNotIn("secret", "\n".join(logs.output))
        self.assertIn("count=2", detection_logs[0])
        self.assertIn("'label': 'person'", detection_logs[0])
        self.assertIn("'confidence': 0.8123", detection_logs[0])
        self.assertIn("'label': 'cell phone'", detection_logs[0])
        self.assertIn("passed=True", rule_logs[0])
        self.assertEqual(1, len(observed_frames))
        self.assertEqual([], fake_cv2.saved_paths)
        self.assertEqual(1, len(fake_cv2.captures))
        self.assertTrue(fake_cv2.captures[0].released)

    def test_monitor_reports_available_labels_after_detection_runs(self) -> None:
        FakeDetector.results = [
            [
                Detection("person", 0.8, (1, 2, 3, 4)),
                Detection("person", 0.7, (5, 6, 7, 8)),
                Detection("cell phone", 0.6, (9, 10, 11, 12)),
            ]
        ]
        observed_labels: list[tuple[str, ...]] = []
        fake_cv2 = FakeCv2(read_frames=[fake_frame()])

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch("cheap_camera_object_detector.monitoring.YoloObjectDetector", FakeDetector),
        ):
            monitor_camera(
                MonitorConfig(
                    source="rtsp://user:secret@camera.local:554/stream1",
                    target="person",
                    frame_interval_seconds=0,
                    warmup_frames=0,
                ),
                NoopNotifier(),
                max_frames=1,
                detection_observer=observed_labels.append,
            )

        self.assertEqual([("person", "cell phone")], observed_labels)

    def test_monitor_keeps_observing_frames_while_detection_is_busy(self) -> None:
        observed_frames: list[VideoFrame] = []
        three_frames_observed = threading.Event()
        release_detection = threading.Event()
        SlowDetector.release_detection = release_detection
        SlowDetector.started_detection.clear()
        SlowDetector.detected_frames = []
        fake_cv2 = FakeCv2(
            read_frames=[fake_frame() for _index in range(15)]
        )

        def observe_frame(frame: VideoFrame) -> None:
            observed_frames.append(frame)
            if len(observed_frames) == 3:
                three_frames_observed.set()

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch("cheap_camera_object_detector.monitoring.YoloObjectDetector", SlowDetector),
            self.assertLogs("cheap_camera_object_detector.monitoring", level="INFO") as logs,
        ):
            monitor_thread = threading.Thread(
                target=monitor_camera,
                args=(
                    MonitorConfig(
                        source="rtsp://user:secret@camera.local:554/stream1",
                        target="dog",
                        frame_interval_seconds=0,
                        warmup_frames=0,
                    ),
                    NoopNotifier(),
                ),
                kwargs={"max_frames": 3, "frame_observer": observe_frame},
            )
            monitor_thread.start()
            self.assertTrue(SlowDetector.started_detection.wait(timeout=1))
            self.assertTrue(three_frames_observed.wait(timeout=1))
            release_detection.set()
            monitor_thread.join(timeout=1)

        self.assertFalse(monitor_thread.is_alive())
        self.assertEqual(3, len(observed_frames))
        self.assertEqual(1, len(SlowDetector.detected_frames))
        skip_logs = [
            line for line in logs.output if "monitor_detection_skipped frame=" in line
        ]
        self.assertEqual(2, len(skip_logs))

    def test_monitor_observes_sampled_frame_without_saving_it(self) -> None:
        sampled_frame = fake_frame()
        observed_frames: list[VideoFrame] = []
        FakeDetector.results = [[]]
        fake_cv2 = FakeCv2(read_frames=[sampled_frame])

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch("cheap_camera_object_detector.monitoring.YoloObjectDetector", FakeDetector),
        ):
            monitor_camera(
                MonitorConfig(
                    source="rtsp://user:secret@camera.local:554/stream1",
                    target="dog",
                    frame_interval_seconds=0,
                    warmup_frames=0,
                    frame_read_attempts=1,
                ),
                NoopNotifier(),
                max_frames=1,
                frame_observer=observed_frames.append,
            )

        self.assertEqual(1, len(observed_frames))
        self.assertIs(sampled_frame, observed_frames[0])
        self.assertEqual([], fake_cv2.saved_frames)

    def test_monitor_observes_first_frame_without_buffer_drain(self) -> None:
        first_frame = high_contrast_frame(fill=30)
        middle_frame = high_contrast_frame(fill=90)
        newest_frame = high_contrast_frame(fill=150)
        observed_frames: list[VideoFrame] = []
        fake_cv2 = FakeCv2(read_frames=[first_frame, middle_frame, newest_frame])
        notifier = RecordingNotifier()

        with patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2):
            monitor_camera(
                MonitorConfig(
                    source="rtsp://user:secret@camera.local:554/stream1",
                    target="",
                    detection_enabled=False,
                    frame_interval_seconds=0,
                    warmup_frames=0,
                    frame_read_attempts=1,
                ),
                notifier,
                max_frames=1,
                frame_observer=observed_frames.append,
            )

        self.assertEqual(1, len(observed_frames))
        self.assertIs(first_frame, observed_frames[0])

    def test_monitor_can_display_frames_without_detection(self) -> None:
        sampled_frame = fake_frame()
        observed_frames: list[VideoFrame] = []
        fake_cv2 = FakeCv2(read_frames=[sampled_frame])
        notifier = RecordingNotifier()

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch(
                "cheap_camera_object_detector.monitoring.YoloObjectDetector",
                side_effect=AssertionError("detector should not be created"),
            ),
            self.assertLogs("cheap_camera_object_detector.monitoring", level="INFO") as logs,
        ):
            monitor_camera(
                MonitorConfig(
                    source="rtsp://user:secret@camera.local:554/stream1",
                    target="",
                    detection_enabled=False,
                    frame_interval_seconds=0,
                    warmup_frames=0,
                    frame_read_attempts=1,
                ),
                notifier,
                max_frames=1,
                frame_observer=observed_frames.append,
            )

        self.assertEqual(1, len(observed_frames))
        self.assertIs(sampled_frame, observed_frames[0])
        self.assertEqual([], notifier.events)
        self.assertNotIn("object_detections frame=", "\n".join(logs.output))

    def test_monitor_reports_iteration_error_and_keeps_trying(self) -> None:
        sampled_frame = fake_frame()
        observed_frames: list[VideoFrame] = []
        observed_errors: list[Exception] = []
        FakeDetector.results = [[]]
        open_capture_results = [
            (FakeCapture([]), "tcp"),
            (FakeCapture([sampled_frame]), "tcp"),
        ]

        with (
            patch(
                "cheap_camera_object_detector.monitoring.open_rtsp_capture",
                side_effect=[
                    open_capture_results[0],
                    open_capture_results[1],
                ],
            ),
            patch(
                "cheap_camera_object_detector.monitoring.read_first_frame",
                side_effect=[
                    CaptureError("camera offline"),
                    FrameReadResult(sampled_frame, reads_used=1, attempts_used=1),
                ],
            ),
            patch("cheap_camera_object_detector.monitoring.YoloObjectDetector", FakeDetector),
        ):
            monitor_camera(
                MonitorConfig(
                    source="rtsp://user:secret@camera.local:554/stream1",
                    target="dog",
                    frame_interval_seconds=0,
                    warmup_frames=0,
                ),
                NoopNotifier(),
                max_frames=2,
                frame_observer=observed_frames.append,
                error_observer=observed_errors.append,
            )

        self.assertEqual(1, len(observed_errors))
        self.assertIn("camera offline", str(observed_errors[0]))
        self.assertEqual(1, len(observed_frames))
        self.assertIs(sampled_frame, observed_frames[0])
        self.assertTrue(open_capture_results[0][0].released)
        self.assertTrue(open_capture_results[1][0].released)

    def test_monitor_observes_first_frame_without_gray_frame_retry(self) -> None:
        first_frame = fake_frame(fill=120)
        second_frame = fake_frame()
        observed_frames: list[VideoFrame] = []
        FakeDetector.results = [[]]
        fake_cv2 = FakeCv2(read_frames=[first_frame, second_frame])

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch("cheap_camera_object_detector.monitoring.YoloObjectDetector", FakeDetector),
        ):
            monitor_camera(
                MonitorConfig(
                    source="rtsp://user:secret@camera.local:554/stream1",
                    target="dog",
                    frame_interval_seconds=0,
                    warmup_frames=0,
                    frame_read_attempts=2,
                ),
                NoopNotifier(),
                max_frames=1,
                frame_observer=observed_frames.append,
        )

        self.assertEqual(1, len(observed_frames))
        self.assertIs(first_frame, observed_frames[0])

    def test_monitor_stops_when_stop_callback_requests_it(self) -> None:
        FakeDetector.results = [[], []]
        fake_cv2 = FakeCv2(read_frames=[fake_frame(), fake_frame()])
        observed_frames: list[VideoFrame] = []
        stop_checks = 0

        def stop_requested() -> bool:
            nonlocal stop_checks
            stop_checks += 1
            return stop_checks > 1

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch("cheap_camera_object_detector.monitoring.YoloObjectDetector", FakeDetector),
        ):
            monitor_camera(
                MonitorConfig(
                    source="rtsp://user:secret@camera.local:554/stream1",
                    target="dog",
                    frame_interval_seconds=0,
                    warmup_frames=0,
                ),
                NoopNotifier(),
                stop_requested=stop_requested,
                frame_observer=observed_frames.append,
            )

        self.assertEqual(2, stop_checks)
        self.assertEqual(1, len(observed_frames))
        self.assertEqual([], fake_cv2.saved_frames)


def fake_frame(*, fill: int | None = None) -> VideoFrame:
    if fill is not None:
        return np.full((2, 2, 3), fill, dtype=np.uint8)
    return high_contrast_frame()


def high_contrast_frame(*, fill: int = 255) -> VideoFrame:
    return np.array([[[0, 0, 0], [fill, 255, 255]]], dtype=np.uint8)


class FakeCapture:
    def __init__(self, frames: list[VideoFrame]) -> None:
        self._frames = frames
        self.released = False

    def read(self) -> tuple[bool, VideoFrame | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class FakeDetector:
    results: list[list[Detection]] = []

    def __init__(
        self,
        model_name: str,
        *,
        confidence: float,
        image_size: int,
    ) -> None:
        pass

    def detect(self, frame: VideoFrame) -> list[Detection]:
        return self.results.pop(0)


class SlowDetector:
    release_detection = threading.Event()
    started_detection = threading.Event()
    detected_frames: list[VideoFrame] = []

    def __init__(
        self,
        model_name: str,
        *,
        confidence: float,
        image_size: int,
    ) -> None:
        pass

    def detect(self, frame: VideoFrame) -> list[Detection]:
        self.detected_frames.append(frame)
        self.started_detection.set()
        self.release_detection.wait(timeout=5)
        return []


class FakeCv2:
    CAP_PROP_OPEN_TIMEOUT_MSEC = 1
    CAP_PROP_READ_TIMEOUT_MSEC = 2
    CAP_FFMPEG = 4

    def __init__(self, read_frames: list[VideoFrame]) -> None:
        self.saved_paths: list[str] = []
        self.saved_frames: list[VideoFrame] = []
        self.captures: list[FakeCapture] = []
        self._read_frames = read_frames

    def VideoCapture(
        self,
        source: str | int,
        apiPreference: int = 0,
        params: tuple[int, ...] = (),
    ) -> FakeCapture:
        capture = FakeCapture(self._read_frames)
        self.captures.append(capture)
        return capture

    def imwrite(self, output_path: str, frame: VideoFrame) -> bool:
        self.saved_paths.append(output_path)
        self.saved_frames.append(frame)
        return True

    def imread(self, input_path: str) -> VideoFrame | None:
        if not self._read_frames:
            return None
        return self._read_frames.pop(0)


class NoopNotifier:
    def notify(self, event: AlertEvent) -> None:
        pass


class RecordingNotifier:
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    def notify(self, event: AlertEvent) -> None:
        self.events.append(event)


if __name__ == "__main__":
    unittest.main()
