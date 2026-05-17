from __future__ import annotations

import unittest
import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from cheap_camera_object_detector.protocols import VideoFrame
from cheap_camera_object_detector.rtsp_capture import (
    CaptureError,
    capture_rtsp_frame,
    capture_rtsp_image,
    open_rtsp_capture,
    read_latest_valid_frame,
)


class RtspCaptureTests(unittest.TestCase):
    def test_capture_rtsp_frame_returns_sampled_frame_without_saving(self) -> None:
        sampled_frame = fake_frame()
        fake_cv2 = FakeCv2(read_frames=[sampled_frame])

        with patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2):
            result = capture_rtsp_frame(
                "rtsp://user:secret@camera.local:554/stream1",
                attempts=1,
                warmup_frames=0,
                prefer_tcp=False,
            )

        self.assertIs(sampled_frame, result.frame)
        self.assertEqual("rtsp://user:secret@camera.local:554/stream1", result.source_url)
        self.assertEqual(1, result.attempts)
        self.assertEqual([], fake_cv2.saved_frames)
        self.assertTrue(fake_cv2.captures[0].released)

    def test_capture_rtsp_image_saves_compatibility_result(self) -> None:
        sampled_frame = fake_frame()
        fake_cv2 = FakeCv2(read_frames=[sampled_frame])

        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "camera.jpg"
            with patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2):
                result = capture_rtsp_image(
                    "rtsp://user:secret@camera.local:554/stream1",
                    output_path,
                    attempts=1,
                    warmup_frames=0,
                    prefer_tcp=False,
                )

        self.assertEqual(output_path.resolve(), result.output_path)
        self.assertEqual([str(output_path)], fake_cv2.saved_paths)
        self.assertIs(sampled_frame, fake_cv2.saved_frames[0])

    def test_open_rtsp_capture_falls_back_to_tcp_when_udp_fails(self) -> None:
        fake_cv2 = FakeCv2(read_frames=[], open_results=[False, True])

        with patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2):
            capture, transport = open_rtsp_capture(
                "rtsp://user:secret@camera.local:554/stream1",
                10_000,
                10_000,
                prefer_tcp=False,
            )

        self.assertEqual("tcp", transport)
        self.assertIs(fake_cv2.captures[1], capture)
        self.assertTrue(fake_cv2.captures[0].released)
        self.assertFalse(fake_cv2.captures[1].released)
        self.assertEqual(
            [
                fake_cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                10_000,
                fake_cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                10_000,
            ],
            fake_cv2.capture_params[0],
        )

    def test_open_rtsp_capture_prefers_tcp_before_udp(self) -> None:
        fake_cv2 = FakeCv2(read_frames=[], open_results=[True])

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch("cheap_camera_object_detector.rtsp_capture._configure_ffmpeg_transport") as configure,
        ):
            _capture, transport = open_rtsp_capture(
                "rtsp://user:secret@camera.local:554/stream1",
                10_000,
                10_000,
                prefer_tcp=True,
            )

        self.assertEqual("tcp", transport)
        configure.assert_called_once_with("tcp")

    def test_open_rtsp_capture_error_lists_attempted_transports(self) -> None:
        fake_cv2 = FakeCv2(read_frames=[], open_results=[False, False])

        with patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2):
            with self.assertRaisesRegex(CaptureError, "UDP, TCP"):
                open_rtsp_capture(
                    "rtsp://user:secret@camera.local:554/stream1",
                    10_000,
                    10_000,
                    prefer_tcp=False,
                )

        self.assertTrue(fake_cv2.captures[0].released)
        self.assertTrue(fake_cv2.captures[1].released)

    def test_open_rtsp_capture_merges_low_latency_ffmpeg_options(self) -> None:
        fake_cv2 = FakeCv2(read_frames=[], open_results=[True])

        with (
            patch("cheap_camera_object_detector.rtsp_capture._cv2", return_value=fake_cv2),
            patch.dict(
                os.environ,
                {
                    "OPENCV_FFMPEG_CAPTURE_OPTIONS": (
                        "stimeout;1000|rtsp_transport;udp|fflags;discardcorrupt"
                    )
                },
            ),
        ):
            open_rtsp_capture(
                "rtsp://user:secret@camera.local:554/stream1",
                10_000,
                10_000,
                prefer_tcp=True,
            )

            options = os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

        self.assertIn("stimeout;1000", options)
        self.assertIn("rtsp_transport;tcp", options)
        self.assertIn("fflags;discardcorrupt+nobuffer", options)
        self.assertIn("flags;low_delay", options)
        self.assertIn("max_delay;0", options)

    def test_read_latest_valid_frame_returns_newest_drained_frame(self) -> None:
        old_frame = fake_frame(fill=10)
        middle_frame = fake_frame(fill=80)
        newest_frame = fake_frame(fill=140)
        capture = FakeCapture([old_frame, middle_frame, newest_frame])

        result = read_latest_valid_frame(
            capture,
            attempts=1,
            warmup_frames=0,
            log_context="test_capture",
            drain_reads=2,
        )

        self.assertIs(newest_frame, result.frame)
        self.assertEqual(3, result.reads_used)
        self.assertEqual(3, result.attempts_used)


def fake_frame(*, fill: int | None = None) -> VideoFrame:
    if fill is not None:
        return np.array([[[fill, 0, 255]]], dtype=np.uint8)
    return np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)


class FakeCapture:
    def __init__(self, frames: list[VideoFrame], *, opened: bool = True) -> None:
        self._frames = frames
        self._opened = opened
        self.released = False

    def read(self) -> tuple[bool, VideoFrame | None]:
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_PROP_OPEN_TIMEOUT_MSEC = 1
    CAP_PROP_READ_TIMEOUT_MSEC = 2
    CAP_FFMPEG = 4

    def __init__(
        self,
        read_frames: list[VideoFrame],
        open_results: list[bool] | None = None,
    ) -> None:
        self.saved_paths: list[str] = []
        self.saved_frames: list[VideoFrame] = []
        self.captures: list[FakeCapture] = []
        self.capture_params: list[list[int]] = []
        self._read_frames = read_frames
        self._open_results = open_results or []

    def VideoCapture(
        self,
        source: str | int,
        apiPreference: int = 0,
        params: Sequence[int] = (),
    ) -> FakeCapture:
        opened = self._open_results.pop(0) if self._open_results else True
        capture = FakeCapture(self._read_frames, opened=opened)
        self.capture_params.append(list(params))
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


if __name__ == "__main__":
    unittest.main()
