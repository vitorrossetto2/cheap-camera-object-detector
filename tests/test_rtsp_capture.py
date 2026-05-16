from __future__ import annotations

import unittest
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from cheap_camera_object_detector.protocols import VideoFrame
from cheap_camera_object_detector.rtsp_capture import capture_rtsp_frame, capture_rtsp_image


class RtspCaptureTests(unittest.TestCase):
    def test_capture_rtsp_frame_returns_sampled_frame_without_saving(self) -> None:
        sampled_frame = fake_frame()
        fake_cv2 = FakeCv2(read_frames=[sampled_frame])

        with patch("image_behaviour_alerts.rtsp_capture._cv2", return_value=fake_cv2):
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
            with patch("image_behaviour_alerts.rtsp_capture._cv2", return_value=fake_cv2):
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


def fake_frame() -> VideoFrame:
    return np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)


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

    def set(self, propId: int, value: float) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_PROP_OPEN_TIMEOUT_MSEC = 1
    CAP_PROP_READ_TIMEOUT_MSEC = 2
    CAP_PROP_BUFFERSIZE = 3
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
        params: Sequence[int] = (),
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


if __name__ == "__main__":
    unittest.main()
