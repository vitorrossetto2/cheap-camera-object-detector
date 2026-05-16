from __future__ import annotations

import unittest

from cheap_camera_object_detector.detection import Detection, matching_detections
from cheap_camera_object_detector.notifications import AlertEvent, CooldownNotifier


class DetectionMatchingTests(unittest.TestCase):
    def test_matching_detections_normalizes_spacing_and_case(self) -> None:
        detections = [
            Detection("Cell Phone", 0.81, (0, 0, 10, 10)),
            Detection("person", 0.7, (0, 0, 20, 20)),
        ]

        matches = matching_detections(detections, " cell_phone ")

        self.assertEqual([match.label for match in matches], ["Cell Phone"])


class CooldownNotifierTests(unittest.TestCase):
    def test_cooldown_suppresses_repeated_events(self) -> None:
        recorder = RecordingNotifier()
        notifier = CooldownNotifier(recorder, cooldown_seconds=5)

        first = AlertEvent("person", (), "0", 100)
        second = AlertEvent("person", (), "0", 102)
        third = AlertEvent("person", (), "0", 106)

        notifier.notify(first)
        notifier.notify(second)
        notifier.notify(third)

        self.assertEqual(recorder.events, [first, third])


class RecordingNotifier:
    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    def notify(self, event: AlertEvent) -> None:
        self.events.append(event)


if __name__ == "__main__":
    unittest.main()
