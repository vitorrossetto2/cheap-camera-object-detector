from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np

from cheap_camera_object_detector.protocols import FloatArray, IntArray, VideoFrame, YoloModelProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


class YoloObjectDetector:
    """Small wrapper around Ultralytics YOLO object detection results."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        *,
        confidence: float = 0.35,
        image_size: int = 640,
    ) -> None:
        if not model_name:
            logger.error("detector_validation_failed parameter=model_name reason=blank")
            raise ValueError("model_name precisa ser informado")
        if confidence <= 0 or confidence > 1:
            logger.error(
                "detector_validation_failed parameter=confidence value=%s reason=out_of_range",
                confidence,
            )
            raise ValueError("confidence precisa estar entre 0 e 1")
        if image_size <= 0:
            logger.error(
                "detector_validation_failed parameter=image_size value=%s reason=not_positive",
                image_size,
            )
            raise ValueError("image_size precisa ser maior que zero")

        self.model_name = model_name
        self.confidence = confidence
        self.image_size = image_size
        logger.info(
            "detector_configured model=%s confidence=%s image_size=%s",
            self.model_name,
            self.confidence,
            self.image_size,
        )

    @cached_property
    def model(self) -> YoloModelProtocol:
        from ultralytics.models.yolo import YOLO

        logger.info("detector_model_load_start model=%s", self.model_name)
        model = cast(YoloModelProtocol, YOLO(self.model_name))
        logger.info("detector_model_load_succeeded model=%s", self.model_name)
        return model

    @property
    def labels(self) -> tuple[str, ...]:
        names = self.model.names
        if isinstance(names, dict):
            labels = tuple(str(names[index]) for index in sorted(names))
        else:
            labels = tuple(str(name) for name in names)
        logger.debug("detector_labels_loaded count=%s labels=%r", len(labels), labels)
        return labels

    def detect(self, frame: VideoFrame) -> list[Detection]:
        logger.debug(
            "detector_predict_start model=%s confidence=%s image_size=%s frame_size=%s",
            self.model_name,
            self.confidence,
            self.image_size,
            getattr(frame, "size", None),
        )
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            save=False,
            verbose=False,
        )
        if not results:
            logger.info("detector_predict_result count=0 reason=no_results")
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            logger.info("detector_predict_result count=0 reason=no_boxes")
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes: IntArray = boxes.cls.cpu().numpy().astype(np.int64)
        names = result.names

        detections: list[Detection] = []
        for index, class_id in enumerate(classes):
            detections.append(
                Detection(
                    label=str(names.get(int(class_id), f"class_{int(class_id)}")),
                    confidence=float(confs[index]),
                    box_xyxy=_box_xyxy(xyxy[index]),
                )
            )
        logger.info(
            "detector_predict_result count=%s labels=%r",
            len(detections),
            [detection.label for detection in detections],
        )
        return detections


def matching_detections(
    detections: list[Detection],
    target: str,
) -> list[Detection]:
    normalized_target = _normalize_label(target)
    if not normalized_target:
        logger.error("matching_rule_validation_failed target=%r reason=blank", target)
        raise ValueError("target precisa ser informado")
    matches = [
        detection
        for detection in detections
        if _normalize_label(detection.label) == normalized_target
    ]
    logger.info(
        "matching_rule_result target=%s normalized_target=%s passed=%s "
        "matches=%s detections=%s",
        target,
        normalized_target,
        bool(matches),
        len(matches),
        len(detections),
    )
    return matches


def _normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").split())


def _box_xyxy(values: FloatArray) -> tuple[float, float, float, float]:
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )
