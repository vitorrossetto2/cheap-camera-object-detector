from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

VideoFrame: TypeAlias = NDArray[np.uint8]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_ArrayType = TypeVar("_ArrayType", bound=np.generic, covariant=True)


class TensorProtocol(Protocol[_ArrayType]):
    def cpu(self) -> TensorProtocol[_ArrayType]:
        ...

    def numpy(self) -> NDArray[_ArrayType]:
        ...


class YoloBoxesProtocol(Protocol):
    xyxy: TensorProtocol[np.float64]
    conf: TensorProtocol[np.float64]
    cls: TensorProtocol[np.float64]

    def __len__(self) -> int:
        ...


class YoloResultProtocol(Protocol):
    boxes: YoloBoxesProtocol | None
    names: Mapping[int, str]


class YoloModelProtocol(Protocol):
    names: Mapping[int, str] | Sequence[str]

    def predict(
        self,
        *,
        source: VideoFrame,
        conf: float,
        imgsz: int,
        save: bool,
        verbose: bool,
    ) -> Sequence[YoloResultProtocol]:
        ...


class VideoCaptureProtocol(Protocol):
    def read(self) -> tuple[bool, VideoFrame | None]:
        ...

    def isOpened(self) -> bool:
        ...

    def release(self) -> None:
        ...


class Cv2Protocol(Protocol):
    CAP_PROP_OPEN_TIMEOUT_MSEC: int
    CAP_PROP_READ_TIMEOUT_MSEC: int
    CAP_FFMPEG: int

    def VideoCapture(
        self,
        source: str | int,
        apiPreference: int = 0,
        params: Sequence[int] = (),
    ) -> VideoCaptureProtocol:
        ...

    def imwrite(self, output_path: str, frame: VideoFrame) -> bool:
        ...

    def imread(self, input_path: str) -> VideoFrame | None:
        ...
