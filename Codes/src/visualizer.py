"""Thin OpenCV video writer wrapper used by the pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2


class VideoSink:
    """Lazy mp4 writer (creates parent dirs and releases on close)."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: Tuple[int, int],
        fourcc: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*fourcc), fps, size
        )
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer at {self.path}")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
