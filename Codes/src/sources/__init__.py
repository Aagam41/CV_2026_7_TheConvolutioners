"""Frame sources: video files OR MOTChallenge-style image folders.

Both implement the same iterator protocol:
    for frame_idx, frame in source: ...

`source.fps`, `source.size`, `source.length` are also available.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Tuple

import cv2

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class FrameSource(ABC):
    fps: float
    size: Tuple[int, int]   # (width, height)
    length: int             # 0 if unknown

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[int, "cv2.Mat"]]:
        ...


class VideoFileSource(FrameSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._cap = cap

    def __iter__(self):
        idx = 0
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                idx += 1
                yield idx, frame
        finally:
            self._cap.release()


class ImageFolderSource(FrameSource):
    """Reads all images in a folder, sorted lexicographically.

    Works out of the box on MOTChallenge sequences (`MOT17/train/<seq>/img1`).
    """

    def __init__(self, folder: str | Path, fps: float = 30.0) -> None:
        self.folder = Path(folder)
        self._files = sorted(
            p for p in self.folder.iterdir() if p.suffix.lower() in _IMG_EXT
        )
        if not self._files:
            raise FileNotFoundError(f"No images found in {self.folder}")
        first = cv2.imread(str(self._files[0]))
        if first is None:
            raise IOError(f"Cannot read {self._files[0]}")
        h, w = first.shape[:2]
        self.fps = float(fps)
        self.size = (w, h)
        self.length = len(self._files)

    def __iter__(self):
        for i, p in enumerate(self._files, start=1):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            yield i, frame


def open_source(path: str | Path, fps: float = 30.0) -> FrameSource:
    """Auto-detect whether `path` is a video file or an image directory."""
    p = Path(path)
    if p.is_dir():
        return ImageFolderSource(p, fps=fps)
    return VideoFileSource(p)
