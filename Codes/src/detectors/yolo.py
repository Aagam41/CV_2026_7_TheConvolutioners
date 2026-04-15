"""Ultralytics YOLO detector wrapper."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from ultralytics import YOLO

from .base import BaseDetector


class YoloDetector(BaseDetector):
    """Single-shot YOLO detector via Ultralytics."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        conf: float = 0.3,
        iou: float = 0.5,
        classes: Optional[Iterable[int]] = None,
        imgsz: int = 640,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = list(classes) if classes is not None else None
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = result.boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
