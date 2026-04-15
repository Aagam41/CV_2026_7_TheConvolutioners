"""SAHI sliced-inference detector wrapping any Ultralytics model."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from .base import BaseDetector


class SahiYoloDetector(BaseDetector):
    """SAHI slicing on top of an Ultralytics YOLO checkpoint.

    Useful when targets are tiny relative to the frame (e.g. aerial / surveillance).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        conf: float = 0.3,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        classes: Optional[Iterable[int]] = None,
        model_type: str = "ultralytics",
        perform_standard_pred: bool = True,
        postprocess_type: str = "GREEDYNMM",
        postprocess_match_metric: str = "IOS",
        postprocess_match_threshold: float = 0.5,
    ) -> None:
        self.model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=conf,
            device=device,
        )
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.classes = set(classes) if classes is not None else None
        self.perform_standard_pred = perform_standard_pred
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = postprocess_match_threshold

    def detect(self, frame: np.ndarray) -> np.ndarray:
        result = get_sliced_prediction(
            image=frame,
            detection_model=self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            perform_standard_pred=self.perform_standard_pred,
            postprocess_type=self.postprocess_type,
            postprocess_match_metric=self.postprocess_match_metric,
            postprocess_match_threshold=self.postprocess_match_threshold,
            verbose=0,
        )

        rows = []
        for obj in result.object_prediction_list:
            cls_id = int(obj.category.id)
            if self.classes is not None and cls_id not in self.classes:
                continue
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            rows.append([x1, y1, x2, y2, float(obj.score.value), cls_id])

        if not rows:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)
