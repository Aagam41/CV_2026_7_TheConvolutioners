"""
detector.py
-----------
YOLO + SAHI custom detector wrapper for BoxMOT.

Produces detections in the format expected by BoxMOT trackers:
    numpy array of shape (N, 6) → [x1, y1, x2, y2, conf, cls]

VisDrone class mapping (0-indexed):
    0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van,
    5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor
"""

import numpy as np
from pathlib import Path
from typing import Optional, List


class YOLOSAHIDetector:
    """
    Wraps a YOLO model with SAHI sliced inference for improved
    small-object detection on UAV/drone imagery.

    Parameters
    ----------
    model_path : str or Path
        Path to YOLO weights (e.g. 'yolov8n.pt', 'yolov8s.pt').
    device : str
        Torch device string: 'cpu', 'cuda:0', '0', etc.
    conf_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        NMS IoU threshold for SAHI postmerge.
    slice_height : int
        Height of each SAHI slice (pixels).
    slice_width : int
        Width of each SAHI slice (pixels).
    overlap_height_ratio : float
        Fractional overlap between vertical slices (0–1).
    overlap_width_ratio : float
        Fractional overlap between horizontal slices (0–1).
    classes : list of int, optional
        If given, keep only detections whose class id is in this list.
        VisDrone has 10 classes (0-9); pass None to keep all.
    model_type : str
        SAHI model type string. 'yolov8' works for YOLOv8-v12 via
        Ultralytics. Use 'yolov5' for YOLOv5, 'yolox' for YOLOX, etc.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        classes: Optional[List[int]] = None,
        model_type: str = "yolov8",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.classes = classes
        self.model_path = str(model_path)

        # Normalise device string for SAHI (needs 'cuda:0' not just '0')
        if device.isdigit():
            device = f"cuda:{device}"
        self.device = device

        # ── Build SAHI AutoDetectionModel ────────────────────────────────
        try:
            from sahi import AutoDetectionModel
        except ImportError as e:
            raise ImportError(
                "SAHI is required. Install with: pip install sahi"
            ) from e

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=self.model_path,
            confidence_threshold=conf_threshold,
            device=self.device,
        )

        print(
            f"[YOLOSAHIDetector] Loaded '{model_path}' on {self.device} "
            f"| slices={slice_height}×{slice_width} "
            f"| overlap=({overlap_height_ratio},{overlap_width_ratio}) "
            f"| conf≥{conf_threshold}"
        )

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run YOLO+SAHI sliced inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3) as returned by cv2.imread / cap.read.

        Returns
        -------
        dets : np.ndarray, shape (N, 6)
            Each row: [x1, y1, x2, y2, conf, cls]  (float32)
            Returns empty (0, 6) array when no detections.
        """
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image=frame,
            detection_model=self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            perform_standard_pred=True,   # also run on full image
            postprocess_type="NMM",       # Non-Maximum Merging for sliced
            postprocess_match_threshold=self.iou_threshold,
            verbose=0,
        )

        preds = result.object_prediction_list
        if not preds:
            return np.empty((0, 6), dtype=np.float32)

        rows = []
        for pred in preds:
            cls_id = int(pred.category.id)
            if self.classes is not None and cls_id not in self.classes:
                continue
            conf = float(pred.score.value)
            if conf < self.conf_threshold:
                continue
            b = pred.bbox
            rows.append([b.minx, b.miny, b.maxx, b.maxy, conf, cls_id])

        if not rows:
            return np.empty((0, 6), dtype=np.float32)

        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"YOLOSAHIDetector(model='{self.model_path}', "
            f"device='{self.device}', conf={self.conf_threshold})"
        )
