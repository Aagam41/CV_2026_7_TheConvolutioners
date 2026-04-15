"""
sahi_detect.py
--------------
Slicing Aided Hyper Inference (SAHI) wrapper around an Ultralytics YOLO model.

SAHI splits a large image into overlapping slices, runs the detector on each
slice at its native resolution, maps boxes back to the original image, and
merges them with NMS. This recovers small objects that get squashed when a
4K VisDrone frame is resized down to the model's imgsz.

Reference: Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for
Small Object Detection" (ICIP 2022). This implementation follows the same
strategy without requiring the `sahi` package:
    * Compute slice grid with given (slice_h, slice_w) and overlap ratio
    * Optionally keep a full-image pass ("SF" — Standard + Fine inference)
    * Class-aware NMS over the union of all detections
"""
import numpy as np
import torch


# ============================== Slice grid ==============================
def compute_slice_coords(img_h, img_w, slice_h, slice_w,
                         overlap_h_ratio=0.2, overlap_w_ratio=0.2):
    """Return a list of (x1, y1, x2, y2) slice boxes covering the image."""
    slices = []
    step_y = max(1, int(slice_h * (1 - overlap_h_ratio)))
    step_x = max(1, int(slice_w * (1 - overlap_w_ratio)))
    y = 0
    while y < img_h:
        x = 0
        y2 = min(y + slice_h, img_h)
        y1 = max(0, y2 - slice_h)
        while x < img_w:
            x2 = min(x + slice_w, img_w)
            x1 = max(0, x2 - slice_w)
            slices.append((x1, y1, x2, y2))
            if x2 >= img_w:
                break
            x += step_x
        if y2 >= img_h:
            break
        y += step_y
    # De-duplicate (edge slices can produce the same box)
    seen = set(); out = []
    for s in slices:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


# =============================== NMS ===============================
def _nms_numpy(boxes, scores, iou_thresh):
    """Greedy NMS on xyxy boxes. Returns kept indices."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, (x2 - x1)) * np.maximum(0.0, (y2 - y1))
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        order = order[1:][iou <= iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def class_aware_nms(dets, iou_thresh=0.5):
    """dets: (N, 6) xyxy+score+cls. Class-aware NMS."""
    if len(dets) == 0:
        return dets
    out = []
    for c in np.unique(dets[:, 5]):
        m = dets[:, 5] == c
        sub = dets[m]
        keep = _nms_numpy(sub[:, :4], sub[:, 4], iou_thresh)
        out.append(sub[keep])
    return np.concatenate(out, axis=0) if out else dets[:0]


# ================================ SAHI ================================
class SAHIDetector:
    """
    Drop-in replacement for `model.predict(...)` in track.py.

    Call:
        det = SAHIDetector(yolo_model,
                           slice_h=640, slice_w=640,
                           overlap_h=0.2, overlap_w=0.2,
                           conf=0.25, iou=0.7,
                           include_full_image=True)
        dets = det(img_bgr, classes=None)   # -> (N,6) xyxy,score,cls
    """
    def __init__(self, yolo_model,
                 slice_h=640, slice_w=640,
                 overlap_h=0.2, overlap_w=0.2,
                 conf=0.25, iou=0.7,
                 include_full_image=True,
                 full_image_imgsz=1280,
                 match_metric_iou=0.5,
                 batch=8,
                 device=None):
        self.model = yolo_model
        self.slice_h = int(slice_h)
        self.slice_w = int(slice_w)
        self.overlap_h = float(overlap_h)
        self.overlap_w = float(overlap_w)
        self.conf = float(conf)
        self.iou = float(iou)
        self.include_full_image = bool(include_full_image)
        self.full_image_imgsz = int(full_image_imgsz)
        self.match_metric_iou = float(match_metric_iou)
        self.batch = int(batch)
        self.device = device

    def __call__(self, img_bgr, classes=None):
        H, W = img_bgr.shape[:2]
        all_dets = []

        # 1) Slice inference
        slices = compute_slice_coords(H, W, self.slice_h, self.slice_w,
                                      self.overlap_h, self.overlap_w)
        # Batch slices to keep GPU busy
        for i in range(0, len(slices), self.batch):
            chunk = slices[i:i + self.batch]
            imgs = [img_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in chunk]
            results = self.model.predict(
                imgs,
                imgsz=max(self.slice_h, self.slice_w),
                conf=self.conf, iou=self.iou,
                verbose=False, classes=classes,
                device=self.device,
            )
            for (x1, y1, x2, y2), res in zip(chunk, results):
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                b = res.boxes.xyxy.cpu().numpy()
                s = res.boxes.conf.cpu().numpy()
                c = res.boxes.cls.cpu().numpy()
                b[:, 0] += x1; b[:, 2] += x1
                b[:, 1] += y1; b[:, 3] += y1
                # Clip to image
                b[:, 0::2] = np.clip(b[:, 0::2], 0, W - 1)
                b[:, 1::2] = np.clip(b[:, 1::2], 0, H - 1)
                all_dets.append(
                    np.concatenate([b, s[:, None], c[:, None]], axis=1))

        # 2) Full-image pass ("SF" mode: small + large objects)
        if self.include_full_image:
            res = self.model.predict(
                img_bgr,
                imgsz=self.full_image_imgsz,
                conf=self.conf, iou=self.iou,
                verbose=False, classes=classes,
                device=self.device,
            )[0]
            if res.boxes is not None and len(res.boxes) > 0:
                b = res.boxes.xyxy.cpu().numpy()
                s = res.boxes.conf.cpu().numpy()
                c = res.boxes.cls.cpu().numpy()
                all_dets.append(
                    np.concatenate([b, s[:, None], c[:, None]], axis=1))

        if not all_dets:
            return np.empty((0, 6), dtype=np.float32)

        dets = np.concatenate(all_dets, axis=0).astype(np.float32)
        # 3) Merge overlapping detections across slices
        dets = class_aware_nms(dets, iou_thresh=self.match_metric_iou)
        return dets
		
