# Usage: python botsort.py data/VisDrone2019-VID-val/sequences/<sequence_name>
#        python botsort.py --images-dir <path> --out-dir runs/<name> --out-mp4 runs/<name>.mp4 --fps 30
"""
botsort.py — Simple BoT-SORT multi-object tracker with Siamese ReID.

Algorithm
---------
1. Constant-velocity Kalman filter predicts each track forward one step.
2. First association  : all tracks  ↔  high-confidence detections
                        cost = IoU cost + cosine-appearance cost (blended)
3. Second association : remaining tracks  ↔  low-confidence detections
                        cost = IoU only
4. Stale tracks (time_since_update > max_age) are deleted.
5. Unmatched high-confidence detections whose score ≥ new_track_thresh
   initialise new (tentative) tracks.
6. A track appears in the output only after it has been confirmed for
   min_hits consecutive frames.

The appearance embeddings come from the pre-trained Siamese ResNet-18
stored in siamese_final.pth.  They are maintained per-track as an
exponential moving average of 128-d L2-normalised feature vectors.

Input / output contract
-----------------------
frame      : np.ndarray  H×W×3  uint8  RGB
detections : np.ndarray  (N, 5) or (N, 6)
             columns: [x1, y1, x2, y2, score [, class_id]]
             coordinates in pixels, score in [0, 1]

BotSort.update() returns
    np.ndarray  (M, 6)  [x1, y1, x2, y2, track_id, score]
or  np.empty((0, 6))   when no confirmed tracks are active.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.optimize import linear_sum_assignment
from typing import List, Optional, Tuple

from model import SiameseNet

# ── ReID pre-processing (mirrors dataset.py) ──────────────────────────────

_REID_TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])


# ═══════════════════════════════════════════════════════════════════════════
# Kalman filter
# ═══════════════════════════════════════════════════════════════════════════

class KalmanBoxTracker:
    """
    Constant-velocity Kalman filter for an axis-aligned bounding box.

    State vector   : [cx, cy, w, h, vcx, vcy, vw, vh]
    Measurement    : [cx, cy, w, h]
    """

    _count: int = 0

    @classmethod
    def reset_count(cls) -> None:
        cls._count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        cx, cy, w, h = self._xyxy_to_cxcywh(bbox)

        # Constant-velocity transition
        self.F = np.eye(8, dtype=float)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Observation selects [cx, cy, w, h]
        self.H = np.zeros((4, 8), dtype=float)
        self.H[:4, :4] = np.eye(4)

        # Process noise — velocity components less certain
        self.Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.01, 0.01])

        # Measurement noise — larger for w, h (more uncertain)
        self.R = np.diag([1., 1., 10., 10.])

        # Initial covariance — high uncertainty on velocities
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4, 1e4])

        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.])

        self.id: int = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1

        self.time_since_update: int = 0
        self.hit_streak: int = 0
        self.age: int = 0

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _xyxy_to_cxcywh(b: np.ndarray) -> List[float]:
        return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                float(b[2] - b[0]), float(b[3] - b[1])]

    def _state_to_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    # ── predict / update ──────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._state_to_xyxy()

    def update(self, bbox: np.ndarray) -> None:
        self.time_since_update = 0
        self.hit_streak += 1
        z = np.array(self._xyxy_to_cxcywh(bbox), dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self._state_to_xyxy()


# ═══════════════════════════════════════════════════════════════════════════
# Track
# ═══════════════════════════════════════════════════════════════════════════

class Track:
    """
    One tracked object: a Kalman filter state plus an EMA appearance feature.
    """

    EMA_ALPHA = 0.9  # weight on the existing (historical) embedding

    def __init__(self, bbox: np.ndarray, score: float, feat: np.ndarray) -> None:
        self.kf = KalmanBoxTracker(bbox)
        self.track_id: int = self.kf.id
        self.score: float = float(score)
        self.feat: np.ndarray = self._norm(feat)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _norm(f: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(f))
        return f / (n + 1e-6)

    # ── public API ────────────────────────────────────────────────────────

    def update(self, bbox: np.ndarray, score: float,
               feat: Optional[np.ndarray]) -> None:
        self.kf.update(bbox)
        self.score = float(score)
        if feat is not None:
            new_f = self._norm(feat)
            self.feat = self._norm(
                self.EMA_ALPHA * self.feat + (1 - self.EMA_ALPHA) * new_f
            )

    def predict(self) -> np.ndarray:
        return self.kf.predict()

    def get_state(self) -> np.ndarray:
        return self.kf.get_state()

    @property
    def time_since_update(self) -> int:
        return self.kf.time_since_update

    @property
    def hit_streak(self) -> int:
        return self.kf.hit_streak


# ═══════════════════════════════════════════════════════════════════════════
# BotSort tracker
# ═══════════════════════════════════════════════════════════════════════════

class BotSort:
    """
    Simple BoT-SORT tracker backed by the pre-trained Siamese ReID model.

    Parameters
    ----------
    model_path        : Path to the saved Siamese weights (siamese_final.pth).
    device            : 'cuda', 'cpu', or None for auto-detect.
    track_high_thresh : Minimum detection score for first-pass association.
    track_low_thresh  : Minimum detection score for second-pass association.
    new_track_thresh  : Minimum score to start a new track.
    max_age           : Frames a track survives without any detection.
    min_hits          : Consecutive hits required before a track is output.
    iou_thresh        : Minimum IoU for a match to be accepted.
    appearance_weight : Blend factor.  0 → IoU only, 1 → appearance only.
    match_cost_thresh : Max blended cost allowed for a first-pass match.
    min_iou_gate      : Loose minimum IoU gate for first-pass appearance matching.
    low_new_track_thresh : Minimum low-confidence score to start a new track.
    spawn_iou_suppression : If a new detection overlaps an existing track above
                            this IoU, skip spawning to avoid duplicates.
    """

    def __init__(
        self,
        model_path: str = "siamese_final.pth",
        device: Optional[str] = None,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        max_age: int = 30,
        min_hits: int = 1,
        iou_thresh: float = 0.3,
        appearance_weight: float = 0.5,
        match_cost_thresh: float = 0.72,
        min_iou_gate: float = 0.05,
        low_new_track_thresh: float = 0.5,
        spawn_iou_suppression: float = 0.8,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ── load Siamese ReID model ────────────────────────────────────────
        self.reid = SiameseNet()
        state = torch.load(model_path, map_location=device, weights_only=True)
        self.reid.load_state_dict(state)
        self.reid.to(device).eval()

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.appearance_weight = float(appearance_weight)
        self.iou_weight = 1.0 - self.appearance_weight
        self.match_cost_thresh = float(match_cost_thresh)
        self.min_iou_gate = float(min_iou_gate)
        self.low_new_track_thresh = float(low_new_track_thresh)
        self.spawn_iou_suppression = float(spawn_iou_suppression)

        self.tracks: List[Track] = []
        self.frame_count: int = 0

    # ── ReID feature extraction ───────────────────────────────────────────

    def _extract_features(
        self, frame: np.ndarray, bboxes: np.ndarray
    ) -> np.ndarray:
        """
        Crop detection patches from *frame* and embed them with the Siamese net.

        Parameters
        ----------
        frame  : H×W×3 uint8 RGB numpy array.
        bboxes : (N, 4) array of [x1, y1, x2, y2] pixel coordinates.

        Returns
        -------
        (N, 128) float32 embedding matrix.
        """
        if len(bboxes) == 0:
            return np.zeros((0, 128), dtype=np.float32)

        h, w = frame.shape[:2]
        crops = []
        for box in bboxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(torch.zeros(3, 128, 128))
            else:
                patch = Image.fromarray(frame[y1:y2, x1:x2])
                crops.append(_REID_TRANSFORM(patch))

        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            feats = self.reid.forward_once(batch).cpu().numpy()
        return feats.astype(np.float32)

    # ── cost matrices ─────────────────────────────────────────────────────

    @staticmethod
    def _iou_matrix(
        boxes_a: np.ndarray, boxes_b: np.ndarray
    ) -> np.ndarray:
        """Return (M, N) IoU matrix for M track boxes and N detection boxes."""
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        a, b = np.array(boxes_a, dtype=float), np.array(boxes_b, dtype=float)
        ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
        iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
        ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
        iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return (inter / (union + 1e-6)).astype(np.float32)

    @staticmethod
    def _cosine_distance(
        feats_a: np.ndarray, feats_b: np.ndarray
    ) -> np.ndarray:
        """Return (M, N) cosine-distance matrix."""
        if len(feats_a) == 0 or len(feats_b) == 0:
            return np.zeros((len(feats_a), len(feats_b)), dtype=np.float32)
        na = feats_a / (np.linalg.norm(feats_a, axis=1, keepdims=True) + 1e-6)
        nb = feats_b / (np.linalg.norm(feats_b, axis=1, keepdims=True) + 1e-6)
        return (1.0 - na @ nb.T).astype(np.float32)

    # ── Hungarian assignment ──────────────────────────────────────────────

    def _associate(
        self,
        tracks: List[Track],
        det_bboxes: np.ndarray,
        det_feats: np.ndarray,
        use_appearance: bool = True,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match *tracks* to detections with the Hungarian algorithm.

        Returns
        -------
        matched        : list of (track_index, det_index) pairs
        unmatched_t    : list of unmatched track indices
        unmatched_d    : list of unmatched detection indices
        """
        nt, nd = len(tracks), len(det_bboxes)
        if nt == 0:
            return [], [], list(range(nd))
        if nd == 0:
            return [], list(range(nt)), []

        track_boxes = np.array([t.get_state() for t in tracks])
        iou_mat = self._iou_matrix(track_boxes, det_bboxes)
        iou_cost = 1.0 - iou_mat
        cost = iou_cost.copy()

        if use_appearance and self.appearance_weight > 0.0:
            track_feats = np.array([t.feat for t in tracks])
            app_cost = self._cosine_distance(track_feats, det_feats)
            cost = self.iou_weight * cost + self.appearance_weight * app_cost

        row_ind, col_ind = linear_sum_assignment(cost)

        matched: List[Tuple[int, int]] = []
        unmatched_t: List[int] = list(range(nt))
        unmatched_d: List[int] = list(range(nd))

        for r, c in zip(row_ind, col_ind):
            if use_appearance and self.appearance_weight > 0.0:
                accept = (
                    iou_mat[r, c] >= self.min_iou_gate
                    and cost[r, c] <= self.match_cost_thresh
                )
            else:
                accept = iou_mat[r, c] >= self.iou_thresh

            if accept:
                matched.append((r, c))
                unmatched_t.remove(r)
                unmatched_d.remove(c)

        return matched, unmatched_t, unmatched_d

    def _should_spawn_track(self, det_bbox: np.ndarray) -> bool:
        """Suppress track birth when a detection strongly overlaps an active track."""
        if len(self.tracks) == 0:
            return True
        track_boxes = np.array([t.get_state() for t in self.tracks], dtype=np.float32)
        det_box = np.array(det_bbox, dtype=np.float32).reshape(1, 4)
        ious = self._iou_matrix(track_boxes, det_box)
        if ious.size == 0:
            return True
        return float(np.max(ious)) < self.spawn_iou_suppression

    # ── main entry point ──────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        detections: np.ndarray,
    ) -> np.ndarray:
        """
        Run one tracking step.

        Parameters
        ----------
        frame      : np.ndarray  H×W×3  uint8  RGB
        detections : np.ndarray  (N, 5) or (N, 6)
                     [x1, y1, x2, y2, score [, class_id]]

        Returns
        -------
        np.ndarray  (M, 6)  [x1, y1, x2, y2, track_id, score]
        Returns an empty (0, 6) array when no confirmed tracks exist.
        """
        self.frame_count += 1

        # ── step 1: predict all tracks forward ────────────────────────────
        for t in self.tracks:
            t.predict()

        if len(detections) == 0:
            self.tracks = [
                t for t in self.tracks if t.time_since_update <= self.max_age
            ]
            return np.empty((0, 6), dtype=np.float32)

        dets = np.array(detections, dtype=np.float32)
        scores = dets[:, 4]
        bboxes = dets[:, :4]

        # ── step 2: split detections by confidence tier ───────────────────
        high_mask = scores >= self.track_high_thresh
        low_mask = (scores >= self.track_low_thresh) & ~high_mask

        high_bboxes = bboxes[high_mask]
        high_scores = scores[high_mask]
        low_bboxes = bboxes[low_mask]
        low_scores = scores[low_mask]

        # ── step 3: extract ReID embeddings (single pass, then split) ─────
        all_feats = self._extract_features(frame, bboxes)
        high_feats = all_feats[high_mask]
        low_feats = all_feats[low_mask]

        # ── step 4: first association — all tracks ↔ high-conf dets ───────
        matched1, unmatched_t1, unmatched_d1 = self._associate(
            self.tracks, high_bboxes, high_feats, use_appearance=True
        )
        for ti, di in matched1:
            self.tracks[ti].update(high_bboxes[di], high_scores[di], high_feats[di])

        # ── step 5: second association — remaining tracks ↔ low-conf dets ─
        remaining = [self.tracks[i] for i in unmatched_t1]
        unmatched_d2 = list(range(len(low_bboxes)))
        if remaining and len(low_bboxes) > 0:
            matched2, _, unmatched_d2 = self._associate(
                remaining, low_bboxes, low_feats, use_appearance=False
            )
            for ri, di in matched2:
                remaining[ri].update(
                    low_bboxes[di],
                    low_scores[di],
                    None,
                )

        # ── step 6: remove tracks that exceeded max_age ───────────────────
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_age
        ]

        # ── step 7: initialise new tracks from high-conf unmatched dets ───
        for di in unmatched_d1:
            if high_scores[di] >= self.new_track_thresh and self._should_spawn_track(high_bboxes[di]):
                self.tracks.append(
                    Track(high_bboxes[di], high_scores[di], high_feats[di])
                )

        # Optional low-confidence births to improve recall on tiny/weak objects.
        for di in unmatched_d2:
            if low_scores[di] >= self.low_new_track_thresh and self._should_spawn_track(low_bboxes[di]):
                self.tracks.append(
                    Track(low_bboxes[di], low_scores[di], low_feats[di])
                )

        # ── step 8: collect output (confirmed tracks only) ────────────────
        out = []
        for t in self.tracks:
            confirmed = (
                t.time_since_update == 0
                and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
            )
            if confirmed:
                box = t.get_state()
                out.append([box[0], box[1], box[2], box[3], t.track_id, t.score])

        return (
            np.array(out, dtype=np.float32)
            if out
            else np.empty((0, 6), dtype=np.float32)
        )

    def reset(self) -> None:
        """Reset all tracker state.  Call between independent sequences."""
        self.tracks.clear()
        self.frame_count = 0
        KalmanBoxTracker.reset_count()


# ═══════════════════════════════════════════════════════════════════════════
# Quick demo — run with:  python botsort.py --images-dir path/to/sequence
# Requires:  opencv-python  (pip install opencv-python)
# EG (short): python botsort.py "data/VisDrone2019-MOT-train/sequences/uav0000013_00000_v"
# EG (full):  python botsort.py --images-dir "data/VisDrone2019-MOT-train/sequences/uav0000013_00000_v" --model siamese_final.pth --out-dir "runs/uav0000013_00000_v" --out-mp4 "runs/uav0000013_00000_v.mp4" --fps 30 --show
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import cv2

    def list_sequence_images(images_dir: str) -> List[str]:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        paths = [
            str(path) for path in sorted(Path(images_dir).iterdir())
            if path.is_file() and path.suffix.lower() in valid_exts
        ]
        if not paths:
            raise SystemExit(f"No image files found in: {images_dir}")
        return paths

    parser = argparse.ArgumentParser(description="BotSort demo with Siamese ReID")
    parser.add_argument(
        "sequence",
        nargs="?",
        default="",
        help="Optional image-sequence folder path (short mode)",
    )
    parser.add_argument(
        "--images-dir",
        default="",
        help="Path to an image-sequence folder (for example a VisDrone sequence)",
    )
    parser.add_argument("--model", default="siamese_final.pth",
                        help="Path to trained Siamese weights")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="YOLO detection confidence threshold")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional folder to save annotated output frames",
    )
    parser.add_argument(
        "--out-mp4",
        default="",
        help="Optional output mp4 file path",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS for the optional output mp4",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames in a window while processing",
    )
    args = parser.parse_args()

    sequence_dir = args.sequence or args.images_dir
    if not sequence_dir:
        raise SystemExit("Provide a sequence path either as positional argument or with --images-dir")

    sequence_name = Path(sequence_dir).name
    if not args.out_dir:
        args.out_dir = os.path.join("runs", sequence_name)
    if not args.out_mp4:
        args.out_mp4 = os.path.join("runs", f"{sequence_name}.mp4")

    # YOLO detector (Ultralytics) — install with: pip install ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed.  pip install ultralytics")

    detector = YOLO("yolov8n.pt")
    tracker = BotSort(model_path=args.model)

    image_paths = list_sequence_images(sequence_dir)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    writer = None
    if args.out_mp4:
        out_parent = Path(args.out_mp4).parent
        if str(out_parent) not in {"", "."}:
            out_parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    id_colors: dict = {}

    for image_path in image_paths:
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"[Warn] Skipping unreadable image: {image_path}")
            continue

        # Convert BGR → RGB for the tracker
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Run YOLO detection (people class = 0; remove filter for all classes)
        results = detector(bgr, conf=args.conf, verbose=False)[0]
        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0])
            dets.append([x1, y1, x2, y2, score])

        tracks = tracker.update(rgb, np.array(dets) if dets else np.empty((0, 5)))

        # Draw
        for row in tracks:
            x1, y1, x2, y2, tid, score = row
            tid = int(tid)
            if tid not in id_colors:
                id_colors[tid] = tuple(int(c) for c in np.random.randint(100, 255, 3))
            color = id_colors[tid]
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(bgr, f"ID {tid}  {score:.2f}",
                        (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if args.out_dir:
            output_path = os.path.join(args.out_dir, os.path.basename(image_path))
            cv2.imwrite(output_path, bgr)

        if args.out_mp4:
            if writer is None:
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    args.out_mp4,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise SystemExit(f"Cannot open output mp4 for writing: {args.out_mp4}")
            writer.write(bgr)

        if args.show:
            cv2.imshow("BotSort + Siamese ReID", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    if writer is not None:
        writer.release()
