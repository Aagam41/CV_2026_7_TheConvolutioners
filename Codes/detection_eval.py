"""
detection_eval.py
-----------------
Evaluate object DETECTION quality on VisDrone-MOT sequences (no tracker
involved). Prints per-sequence metrics and an overall summary, plus
saves a CSV. Supports plain Ultralytics YOLO and the SAHI sliced
detector from sahi_detect.py.

Metrics reported:
    Precision, Recall, F1
    AP@0.5, AP@0.75, mAP@[.5:.95]   (COCO-style, all-class average)
    AR@100                            (max-detections=100)
    Optional per-class AP@0.5
    Mean inference time per frame

Usage examples:
    # Plain YOLO11n
    python detection_eval.py --dataset /path/to/dataset \\
        --split VisDrone2019-MOT-val \\
        --weights yolo11n.pt \\
        --output det_eval/yolo11n

    # YOLO11n fine-tuned on VisDrone
    python detection_eval.py --dataset /path/to/dataset \\
        --split VisDrone2019-MOT-val \\
        --weights yolo11_visdrone.pt \\
        --output det_eval/yolo11_visdrone

    # YOLOv8n with SAHI
    python detection_eval.py --dataset /path/to/dataset \\
        --split VisDrone2019-MOT-val \\
        --weights yolov8n.pt \\
        --sahi --slice_h 640 --slice_w 640 --overlap_h 0.2 --overlap_w 0.2 \\
        --output det_eval/yolov8n_sahi

Notes on class mapping:
    VisDrone categories  : 1=pedestrian 2=people 3=bicycle 4=car 5=van
                           6=truck 7=tricycle 8=awning-tricycle 9=bus
                           10=motor   (0=ignored, 11=others)
    COCO model mapping defaults are below; pass --class_map "coco_id:vd_id,..."
    to override.
"""
import os
import json
import time
import argparse
from collections import defaultdict
import numpy as np
import cv2
from tqdm import tqdm

from ultralytics import YOLO
from sahi_detect import SAHIDetector


# ============================ Drawing ============================
# Distinct color per VisDrone class
_CLASS_COLORS = {
    1: (255, 64, 64),     # pedestrian   — red
    2: (255, 128, 0),     # people       — orange
    3: (255, 255, 0),     # bicycle      — yellow
    4: (0, 255, 0),       # car          — green
    5: (0, 200, 255),     # van          — cyan
    6: (255, 0, 255),     # truck        — magenta
    7: (0, 128, 255),     # tricycle     — blue
    8: (128, 0, 255),     # awning-tri   — purple
    9: (255, 0, 128),     # bus          — pink
    10: (0, 255, 128),    # motor        — teal
}


def _color_for(c):
    return _CLASS_COLORS.get(int(c),
                             (int((c * 73) % 255),
                              int((c * 113 + 50) % 255),
                              int((c * 53 + 100) % 255)))


def draw_predictions(img, dets, gt_xyxy=None, gt_cls=None, show_gt=False,
                     show_score=True, gt_match_iou=0.5):
    """
    Draw predictions and (optionally) GT boxes on a copy of img.

    GT boxes are color-coded by detection status (very useful for finding
    missed detections at a glance):
        GREEN = GT matched by a prediction of the same class with IoU >= gt_match_iou
        RED   = GT missed (no prediction of the same class above the IoU threshold)

    Predictions are drawn with per-VisDrone-class colors and labeled with
    "<class> <score>".
    """
    out = img.copy()

    # ---- GT boxes with match-status coloring ----
    if show_gt and gt_xyxy is not None and len(gt_xyxy) > 0:
        matched = np.zeros(len(gt_xyxy), dtype=bool)
        if len(dets) > 0:
            p_box = dets[:, :4]
            p_cls = dets[:, 5].astype(np.int32)
            iou = box_iou_matrix(p_box, gt_xyxy)   # P x G
            # GT is "matched" if any same-class prediction has IoU >= thresh
            for gi in range(len(gt_xyxy)):
                same_cls = np.where(p_cls == int(gt_cls[gi]))[0]
                if len(same_cls) and iou[same_cls, gi].max() >= gt_match_iou:
                    matched[gi] = True
        for gi, b in enumerate(gt_xyxy):
            x1, y1, x2, y2 = map(int, b)
            col = (0, 200, 0) if matched[gi] else (0, 0, 255)  # BGR
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 1)
            if not matched[gi]:
                # Tag missed GT with class name so it's obvious why it matters
                tag = 'MISS:' + VD_CLASSES.get(int(gt_cls[gi]),
                                               str(int(gt_cls[gi])))
                (tw, th), _ = cv2.getTextSize(tag,
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              0.4, 1)
                cv2.rectangle(out, (x1, y2),
                              (x1 + tw + 4, y2 + th + 4), (0, 0, 255), -1)
                cv2.putText(out, tag, (x1 + 2, y2 + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1)

    # ---- Predictions (per-class color) ----
    for d in dets:
        x1, y1, x2, y2, sc, c = d
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        col = _color_for(c)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cls_name = VD_CLASSES.get(int(c), str(int(c)))
        label = f'{cls_name} {sc:.2f}' if show_score else cls_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)),
                      (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, max(th, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # ---- Legend strip (top-left) ----
    if show_gt:
        legend_h = 22
        legend = np.zeros((legend_h, 360, 3), dtype=np.uint8)
        cv2.rectangle(legend, (4, 4), (16, 18), (0, 200, 0), -1)
        cv2.putText(legend, 'GT matched', (22, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(legend, (130, 4), (142, 18), (0, 0, 255), -1)
        cv2.putText(legend, 'GT MISSED', (148, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(legend, (250, 4), (262, 18), (0, 255, 255), -1)
        cv2.putText(legend, 'pred', (268, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        h_avail = min(legend_h, out.shape[0])
        w_avail = min(legend.shape[1], out.shape[1])
        out[:h_avail, :w_avail] = legend[:h_avail, :w_avail]

    return out


# ============================ VisDrone GT IO ============================
# VisDrone categories used for evaluation (drop 0 ignored, 11 others)
VD_CLASSES = {
    1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
    6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor',
}
IGNORE_CATS = {0, 11}


# Default mapping: COCO model class id -> VisDrone class id
# Coverage is partial: we map the COCO classes that have a VisDrone equivalent.
# Anything not in this dict gets dropped from predictions before scoring.
DEFAULT_COCO_TO_VD = {
    0: 1,   # person -> pedestrian
    1: 3,   # bicycle -> bicycle
    2: 4,   # car -> car
    3: 10,  # motorcycle -> motor
    5: 9,   # bus -> bus
    7: 6,   # truck -> truck
}


def parse_class_map(spec):
    """Parse 'coco_id:vd_id,coco_id:vd_id,...' into {int:int}."""
    if not spec:
        return None
    out = {}
    for kv in spec.split(','):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(':')
        out[int(k)] = int(v)
    return out


def load_gt(ann_file):
    """{frame: [(x,y,w,h,cat,truncation,occlusion), ...]}.
    Drops ignored categories and score==0 rows.
    """
    data = {}
    if not os.path.isfile(ann_file):
        return data
    with open(ann_file) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8:
                continue
            frame = int(p[0])
            x, y, w, h = map(float, p[2:6])
            score = int(p[6])
            cat = int(p[7])
            if score == 0 or cat in IGNORE_CATS:
                continue
            if w < 1 or h < 1:
                continue
            data.setdefault(frame, []).append(
                (x, y, w, h, cat))
    return data


def list_frames(seq_dir):
    files = sorted([f for f in os.listdir(seq_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return [os.path.join(seq_dir, f) for f in files]


# ============================ Detector wrappers ============================
class PlainYOLO:
    def __init__(self, weights, conf, iou, imgsz, device=None, classes=None):
        self.model = YOLO(weights)
        self.conf = conf; self.iou = iou; self.imgsz = imgsz
        self.device = device; self.classes = classes

    def __call__(self, img):
        res = self.model.predict(img, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, verbose=False,
                                 classes=self.classes, device=self.device)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        b = res.boxes.xyxy.cpu().numpy()
        s = res.boxes.conf.cpu().numpy()
        c = res.boxes.cls.cpu().numpy()
        return np.concatenate([b, s[:, None], c[:, None]],
                              axis=1).astype(np.float32)


class SAHIYOLO:
    def __init__(self, weights, conf, iou, imgsz, device=None, classes=None,
                 slice_h=640, slice_w=640, overlap_h=0.2, overlap_w=0.2,
                 include_full=True, batch=8):
        self.classes = classes
        model = YOLO(weights)
        self.det = SAHIDetector(model, slice_h=slice_h, slice_w=slice_w,
                                overlap_h=overlap_h, overlap_w=overlap_w,
                                conf=conf, iou=iou,
                                include_full_image=include_full,
                                full_image_imgsz=imgsz, batch=batch,
                                device=device)

    def __call__(self, img):
        return self.det(img, classes=self.classes)


# ============================ Box helpers ============================
def xywh_to_xyxy(boxes):
    """N×4 (x,y,w,h) -> N×4 (x1,y1,x2,y2)."""
    if len(boxes) == 0:
        return boxes
    out = boxes.copy().astype(np.float32)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def box_iou_matrix(a, b):
    """N×4 vs M×4 (xyxy) -> N×M IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.astype(np.float32); b = b.astype(np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    iou = np.zeros((len(a), len(b)), dtype=np.float32)
    for i in range(len(a)):
        x1 = np.maximum(a[i, 0], b[:, 0])
        y1 = np.maximum(a[i, 1], b[:, 1])
        x2 = np.minimum(a[i, 2], b[:, 2])
        y2 = np.minimum(a[i, 3], b[:, 3])
        iw = (x2 - x1).clip(0); ih = (y2 - y1).clip(0)
        inter = iw * ih
        union = area_a[i] + area_b - inter
        iou[i] = np.where(union > 0, inter / union, 0.0)
    return iou


# ============================ AP computation ============================
def compute_ap(recall, precision):
    """COCO-style 101-point interpolation AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # 101 recall thresholds
    rec_thr = np.linspace(0, 1, 101)
    ap = 0.0
    for r in rec_thr:
        idx = np.searchsorted(mrec, r, side='left')
        p = mpre[idx] if idx < len(mpre) else 0.0
        ap += p
    return ap / 101.0


def compute_ap_for_class(records, n_gt, iou_thresholds):
    """
    records: list of (score, is_tp_per_iou_bool_array) from all images
             where is_tp_per_iou_bool_array has shape (len(iou_thresholds),)
    n_gt: total number of GT instances of this class across all images
    Returns: list of AP per iou threshold.
    """
    if n_gt == 0 or len(records) == 0:
        return [0.0] * len(iou_thresholds)
    records.sort(key=lambda x: -x[0])  # by score desc
    tp_per_iou = np.array([r[1] for r in records], dtype=np.float32)  # N×T
    fp_per_iou = 1.0 - tp_per_iou

    aps = []
    for ti in range(len(iou_thresholds)):
        tp_cum = np.cumsum(tp_per_iou[:, ti])
        fp_cum = np.cumsum(fp_per_iou[:, ti])
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        aps.append(compute_ap(recall, precision))
    return aps


def compute_ar(records, n_gt, iou_thresholds, max_dets=100):
    """Average Recall at max_dets. records already sorted by score desc upstream
    is fine; we'll re-sort to be safe and per-image limit."""
    if n_gt == 0 or len(records) == 0:
        return 0.0
    # max_dets is per-image; we approximated via per-image limit at match time.
    records_sorted = sorted(records, key=lambda x: -x[0])
    tp_per_iou = np.array([r[1] for r in records_sorted], dtype=np.float32)
    recalls = []
    for ti in range(len(iou_thresholds)):
        tp = tp_per_iou[:, ti].sum()
        recalls.append(tp / n_gt)
    return float(np.mean(recalls))


# ============================ Per-image matching ============================
def match_image(pred_xyxy, pred_scores, pred_cls,
                gt_xyxy, gt_cls, iou_thresholds):
    """
    Returns dict: {class_id: list of (score, tp_per_iou_array)}
    Greedy match per IoU threshold (COCO-style: highest score first).
    """
    out = defaultdict(list)
    classes = set(pred_cls.tolist()) | set(gt_cls.tolist())
    for c in classes:
        p_idx = np.where(pred_cls == c)[0]
        g_idx = np.where(gt_cls == c)[0]
        if len(p_idx) == 0:
            continue
        # Sort preds by score desc
        order = p_idx[np.argsort(-pred_scores[p_idx])]
        p_boxes = pred_xyxy[order]
        p_scores = pred_scores[order]
        if len(g_idx) == 0:
            for s in p_scores:
                out[c].append((float(s),
                               np.zeros(len(iou_thresholds), dtype=np.float32)))
            continue
        g_boxes = gt_xyxy[g_idx]
        iou = box_iou_matrix(p_boxes, g_boxes)   # P×G
        for ti, thr in enumerate(iou_thresholds):
            taken = np.zeros(len(g_boxes), dtype=bool)
            for pi in range(len(p_boxes)):
                # find best unmatched GT above threshold
                ious = iou[pi].copy()
                ious[taken] = -1
                best = ious.argmax() if len(ious) else -1
                tp = (best >= 0) and (ious[best] >= thr)
                if tp:
                    taken[best] = True
                # We need to record per-pred per-iou TP. Build later.
        # Easier: rebuild per-pred record across all thresholds at once.
        records = []
        # Per IoU threshold, repeat the greedy match (consistent with COCO)
        tp_table = np.zeros((len(p_boxes), len(iou_thresholds)),
                            dtype=np.float32)
        for ti, thr in enumerate(iou_thresholds):
            taken = np.zeros(len(g_boxes), dtype=bool)
            for pi in range(len(p_boxes)):
                ious = iou[pi].copy()
                ious[taken] = -1
                if len(ious) == 0:
                    continue
                best = ious.argmax()
                if ious[best] >= thr:
                    taken[best] = True
                    tp_table[pi, ti] = 1.0
        for pi in range(len(p_boxes)):
            out[c].append((float(p_scores[pi]), tp_table[pi]))
    return out


# ============================ Main eval routine ============================
def evaluate_split(detector, dataset, split, output_dir,
                   class_map, score_thresh_for_pr=0.25,
                   iou_for_pr=0.5,
                   max_frames=None,
                   save_video=False, video_fps=30, draw_score_thresh=0.25,
                   draw_gt=True, gt_match_iou=0.5):
    seq_root = os.path.join(dataset, split, 'sequences')
    ann_root = os.path.join(dataset, split, 'annotations')
    assert os.path.isdir(seq_root), seq_root
    assert os.path.isdir(ann_root), ann_root

    iou_thresholds = np.linspace(0.5, 0.95, 10)   # COCO-style
    os.makedirs(output_dir, exist_ok=True)
    vid_dir = os.path.join(output_dir, 'videos')
    if save_video:
        os.makedirs(vid_dir, exist_ok=True)

    all_records = defaultdict(list)   # class -> list of (score, tp_per_iou)
    all_n_gt = defaultdict(int)
    seq_summaries = []

    seqs = sorted([d for d in os.listdir(seq_root)
                   if os.path.isdir(os.path.join(seq_root, d))])

    total_frames = 0
    total_inf_time = 0.0

    for seq in tqdm(seqs, desc='Sequences'):
        seq_dir = os.path.join(seq_root, seq)
        gt = load_gt(os.path.join(ann_root, f'{seq}.txt'))
        if not gt:
            continue
        frames = list_frames(seq_dir)
        if max_frames:
            frames = frames[:max_frames]

        seq_records = defaultdict(list)
        seq_n_gt = defaultdict(int)
        seq_tp = 0; seq_fp = 0; seq_fn = 0
        seq_inf = 0.0

        writer = None
        out_vid_path = (os.path.join(vid_dir, f'{seq}.mp4')
                        if save_video else None)

        for fidx, fpath in enumerate(frames, 1):
            img = cv2.imread(fpath)
            if img is None:
                continue

            t0 = time.time()
            dets = detector(img)            # Nx6 xyxy,score,cls
            seq_inf += time.time() - t0

            # GT for this frame
            g = gt.get(fidx, [])
            gt_xyxy = (xywh_to_xyxy(np.array([[q[0], q[1], q[2], q[3]]
                                              for q in g], dtype=np.float32))
                       if g else np.empty((0, 4), dtype=np.float32))
            gt_cls = (np.array([q[4] for q in g], dtype=np.int32)
                      if g else np.empty((0,), dtype=np.int32))

            # Map predicted COCO classes → VisDrone classes; drop unmapped
            if len(dets) > 0:
                if class_map is not None:
                    keep = []
                    for i, c in enumerate(dets[:, 5].astype(int)):
                        if c in class_map:
                            keep.append(i)
                    dets = dets[keep]
                    if len(dets) > 0:
                        dets[:, 5] = np.array(
                            [class_map[int(c)] for c in dets[:, 5]],
                            dtype=np.float32)
            pred_xyxy = dets[:, :4] if len(dets) else np.empty((0, 4),
                                                               dtype=np.float32)
            pred_scores = dets[:, 4] if len(dets) else np.empty((0,),
                                                                dtype=np.float32)
            pred_cls = (dets[:, 5].astype(np.int32) if len(dets)
                        else np.empty((0,), dtype=np.int32))

            # Track per-class GT counts
            for c in gt_cls.tolist():
                seq_n_gt[c] += 1
                all_n_gt[c] += 1

            # Per-image AP records
            recs = match_image(pred_xyxy, pred_scores, pred_cls,
                               gt_xyxy, gt_cls, iou_thresholds)
            for c, lst in recs.items():
                seq_records[c].extend(lst)
                all_records[c].extend(lst)

            # Quick precision/recall/F1 at fixed score & IoU
            keep = pred_scores >= score_thresh_for_pr
            if keep.any():
                p_boxes = pred_xyxy[keep]
                p_cls = pred_cls[keep]
            else:
                p_boxes = np.empty((0, 4), dtype=np.float32)
                p_cls = np.empty((0,), dtype=np.int32)

            tp_img = 0
            matched_gt = np.zeros(len(gt_xyxy), dtype=bool)
            if len(p_boxes) and len(gt_xyxy):
                iou = box_iou_matrix(p_boxes, gt_xyxy)
                # Greedy by class
                for ci in range(len(p_boxes)):
                    cand = np.where(
                        (gt_cls == p_cls[ci]) & ~matched_gt)[0]
                    if len(cand) == 0:
                        continue
                    best = cand[iou[ci, cand].argmax()]
                    if iou[ci, best] >= iou_for_pr:
                        matched_gt[best] = True
                        tp_img += 1
            fp_img = len(p_boxes) - tp_img
            fn_img = len(gt_xyxy) - tp_img
            seq_tp += tp_img; seq_fp += fp_img; seq_fn += fn_img

            # Write annotated frame
            if save_video:
                draw_keep = pred_scores >= draw_score_thresh
                draw_dets = (np.concatenate(
                    [pred_xyxy[draw_keep],
                     pred_scores[draw_keep, None],
                     pred_cls[draw_keep, None].astype(np.float32)],
                    axis=1) if draw_keep.any()
                    else np.empty((0, 6), dtype=np.float32))
                vis = draw_predictions(img, draw_dets,
                                       gt_xyxy=gt_xyxy, gt_cls=gt_cls,
                                       show_gt=draw_gt, show_score=True,
                                       gt_match_iou=gt_match_iou)
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(out_vid_path, fourcc,
                                             video_fps, (w, h))
                writer.write(vis)

        if writer is not None:
            writer.release()

        # Sequence-level AP
        seq_aps_50 = []
        seq_aps_75 = []
        seq_aps_5095 = []
        for c in seq_records.keys():
            aps = compute_ap_for_class(seq_records[c], seq_n_gt[c],
                                       iou_thresholds)
            seq_aps_50.append(aps[0])
            seq_aps_75.append(aps[5])     # 0.75 is the 6th (idx 5)
            seq_aps_5095.append(np.mean(aps))

        prec = seq_tp / max(seq_tp + seq_fp, 1)
        rec = seq_tp / max(seq_tp + seq_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        n_frames = len(frames)
        total_frames += n_frames
        total_inf_time += seq_inf

        seq_summaries.append({
            'sequence': seq,
            'frames': n_frames,
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'AP@0.5': round(float(np.mean(seq_aps_50)) if seq_aps_50 else 0, 4),
            'AP@0.75': round(float(np.mean(seq_aps_75)) if seq_aps_75 else 0, 4),
            'mAP@.5:.95': round(float(np.mean(seq_aps_5095)) if seq_aps_5095
                                else 0, 4),
            'inf_ms_per_frame': round(1000 * seq_inf / max(n_frames, 1), 2),
        })

    # ── Overall metrics across the whole split ──────────────────────────
    overall_aps_50 = []; overall_aps_75 = []; overall_aps_5095 = []
    per_class_ap50 = {}
    overall_records_flat = []
    overall_n_gt_total = 0
    for c, recs in all_records.items():
        aps = compute_ap_for_class(recs, all_n_gt[c], iou_thresholds)
        overall_aps_50.append(aps[0])
        overall_aps_75.append(aps[5])
        overall_aps_5095.append(np.mean(aps))
        per_class_ap50[VD_CLASSES.get(c, str(c))] = round(float(aps[0]), 4)
        overall_records_flat.extend(recs)
        overall_n_gt_total += all_n_gt[c]

    ar100 = compute_ar(overall_records_flat, overall_n_gt_total,
                       iou_thresholds, max_dets=100)

    # Overall PR/F1: aggregate sequence TP/FP/FN
    tot_tp = sum(s['precision'] * 0 for s in seq_summaries)  # placeholder
    # Better: re-aggregate from per-sequence (we lost raw counts; recompute
    # from precision/recall is messy). Use micro from records at IoU=0.5.
    # Approximation: micro precision/recall using AP@0.5 records and a
    # threshold of `score_thresh_for_pr` is not quite the same. We'll
    # report mean of per-sequence values plus AP-derived numbers.
    mean_prec = float(np.mean([s['precision'] for s in seq_summaries])) \
        if seq_summaries else 0.0
    mean_rec = float(np.mean([s['recall'] for s in seq_summaries])) \
        if seq_summaries else 0.0
    mean_f1 = float(np.mean([s['f1'] for s in seq_summaries])) \
        if seq_summaries else 0.0

    overall = {
        'split': split,
        'sequences': len(seq_summaries),
        'frames': total_frames,
        'mean_precision': round(mean_prec, 4),
        'mean_recall': round(mean_rec, 4),
        'mean_f1': round(mean_f1, 4),
        'AP@0.5': round(float(np.mean(overall_aps_50))
                        if overall_aps_50 else 0, 4),
        'AP@0.75': round(float(np.mean(overall_aps_75))
                         if overall_aps_75 else 0, 4),
        'mAP@.5:.95': round(float(np.mean(overall_aps_5095))
                            if overall_aps_5095 else 0, 4),
        'AR@100': round(float(ar100), 4),
        'inf_ms_per_frame': round(1000 * total_inf_time
                                  / max(total_frames, 1), 2),
        'fps': round(total_frames / max(total_inf_time, 1e-9), 2),
        'per_class_AP@0.5': per_class_ap50,
        'score_threshold_for_PR': score_thresh_for_pr,
        'iou_threshold_for_PR': iou_for_pr,
    }

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'overall.json'), 'w') as f:
        json.dump(overall, f, indent=2)

    csv_path = os.path.join(output_dir, 'per_sequence.csv')
    if seq_summaries:
        cols = list(seq_summaries[0].keys())
        with open(csv_path, 'w') as f:
            f.write(','.join(cols) + '\n')
            for s in seq_summaries:
                f.write(','.join(str(s[c]) for c in cols) + '\n')

    # Print
    print('\n' + '=' * 78)
    print(f'  DETECTION EVAL — {split}')
    print('=' * 78)
    print(f'{"sequence":<32} {"frames":>7} {"P":>6} {"R":>6} {"F1":>6} '
          f'{"AP50":>6} {"AP75":>6} {"mAP":>6} {"ms/fr":>7}')
    print('-' * 78)
    for s in seq_summaries:
        print(f'{s["sequence"][:32]:<32} {s["frames"]:>7} '
              f'{s["precision"]:>6.3f} {s["recall"]:>6.3f} {s["f1"]:>6.3f} '
              f'{s["AP@0.5"]:>6.3f} {s["AP@0.75"]:>6.3f} '
              f'{s["mAP@.5:.95"]:>6.3f} {s["inf_ms_per_frame"]:>7.1f}')
    print('-' * 78)
    print(f'{"OVERALL":<32} {overall["frames"]:>7} '
          f'{overall["mean_precision"]:>6.3f} '
          f'{overall["mean_recall"]:>6.3f} '
          f'{overall["mean_f1"]:>6.3f} '
          f'{overall["AP@0.5"]:>6.3f} '
          f'{overall["AP@0.75"]:>6.3f} '
          f'{overall["mAP@.5:.95"]:>6.3f} '
          f'{overall["inf_ms_per_frame"]:>7.1f}')
    print('=' * 78)
    print(f'AR@100: {overall["AR@100"]:.4f}    FPS: {overall["fps"]:.2f}')
    print('\nPer-class AP@0.5:')
    for cls_name, ap in sorted(per_class_ap50.items(),
                               key=lambda x: -x[1]):
        print(f'  {cls_name:<20} {ap:.4f}')
    print(f'\nSaved: {os.path.join(output_dir, "overall.json")}')
    print(f'Saved: {csv_path}')


# ================================== CLI ==================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', default='VisDrone2019-MOT-val')
    ap.add_argument('--weights', required=True,
                    help='YOLO .pt (e.g. yolov8n.pt, yolo11n.pt, '
                         'yolo11_visdrone.pt)')
    ap.add_argument('--output', required=True,
                    help='Output directory for metrics')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--device', default=None,
                    help='cuda:0, cpu, etc. (default: auto)')
    ap.add_argument('--classes', type=int, nargs='*', default=None,
                    help='Optional YOLO class filter applied at predict()')
    ap.add_argument('--class_map', type=str, default=None,
                    help='Override COCO->VisDrone mapping. Format: '
                         '"0:1,1:3,2:4,...". Pass empty to disable mapping '
                         '(use predicted class ids as-is, useful for '
                         'VisDrone-finetuned weights whose classes already '
                         'match VisDrone ids).')
    ap.add_argument('--no_class_map', action='store_true',
                    help='Use predicted class ids directly (for VisDrone-'
                         'finetuned weights).')
    ap.add_argument('--score_thresh_for_pr', type=float, default=0.25,
                    help='Confidence threshold for the P/R/F1 columns')
    ap.add_argument('--iou_for_pr', type=float, default=0.5,
                    help='IoU threshold for the P/R/F1 columns')
    ap.add_argument('--max_frames', type=int, default=None,
                    help='Limit frames per sequence (for quick smoke tests)')
    # Video output
    ap.add_argument('--save_video', action='store_true',
                    help='Write annotated MP4 per sequence to <output>/videos/')
    ap.add_argument('--video_fps', type=int, default=30)
    ap.add_argument('--draw_score_thresh', type=float, default=0.25,
                    help='Min score to draw a box in the saved video')
    ap.add_argument('--no_gt', action='store_true',
                    help='Do NOT draw GT boxes on saved videos '
                         '(GT is on by default with --save_video)')
    ap.add_argument('--gt_match_iou', type=float, default=0.5,
                    help='IoU threshold for GT to count as matched '
                         '(green) vs missed (red) on saved videos')
    # SAHI
    ap.add_argument('--sahi', action='store_true')
    ap.add_argument('--slice_h', type=int, default=640)
    ap.add_argument('--slice_w', type=int, default=640)
    ap.add_argument('--overlap_h', type=float, default=0.2)
    ap.add_argument('--overlap_w', type=float, default=0.2)
    ap.add_argument('--no_full_image', action='store_true')
    ap.add_argument('--sahi_batch', type=int, default=8)
    args = ap.parse_args()

    # Class map
    if args.no_class_map:
        class_map = None
        print('[class_map] DISABLED — predicted class ids used as-is '
              '(expect for VisDrone-finetuned weights).')
    elif args.class_map:
        class_map = parse_class_map(args.class_map)
        print(f'[class_map] custom: {class_map}')
    else:
        class_map = DEFAULT_COCO_TO_VD
        print(f'[class_map] default COCO->VisDrone: {class_map}')

    # Build detector
    if args.sahi:
        detector = SAHIYOLO(args.weights, conf=args.conf, iou=args.iou,
                            imgsz=args.imgsz, device=args.device,
                            classes=args.classes,
                            slice_h=args.slice_h, slice_w=args.slice_w,
                            overlap_h=args.overlap_h, overlap_w=args.overlap_w,
                            include_full=not args.no_full_image,
                            batch=args.sahi_batch)
        print(f'[detector] SAHI YOLO  weights={args.weights}  '
              f'slice=({args.slice_h},{args.slice_w}) '
              f'overlap=({args.overlap_h},{args.overlap_w}) '
              f'full_image={not args.no_full_image}')
    else:
        detector = PlainYOLO(args.weights, conf=args.conf, iou=args.iou,
                             imgsz=args.imgsz, device=args.device,
                             classes=args.classes)
        print(f'[detector] YOLO  weights={args.weights}  imgsz={args.imgsz}')

    evaluate_split(detector, args.dataset, args.split, args.output,
                   class_map=class_map,
                   score_thresh_for_pr=args.score_thresh_for_pr,
                   iou_for_pr=args.iou_for_pr,
                   max_frames=args.max_frames,
                   save_video=args.save_video,
                   video_fps=args.video_fps,
                   draw_score_thresh=args.draw_score_thresh,
                   draw_gt=(not args.no_gt),
                   gt_match_iou=args.gt_match_iou)


if __name__ == '__main__':
    main()
