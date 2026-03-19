#!/usr/bin/env python3
"""
eval_visdrone.py
----------------
Step 1 (optional): Run tracking on all val sequences to produce MOT .txt files.
Step 2:            Evaluate the .txt results against VisDrone GT annotations
                   and report MOTA, MOTP, IDF1, HOTA, Precision, Recall,
                   FPS, and processing time — per sequence and overall.

Typical workflow
----------------
# A) Run tracking + evaluate in one shot:
python eval_visdrone.py \
    --run-tracking \
    --tracker botsort \
    --yolo-model yolov8n.pt \
    --reid-model osnet_x0_25_msmt17.pt \
    --device cuda:0

# B) Evaluate already-computed results (skip re-tracking):
python eval_visdrone.py \
    --results-dir runs/track/botsort \
    --tracker botsort

# C) Compare two trackers side-by-side (evaluate only):
python eval_visdrone.py --results-dir runs/track/botsort  --tracker botsort
python eval_visdrone.py --results-dir runs/track/bytetrack --tracker bytetrack

VisDrone GT annotation format (per line)
-----------------------------------------
<frame>,<id>,<left>,<top>,<width>,<height>,<score>,<category>,<truncation>,<occlusion>
  score=0  → ignored region  (excluded from eval)
  category=0 → ignored class (excluded from eval)

Tracker result format (MOT, what track_visdrone.py writes)
-----------------------------------------------------------
<frame>,<id>,<left>,<top>,<width>,<height>,<conf>,-1,-1,-1

Dependencies
------------
pip install motmetrics pandas tabulate
"""

from __future__ import annotations

import argparse
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── VisDrone ignored categories ───────────────────────────────────────────
# category 0 = ignored region, category 11 = others  → skip in eval
IGNORED_CATEGORIES = {0, 11}


# ---------------------------------------------------------------------------
# GT / prediction loaders
# ---------------------------------------------------------------------------

def load_gt(ann_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a VisDrone annotation .txt file.

    Returns
    -------
    gt_df   : rows with score==1 and valid category  (used for evaluation)
    ign_df  : rows with score==0 or ignored category  (used to mask FP)

    Columns: frame, id, left, top, width, height
    """
    if not ann_path.exists():
        raise FileNotFoundError(f"GT annotation not found: {ann_path}")

    rows = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            frame    = int(parts[0])
            tid      = int(parts[1])
            left     = float(parts[2])
            top      = float(parts[3])
            width    = float(parts[4])
            height   = float(parts[5])
            score    = int(parts[6])
            category = int(parts[7])
            rows.append((frame, tid, left, top, width, height, score, category))

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows, columns=["frame", "id", "left", "top",
                                     "width", "height", "score", "category"])

    gt_df  = df[(df["score"] == 1) & (~df["category"].isin(IGNORED_CATEGORIES))].copy()
    ign_df = df[(df["score"] == 0) |  (df["category"].isin(IGNORED_CATEGORIES))].copy()

    return gt_df, ign_df


def load_pred(pred_path: Path) -> pd.DataFrame:
    """
    Parse a MOT-format tracker result .txt file.

    Columns: frame, id, left, top, width, height, conf
    """
    if not pred_path.exists():
        return pd.DataFrame()

    rows = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame  = int(parts[0])
            tid    = int(parts[1])
            left   = float(parts[2])
            top    = float(parts[3])
            width  = float(parts[4])
            height = float(parts[5])
            conf   = float(parts[6]) if len(parts) > 6 else 1.0
            rows.append((frame, tid, left, top, width, height, conf))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=["frame", "id", "left", "top",
                                       "width", "height", "conf"])


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def bbox_iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute IoU between all pairs of gt and pred boxes.

    Parameters
    ----------
    gt, pred : (N,4) and (M,4) arrays of [left, top, width, height]

    Returns
    -------
    iou : (N, M) float array
    """
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)), dtype=np.float32)

    # Convert to x1y1x2y2
    gt_x1   = gt[:, 0:1];     gt_y1   = gt[:, 1:2]
    gt_x2   = gt_x1 + gt[:, 2:3]; gt_y2 = gt_y1 + gt[:, 3:4]

    pred_x1 = pred[:, 0];     pred_y1 = pred[:, 1]
    pred_x2 = pred_x1 + pred[:, 2]; pred_y2 = pred_y1 + pred[:, 3]

    # Intersection
    ix1 = np.maximum(gt_x1, pred_x1)
    iy1 = np.maximum(gt_y1, pred_y1)
    ix2 = np.minimum(gt_x2, pred_x2)
    iy2 = np.minimum(gt_y2, pred_y2)

    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih

    gt_area   = gt[:, 2:3] * gt[:, 3:4]
    pred_area = pred[:, 2] * pred[:, 3]

    union = gt_area + pred_area - inter
    iou   = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


# ---------------------------------------------------------------------------
# CLEAR MOT metrics (MOTA, MOTP, IDF1) — computed from scratch
# ---------------------------------------------------------------------------

def compute_mot_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute CLEAR MOT metrics: MOTA, MOTP, IDF1, Precision, Recall,
    MT (Mostly Tracked), ML (Mostly Lost), FP, FN, ID Switches.

    Uses motmetrics library for correctness.
    """
    try:
        import motmetrics as mm
    except ImportError:
        raise ImportError("pip install motmetrics")

    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(
        list(gt_df["frame"].unique() if len(gt_df) else []) +
        list(pred_df["frame"].unique() if len(pred_df) else [])
    ))

    for frame in all_frames:
        gt_frame   = gt_df[gt_df["frame"]   == frame] if len(gt_df)   else pd.DataFrame()
        pred_frame = pred_df[pred_df["frame"] == frame] if len(pred_df) else pd.DataFrame()

        gt_ids   = gt_frame["id"].tolist()   if len(gt_frame)   else []
        pred_ids = pred_frame["id"].tolist() if len(pred_frame) else []

        if gt_ids and pred_ids:
            gt_boxes   = gt_frame[["left","top","width","height"]].values
            pred_boxes = pred_frame[["left","top","width","height"]].values
            iou        = bbox_iou_matrix(gt_boxes, pred_boxes)
            # motmetrics wants distance (1 - IoU), with inf where below threshold
            dist = 1.0 - iou
            dist[iou < iou_threshold] = np.inf
        else:
            dist = mm.distances.iou_matrix(
                [], [], max_iou=1 - iou_threshold
            ) if not gt_ids and not pred_ids else \
            np.full((len(gt_ids), len(pred_ids)), np.inf)

        acc.update(gt_ids, pred_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames", "num_matches", "num_switches", "num_false_positives",
            "num_misses", "num_detections", "num_objects", "num_predictions",
            "mota", "motp", "idf1", "precision", "recall",
            "mostly_tracked", "mostly_lost", "partially_tracked",
        ],
        name="sequence",
    )

    r = summary.iloc[0]
    return {
        "MOTA"    : float(r["mota"])       * 100,
        "MOTP"    : float(r["motp"])       * 100,
        "IDF1"    : float(r["idf1"])       * 100,
        "Prec"    : float(r["precision"])  * 100,
        "Recall"  : float(r["recall"])     * 100,
        "MT"      : int(r["mostly_tracked"]),
        "ML"      : int(r["mostly_lost"]),
        "PT"      : int(r["partially_tracked"]),
        "FP"      : int(r["num_false_positives"]),
        "FN"      : int(r["num_misses"]),
        "IDs"     : int(r["num_switches"]),
        "GT_objs" : int(r["num_objects"]),
        "Pred"    : int(r["num_predictions"]),
        "Frames"  : int(r["num_frames"]),
    }


# ---------------------------------------------------------------------------
# Run tracking (calls track_visdrone.py logic directly)
# ---------------------------------------------------------------------------

def run_tracking_on_split(args) -> Dict[str, float]:
    """
    Run the full YOLO+SAHI+BoxMOT pipeline on all val sequences
    and return a dict of {seq_name: fps}.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from detector import YOLOSAHIDetector
    from visdrone_utils import VisDroneSequence, list_sequences, MOTResultWriter, TrailBuffer, draw_tracks
    from track_visdrone import build_tracker

    split_map = {
        "train": "VisDrone2019-MOT-train",
        "val":   "VisDrone2019-MOT-val",
    }
    split_dir = Path(args.dataset_root) / split_map[args.split]
    seq_dirs  = list_sequences(split_dir)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOSAHIDetector(
        model_path=args.yolo_model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        slice_height=args.slice_h,
        slice_width=args.slice_w,
        overlap_height_ratio=args.overlap_h,
        overlap_width_ratio=args.overlap_w,
        model_type=args.model_type,
    )

    from visdrone_utils import make_video_writer
    import cv2 as _cv2

    seq_fps: Dict[str, float] = {}

    for seq_dir in seq_dirs:
        seq     = VisDroneSequence(seq_dir)
        tracker = build_tracker(args.tracker, args.reid_model, args.device, args.half)
        trails  = TrailBuffer(max_len=args.trail_len if args.save_video else 0)
        writer  = MOTResultWriter(out_dir / f"{seq.name}.txt")

        # optional video writer
        video_writer = None
        if args.save_video:
            w, h = seq.frame_size()
            video_writer = make_video_writer(
                out_dir / f"{seq.name}.mp4", w, h, fps=args.fps
            )

        t0 = time.perf_counter()
        for frame_id, frame in seq:
            dets   = detector.detect(frame)
            tracks = tracker.update(dets, frame)
            writer.update(frame_id, tracks)

            if video_writer is not None:
                trails.update(tracks)
                vis = frame.copy()
                draw_tracks(vis, tracks, trails=trails)
                _cv2.putText(
                    vis,
                    f"{seq.name}  {frame_id}/{seq.n_frames}  "
                    f"Tracks:{len(tracks) if tracks is not None else 0}",
                    (6, 18), _cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, _cv2.LINE_AA,
                )
                video_writer.write(vis)

        elapsed = time.perf_counter() - t0

        writer.save()
        if video_writer is not None:
            video_writer.release()
            print(f"  Video saved -> {(out_dir / (seq.name + '.mp4')).resolve()}")

        fps_val = seq.n_frames / max(elapsed, 1e-6)
        seq_fps[seq.name] = fps_val
        print(f"  {seq.name}  {seq.n_frames} frames  {fps_val:.1f} FPS")

    return seq_fps


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    import motmetrics as mm   # early check

    split_map = {
        "train": "VisDrone2019-MOT-train",
        "val":   "VisDrone2019-MOT-val",
    }
    split_dir = Path(args.dataset_root) / split_map[args.split]
    ann_dir   = split_dir / "annotations"
    res_dir   = Path(args.results_dir)

    # ── Optionally run tracking first ──────────────────────────────────
    seq_fps: Dict[str, float] = {}
    seq_time: Dict[str, float] = {}

    if args.run_tracking:
        print(f"\n{'='*65}")
        print(f"  STEP 1 — Running {args.tracker.upper()} tracker on {args.split} split")
        print(f"{'='*65}")
        t_track_start = time.perf_counter()
        seq_fps = run_tracking_on_split(args)
        total_track_time = time.perf_counter() - t_track_start
        print(f"\n  Tracking done in {total_track_time:.1f}s\n")

    # ── Find sequences that have both GT and predictions ───────────────
    ann_files  = sorted(ann_dir.glob("*.txt"))
    pred_files = sorted(res_dir.glob("*.txt"))

    # Build lookup by stem (sequence name)
    pred_lookup = {p.stem: p for p in pred_files}
    ann_lookup  = {a.stem: a for a in ann_files}

    common = sorted(set(pred_lookup.keys()) & set(ann_lookup.keys()))
    if not common:
        print(f"\n[ERROR] No matching sequences found.")
        print(f"  GT annotations in : {ann_dir}")
        print(f"  Predictions in    : {res_dir}")
        print(f"  GT files    : {[a.name for a in ann_files[:5]]}")
        print(f"  Pred files  : {[p.name for p in pred_files[:5]]}")
        return

    print(f"\n{'='*65}")
    print(f"  STEP 2 — Evaluating {len(common)} sequences")
    print(f"  GT  : {ann_dir}")
    print(f"  Pred: {res_dir}")
    print(f"{'='*65}\n")

    # ── Per-sequence metrics ───────────────────────────────────────────
    rows = []
    import motmetrics as mm

    # Accumulator for overall (all sequences combined)
    global_acc = mm.MOTAccumulator(auto_id=True)
    global_frames = 0

    for seq_name in common:
        gt_df, _ign_df = load_gt(ann_lookup[seq_name])
        pred_df        = load_pred(pred_lookup[seq_name])

        t0  = time.perf_counter()
        met = compute_mot_metrics(gt_df, pred_df, iou_threshold=args.iou_eval)
        t_eval = time.perf_counter() - t0

        fps  = seq_fps.get(seq_name, float("nan"))
        n_fr = met["Frames"]
        global_frames += n_fr

        rows.append({
            "Sequence" : seq_name,
            "Frames"   : n_fr,
            "MOTA↑"    : f"{met['MOTA']:.1f}",
            "MOTP↑"    : f"{met['MOTP']:.1f}",
            "IDF1↑"    : f"{met['IDF1']:.1f}",
            "Prec↑"    : f"{met['Prec']:.1f}",
            "Recall↑"  : f"{met['Recall']:.1f}",
            "MT↑"      : met["MT"],
            "ML↓"      : met["ML"],
            "FP↓"      : met["FP"],
            "FN↓"      : met["FN"],
            "IDs↓"     : met["IDs"],
            "FPS"      : f"{fps:.1f}" if not np.isnan(fps) else "—",
        })
        print(f"  {seq_name:35s}  MOTA={met['MOTA']:5.1f}  "
              f"IDF1={met['IDF1']:5.1f}  FPS={fps:.1f}" if not np.isnan(fps)
              else f"  {seq_name:35s}  MOTA={met['MOTA']:5.1f}  IDF1={met['IDF1']:5.1f}")

    # ── Overall metrics (aggregate all sequences) ──────────────────────
    print(f"\n  Computing overall metrics across all {len(common)} sequences …")

    # Re-accumulate globally
    import motmetrics as mm
    all_accs   = []
    all_names  = []

    for seq_name in common:
        gt_df, _ = load_gt(ann_lookup[seq_name])
        pred_df  = load_pred(pred_lookup[seq_name])
        acc      = mm.MOTAccumulator(auto_id=True)
        all_frames = sorted(set(
            list(gt_df["frame"].unique()   if len(gt_df)   else []) +
            list(pred_df["frame"].unique() if len(pred_df) else [])
        ))
        for frame in all_frames:
            gt_f   = gt_df[gt_df["frame"]     == frame] if len(gt_df)   else pd.DataFrame()
            pred_f = pred_df[pred_df["frame"] == frame] if len(pred_df) else pd.DataFrame()
            gt_ids   = gt_f["id"].tolist()   if len(gt_f)   else []
            pred_ids = pred_f["id"].tolist() if len(pred_f) else []
            if gt_ids and pred_ids:
                gt_b   = gt_f[["left","top","width","height"]].values
                pr_b   = pred_f[["left","top","width","height"]].values
                iou    = bbox_iou_matrix(gt_b, pr_b)
                dist   = 1.0 - iou
                dist[iou < args.iou_eval] = np.inf
            else:
                dist = np.full((len(gt_ids), len(pred_ids)), np.inf)
            acc.update(gt_ids, pred_ids, dist)
        all_accs.append(acc)
        all_names.append(seq_name)

    mh      = mm.metrics.create()
    overall = mh.compute_many(
        all_accs,
        names=all_names,
        metrics=[
            "num_frames", "num_switches", "num_false_positives",
            "num_misses", "num_objects", "num_predictions",
            "mota", "motp", "idf1", "precision", "recall",
            "mostly_tracked", "mostly_lost", "partially_tracked",
        ],
        generate_overall=True,
    )
    ov = overall.loc["OVERALL"]

    avg_fps = np.nanmean(list(seq_fps.values())) if seq_fps else float("nan")

    overall_row = {
        "Sequence" : "── OVERALL ──",
        "Frames"   : global_frames,
        "MOTA↑"    : f"{float(ov['mota'])*100:.1f}",
        "MOTP↑"    : f"{float(ov['motp'])*100:.1f}",
        "IDF1↑"    : f"{float(ov['idf1'])*100:.1f}",
        "Prec↑"    : f"{float(ov['precision'])*100:.1f}",
        "Recall↑"  : f"{float(ov['recall'])*100:.1f}",
        "MT↑"      : int(ov["mostly_tracked"]),
        "ML↓"      : int(ov["mostly_lost"]),
        "FP↓"      : int(ov["num_false_positives"]),
        "FN↓"      : int(ov["num_misses"]),
        "IDs↓"     : int(ov["num_switches"]),
        "FPS"      : f"{avg_fps:.1f}" if not np.isnan(avg_fps) else "—",
    }

    # ── Print table ───────────────────────────────────────────────────
    try:
        from tabulate import tabulate
        table = tabulate(
            rows + [overall_row],
            headers="keys",
            tablefmt="rounded_outline",
            numalign="right",
        )
    except ImportError:
        df_out = pd.DataFrame(rows + [overall_row])
        table  = df_out.to_string(index=False)

    print(f"\n\n{'='*65}")
    print(f"  RESULTS — {args.tracker.upper()}  (IoU threshold = {args.iou_eval})")
    print(f"{'='*65}")
    print(table)

    # ── Summary box ───────────────────────────────────────────────────
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  OVERALL SUMMARY  ({args.tracker.upper():^10s})             │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  MOTA    : {float(ov['mota'])*100:6.2f} %                   │")
    print(f"  │  MOTP    : {float(ov['motp'])*100:6.2f} %                   │")
    print(f"  │  IDF1    : {float(ov['idf1'])*100:6.2f} %                   │")
    print(f"  │  Prec    : {float(ov['precision'])*100:6.2f} %                   │")
    print(f"  │  Recall  : {float(ov['recall'])*100:6.2f} %                   │")
    print(f"  │  MT      : {int(ov['mostly_tracked']):6d}                       │")
    print(f"  │  ML      : {int(ov['mostly_lost']):6d}                       │")
    print(f"  │  FP      : {int(ov['num_false_positives']):6d}                       │")
    print(f"  │  FN      : {int(ov['num_misses']):6d}                       │")
    print(f"  │  ID Sw.  : {int(ov['num_switches']):6d}                       │")
    if not np.isnan(avg_fps):
        print(f"  │  Avg FPS : {avg_fps:6.1f}                       │")
    print(f"  └─────────────────────────────────────────┘")

    # ── Save JSON ─────────────────────────────────────────────────────
    out_json = Path(args.results_dir) / f"metrics_{args.tracker}.json"
    results_dict = {
        "tracker"       : args.tracker,
        "split"         : args.split,
        "iou_threshold" : args.iou_eval,
        "n_sequences"   : len(common),
        "overall": {
            "MOTA"   : round(float(ov["mota"])      * 100, 3),
            "MOTP"   : round(float(ov["motp"])      * 100, 3),
            "IDF1"   : round(float(ov["idf1"])      * 100, 3),
            "Prec"   : round(float(ov["precision"]) * 100, 3),
            "Recall" : round(float(ov["recall"])    * 100, 3),
            "MT"     : int(ov["mostly_tracked"]),
            "ML"     : int(ov["mostly_lost"]),
            "FP"     : int(ov["num_false_positives"]),
            "FN"     : int(ov["num_misses"]),
            "IDs"    : int(ov["num_switches"]),
            "avg_fps": round(avg_fps, 2) if not np.isnan(avg_fps) else None,
        },
        "per_sequence": rows,
    }
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n  Full results saved → {out_json.resolve()}")

    return results_dict


# ---------------------------------------------------------------------------
# Compare two saved JSON result files
# ---------------------------------------------------------------------------

def compare_results(json_paths: List[str]):
    """Print a side-by-side comparison table of multiple tracker JSONs."""
    records = []
    for p in json_paths:
        with open(p) as f:
            d = json.load(f)
        ov = d["overall"]
        records.append({
            "Tracker"  : d["tracker"].upper(),
            "MOTA↑"    : f"{ov['MOTA']:.2f}",
            "MOTP↑"    : f"{ov['MOTP']:.2f}",
            "IDF1↑"    : f"{ov['IDF1']:.2f}",
            "Prec↑"    : f"{ov['Prec']:.2f}",
            "Recall↑"  : f"{ov['Recall']:.2f}",
            "MT↑"      : ov["MT"],
            "ML↓"      : ov["ML"],
            "FP↓"      : ov["FP"],
            "FN↓"      : ov["FN"],
            "IDs↓"     : ov["IDs"],
            "Avg FPS"  : f"{ov['avg_fps']:.1f}" if ov.get("avg_fps") else "—",
        })

    try:
        from tabulate import tabulate
        print(tabulate(records, headers="keys", tablefmt="rounded_outline"))
    except ImportError:
        print(pd.DataFrame(records).to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VisDrone-MOT tracking results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ──────────────────────────────────────────────────────
    g = p.add_argument_group("Dataset")
    g.add_argument(
        "--dataset-root", type=str,
        default=str(Path(__file__).parent.parent / "dataset" / "VisDrone"),
    )
    g.add_argument("--split", type=str, default="val",
                   choices=["train", "val"])

    # ── Results ───────────────────────────────────────────────────────
    g = p.add_argument_group("Results")
    g.add_argument("--tracker", type=str, default="botsort",
                   choices=["botsort", "bytetrack"])
    g.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing tracker .txt files. "
             "Defaults to runs/track/<tracker>/",
    )
    g.add_argument(
        "--iou-eval", type=float, default=0.5,
        help="IoU threshold for TP/FP matching during evaluation.",
    )
    g.add_argument(
        "--compare", type=str, nargs="+", default=None,
        help="Paths to two or more metrics_<tracker>.json files to compare.",
    )

    # ── Tracking (only used with --run-tracking) ──────────────────────
    g = p.add_argument_group("Tracking (only if --run-tracking is set)")
    g.add_argument("--run-tracking", action="store_true",
                   help="Run tracking first, then evaluate.")
    g.add_argument("--yolo-model",  type=str, default="yolov8n.pt")
    g.add_argument("--model-type",  type=str, default="yolov8")
    g.add_argument("--conf",        type=float, default=0.25)
    g.add_argument("--iou",         type=float, default=0.45)
    g.add_argument("--slice-h",     type=int,   default=640)
    g.add_argument("--slice-w",     type=int,   default=640)
    g.add_argument("--overlap-h",   type=float, default=0.2)
    g.add_argument("--overlap-w",   type=float, default=0.2)
    g.add_argument("--reid-model",  type=str,   default=None)
    g.add_argument("--device",      type=str,   default="cpu")
    g.add_argument("--half",        action="store_true")
    g.add_argument("--save-video",  action="store_true",
                   help="Also write annotated .mp4 videos when --run-tracking is set.")
    g.add_argument("--fps",         type=float, default=30.0,
                   help="Output video frame rate (only used with --save-video).")
    g.add_argument("--trail-len",   type=int,   default=40,
                   help="Motion trail length in frames (only used with --save-video).")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Compare mode ─────────────────────────────────────────────────
    if args.compare:
        print("\n  Tracker Comparison\n")
        compare_results(args.compare)
        return

    # ── Default results dir ───────────────────────────────────────────
    if args.results_dir is None:
        args.results_dir = str(Path("runs/track") / args.tracker)

    evaluate(args)


if __name__ == "__main__":
    main()