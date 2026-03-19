#!/usr/bin/env python3
"""
track_visdrone.py
-----------------
End-to-end MOT pipeline for VisDrone using YOLO + SAHI as the
custom detector and BoxMOT (BotSort / ByteTrack) as the tracker.

This script is the drop-in replacement for `boxmot track` that plugs
in the SAHI-augmented detector instead of a plain YOLO model.

Usage examples
--------------
# BotSort on the validation split (all sequences):
python track_visdrone.py \
    --split val \
    --tracker botsort \
    --yolo-model yolov8n.pt \
    --reid-model osnet_x0_25_msmt17.pt \
    --device cuda:0

# ByteTrack on a single sequence:
python track_visdrone.py \
    --split train \
    --seq uav0000013_00000_v \
    --tracker bytetrack \
    --yolo-model yolov8s.pt \
    --device 0

# Use a custom dataset root (override default ../dataset path):
python track_visdrone.py \
    --dataset-root /data/VisDrone \
    --split val \
    --tracker botsort \
    --yolo-model yolov8n.pt \
    --reid-model osnet_x0_25_msmt17.pt

Output
------
runs/
└── track/
    └── botsort/              (or bytetrack/)
        ├── uav0000013_00000_v.mp4    ← annotated video
        ├── uav0000013_00000_v.txt    ← MOT-format tracking result
        └── ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ── Local modules ──────────────────────────────────────────────────────────
from detector import YOLOSAHIDetector
from visdrone_utils import (
    VisDroneSequence,
    list_sequences,
    MOTResultWriter,
    make_video_writer,
    draw_tracks,
    TrailBuffer,
)


# ── BoxMOT tracker factory ─────────────────────────────────────────────────

def build_tracker(
        tracker_name: str,
        reid_model: str | None,
        device: str,
        half: bool,
):
    """
    Instantiate the requested BoxMOT tracker.

    BotSort   → Motion + Appearance (needs a ReID model)
    ByteTrack → Motion-only (no ReID needed)

    Returns a fresh tracker instance (stateless across sequences – callers
    must call this once per sequence so state is reset).
    """
    tracker_name = tracker_name.lower()

    if tracker_name == "botsort":
        from boxmot import BotSort
        if reid_model is None:
            reid_model = "osnet_x0_25_msmt17.pt"
            print(f"[build_tracker] No --reid-model given; defaulting to '{reid_model}'")
        return BotSort(
            reid_weights=Path(reid_model),
            device=device,
            half=half,
        )

    elif tracker_name == "bytetrack":
        from boxmot import ByteTrack
        return ByteTrack(
            device=device,
            half=half,
        )

    else:
        raise ValueError(
            f"Unsupported tracker '{tracker_name}'. "
            "Choose from: botsort, bytetrack"
        )


# ── Per-sequence tracking loop ─────────────────────────────────────────────

def track_sequence(
        seq: VisDroneSequence,
        detector: YOLOSAHIDetector,
        tracker_name: str,
        reid_model: str | None,
        device: str,
        half: bool,
        out_dir: Path,
        save_video: bool,
        save_txt: bool,
        show: bool,
        fps: float,
        no_draw_det: bool,
        trail_len: int = 40,
) -> dict:
    """
    Run the full detect → track → annotate pipeline on one sequence.

    Returns a dict with timing statistics.
    """
    print(f"\n{'─' * 60}")
    print(f"  Sequence : {seq.name}  ({seq.n_frames} frames)")
    print(f"  Tracker  : {tracker_name.upper()}")
    print(f"{'─' * 60}")

    # Build a fresh tracker (resets internal state for each sequence)
    tracker = build_tracker(tracker_name, reid_model, device, half)

    # Output paths
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{seq.name}.mp4"
    txt_path = out_dir / f"{seq.name}.txt"

    mot_writer = MOTResultWriter(txt_path) if save_txt else None

    w, h = seq.frame_size()
    video_writer = make_video_writer(video_path, w, h, fps) if save_video else None

    # Trail buffer – reset at the start of each sequence
    trails = TrailBuffer(max_len=trail_len)

    t_det_total = 0.0
    t_trk_total = 0.0
    n_det_total = 0

    for frame_id, frame in seq:
        # ── Detection ──────────────────────────────────────────────
        t0 = time.perf_counter()
        dets = detector.detect(frame)  # (N, 6)  [x1,y1,x2,y2,conf,cls]
        t_det_total += time.perf_counter() - t0
        n_det_total += len(dets)

        # ── Tracking ───────────────────────────────────────────────
        t0 = time.perf_counter()
        tracks = tracker.update(dets, frame)  # (M, 8)  [x1,y1,x2,y2,id,conf,cls,ind]
        t_trk_total += time.perf_counter() - t0

        # ── Record results ─────────────────────────────────────────
        if mot_writer is not None:
            mot_writer.update(frame_id, tracks)

        # ── Update trail buffer ────────────────────────────────────
        trails.update(tracks)

        # ── Visualisation ──────────────────────────────────────────
        vis_frame = frame.copy()
        if not no_draw_det and len(dets) > 0:
            # Draw raw detections (white boxes) before tracker filters
            for d in dets:
                cv2.rectangle(
                    vis_frame,
                    (int(d[0]), int(d[1])), (int(d[2]), int(d[3])),
                    (220, 220, 220), 1,
                )
        draw_tracks(vis_frame, tracks, trails=trails)

        # Frame info overlay
        info = (
            f"Seq: {seq.name}  Frame: {frame_id}/{seq.n_frames}  "
            f"Dets: {len(dets)}  Tracks: {len(tracks) if tracks is not None else 0}"
        )
        cv2.putText(vis_frame, info, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if video_writer is not None:
            video_writer.write(vis_frame)

        if show:
            cv2.imshow("BoxMOT + SAHI", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  [show] User quit.")
                break

        if frame_id % 50 == 0:
            fps_det = frame_id / max(t_det_total, 1e-6)
            print(f"  Frame {frame_id:4d}/{seq.n_frames}  "
                  f"det FPS={fps_det:.1f}  "
                  f"active tracks={len(tracks) if tracks is not None else 0}")

    # ── Finalise ───────────────────────────────────────────────────
    if video_writer is not None:
        video_writer.release()
        print(f"  Video saved → {video_path}")

    if mot_writer is not None:
        mot_writer.save()

    if show:
        cv2.destroyAllWindows()

    stats = {
        "sequence": seq.name,
        "n_frames": seq.n_frames,
        "avg_dets/frame": n_det_total / max(seq.n_frames, 1),
        "det_fps": seq.n_frames / max(t_det_total, 1e-6),
        "trk_fps": seq.n_frames / max(t_trk_total, 1e-6),
    }
    print(f"  Done. avg_dets={stats['avg_dets/frame']:.1f}  "
          f"det_fps={stats['det_fps']:.1f}  trk_fps={stats['trk_fps']:.1f}")
    return stats


# ── CLI entry point ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track VisDrone-MOT sequences with YOLO+SAHI detector "
                    "and BoxMOT trackers (BotSort / ByteTrack).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ──────────────────────────────────────────────────────
    g = p.add_argument_group("Dataset")
    g.add_argument(
        "--dataset-root", type=str,
        default=str(Path(__file__).parent.parent / "dataset" / "VisDrone"),
        help="Root of the VisDrone dataset "
             "(contains VisDrone2019-MOT-train, VisDrone2019-MOT-val, …).",
    )
    g.add_argument(
        "--split", type=str, default="val",
        choices=["train", "val"],
        help="Which split to process.",
    )
    g.add_argument(
        "--seq", type=str, default=None,
        help="Process only this single sequence name (e.g. uav0000013_00000_v). "
             "If not set, all sequences in the split are processed.",
    )

    # ── Detector (YOLO + SAHI) ────────────────────────────────────────
    g = p.add_argument_group("Detector (YOLO + SAHI)")
    g.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                   help="Path to YOLO weights.")
    g.add_argument("--model-type", type=str, default="yolov8",
                   help="SAHI model type: yolov8, yolov5, yolox, rtdetr, …")
    g.add_argument("--conf", type=float, default=0.25,
                   help="Detector confidence threshold.")
    g.add_argument("--iou", type=float, default=0.45,
                   help="NMS / postmerge IoU threshold.")
    g.add_argument("--slice-h", type=int, default=640,
                   help="SAHI slice height (px).")
    g.add_argument("--slice-w", type=int, default=640,
                   help="SAHI slice width (px).")
    g.add_argument("--overlap-h", type=float, default=0.2,
                   help="SAHI vertical overlap ratio.")
    g.add_argument("--overlap-w", type=float, default=0.2,
                   help="SAHI horizontal overlap ratio.")
    g.add_argument("--classes", type=int, nargs="+", default=None,
                   help="Keep only these VisDrone class IDs "
                        "(0=pedestrian 1=people 2=bicycle 3=car … 9=motor). "
                        "Default: keep all.")

    # ── Tracker ───────────────────────────────────────────────────────
    g = p.add_argument_group("Tracker")
    g.add_argument("--tracker", type=str, default="botsort",
                   choices=["botsort", "bytetrack"],
                   help="BoxMOT tracker to use.")
    g.add_argument("--reid-model", type=str, default=None,
                   help="ReID model weights (only used by BotSort). "
                        "Auto-downloaded if not present.")

    # ── Hardware ──────────────────────────────────────────────────────
    g = p.add_argument_group("Hardware")
    g.add_argument("--device", type=str, default="cpu",
                   help="Torch device: 'cpu', 'cuda:0', '0', etc.")
    g.add_argument("--half", action="store_true",
                   help="Use FP16 inference (GPU only).")

    # ── Output ────────────────────────────────────────────────────────
    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", type=str, default="runs/track",
                   help="Root output directory.")
    g.add_argument("--no-save-video", action="store_true",
                   help="Do not write annotated output videos.")
    g.add_argument("--no-save-txt", action="store_true",
                   help="Do not write MOT-format .txt result files.")
    g.add_argument("--show", action="store_true",
                   help="Display frames in an OpenCV window (slow).")
    g.add_argument("--fps", type=float, default=30.0,
                   help="Output video frame rate.")
    g.add_argument("--no-draw-det", action="store_true",
                   help="Skip drawing raw detections (only draw tracks).")
    g.add_argument("--trail-len", type=int, default=40,
                   help="Number of past frames to show in the motion trail (0 = off).")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve sequences ─────────────────────────────────────────────
    split_map = {"train": "VisDrone2019-MOT-train", "val": "VisDrone2019-MOT-val"}
    split_dir = Path(args.dataset_root) / split_map[args.split]

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            "Check --dataset-root and --split arguments."
        )

    if args.seq:
        seq_dirs = [split_dir / "sequences" / args.seq]
        if not seq_dirs[0].exists():
            raise FileNotFoundError(f"Sequence not found: {seq_dirs[0]}")
    else:
        seq_dirs = list_sequences(split_dir)

    print(f"\n{'=' * 60}")
    print(f"  VisDrone MOT Tracking with YOLO+SAHI + BoxMOT")
    print(f"{'=' * 60}")
    print(f"  Split    : {args.split}  ({len(seq_dirs)} sequences)")
    print(f"  Detector : {args.yolo_model} via SAHI ({args.model_type})")
    print(f"  Tracker  : {args.tracker.upper()}")
    print(f"  Device   : {args.device}")
    print(f"  Output   : {args.output_dir}/{args.tracker}/")
    print(f"{'=' * 60}\n")

    # ── Build detector (shared across sequences) ──────────────────────
    detector = YOLOSAHIDetector(
        model_path=args.yolo_model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        slice_height=args.slice_h,
        slice_width=args.slice_w,
        overlap_height_ratio=args.overlap_h,
        overlap_width_ratio=args.overlap_w,
        classes=args.classes,
        model_type=args.model_type,
    )

    # ── Output directory for this run ────────────────────────────────
    out_dir = Path(args.output_dir) / args.tracker

    # ── Process each sequence ────────────────────────────────────────
    all_stats = []
    t_total = time.perf_counter()

    for seq_dir in seq_dirs:
        seq = VisDroneSequence(seq_dir)
        stats = track_sequence(
            seq=seq,
            detector=detector,
            tracker_name=args.tracker,
            reid_model=args.reid_model,
            device=args.device,
            half=args.half,
            out_dir=out_dir,
            save_video=not args.no_save_video,
            save_txt=not args.no_save_txt,
            show=args.show,
            fps=args.fps,
            no_draw_det=args.no_draw_det,
            trail_len=args.trail_len,
        )
        all_stats.append(stats)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total
    total_frames = sum(s["n_frames"] for s in all_stats)
    avg_det_fps = np.mean([s["det_fps"] for s in all_stats])

    print(f"\n{'=' * 60}")
    print(f"  DONE  –  {len(all_stats)} sequences  |  {total_frames} frames")
    print(f"  Total wall time : {elapsed:.1f}s")
    print(f"  Avg det FPS     : {avg_det_fps:.1f}")
    print(f"  Results in      : {out_dir.resolve()}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
