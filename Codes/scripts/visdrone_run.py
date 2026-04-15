"""Run a tracker over every sequence of a VisDrone-MOT split.

Two paths supported by ``--mode``:
    boxmot   → uses the official Boxmot CLI engine for each sequence
               (consistent with BoxMOT's published numbers; needs a built-in
               or VisDrone-trained detector accessible to BoxMOT).
    custom   → uses TrackingPipeline with our SAHI+YOLO detector
               (good for tiny aerial targets).

After tracking, optionally:
    1. Convert VisDrone GT to MOT-Challenge format under ``<gt_root>``.
    2. Evaluate the run with ``Boxmot.val(eval_existing=True)``.

Examples
--------
    # Path A — eval BoxMOT's official ablation pipeline on val
    python scripts/visdrone_run.py \
        --root /data/VisDrone --split val \
        --mode boxmot \
        --detector yolox_x_visdrone --reid lmbn_n_duke --tracker botsort \
        --evaluate

    # Path B — SAHI+YOLOv8 + BotSort on test-dev sequences
    python scripts/visdrone_run.py \
        --root /data/VisDrone --split test-dev \
        --mode custom \
        --detector yolov8n.pt --reid-weights osnet_x0_25_msmt17.pt \
        --tracker botsort --classes 1 2 4 5 6 9 \
        --evaluate
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import Boxmot, TrackingPipeline                # noqa: E402
from src.datasets import VisDroneMOT                    # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def _run_boxmot(seqs, args, out_root):
    bm = Boxmot(detector=args.detector, reid=args.reid, tracker=args.tracker,
                device=args.device, half=args.half)
    for seq in seqs:
        run = bm.track(
            source=str(seq.img_dir),
            save=args.save_video, save_txt=True,
            save_crop=args.save_crop, save_trajectories=args.save_trajectories,
            show_trajectories=True, show_labels=True,
            classes=args.classes, per_class=args.per_class,
            conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            project=str(out_root), name=seq.name, exist_ok=True,
            verbose=False,
        )
        logging.info("[%s] -> %s", seq.name, run.save_dir)


def _run_custom(seqs, args, out_root):
    pipe = TrackingPipeline(
        detector_cfg={
            "type": "sahi_yolo",
            "model_path": args.detector,
            "device": args.device,
            "confidence_threshold": args.conf,
            "classes": args.classes,
            "sahi": {
                "slice_height": args.slice_size, "slice_width": args.slice_size,
                "overlap_height_ratio": 0.2, "overlap_width_ratio": 0.2,
                "model_type": "ultralytics",
            },
        },
        tracker_cfg={
            "type": args.tracker,
            "reid_weights": args.reid_weights,
            "device": args.device, "half": args.half,
        },
    )
    for seq in seqs:
        pipe.run(
            source=seq.img_dir, output_dir=out_root,
            save_video=args.save_video, save_mot=True,
            show_trajectories=True, run_name=seq.name,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run tracking over a VisDrone-MOT split.")
    ap.add_argument("--root", required=True, type=Path,
                    help="VisDrone root containing VisDrone2019-MOT-{train,val,test-dev}/")
    ap.add_argument("--split", choices=["train", "val", "test-dev"], default="val")
    ap.add_argument("--mode", choices=["boxmot", "custom"], default="boxmot")

    ap.add_argument("--detector", default="yolox_x_visdrone",
                    help="boxmot mode: short name; custom mode: .pt path")
    ap.add_argument("--reid", default="lmbn_n_duke",
                    help="boxmot mode: short name")
    ap.add_argument("--reid-weights", default="osnet_x0_25_msmt17.pt",
                    help="custom mode: .pt path")
    ap.add_argument("--tracker", default="botsort")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", action="store_true")

    ap.add_argument("--classes", nargs="*", type=int,
                    default=[1, 2, 4, 5, 6, 9],
                    help="VisDrone categories to keep (1=ped, 2=people, 4=car, ...)")
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--slice-size", type=int, default=640,
                    help="custom mode: SAHI slice size")
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--save-crop", action="store_true")
    ap.add_argument("--save-trajectories", action="store_true", default=True)

    ap.add_argument("--out", type=Path,
                    default=Path("outputs/visdrone"),
                    help="Output root for predictions / videos.")
    ap.add_argument("--evaluate", action="store_true",
                    help="After tracking, build MOT-Challenge GT and run val(eval_existing).")
    ap.add_argument("--gt-out", type=Path,
                    default=Path("outputs/visdrone/gt"),
                    help="Where to write converted MOT-Challenge GT.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N sequences (0=all). Handy for sanity checks.")
    args = ap.parse_args()

    ds = VisDroneMOT(args.root, split=args.split)
    seqs = ds.sequences()
    if args.limit:
        seqs = seqs[: args.limit]
    logging.info("Split=%s  sequences=%d  mode=%s", args.split, len(seqs), args.mode)

    out_root = args.out / args.split / args.tracker
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "boxmot":
        _run_boxmot(seqs, args, out_root)
    else:
        _run_custom(seqs, args, out_root)

    if not args.evaluate:
        return

    if args.split == "train":
        logging.warning("Skipping eval on 'train' split (typically used for fitting).")
        return

    # Convert GT once.
    gt_root = ds.export_motchallenge_gt(args.gt_out, benchmark_name="VisDrone-MOT")
    logging.info("Converted GT -> %s", gt_root)

    # Evaluate with BoxMOT's official engine.
    result = Boxmot().val(
        benchmark=f"VisDrone-MOT-{args.split.replace('-', '')}",
        eval_existing=True,
        project=str(out_root.parent), name=out_root.name,
        verbose=True,
        extra={"--gt-folder": str(args.gt_out)},
    )
    logging.info("Metrics: %s", result.metrics)


if __name__ == "__main__":
    main()
