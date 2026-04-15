"""Run a tracker over every sequence of a VisDrone-MOT split.

Path A (--mode boxmot)
    Shells out to BoxMOT's CLI (`python -m boxmot.engine.cli track ...`)
    per sequence. The CLI exposes flags the Python facade hides:
    --show-trajectories, --show-labels, --show-conf, --save-crop,
    --target-id, --per-class, --postprocessing, etc.

    Detector resolution: if you pass a name like 'yolov8n', BoxMOT looks
    up a detector profile by that name. If you pass a .pt path or a name
    ending in .pt (e.g. 'yolov8n.pt'), BoxMOT auto-resolves via the
    ultralytics/default.yaml family profile and downloads weights as
    needed. So 'yolov8n.pt' just works without any custom YAML.

Path B (--mode custom)
    Uses TrackingPipeline with our SAHI+YOLO detector + custom drawing.
    Good for tiny aerial targets.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import Boxmot, TrackingPipeline                # noqa: E402
from src.datasets import VisDroneMOT                    # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


# ---------------------------------------------------------------------------
def _normalize_detector(spec: str) -> str:
    """Turn bare model names into .pt so BoxMOT's family resolver kicks in.

    BoxMOT 17 looks up detector configs in this order:
        1. exact match by filename in detectors/<family>/
        2. family default (e.g. ultralytics/default.yaml) when filename has .pt

    Bundled exact-match profiles (BoxMOT 17): yolox_x_visdrone,
    yolox_x_mot17_ablation, yolox_x_mot20_ablation, yolox_x_dancetrack_ablation,
    yolo11l_3ch, yolo11s_obb. Anything else should be passed as <name>.pt so
    the family default handles it.
    """
    bundled_exact = {
        "yolox_x_visdrone", "yolox_x_mot17_ablation", "yolox_x_mot20_ablation",
        "yolox_x_dancetrack_ablation", "yolo11l_3ch", "yolo11s_obb",
    }
    if spec in bundled_exact:
        return spec
    if spec.endswith((".pt", ".onnx", ".engine")) or "/" in spec:
        return spec
    # Bare name like 'yolov8n' / 'yolo11s' → append .pt for family resolution
    return f"{spec}.pt"


# ---------------------------------------------------------------------------
def _run_boxmot(seqs, args, out_root):
    """Track every sequence via BoxMOT CLI subprocess.

    The CLI surfaces flags the Python facade hides — most importantly
    --show-trajectories, --show-labels, --show-conf, --save-crop, and
    --target-id. We invoke `python -m boxmot.engine.cli track` per
    sequence so each output is named after the sequence.
    """
    detector = _normalize_detector(args.detector)

    for seq in seqs:
        cmd = [
            sys.executable, "-m", "boxmot.engine.cli", "track",
            "--detector", detector,
            "--reid", args.reid,
            "--tracker", args.tracker,
            "--source", str(seq.img_dir),
            "--imgsz", str(args.imgsz),
            "--conf", str(args.conf),
            "--iou", str(args.iou),
            "--device", args.device,
            "--project", str(out_root),
            "--name", seq.name,
            "--exist-ok",
            "--save",                      # write annotated video
            "--save-txt",                  # write MOT-format txt
            "--show-trajectories",         # draw trails
            "--show-labels",
            "--show-conf",
        ]
        if args.half:
            cmd.append("--half")
        if args.per_class:
            cmd.append("--per-class")
        if args.save_crop:
            cmd.append("--save-crop")
        if args.target_id is not None:
            cmd += ["--target-id", str(args.target_id)]
        if args.classes:
            cmd += ["--classes", ",".join(str(c) for c in args.classes)]

        logging.info("[%s] launching BoxMOT track CLI", seq.name)
        subprocess.run(cmd, check=True)

        # Normalize output names: BoxMOT writes <project>/<name>/<source_stem>.mp4
        # since name=seq.name and source_stem also = seq.name, mp4 is already
        # named correctly. But we ensure a top-level <seq>.mp4 + <seq>.txt
        # exist for predictable downstream consumption.
        run_dir = out_root / seq.name
        if run_dir.exists():
            for mp4 in list(run_dir.glob("*.mp4")):
                target = run_dir / f"{seq.name}.mp4"
                if mp4 != target and not target.exists():
                    shutil.move(str(mp4), str(target))
                    logging.info("[%s] renamed %s -> %s",
                                 seq.name, mp4.name, target.name)
            # MOT-Challenge txt is usually under labels/<seq>.txt
            for txt in run_dir.rglob("*.txt"):
                target = run_dir / f"{seq.name}.txt"
                if (txt != target and not target.exists()
                        and txt.parent.name != "labels" or txt.parent == run_dir):
                    if txt != target:
                        try:
                            shutil.copy(str(txt), str(target))
                        except Exception:
                            pass
        logging.info("[%s] -> %s", seq.name, run_dir)


# ---------------------------------------------------------------------------
def _run_custom(seqs, args, out_root):
    """SAHI sliced inference + BoxMOT tracker via our TrackingPipeline."""
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
            "per_class": args.per_class,
        },
    )
    for seq in seqs:
        pipe.run(
            source=seq.img_dir, output_dir=out_root,
            save_video=args.save_video, save_mot=True,
            show_trajectories=True, run_name=seq.name,
        )


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Run tracking over a VisDrone-MOT split.")
    ap.add_argument("--root", required=True, type=Path,
                    help="VisDrone root containing VisDrone2019-MOT-{train,val,test-dev}/")
    ap.add_argument("--split", choices=["train", "val", "test-dev"], default="val")
    ap.add_argument("--mode", choices=["boxmot", "custom"], default="boxmot")

    ap.add_argument("--detector", default="yolox_x_visdrone",
                    help="boxmot mode: short name (yolox_x_visdrone, yolo11l_3ch) "
                         "or 'yolov8n', 'yolo11s' (auto-suffixed .pt for family "
                         "resolution); custom mode: .pt path")
    ap.add_argument("--reid", default="lmbn_n_duke",
                    help="boxmot mode: short name like 'lmbn_n_duke' or "
                         "'osnet_x0_25_msmt17'")
    ap.add_argument("--reid-weights", default="osnet_x0_25_msmt17.pt",
                    help="custom mode: .pt path")
    ap.add_argument("--tracker", default="botsort")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", action="store_true")

    ap.add_argument("--classes", nargs="*", type=int,
                    default=[1, 2, 4, 5, 6, 9])
    ap.add_argument("--per-class", action="store_true")
    ap.add_argument("--target-id", type=int, default=None,
                    help="Highlight a single track ID in green (boxmot mode).")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--slice-size", type=int, default=640,
                    help="custom mode: SAHI slice size")
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--save-crop", action="store_true")
    ap.add_argument("--save-trajectories", action="store_true", default=True,
                    help="Always on for boxmot mode (--show-trajectories).")

    ap.add_argument("--out", type=Path, default=Path("outputs/visdrone"))
    ap.add_argument("--evaluate", action="store_true",
                    help="After tracking, build MOT-Challenge GT and run val.")
    ap.add_argument("--gt-out", type=Path, default=Path("outputs/visdrone/gt"))
    ap.add_argument("--eval-benchmark", default="visdrone-ablation",
                    help="Benchmark id passed to bm.val(). Default uses "
                         "BoxMOT's bundled visdrone-ablation pipeline.")
    ap.add_argument("--limit", type=int, default=0)
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
        logging.warning("Skipping eval on 'train' split.")
        return

    # Convert VisDrone GT → MOT-Challenge layout (TrackEval expects this)
    gt_root = ds.export_motchallenge_gt(args.gt_out, benchmark_name="VisDrone-MOT")
    logging.info("Converted GT -> %s", gt_root)

    # Evaluate using BoxMOT's bundled visdrone-ablation benchmark by default.
    # If you want to score against your own converted GT, pass
    # --eval-benchmark with the id of a benchmark you registered separately.
    bm = Boxmot(
        detector=_normalize_detector(args.detector),
        reid=args.reid,
        tracker=args.tracker,
        classes=args.classes,
        project=str(out_root.parent),
    )
    result = bm.val(
        benchmark=args.eval_benchmark,
        device=args.device,
        half=args.half,
        verbose=True,
    )
    logging.info("Metrics: %s", result)


if __name__ == "__main__":
    main()
