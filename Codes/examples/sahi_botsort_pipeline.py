"""Path B example: SAHI sliced YOLO detector → BoxMOT BotSort tracker.

Useful when:
  - targets are tiny (aerial, drone, surveillance) and you need SAHI slicing
  - you want full Python control over the inference loop
  - BoxMOT's CLI doesn't natively cover your detector

Run:
    python examples/sahi_botsort_pipeline.py --source video.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import TrackingPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Video file, image folder, or stream URL.")
    ap.add_argument("--out", default="outputs/sahi_botsort")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reid-weights", default="osnet_x0_25_msmt17.pt")
    ap.add_argument("--detector", default="yolov8n.pt")
    ap.add_argument("--classes", nargs="+", type=int, default=[0])
    ap.add_argument("--slice-size", type=int, default=640)
    args = ap.parse_args()

    pipe = TrackingPipeline(
        detector_cfg={
            "type": "sahi_yolo",
            "model_path": args.detector,
            "device": args.device,
            "confidence_threshold": 0.3,
            "classes": args.classes,
            "sahi": {
                "slice_height": args.slice_size,
                "slice_width": args.slice_size,
                "overlap_height_ratio": 0.2,
                "overlap_width_ratio": 0.2,
                "model_type": "ultralytics",
            },
        },
        tracker_cfg={
            "type": "botsort",
            "reid_weights": args.reid_weights,
            "device": args.device,
            "half": False,
        },
    )

    res = pipe.run(
        source=args.source,
        output_dir=args.out,
        save_video=True,
        save_mot=True,
        show_trajectories=True,
        run_name="sahi_botsort",
    )
    print(f"Annotated video: {res.video_path}")
    print(f"MOT predictions: {res.mot_path}")
    print(f"Frames processed: {res.n_frames}")


if __name__ == "__main__":
    main()
