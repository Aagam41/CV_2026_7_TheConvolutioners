#!/usr/bin/env python3
"""
test_detector.py
----------------
Smoke-test the YOLOSAHIDetector on a single image or the first frame
of a VisDrone sequence, without running the full tracking pipeline.

Usage
-----
# Test on a plain image:
python test_detector.py --image path/to/frame.jpg

# Test on first frame of a sequence:
python test_detector.py \
    --seq-dir ../dataset/VisDrone/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
    --yolo-model yolov8n.pt \
    --device cpu \
    --show
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

from detector import YOLOSAHIDetector
from visdrone_utils import PALETTE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--seq-dir", type=str, default=None)
    p.add_argument("--yolo-model", type=str, default="yolov8n.pt")
    p.add_argument("--model-type", type=str, default="yolov8")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--slice-h", type=int, default=640)
    p.add_argument("--slice-w", type=int, default=640)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load frame ────────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        label = Path(args.image).name
    elif args.seq_dir:
        seq_dir = Path(args.seq_dir)
        imgs = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.png"))
        if not imgs:
            raise FileNotFoundError(f"No images in {args.seq_dir}")
        frame = cv2.imread(str(imgs[0]))
        label = f"{seq_dir.name} / {imgs[0].name}"
    else:
        raise ValueError("Provide --image or --seq-dir")

    print(f"Frame: {label}  shape={frame.shape}")

    # ── Detect ────────────────────────────────────────────────────────
    det = YOLOSAHIDetector(
        model_path=args.yolo_model,
        device=args.device,
        conf_threshold=args.conf,
        slice_height=args.slice_h,
        slice_width=args.slice_w,
        model_type=args.model_type,
    )
    dets = det.detect(frame)
    print(f"Detections: {len(dets)}")
    for i, d in enumerate(dets[:10]):
        print(f"  [{i}] cls={int(d[5])} conf={d[4]:.3f} "
              f"box=({d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f})")
    if len(dets) > 10:
        print(f"  … and {len(dets)-10} more")

    # ── Draw ──────────────────────────────────────────────────────────
    vis = frame.copy()
    for d in dets:
        color = PALETTE[int(d[5]) % len(PALETTE)]
        cv2.rectangle(vis, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), color, 1)
        cv2.putText(vis, f"cls{int(d[5])} {d[4]:.2f}",
                    (int(d[0]), max(int(d[1])-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    out_path = Path("runs/test_detector_output.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"Annotated frame saved → {out_path}")

    if args.show:
        cv2.imshow("YOLO+SAHI detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
