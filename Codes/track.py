"""
track.py
--------
Run the YOLO11 detector + BoT-SORT-RF tracker on a single VisDrone sequence
(folder of images) or a video file, and write an annotated output video.
Optionally writes MOTChallenge-format results for use by eval.py.

Usage:
    python track.py --source /path/to/dataset/VisDrone2019-MOT-val/sequences/uav0000009_03358_v \
                    --yolo   yolo11_visdrone.pt \
                    --rf     rf_hsv.pkl \
                    --output output.mp4 \
                    --results output.txt
"""
import os
import argparse
import numpy as np
import cv2

from ultralytics import YOLO
from bot_sort_rf import BoTSORT_RF
from sahi_detect import SAHIDetector


def _color(i):
    return (int((i * 37) % 255), int((i * 17 + 90) % 255), int((i * 53 + 30) % 255))


def draw_tracks(img, tracks):
    for t in tracks:
        x, y, w, h = t.tlwh
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        c = _color(t.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        label = f'ID {t.track_id}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(img, label, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def iter_frames(source):
    """Yield (frame_idx_1based, BGR_image)."""
    if os.path.isdir(source):
        files = sorted([f for f in os.listdir(source)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for i, f in enumerate(files, 1):
            img = cv2.imread(os.path.join(source, f))
            if img is None:
                continue
            yield i, img
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(source)
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            i += 1
            yield i, img
        cap.release()


def run(source, yolo_path, rf_path, output_video,
        conf=0.25, iou=0.7, imgsz=1280, fps=30,
        results_file=None, show_progress=True, classes=None,
        sahi=False, slice_h=640, slice_w=640,
        overlap_h=0.2, overlap_w=0.2,
        sahi_full_image=True, sahi_batch=8):
    model = YOLO(yolo_path)
    tracker = BoTSORT_RF(rf_path, frame_rate=fps)

    sahi_det = None
    if sahi:
        sahi_det = SAHIDetector(model,
                                slice_h=slice_h, slice_w=slice_w,
                                overlap_h=overlap_h, overlap_w=overlap_w,
                                conf=conf, iou=iou,
                                include_full_image=sahi_full_image,
                                full_image_imgsz=imgsz,
                                batch=sahi_batch)
        print(f'[SAHI] slice=({slice_h},{slice_w}) '
              f'overlap=({overlap_h},{overlap_w}) '
              f'full_image={sahi_full_image}')

    writer = None
    lines = []
    n = 0
    for frame_idx, img in iter_frames(source):
        n += 1
        if sahi_det is not None:
            dets = sahi_det(img, classes=classes)
            if dets.shape[0] == 0:
                dets = np.empty((0, 6), dtype=np.float32)
        else:
            res = model.predict(img, conf=conf, iou=iou, imgsz=imgsz,
                                verbose=False, classes=classes)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                dets = np.concatenate(
                    [xyxy, scores[:, None], cls[:, None]], axis=1
                ).astype(np.float32)
            else:
                dets = np.empty((0, 6), dtype=np.float32)

        tracks = tracker.update(dets, img)
        vis = draw_tracks(img.copy(), tracks)

        if writer is None:
            h, w = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            os.makedirs(os.path.dirname(os.path.abspath(output_video)) or '.',
                        exist_ok=True)
            writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        writer.write(vis)

        for t in tracks:
            x, y, w_, h_ = t.tlwh
            lines.append(f'{frame_idx},{t.track_id},{x:.2f},{y:.2f},'
                         f'{w_:.2f},{h_:.2f},{t.score:.4f},-1,-1,-1')

        if show_progress and n % 25 == 0:
            print(f'  frame {n}, active tracks={len(tracks)}')

    if writer is not None:
        writer.release()
    if results_file:
        os.makedirs(os.path.dirname(os.path.abspath(results_file)) or '.',
                    exist_ok=True)
        with open(results_file, 'w') as f:
            f.write('\n'.join(lines))
    print(f'Done. Frames={n}  video -> {output_video}'
          + (f'  results -> {results_file}' if results_file else ''))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True,
                    help='Folder of frames (VisDrone sequence) or a video file')
    ap.add_argument('--yolo', default='yolo11_visdrone.pt')
    ap.add_argument('--rf', default='rf_hsv.pkl')
    ap.add_argument('--output', default='output.mp4')
    ap.add_argument('--results', default=None,
                    help='Optional MOTChallenge-format results .txt')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--classes', type=int, nargs='*', default=None,
                    help='Optional YOLO class filter (e.g. 0 1 3 4)')
    # --- SAHI slicing options ---
    ap.add_argument('--sahi', action='store_true',
                    help='Enable SAHI slicing inference')
    ap.add_argument('--slice_h', type=int, default=640)
    ap.add_argument('--slice_w', type=int, default=640)
    ap.add_argument('--overlap_h', type=float, default=0.2)
    ap.add_argument('--overlap_w', type=float, default=0.2)
    ap.add_argument('--no_full_image', action='store_true',
                    help='Disable full-image pass (slice-only inference)')
    ap.add_argument('--sahi_batch', type=int, default=8,
                    help='Number of slices per YOLO forward batch')
    args = ap.parse_args()

    run(args.source, args.yolo, args.rf, args.output,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, fps=args.fps,
        results_file=args.results, classes=args.classes,
        sahi=args.sahi, slice_h=args.slice_h, slice_w=args.slice_w,
        overlap_h=args.overlap_h, overlap_w=args.overlap_w,
        sahi_full_image=not args.no_full_image,
        sahi_batch=args.sahi_batch)


if __name__ == '__main__':
    main()
