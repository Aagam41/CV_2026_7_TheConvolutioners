"""
tracker_eval.py
---------------
Evaluate the *tracker only* (no detector). Ground-truth boxes are fed into
BoT-SORT-RF as if they were perfect detections; the tracker assigns its own
IDs, and we measure how well those IDs match the GT IDs.

This isolates association quality (IDF1, ID switches, fragmentations) from
detector quality. With perfect detections, MOTA / Recall / Precision will be
near-perfect by construction; the interesting numbers are IDF1, IDP, IDR,
IDs (id switches), Frag, MT, ML.

Usage:
    python tracker_eval.py --dataset /path/to/dataset \
                           --split VisDrone2019-MOT-val \
                           --rf rf_hsv.pkl \
                           --output tracker_eval_out
"""
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# --- NumPy 2.0 compatibility shim for motmetrics (np.asfarray removed) ---
if not hasattr(np, 'asfarray'):
    def _asfarray(a, dtype=np.float64):
        return np.asarray(a, dtype=dtype)
    np.asfarray = _asfarray

import motmetrics as mm

from bot_sort_rf import BoTSORT_RF


# VisDrone ignore categories: 0=ignored-regions, 11=others
IGNORE_CATS = {0, 11}


# --------------------------- IO helpers ---------------------------
def load_gt(path):
    """{frame: [(id, x, y, w, h, cat), ...]}  (filtered, score>0)."""
    data = {}
    if not os.path.isfile(path):
        return data
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8:
                continue
            frame = int(p[0]); tid = int(p[1])
            x, y, w, h = map(float, p[2:6])
            score = int(p[6]); cat = int(p[7])
            if score == 0 or cat in IGNORE_CATS or tid <= 0:
                continue
            data.setdefault(frame, []).append((tid, x, y, w, h, cat))
    return data


def list_frames(seq_dir):
    files = sorted([f for f in os.listdir(seq_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return [os.path.join(seq_dir, f) for f in files]


# --------------------------- Drawing ---------------------------
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


# --------------------------- Per-sequence tracking ---------------------------
def run_sequence(seq_dir, gt, rf_path, out_video=None, out_results=None,
                 fps=30, save_video=False, perturb=0.0):
    """
    Feed GT boxes as detections to the tracker, return list of MOT-format lines.
    perturb: optional gaussian noise (fraction of box w/h) added to GT to
             stress-test the tracker. 0.0 = perfect detections.
    """
    frame_paths = list_frames(seq_dir)
    n_frames = len(frame_paths)
    tracker = BoTSORT_RF(rf_path, frame_rate=fps,
                         track_high_thresh=0.5,
                         new_track_thresh=0.6)

    writer = None
    lines = []
    rng = np.random.default_rng(0)

    for fidx in range(1, n_frames + 1):
        img = cv2.imread(frame_paths[fidx - 1])
        if img is None:
            continue
        H, W = img.shape[:2]

        gt_items = gt.get(fidx, [])
        if gt_items:
            arr = np.array([[g[1], g[2], g[3], g[4]] for g in gt_items],
                           dtype=np.float32)  # x,y,w,h
            if perturb > 0.0:
                noise = rng.normal(0.0, perturb, size=arr.shape).astype(np.float32)
                arr[:, 0] += noise[:, 0] * arr[:, 2]
                arr[:, 1] += noise[:, 1] * arr[:, 3]
                arr[:, 2] *= (1.0 + 0.5 * noise[:, 2])
                arr[:, 3] *= (1.0 + 0.5 * noise[:, 3])
                arr[:, 2:] = np.clip(arr[:, 2:], 2.0, None)
            x1 = arr[:, 0]
            y1 = arr[:, 1]
            x2 = arr[:, 0] + arr[:, 2]
            y2 = arr[:, 1] + arr[:, 3]
            scores = np.ones(len(arr), dtype=np.float32)         # perfect dets
            cls = np.array([g[5] for g in gt_items], dtype=np.float32)
            dets = np.stack([x1, y1, x2, y2, scores, cls], axis=1)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        tracks = tracker.update(dets, img)

        for t in tracks:
            x, y, w_, h_ = t.tlwh
            lines.append(f'{fidx},{t.track_id},{x:.2f},{y:.2f},'
                         f'{w_:.2f},{h_:.2f},{t.score:.4f},-1,-1,-1')

        if save_video and out_video is not None:
            vis = draw_tracks(img.copy(), tracks)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                os.makedirs(os.path.dirname(os.path.abspath(out_video)) or '.',
                            exist_ok=True)
                writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))
            writer.write(vis)

    if writer is not None:
        writer.release()

    if out_results:
        os.makedirs(os.path.dirname(os.path.abspath(out_results)) or '.',
                    exist_ok=True)
        with open(out_results, 'w') as f:
            f.write('\n'.join(lines))

    return lines


# --------------------------- Metrics ---------------------------
def load_res(path):
    data = {}
    if not os.path.isfile(path):
        return data
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6:
                continue
            frame = int(p[0]); tid = int(p[1])
            x, y, w, h = map(float, p[2:6])
            data.setdefault(frame, []).append((tid, x, y, w, h))
    return data


def accumulate(gt, res, iou_thresh=0.5):
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(list(gt.keys()) + list(res.keys())))
    for f in frames:
        g = gt.get(f, [])
        r = res.get(f, [])
        gids = [x[0] for x in g]
        rids = [x[0] for x in r]
        gb = np.array([[x[1], x[2], x[3], x[4]] for x in g],
                      dtype=float) if g else np.empty((0, 4))
        rb = np.array([x[1:5] for x in r],
                      dtype=float) if r else np.empty((0, 4))
        if len(gb) and len(rb):
            dist = mm.distances.iou_matrix(gb, rb, max_iou=1.0 - iou_thresh)
        else:
            dist = np.empty((len(gb), len(rb)))
        acc.update(gids, rids, dist)
    return acc


# --------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', default='VisDrone2019-MOT-val')
    ap.add_argument('--rf', default='rf_hsv.pkl')
    ap.add_argument('--output', default='tracker_eval_out')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--save_video', action='store_true',
                    help='Write annotated MP4 per sequence (slower)')
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--perturb', type=float, default=0.0,
                    help='Gaussian noise on GT boxes (fraction of w/h). '
                         '0.0 = perfect detections; try 0.05 for stress test.')
    args = ap.parse_args()

    seq_root = os.path.join(args.dataset, args.split, 'sequences')
    ann_root = os.path.join(args.dataset, args.split, 'annotations')
    assert os.path.isdir(seq_root), f'Missing {seq_root}'
    assert os.path.isdir(ann_root), f'Missing {ann_root}'

    vid_dir = os.path.join(args.output, 'videos'); os.makedirs(vid_dir, exist_ok=True)
    res_dir = os.path.join(args.output, 'results'); os.makedirs(res_dir, exist_ok=True)

    seqs = sorted([d for d in os.listdir(seq_root)
                   if os.path.isdir(os.path.join(seq_root, d))])

    accs, names = [], []
    for seq in tqdm(seqs, desc='Sequences'):
        seq_dir = os.path.join(seq_root, seq)
        gt_file = os.path.join(ann_root, f'{seq}.txt')
        out_res = os.path.join(res_dir, f'{seq}.txt')
        out_vid = os.path.join(vid_dir, f'{seq}.mp4') if args.save_video else None

        gt = load_gt(gt_file)
        if not gt:
            print(f'  [warn] no GT for {seq}, skipping')
            continue

        if not (args.skip_existing and os.path.isfile(out_res)):
            print(f'\n[Tracking GT-as-det] {seq}')
            run_sequence(seq_dir, gt, args.rf,
                         out_video=out_vid, out_results=out_res,
                         fps=args.fps, save_video=args.save_video,
                         perturb=args.perturb)

        # GT for metrics: drop the cat field
        gt_for_metrics = {f: [(g[0], g[1], g[2], g[3], g[4]) for g in v]
                          for f, v in gt.items()}
        res = load_res(out_res)
        accs.append(accumulate(gt_for_metrics, res))
        names.append(seq)

    if not accs:
        print('No sequences with GT to evaluate.')
        return

    mh = mm.metrics.create()
    wanted = ['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr',
              'recall', 'precision', 'num_unique_objects',
              'mostly_tracked', 'partially_tracked', 'mostly_lost',
              'num_false_positives', 'num_misses',
              'num_switches', 'num_fragmentations']
    optional = ['hota', 'deta', 'assa', 'hota_alpha']
    available = set(mh.names)
    metrics = [m for m in wanted if m in available]
    metrics += [m for m in optional if m in available]
    if not any(m in available for m in optional):
        print('[info] HOTA not available in this motmetrics version. '
              'Use TrackEval for HOTA.')

    summary = mh.compute_many(accs, names=names, metrics=metrics,
                              generate_overall=True)
    print()
    print(mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names))
    csv_path = os.path.join(args.output, 'metrics.csv')
    summary.to_csv(csv_path)
    print(f'\nSaved metrics -> {csv_path}')
    print('\nNote: with GT-as-detections MOTA/Recall/Precision saturate near 100%; '
          'focus on IDF1, IDs (switches), Frag, MT/ML for tracker quality.')


if __name__ == '__main__':
    main()
