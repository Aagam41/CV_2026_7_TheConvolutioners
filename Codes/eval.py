"""
eval.py
-------
Run the BoT-SORT-RF tracker on every sequence of a VisDrone split and report
MOT metrics (MOTA, IDF1, MOTP, MT/ML, ID switches, fragmentations, ...).
HOTA is reported automatically if the installed version of `motmetrics`
supports it; otherwise install TrackEval for HOTA.

Usage:
    python eval.py --dataset /path/to/dataset \
                   --split VisDrone2019-MOT-val \
                   --yolo yolo11_visdrone.pt \
                   --rf   rf_hsv.pkl \
                   --output eval_out
"""
import os
import argparse
import numpy as np

# --- NumPy 2.0 compatibility shim for motmetrics ---
# motmetrics.distances.iou_matrix calls np.asfarray, which was removed in
# NumPy 2.0. Patch it back before importing motmetrics.
if not hasattr(np, 'asfarray'):
    def _asfarray(a, dtype=np.float64):
        return np.asarray(a, dtype=dtype)
    np.asfarray = _asfarray

import motmetrics as mm
from tqdm import tqdm

from track import run as run_track


# VisDrone ignore categories: 0=ignored-regions, 11=others
IGNORE_CATS = {0, 11}


def load_gt(path):
    """VisDrone GT -> {frame: [(id, x, y, w, h), ...]}."""
    data = {}
    if not os.path.isfile(path):
        return data
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8:
                continue
            frame = int(p[0])
            tid = int(p[1])
            x, y, w, h = map(float, p[2:6])
            score = int(p[6])
            cat = int(p[7])
            if score == 0 or cat in IGNORE_CATS or tid <= 0:
                continue
            data.setdefault(frame, []).append((tid, x, y, w, h))
    return data


def load_res(path):
    """Tracker output (MOTChallenge) -> {frame: [(id, x, y, w, h), ...]}."""
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
        gb = np.array([x[1:] for x in g], dtype=float) if g else np.empty((0, 4))
        rb = np.array([x[1:] for x in r], dtype=float) if r else np.empty((0, 4))
        if len(gb) and len(rb):
            dist = mm.distances.iou_matrix(gb, rb, max_iou=1.0 - iou_thresh)
        else:
            dist = np.empty((len(gb), len(rb)))
        acc.update(gids, rids, dist)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', default='VisDrone2019-MOT-val')
    ap.add_argument('--yolo', default='yolo11_visdrone.pt')
    ap.add_argument('--rf', default='rf_hsv.pkl')
    ap.add_argument('--output', default='eval_out')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--skip_existing', action='store_true',
                    help='Skip tracking if a results .txt already exists')
    ap.add_argument('--classes', type=int, nargs='*', default=None)
    # --- SAHI slicing options ---
    ap.add_argument('--sahi', action='store_true',
                    help='Enable SAHI slicing inference')
    ap.add_argument('--slice_h', type=int, default=640)
    ap.add_argument('--slice_w', type=int, default=640)
    ap.add_argument('--overlap_h', type=float, default=0.2)
    ap.add_argument('--overlap_w', type=float, default=0.2)
    ap.add_argument('--no_full_image', action='store_true')
    ap.add_argument('--sahi_batch', type=int, default=8)
    args = ap.parse_args()

    seq_root = os.path.join(args.dataset, args.split, 'sequences')
    ann_root = os.path.join(args.dataset, args.split, 'annotations')
    assert os.path.isdir(seq_root), f'Missing {seq_root}'
    assert os.path.isdir(ann_root), f'Missing {ann_root}'

    vid_dir = os.path.join(args.output, 'videos'); os.makedirs(vid_dir, exist_ok=True)
    res_dir = os.path.join(args.output, 'results'); os.makedirs(res_dir, exist_ok=True)

    seqs = sorted([d for d in os.listdir(seq_root)
                   if os.path.isdir(os.path.join(seq_root, d))])

    accs = []
    names = []
    for seq in tqdm(seqs, desc='Sequences'):
        seq_dir = os.path.join(seq_root, seq)
        out_vid = os.path.join(vid_dir, f'{seq}.mp4')
        out_res = os.path.join(res_dir, f'{seq}.txt')

        if not (args.skip_existing and os.path.isfile(out_res)):
            print(f'\n[Tracking] {seq}')
            run_track(seq_dir, args.yolo, args.rf, out_vid,
                      conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                      fps=args.fps, results_file=out_res,
                      show_progress=False, classes=args.classes,
                      sahi=args.sahi,
                      slice_h=args.slice_h, slice_w=args.slice_w,
                      overlap_h=args.overlap_h, overlap_w=args.overlap_w,
                      sahi_full_image=not args.no_full_image,
                      sahi_batch=args.sahi_batch)

        gt = load_gt(os.path.join(ann_root, f'{seq}.txt'))
        res = load_res(out_res)
        if not gt:
            print(f'  [warn] no GT for {seq}, skipping metrics')
            continue
        accs.append(accumulate(gt, res))
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
    # HOTA family (available in newer motmetrics builds)
    optional = ['hota', 'deta', 'assa', 'hota_alpha']
    available = set(mh.names)
    metrics = [m for m in wanted if m in available]
    metrics += [m for m in optional if m in available]
    if not any(m in available for m in optional):
        print('[info] HOTA not available in this motmetrics version. '
              'For HOTA, use TrackEval (https://github.com/JonathonLuiten/TrackEval).')

    summary = mh.compute_many(accs, names=names, metrics=metrics,
                              generate_overall=True)
    print()
    print(mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names))
    csv_path = os.path.join(args.output, 'metrics.csv')
    summary.to_csv(csv_path)
    print(f'\nSaved metrics -> {csv_path}')


if __name__ == '__main__':
    main()
