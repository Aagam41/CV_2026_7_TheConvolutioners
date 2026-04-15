"""
rf_train.py
-----------
Train a Random Forest same-identity classifier on SIFT Bag-of-Visual-Words
features of VisDrone-MOT ground-truth crops. Saved pickle contains the RF,
the KMeans vocabulary, and n_clusters — all of which bot_sort_rf.py needs
at inference time to compute matching BoW vectors.

Usage:
    python rf_train.py --dataset /path/to/dataset \
                       --split VisDrone2019-MOT-train \
                       --out rf_sift.pkl
"""
import os
import argparse
import pickle
import random
import time
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# ============================ SIFT BoW feature ============================
def sift_descriptors(img_bgr, sift):
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, desc = sift.detectAndCompute(gray, None)
    return desc


def sift_bow(img_bgr, sift, kmeans, n_clusters):
    """Normalized BoW histogram (float32, length=n_clusters)."""
    hist = np.zeros(n_clusters, dtype=np.float32)
    desc = sift_descriptors(img_bgr, sift)
    if desc is None or len(desc) == 0:
        return hist
    labels = kmeans.predict(desc.astype(np.float32))
    for lbl in labels:
        hist[lbl] += 1
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


# ========================= VisDrone annotation IO =========================
IGNORE_CATS = {0, 11}


def load_annotations(ann_file):
    data = {}
    with open(ann_file) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 8:
                continue
            frame = int(p[0]); tid = int(p[1])
            x, y, w, h = map(float, p[2:6])
            score = int(p[6]); cat = int(p[7])
            if score == 0 or cat in IGNORE_CATS or tid <= 0:
                continue
            data.setdefault(frame, []).append((tid, x, y, w, h))
    return data


def iter_crops(seq_root, ann_root, crop_size=(64, 64)):
    """Generator: yields (track_key, crop_bgr) over a whole split."""
    seqs = sorted([d for d in os.listdir(seq_root)
                   if os.path.isdir(os.path.join(seq_root, d))])
    for seq in seqs:
        seq_dir = os.path.join(seq_root, seq)
        ann_file = os.path.join(ann_root, seq + '.txt')
        if not os.path.isfile(ann_file):
            continue
        anns = load_annotations(ann_file)
        frames = sorted([f for f in os.listdir(seq_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        frame_map = {i + 1: os.path.join(seq_dir, f)
                     for i, f in enumerate(frames)}
        for fidx in sorted(anns.keys()):
            if fidx not in frame_map:
                continue
            img = cv2.imread(frame_map[fidx])
            if img is None:
                continue
            H, W = img.shape[:2]
            for tid, x, y, w, h in anns[fidx]:
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(W, int(x + w)), min(H, int(y + h))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = cv2.resize(img[y1:y2, x1:x2], crop_size)
                yield f'{seq}_{tid}', crop


# ============================== Vocabulary ==============================
def build_vocabulary(seq_root, ann_root, n_clusters,
                     max_descriptors, sift_nfeatures, crop_size):
    print(f'[1/3] Building SIFT vocabulary '
          f'(target ~{max_descriptors:,} descriptors, k={n_clusters})...')
    sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
    pool = []
    total = 0
    for _, crop in tqdm(iter_crops(seq_root, ann_root, crop_size),
                        desc='Scanning crops for vocab'):
        desc = sift_descriptors(crop, sift)
        if desc is not None:
            pool.append(desc)
            total += len(desc)
        if total >= max_descriptors:
            break
    if total == 0:
        raise RuntimeError('No SIFT descriptors collected. '
                           'Check dataset paths.')
    print(f'      collected {total:,} descriptors. Fitting MiniBatchKMeans...')
    stacked = np.vstack(pool).astype(np.float32)
    del pool
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=10000,
        n_init=3,
        verbose=0,
    )
    kmeans.fit(stacked)
    del stacked
    return kmeans


# =========================== Per-crop features ===========================
def build_feature_db(seq_root, ann_root, kmeans, n_clusters,
                     sift_nfeatures, crop_size, max_per_track=12):
    print('[2/3] Computing SIFT BoW vector for every crop...')
    sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
    db = {}
    for key, crop in tqdm(iter_crops(seq_root, ann_root, crop_size),
                          desc='BoW per crop'):
        vec = sift_bow(crop, sift, kmeans, n_clusters)
        db.setdefault(key, []).append(vec)
    rng = random.Random(42)
    for k in list(db.keys()):
        if len(db[k]) > max_per_track:
            db[k] = rng.sample(db[k], max_per_track)
        if len(db[k]) < 2:
            db.pop(k, None)
    print(f'      tracks with >=2 crops: {len(db):,}')
    return db


# =============================== Pair sampling ===============================
def make_pairs(db, n_pos, n_neg, seed=42):
    rng = random.Random(seed)
    keys = list(db.keys())
    X, y = [], []
    pos = 0; tries = 0
    while pos < n_pos and tries < n_pos * 20:
        tries += 1
        k = rng.choice(keys)
        a, b = rng.sample(db[k], 2)
        X.append(np.abs(a - b)); y.append(1); pos += 1
    neg = 0; tries = 0
    while neg < n_neg and tries < n_neg * 20:
        tries += 1
        k1, k2 = rng.sample(keys, 2)
        a = rng.choice(db[k1]); b = rng.choice(db[k2])
        X.append(np.abs(a - b)); y.append(0); neg += 1
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    idx = np.arange(len(y))
    np.random.default_rng(seed).shuffle(idx)
    return X[idx], y[idx]


# ================================== Main ==================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    help='Root containing VisDrone2019-MOT-* folders')
    ap.add_argument('--split', default='VisDrone2019-MOT-train')
    ap.add_argument('--out', default='rf_sift.pkl')
    ap.add_argument('--n_clusters', type=int, default=100)
    ap.add_argument('--max_descriptors', type=int, default=2_000_000)
    ap.add_argument('--sift_nfeatures', type=int, default=100)
    ap.add_argument('--crop_size', type=int, nargs=2, default=[64, 64])
    ap.add_argument('--n_pos', type=int, default=60000)
    ap.add_argument('--n_neg', type=int, default=60000)
    ap.add_argument('--n_estimators', type=int, default=200)
    ap.add_argument('--max_depth', type=int, default=None)
    ap.add_argument('--max_per_track', type=int, default=12)
    args = ap.parse_args()

    crop_size = tuple(args.crop_size)
    seq_root = os.path.join(args.dataset, args.split, 'sequences')
    ann_root = os.path.join(args.dataset, args.split, 'annotations')
    assert os.path.isdir(seq_root), f'Missing {seq_root}'
    assert os.path.isdir(ann_root), f'Missing {ann_root}'

    t0 = time.time()
    kmeans = build_vocabulary(seq_root, ann_root,
                              n_clusters=args.n_clusters,
                              max_descriptors=args.max_descriptors,
                              sift_nfeatures=args.sift_nfeatures,
                              crop_size=crop_size)

    db = build_feature_db(seq_root, ann_root, kmeans,
                          n_clusters=args.n_clusters,
                          sift_nfeatures=args.sift_nfeatures,
                          crop_size=crop_size,
                          max_per_track=args.max_per_track)
    if len(db) < 10:
        raise RuntimeError('Too few tracks; check dataset paths.')

    print(f'[3/3] Sampling pairs (+{args.n_pos} / -{args.n_neg}) '
          f'and training RF...')
    X, y = make_pairs(db, args.n_pos, args.n_neg)
    print(f'      pair matrix: {X.shape}, '
          f'pos={int(y.sum())}, neg={int((1-y).sum())}')

    n = len(X); cut = int(n * 0.9)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
        min_samples_leaf=2,
    )
    clf.fit(X[:cut], y[:cut])
    acc_tr = accuracy_score(y[:cut], clf.predict(X[:cut]))
    acc_va = accuracy_score(y[cut:], clf.predict(X[cut:]))
    print(f'      train acc = {acc_tr:.4f}   val acc = {acc_va:.4f}')

    blob = {
        'model': clf,
        'kmeans': kmeans,
        'n_clusters': args.n_clusters,
        'sift_nfeatures': args.sift_nfeatures,
        'crop_size': crop_size,
        'feature_type': 'sift_bow',
    }
    with open(args.out, 'wb') as f:
        pickle.dump(blob, f)
    print(f'Saved -> {args.out}   (elapsed {(time.time()-t0)/60:.1f} min)')


if __name__ == '__main__':
    main()
