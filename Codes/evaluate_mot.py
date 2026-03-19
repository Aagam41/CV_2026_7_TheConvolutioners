"""
VisDrone MOT Evaluation Script  (folder mode)
===============================================
Computes metrics for every sequence in the val set, then prints an
aggregated overall summary.

Metrics computed
----------------
  DETECTION  (dets/ vs GT)  → Precision, Recall, F1, AP@0.5, AP@0.75, mAP@[.5:.95]
  TRACKING   (MOT/  vs GT)  → MOTA, MOTP, IDF1, HOTA, DetA, AssA, MT, ML, FP, FN, IDs, Frag
  EMBEDDINGS (embs/ files)  → Emb-dim, L2-norm, intra/inter cosine sim, discriminability

Usage
-----
    python evaluate_mot.py \\
        --gt   path/to/annotations/          \\   # folder with gt_*.txt files
        --dets path/to/runs/.../dets/        \\   # folder with dets *.txt files
        --mot  path/to/runs/MOT/             \\   # folder with MOT *.txt files
        --embs path/to/runs/.../embs/        \\   # (optional) folder with embs files
        --fps  30                            \\   # video FPS
        --iou  0.5                               # IoU match threshold

Sequence matching
-----------------
  Sequences are discovered from the --gt folder.
  Files in --dets / --mot / --embs are matched by stem name.
  GT files may be prefixed with "gt_" (e.g. gt_uav0000086_00000_v.txt)
  or not (uav0000086_00000_v.txt) — both are handled.

Output
------
  Console: per-sequence tables + overall summary table
  CSV:     evaluate_results.csv  (written next to this script, override with --out)
"""

import argparse
import csv
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── numpy 2.0 compatibility fix for motmetrics ────────────────
# np.asfarray was removed in NumPy 2.0; patch it back before
# motmetrics is imported so its internals don't crash.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
# ─────────────────────────────────────────────────────────────

import motmetrics as mm
from tabulate import tabulate

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# FILE LOADERS
# ══════════════════════════════════════════════════════════════

def load_gt(path):
    """
    VisDrone GT  (comma-sep, 10 cols):
      frame, track_id, x, y, w, h, consider, class, truncation, occlusion
    Skips rows where consider==0 (ignored regions).
    Returns  dict[frame] -> [[tid, x, y, w, h, cls], ...]
    """
    gt = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split(',')
            if len(p) < 8:
                continue
            frame    = int(p[0])
            tid      = int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            consider = int(p[6])
            cls      = int(p[7])
            if consider == 0:
                continue
            gt[frame].append([tid, x, y, w, h, cls])
    return gt


def load_dets(path):
    """
    BoxMOT dets  (space-sep, 7 cols):
      frame  x1  y1  x2  y2  conf  class_id(COCO)
    Converts xyxy -> xywh.
    Returns  dict[frame] -> [[x, y, w, h, conf, cls], ...]
    """
    dets = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) < 7:
                continue
            frame = int(float(p[0]))
            x1, y1, x2, y2 = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            conf  = float(p[5])
            cls   = int(float(p[6]))
            dets[frame].append([x1, y1, x2 - x1, y2 - y1, conf, cls])
    return dets


def load_mot(path):
    """
    MOTChallenge  (comma-sep, 9 cols):
      frame, track_id, x, y, w, h, conf, class, score
    Returns  dict[frame] -> [[tid, x, y, w, h, conf], ...]
    """
    mot = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split(',')
            if len(p) < 6:
                continue
            frame = int(p[0])
            tid   = int(p[1])
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            conf  = float(p[6]) if len(p) > 6 else 1.0
            mot[frame].append([tid, x, y, w, h, conf])
    return mot


def load_embs(path, max_rows=10_000):
    """
    Embeddings file.  Supports  .npy / .npz / .txt
    Returns np.ndarray  (N, D).
    For large .txt files only the first max_rows rows are loaded.
    """
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.npz'):
        data = np.load(path)
        return data[list(data.keys())[0]]
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append(list(map(float, line.split())))
    return np.array(rows, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def xywh_iou(a, b):
    ax1, ay1 = a[0], a[1];  ax2, ay2 = ax1 + a[2], ay1 + a[3]
    bx1, by1 = b[0], b[1];  bx2, by2 = bx1 + b[2], by1 + b[3]
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(gt_boxes, det_boxes):
    G, D = len(gt_boxes), len(det_boxes)
    mat  = np.zeros((G, D))
    for i, g in enumerate(gt_boxes):
        for j, d in enumerate(det_boxes):
            mat[i, j] = xywh_iou(g, d)
    return mat


# ══════════════════════════════════════════════════════════════
# 1.  DETECTION METRICS
# ══════════════════════════════════════════════════════════════

def _match_frame(gt_boxes, det_rows_sorted, iou_thresh):
    """Greedy det matching for one frame. Returns (tp, fp, fn, frame_matches)."""
    det_boxes     = [[r[0], r[1], r[2], r[3]] for r in det_rows_sorted]
    matched_gt    = set()
    frame_matches = [False] * len(det_boxes)

    if gt_boxes and det_boxes:
        iou = iou_matrix(gt_boxes, det_boxes)
        for d_idx in range(len(det_boxes)):
            best_iou = 0; best_g = -1
            for g_idx in range(len(gt_boxes)):
                if g_idx in matched_gt:
                    continue
                if iou[g_idx, d_idx] > best_iou:
                    best_iou = iou[g_idx, d_idx]
                    best_g   = g_idx
            if best_iou >= iou_thresh and best_g >= 0:
                matched_gt.add(best_g)
                frame_matches[d_idx] = True

    tp = sum(frame_matches)
    return tp, len(det_boxes) - tp, len(gt_boxes) - tp, frame_matches


def compute_detection_metrics(gt_dict, det_dict, iou_thresh=0.5):
    """
    Frame-by-frame greedy matching.
    Returns raw counts (prefixed _) for aggregation + formatted metrics.
    """
    all_frames = sorted(set(gt_dict.keys()) | set(det_dict.keys()))
    conf_list  = []; match_list = []
    total_gt   = 0
    tp_sum = fp_sum = fn_sum = 0

    for frame in all_frames:
        gt_rows  = gt_dict.get(frame, [])
        det_rows = sorted(det_dict.get(frame, []), key=lambda r: -r[4])
        gt_boxes = [[r[0], r[1], r[2], r[3]] for r in gt_rows]

        tp, fp, fn, matched = _match_frame(gt_boxes, det_rows, iou_thresh)
        tp_sum += tp; fp_sum += fp; fn_sum += fn
        total_gt += len(gt_boxes)

        for r, m in zip(det_rows, matched):
            conf_list.append(r[4]); match_list.append(m)

    prec = tp_sum / (tp_sum + fp_sum + 1e-9)
    rec  = tp_sum / (tp_sum + fn_sum + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    confs   = np.array(conf_list)
    matches = np.array(match_list, dtype=float)
    order   = np.argsort(-confs)
    confs   = confs[order]; matches = matches[order]
    cum_tp  = np.cumsum(matches)
    cum_fp  = np.cumsum(1 - matches)
    pc      = cum_tp / (cum_tp + cum_fp + 1e-9)
    rc      = cum_tp / (total_gt + 1e-9)
    ap = sum(pc[rc >= t].max() if (rc >= t).any() else 0
             for t in np.linspace(0, 1, 101)) / 101

    return {
        "_tp": tp_sum, "_fp": fp_sum, "_fn": fn_sum,
        "_total_gt": total_gt, "_total_dets": len(conf_list),
        "TP": tp_sum, "FP": fp_sum, "FN": fn_sum,
        "Precision":    round(prec * 100, 2),
        "Recall":       round(rec  * 100, 2),
        "F1":           round(f1   * 100, 2),
        f"AP@{iou_thresh}": round(ap * 100, 2),
    }


def compute_ap_multi_thresh(gt_dict, det_dict):
    r = compute_detection_metrics(gt_dict, det_dict, 0.50)
    r75 = compute_detection_metrics(gt_dict, det_dict, 0.75)
    r["AP@0.75"] = r75["AP@0.75"]
    aps = []
    for t in np.arange(0.50, 1.00, 0.05):
        rx = compute_detection_metrics(gt_dict, det_dict, round(t, 2))
        aps.append(rx[f"AP@{round(t,2)}"])
    r["mAP@[.5:.95]"] = round(float(np.mean(aps)), 2)
    return r


# ══════════════════════════════════════════════════════════════
# 2.  TRACKING METRICS  (MOTA / MOTP / IDF1 via motmetrics)
# ══════════════════════════════════════════════════════════════

def _build_accumulator(gt_dict, mot_dict, iou_thresh=0.5):
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(gt_dict.keys()) | set(mot_dict.keys())):
        gt_rows  = gt_dict.get(frame, [])
        mot_rows = mot_dict.get(frame, [])
        gt_ids   = [r[0] for r in gt_rows]
        mot_ids  = [r[0] for r in mot_rows]
        gt_boxes = [[r[1], r[2], r[3], r[4]] for r in gt_rows]
        mot_boxes= [[r[1], r[2], r[3], r[4]] for r in mot_rows]

        if gt_boxes and mot_boxes:
            dist = mm.distances.iou_matrix(
                np.array(gt_boxes), np.array(mot_boxes), max_iou=iou_thresh)
        else:
            dist = np.empty((len(gt_boxes), len(mot_boxes)))
            dist[:] = np.nan
        acc.update(gt_ids, mot_ids, dist)
    return acc


def _extract_mot_metrics(acc, name="seq"):
    mh      = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=name)
    res = summary.iloc[0]
    return {
        "MOTA":  round(float(res["mota"])      * 100, 2),
        "MOTP":  round(float(res["motp"])      * 100, 2),
        "IDF1":  round(float(res["idf1"])      * 100, 2),
        "MT":    int(res["mostly_tracked"]),
        "ML":    int(res["mostly_lost"]),
        "PT":    int(res["partially_tracked"]),
        "FP_t":  int(res["num_false_positives"]),
        "FN_t":  int(res["num_misses"]),
        "IDs":   int(res["num_switches"]),
        "Frag":  int(res["num_fragmentations"]),
        "Prec_t":round(float(res["precision"]) * 100, 2),
        "Rec_t": round(float(res["recall"])    * 100, 2),
    }


# ══════════════════════════════════════════════════════════════
# 3.  HOTA
# ══════════════════════════════════════════════════════════════

def compute_hota(gt_dict, mot_dict, alphas=None):
    if alphas is None:
        alphas = np.arange(0.05, 1.00, 0.05)

    all_frames = sorted(set(gt_dict.keys()) | set(mot_dict.keys()))
    hota_vals  = []

    for alpha in alphas:
        TP = FP = FN = 0
        pair_tpa   = defaultdict(int)
        gt_total   = defaultdict(int)
        pred_total = defaultdict(int)

        for frame in all_frames:
            gt_rows  = gt_dict.get(frame,  [])
            mot_rows = mot_dict.get(frame, [])
            gt_ids   = [r[0] for r in gt_rows]
            mot_ids  = [r[0] for r in mot_rows]
            gt_boxes  = [[r[1],r[2],r[3],r[4]] for r in gt_rows]
            mot_boxes = [[r[1],r[2],r[3],r[4]] for r in mot_rows]

            if not gt_boxes and not mot_boxes:
                continue
            iou = iou_matrix(gt_boxes, mot_boxes) if (gt_boxes and mot_boxes) \
                  else np.zeros((len(gt_boxes), len(mot_boxes)))

            matched_g = set(); matched_d = set()
            pairs_s = sorted(
                [(iou[g, d], g, d) for g in range(len(gt_ids))
                                   for d in range(len(mot_ids))],
                reverse=True)
            for val, g, d in pairs_s:
                if val >= alpha and g not in matched_g and d not in matched_d:
                    matched_g.add(g); matched_d.add(d)
                    pair_tpa[(gt_ids[g], mot_ids[d])] += 1

            tp = len(matched_g)
            TP += tp;  FP += len(mot_ids) - tp;  FN += len(gt_ids) - tp
            for gid in gt_ids:  gt_total[gid]   += 1
            for pid in mot_ids: pred_total[pid]  += 1

        DetA = TP / (TP + FP + FN + 1e-9)
        assa_n = assa_d = 0.0
        for (gid, pid), tpa in pair_tpa.items():
            fpa = max(0, pred_total.get(pid, tpa) - tpa)
            fna = max(0, gt_total.get(gid,  tpa) - tpa)
            assa_n += tpa;  assa_d += tpa + fpa + fna
        AssA = assa_n / (assa_d + 1e-9) if assa_d > 0 else 0.0
        hota_vals.append((np.sqrt(DetA * AssA), DetA, AssA))

    h = np.array(hota_vals)
    return {
        "HOTA": round(float(h[:, 0].mean()) * 100, 2),
        "DetA": round(float(h[:, 1].mean()) * 100, 2),
        "AssA": round(float(h[:, 2].mean()) * 100, 2),
    }


# ══════════════════════════════════════════════════════════════
# 4.  EMBEDDING METRICS
# ══════════════════════════════════════════════════════════════

def compute_emb_metrics(emb_path, mot_dict, n_sample=5000):
    try:
        embs = load_embs(emb_path)
    except Exception as e:
        return {"Error": str(e)}

    N, D  = embs.shape
    norms = np.linalg.norm(embs, axis=1)

    row_idx    = 0
    track_embs = defaultdict(list)
    for frame in sorted(mot_dict.keys()):
        for t in mot_dict[frame]:
            if row_idx < N:
                track_embs[t[0]].append(row_idx)
                row_idx += 1

    def cosine(a, b):
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b))

    track_ids = [tid for tid, idxs in track_embs.items() if len(idxs) >= 2]
    rng       = np.random.default_rng(42)

    intra, inter = [], []
    for tid in track_ids:
        idxs = track_embs[tid]
        for _ in range(min(10, len(idxs))):
            i, j = rng.choice(len(idxs), size=2, replace=False)
            intra.append(cosine(embs[idxs[i]], embs[idxs[j]]))
        if len(intra) > n_sample:
            break

    if len(track_ids) >= 2:
        for _ in range(min(n_sample, 5000)):
            t1, t2 = rng.choice(len(track_ids), size=2, replace=False)
            i1 = rng.choice(track_embs[track_ids[t1]])
            i2 = rng.choice(track_embs[track_ids[t2]])
            inter.append(cosine(embs[i1], embs[i2]))

    intra_m = float(np.mean(intra)) if intra else 0.0
    inter_m = float(np.mean(inter)) if inter else 0.0

    return {
        "Emb-dim":             D,
        "N vectors":           N,
        "Mean L2 norm":        round(float(np.mean(norms)), 4),
        "Intra-track cos-sim": round(intra_m, 4),
        "Inter-track cos-sim": round(inter_m, 4),
        "Discriminability":    round(intra_m - inter_m, 4),
        "Tracks sampled":      len(track_ids),
    }


# ══════════════════════════════════════════════════════════════
# SEQUENCE DISCOVERY
# ══════════════════════════════════════════════════════════════

def discover_sequences(gt_dir, dets_dir, mot_dir, embs_dir=None):
    """
    Scan gt_dir for *.txt files.  Strip optional "gt_" prefix to get seq stem.
    Match stems against dets_dir and mot_dir.
    Returns list of dicts: seq_name, gt, dets, mot, embs (or None).
    """
    gt_files = sorted(Path(gt_dir).glob("*.txt"))
    if not gt_files:
        sys.exit(f"[ERROR] No .txt files found in GT folder: {gt_dir}")

    seqs    = []
    missing = []
    for gt_path in gt_files:
        stem     = gt_path.stem
        seq_name = stem[3:] if stem.startswith("gt_") else stem

        dets_path = Path(dets_dir) / f"{seq_name}.txt"
        mot_path  = Path(mot_dir)  / f"{seq_name}.txt"

        skip = False
        if not dets_path.exists():
            missing.append(f"  DETS missing: {dets_path}"); skip = True
        if not mot_path.exists():
            missing.append(f"  MOT  missing: {mot_path}");  skip = True
        if skip:
            continue

        embs_path = None
        if embs_dir:
            for ext in [".txt", ".npy", ".npz"]:
                c = Path(embs_dir) / f"{seq_name}{ext}"
                if c.exists():
                    embs_path = str(c); break

        seqs.append(dict(seq_name=seq_name, gt=str(gt_path),
                         dets=str(dets_path), mot=str(mot_path),
                         embs=embs_path))

    if missing:
        print("\n[WARN] Skipped sequences (missing files):")
        for m in missing: print(m)

    return seqs


# ══════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════

def agg_det_metrics(det_results_list):
    tp = fp = fn = gt_t = det_t = 0
    ap50_v = []; ap75_v = []; map_v = []
    for r in det_results_list:
        tp   += r["_tp"];  fp += r["_fp"];  fn += r["_fn"]
        gt_t += r["_total_gt"];  det_t += r["_total_dets"]
        ap50_v.append(r.get("AP@0.5",        0))
        ap75_v.append(r.get("AP@0.75",       0))
        map_v.append( r.get("mAP@[.5:.95]",  0))
    prec = tp / (tp + fp + 1e-9) * 100
    rec  = tp / (tp + fn + 1e-9) * 100
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision (%)":     round(prec, 2),
        "Recall (%)":        round(rec,  2),
        "F1 (%)":            round(f1,   2),
        "AP@0.5 (macro %)":  round(float(np.mean(ap50_v)), 2),
        "AP@0.75 (macro %)": round(float(np.mean(ap75_v)), 2),
        "mAP@[.5:.95] (%)":  round(float(np.mean(map_v)),  2),
        "Total GT boxes":    gt_t,
        "Total detections":  det_t,
    }


def agg_mot_metrics(accumulator_list, hota_list):
    mh      = mm.metrics.create()
    summary = mh.compute_many(
        accumulator_list,
        metrics=mm.metrics.motchallenge_metrics,
        names=[f"seq{i}" for i in range(len(accumulator_list))],
        generate_overall=True,
    )
    ov = summary.loc["OVERALL"]
    mot = {
        "MOTA (%)":   round(float(ov["mota"])      * 100, 2),
        "MOTP (%)":   round(float(ov["motp"])      * 100, 2),
        "IDF1 (%)":   round(float(ov["idf1"])      * 100, 2),
        "MT":         int(ov["mostly_tracked"]),
        "ML":         int(ov["mostly_lost"]),
        "PT":         int(ov["partially_tracked"]),
        "FP (track)": int(ov["num_false_positives"]),
        "FN (track)": int(ov["num_misses"]),
        "ID-sw":      int(ov["num_switches"]),
        "Frag":       int(ov["num_fragmentations"]),
    }
    if hota_list:
        mot["HOTA (%)"] = round(float(np.mean([h["HOTA"] for h in hota_list])), 2)
        mot["DetA (%)"] = round(float(np.mean([h["DetA"] for h in hota_list])), 2)
        mot["AssA (%)"] = round(float(np.mean([h["AssA"] for h in hota_list])), 2)
    return mot


def agg_emb_metrics(emb_results_list):
    valid = [r for r in emb_results_list if r and "Error" not in r]
    if not valid: return {}
    keys = ["Intra-track cos-sim", "Inter-track cos-sim", "Discriminability"]
    out  = {}
    for k in keys:
        vals = [r[k] for r in valid if k in r]
        out[k + " (mean)"] = round(float(np.mean(vals)), 4) if vals else "N/A"
    out["Sequences with embs"] = len(valid)
    return out


# ══════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ══════════════════════════════════════════════════════════════

W = 72

def banner(title):
    print("\n╔" + "═"*(W-2) + "╗")
    print("║" + title.center(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")

def section(title):
    print("\n" + "─"*W)
    print(f"  {title}")
    print("─"*W)

def print_kv(d):
    print(tabulate([[k, v] for k, v in d.items()],
                   headers=["Metric", "Value"],
                   tablefmt="rounded_outline"))

def print_wide(rows, headers):
    print(tabulate(rows, headers=headers,
                   tablefmt="rounded_outline",
                   floatfmt=".2f", numalign="right"))


# ══════════════════════════════════════════════════════════════
# CSV SAVE
# ══════════════════════════════════════════════════════════════

def save_csv(per_seq_rows, overall_det, overall_mot, overall_emb, out_path):
    all_rows = per_seq_rows + [
        {"Sequence": "OVERALL (detection)", **overall_det},
        {"Sequence": "OVERALL (tracking)",  **overall_mot},
    ]
    if overall_emb:
        all_rows.append({"Sequence": "OVERALL (emb)", **overall_emb})

    fields = sorted({k for row in all_rows for k in row},
                    key=lambda x: (x != "Sequence", x))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            if row: w.writerow(row)
    print(f"\n  Results saved → {out_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VisDrone MOT Evaluation — folder mode")
    parser.add_argument("--gt",   required=True, help="Folder: GT annotation .txt files")
    parser.add_argument("--dets", required=True, help="Folder: dets .txt files  (runs/.../dets/)")
    parser.add_argument("--mot",  required=True, help="Folder: MOT output .txt files  (runs/MOT/...)")
    parser.add_argument("--embs", default=None,  help="Folder: embs files  (optional)")
    parser.add_argument("--fps",  type=float, default=30.0, help="Video FPS (default: 30)")
    parser.add_argument("--iou",  type=float, default=0.5,  help="IoU threshold (default: 0.5)")
    parser.add_argument("--out",  default="evaluate_results.csv", help="CSV output path")
    args = parser.parse_args()

    banner("VisDrone MOT Evaluation Suite  │  Val-Set Folder Mode")
    print(f"  GT folder   : {args.gt}")
    print(f"  Dets folder : {args.dets}")
    print(f"  MOT folder  : {args.mot}")
    print(f"  Embs folder : {args.embs or '(none)'}")
    print(f"  FPS: {args.fps}   IoU threshold: {args.iou}")

    # ── Discover sequences ───────────────────────────────────
    seqs = discover_sequences(args.gt, args.dets, args.mot, args.embs)
    if not seqs:
        sys.exit("[ERROR] No matching sequences found. Check folder paths.")
    n_seqs = len(seqs)
    print(f"\n  Found {n_seqs} sequence(s):")
    for s in seqs:
        emb_tag = "  [embs ✓]" if s["embs"] else ""
        print(f"    • {s['seq_name']}{emb_tag}")

    # ── Storage ──────────────────────────────────────────────
    det_results_all  = []
    accumulators     = []
    hota_results_all = []
    emb_results_all  = []
    per_seq_csv_rows = []

    # Wide table data
    det_rows = []; mot_rows = []; emb_rows = []
    det_hdr  = ["Sequence", "Frames", "GT-boxes", "Dets",
                "Prec%", "Rec%", "F1%", "AP@.5%", "AP@.75%", "mAP%"]
    mot_hdr  = ["Sequence", "MOTA%", "MOTP%", "IDF1%",
                "HOTA%", "DetA%", "AssA%", "MT", "ML", "IDs", "Frag"]
    emb_hdr  = ["Sequence", "Emb-dim", "N-vecs",
                "Intra-sim", "Inter-sim", "Discrim"]

    t_global = time.time()

    # ── Per-sequence loop ────────────────────────────────────
    for idx, seq in enumerate(seqs, 1):
        name = seq["seq_name"]
        section(f"[{idx}/{n_seqs}]  {name}")
        t0 = time.time()

        # Load
        print(f"  Loading GT   …", end=" ", flush=True)
        gt_dict  = load_gt(seq["gt"])
        print(f"{sum(len(v) for v in gt_dict.values())} boxes / {len(gt_dict)} frames")

        print(f"  Loading dets …", end=" ", flush=True)
        det_dict = load_dets(seq["dets"])
        print(f"{sum(len(v) for v in det_dict.values())} dets / {len(det_dict)} frames")

        print(f"  Loading MOT  …", end=" ", flush=True)
        mot_dict = load_mot(seq["mot"])
        print(f"{sum(len(v) for v in mot_dict.values())} tracked boxes / {len(mot_dict)} frames")

        n_frames = max(
            max(gt_dict.keys())  if gt_dict  else 0,
            max(mot_dict.keys()) if mot_dict else 0,
        )

        # ── Detection ──────────────────────────────────────
        print("  Detection metrics  …", end=" ", flush=True)
        td = time.time()
        dm = compute_ap_multi_thresh(gt_dict, det_dict)
        det_results_all.append(dm)
        print(f"{time.time()-td:.1f}s  "
              f"Prec={dm['Precision']}%  Rec={dm['Recall']}%  "
              f"AP@.5={dm['AP@0.5']}%  mAP={dm['mAP@[.5:.95]']}%")

        det_rows.append([
            name, n_frames, dm["_total_gt"], dm["_total_dets"],
            dm["Precision"], dm["Recall"], dm["F1"],
            dm["AP@0.5"], dm.get("AP@0.75", 0), dm.get("mAP@[.5:.95]", 0),
        ])

        # ── MOTA / MOTP / IDF1 ────────────────────────────
        print("  MOTA / MOTP / IDF1 …", end=" ", flush=True)
        tm = time.time()
        acc    = _build_accumulator(gt_dict, mot_dict, args.iou)
        accumulators.append(acc)
        mm_res = _extract_mot_metrics(acc, name)
        print(f"{time.time()-tm:.1f}s  "
              f"MOTA={mm_res['MOTA']}%  MOTP={mm_res['MOTP']}%  IDF1={mm_res['IDF1']}%")

        # ── HOTA ──────────────────────────────────────────
        print("  HOTA / DetA / AssA …", end=" ", flush=True)
        th    = time.time()
        hota  = compute_hota(gt_dict, mot_dict)
        hota_results_all.append(hota)
        print(f"{time.time()-th:.1f}s  "
              f"HOTA={hota['HOTA']}%  DetA={hota['DetA']}%  AssA={hota['AssA']}%")

        gt_tracks   = set(r[0] for v in gt_dict.values()  for r in v)
        pred_tracks = set(r[0] for v in mot_dict.values() for r in v)

        mot_rows.append([
            name,
            mm_res["MOTA"], mm_res["MOTP"], mm_res["IDF1"],
            hota["HOTA"],   hota["DetA"],   hota["AssA"],
            mm_res["MT"],   mm_res["ML"],
            mm_res["IDs"],  mm_res["Frag"],
        ])

        # ── Embeddings ────────────────────────────────────
        emb_res = {}
        if seq["embs"]:
            print(f"  Emb metrics        …", end=" ", flush=True)
            te     = time.time()
            emb_res = compute_emb_metrics(seq["embs"], mot_dict)
            emb_results_all.append(emb_res)
            print(f"{time.time()-te:.1f}s  "
                  f"Discrim={emb_res.get('Discriminability', 'N/A')}")
            emb_rows.append([
                name,
                emb_res.get("Emb-dim",             "—"),
                emb_res.get("N vectors",            "—"),
                emb_res.get("Intra-track cos-sim",  "—"),
                emb_res.get("Inter-track cos-sim",  "—"),
                emb_res.get("Discriminability",     "—"),
            ])

        # CSV row
        csv_row = {
            "Sequence":     name,
            "Frames":       n_frames,
            "GT-tracks":    len(gt_tracks),
            "Pred-tracks":  len(pred_tracks),
        }
        csv_row.update({k: v for k, v in dm.items()     if not k.startswith("_")})
        csv_row.update(mm_res)
        csv_row.update(hota)
        if emb_res and "Error" not in emb_res:
            csv_row.update(emb_res)
        per_seq_csv_rows.append(csv_row)

        print(f"  ── done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════
    # OVERALL SUMMARY
    # ══════════════════════════════════════════════════════
    banner(f"Overall Val-Set Results  │  {n_seqs} sequence(s)")

    section("1 ▸ DETECTION METRICS — Overall + Per-Sequence")
    overall_det = agg_det_metrics(det_results_all)
    print("  Overall:")
    print_kv(overall_det)
    print("\n  Per-sequence:")
    print_wide(det_rows, det_hdr)

    section("2 ▸ TRACKING METRICS — Overall + Per-Sequence")
    overall_mot = agg_mot_metrics(accumulators, hota_results_all)
    print("  Overall:")
    print_kv(overall_mot)
    print("\n  Per-sequence:")
    print_wide(mot_rows, mot_hdr)

    overall_emb = {}
    if emb_rows:
        section("3 ▸ EMBEDDING METRICS — Overall + Per-Sequence")
        overall_emb = agg_emb_metrics(emb_results_all)
        if overall_emb:
            print("  Overall:")
            print_kv(overall_emb)
        print("\n  Per-sequence:")
        print_wide(emb_rows, emb_hdr)

    section("Metric Glossary")
    glossary = [
        ("MOTA  (%)",          "Multi-Object Tracking Accuracy (FP+FN+IDs). ↑ better"),
        ("MOTP  (%)",          "Multi-Object Tracking Precision (mean IoU of matched pairs). ↑ better"),
        ("IDF1  (%)",          "ID F1 — correctly-IDed dets / avg(GT, pred). ↑ better"),
        ("HOTA  (%)",          "sqrt(DetA × AssA) — balanced detection + association. ↑ better"),
        ("DetA  (%)",          "Detection Accuracy component of HOTA"),
        ("AssA  (%)",          "Association Accuracy component of HOTA (trajectory continuity)"),
        ("MT",                 "Mostly Tracked: GT tracks covered >80% of their lifetime"),
        ("ML",                 "Mostly Lost: GT tracks covered <20% of their lifetime"),
        ("IDs",                "ID Switches: tracker gives wrong ID to an existing target"),
        ("Frag",               "Fragmentations: track interrupted then resumed"),
        ("AP@0.5  (%)",        "Average Precision at IoU=0.50. ↑ better"),
        ("mAP@[.5:.95] (%)",   "COCO-style mAP over IoU 0.50→0.95. ↑ better"),
        ("Discriminability",   "Intra−Inter cosine sim: ReID ID separation quality. ↑ better"),
    ]
    for k, v in glossary:
        print(f"  {k:<24}  {v}")

    save_csv(per_seq_csv_rows, overall_det, overall_mot, overall_emb, args.out)

    total = time.time() - t_global
    print(f"\n  Total time: {total:.1f}s across {n_seqs} sequence(s)\n")


if __name__ == "__main__":
    main()
