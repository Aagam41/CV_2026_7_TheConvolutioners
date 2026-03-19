# Usage: python eval_botsort.py --split val
#        python eval_botsort.py --split both
#        python eval_botsort.py --split val --max-seqs 1   # quick smoke test
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from botsort import BotSort

# Repo root == directory containing this script.
_REPO_ROOT = Path(__file__).resolve().parent


# motmetrics versions used in many environments still call np.asfarray,
# which was removed in NumPy 2.0.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)


@dataclass
class SequenceStats:
    split: str
    sequence: str
    frames: int
    runtime_sec: float
    fps: float
    mota: float
    motp: float
    idf1: float
    idp: float
    idr: float
    num_switches: float
    num_false_positives: float
    num_misses: float
    mostly_tracked: float
    partially_tracked: float
    mostly_lost: float
    num_fragmentations: float
    deta_50: float
    assa_50: float
    hota_50_approx: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate custom BotSort on VisDrone VID splits (val/test-dev)."
    )
    parser.add_argument("--data-root", default=str(_REPO_ROOT / "data"), help="Path to project data folder")
    parser.add_argument(
        "--split",
        choices=["val", "test-dev", "both"],
        default="both",
        help="Which VID split to evaluate",
    )
    parser.add_argument(
        "--model",
        default=str(_REPO_ROOT / "siamese_final.pth"),
        help="Path to trained Siamese model weights",
    )
    parser.add_argument(
        "--detector",
        default=str(_REPO_ROOT / "yolov8n.pt"),
        help="Ultralytics detector model path/name",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--max-seqs", type=int, default=0, help="Limit number of sequences per split (0 = all)")
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for MOT matching and HOTA approximation",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Optional comma-separated VisDrone class ids to evaluate (e.g. 1,4,5). Empty = all classes.",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Save tracker predictions in MOT txt format under eval_outputs/preds",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "eval_botsort"),
        help="Directory for metrics reports and optional prediction dumps",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for BotSort Siamese encoder (cuda/cpu). Default: auto",
    )
    return parser.parse_args()


def parse_class_filter(classes_arg: str) -> set[int] | None:
    if not classes_arg.strip():
        return None
    return {int(x.strip()) for x in classes_arg.split(",") if x.strip()}


def list_sequences(split_root: Path, max_seqs: int) -> List[str]:
    ann_dir = split_root / "annotations"
    seq_names = sorted(p.stem for p in ann_dir.glob("*.txt"))
    if max_seqs > 0:
        seq_names = seq_names[:max_seqs]
    return seq_names


def load_gt_by_frame(annotation_path: Path, class_filter: set[int] | None) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    by_frame: Dict[int, List[Tuple[int, np.ndarray]]] = {}

    with annotation_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue

            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            score = float(row[6])
            cls_id = int(float(row[7]))

            if score <= 0 or w <= 0 or h <= 0:
                continue
            if class_filter is not None and cls_id not in class_filter:
                continue

            by_frame.setdefault(frame_id, []).append((track_id, np.array([x, y, w, h], dtype=np.float32)))

    return by_frame


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def match_frame_iou(
    gt_boxes_xywh: List[np.ndarray],
    pr_boxes_xywh: List[np.ndarray],
    iou_thresh: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    n_gt = len(gt_boxes_xywh)
    n_pr = len(pr_boxes_xywh)

    if n_gt == 0:
        return [], [], list(range(n_pr))
    if n_pr == 0:
        return [], list(range(n_gt)), []

    iou_mat = np.zeros((n_gt, n_pr), dtype=np.float32)
    for i, g in enumerate(gt_boxes_xywh):
        for j, p in enumerate(pr_boxes_xywh):
            iou_mat[i, j] = iou_xywh(g, p)

    cost = 1.0 - iou_mat
    rows, cols = linear_sum_assignment(cost)

    matches: List[Tuple[int, int]] = []
    unmatched_gt = list(range(n_gt))
    unmatched_pr = list(range(n_pr))

    for r, c in zip(rows, cols):
        if iou_mat[r, c] >= iou_thresh:
            matches.append((r, c))
            unmatched_gt.remove(r)
            unmatched_pr.remove(c)

    return matches, unmatched_gt, unmatched_pr


def evaluate_sequence(
    split_name: str,
    seq_name: str,
    split_root: Path,
    detector: YOLO,
    tracker: BotSort,
    class_filter: set[int] | None,
    iou_thresh: float,
    det_conf: float,
    save_preds_dir: Path | None,
) -> tuple[SequenceStats, mm.MOTAccumulator]:
    seq_dir = split_root / "sequences" / seq_name
    ann_path = split_root / "annotations" / f"{seq_name}.txt"
    frame_paths = sorted(seq_dir.glob("*.jpg"))
    gt_by_frame = load_gt_by_frame(ann_path, class_filter)

    if save_preds_dir is not None:
        save_preds_dir.mkdir(parents=True, exist_ok=True)
        pred_file = (save_preds_dir / f"{seq_name}.txt").open("w", newline="")
    else:
        pred_file = None

    acc = mm.MOTAccumulator(auto_id=True)

    # For HOTA-like approximation at alpha=0.5.
    gt_life: Dict[str, int] = {}
    pr_life: Dict[str, int] = {}
    pair_tp: Dict[Tuple[str, str], int] = {}
    tp_total = 0
    fp_total = 0
    fn_total = 0

    tracker.reset()
    t0 = time.perf_counter()

    for frame_idx, frame_path in enumerate(frame_paths, start=1):
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        det = detector(bgr, conf=det_conf, verbose=False)[0]
        det_rows = []
        for box in det.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            det_rows.append([x1, y1, x2, y2, conf])

        det_np = np.array(det_rows, dtype=np.float32) if det_rows else np.empty((0, 5), dtype=np.float32)
        tracks = tracker.update(rgb, det_np)

        pred_ids: List[int] = []
        pred_boxes_xywh: List[np.ndarray] = []
        for row in tracks:
            x1, y1, x2, y2, tid, score = row
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            tid_int = int(tid)
            pred_ids.append(tid_int)
            pred_boxes_xywh.append(np.array([x1, y1, w, h], dtype=np.float32))

            if pred_file is not None:
                pred_file.write(
                    f"{frame_idx},{tid_int},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{float(score):.6f},-1,-1,-1\n"
                )

        gt_entries = gt_by_frame.get(frame_idx, [])
        gt_ids = [g[0] for g in gt_entries]
        gt_boxes_xywh = [g[1] for g in gt_entries]

        # motmetrics update
        gt_arr = np.asarray(gt_boxes_xywh, dtype=np.float32).reshape(-1, 4)
        pr_arr = np.asarray(pred_boxes_xywh, dtype=np.float32).reshape(-1, 4)
        dist = mm.distances.iou_matrix(
            gt_arr,
            pr_arr,
            max_iou=iou_thresh,
        )
        acc.update(gt_ids, pred_ids, dist)

        # HOTA approximation bookkeeping
        seq_gt_id_keys = [f"{split_name}/{seq_name}/gt/{gid}" for gid in gt_ids]
        seq_pr_id_keys = [f"{split_name}/{seq_name}/pr/{pid}" for pid in pred_ids]

        for gk in seq_gt_id_keys:
            gt_life[gk] = gt_life.get(gk, 0) + 1
        for pk in seq_pr_id_keys:
            pr_life[pk] = pr_life.get(pk, 0) + 1

        matches, unmatched_gt, unmatched_pr = match_frame_iou(gt_boxes_xywh, pred_boxes_xywh, iou_thresh)
        tp_total += len(matches)
        fn_total += len(unmatched_gt)
        fp_total += len(unmatched_pr)

        for g_i, p_i in matches:
            pair = (seq_gt_id_keys[g_i], seq_pr_id_keys[p_i])
            pair_tp[pair] = pair_tp.get(pair, 0) + 1

    runtime_sec = time.perf_counter() - t0
    frames = len(frame_paths)
    fps = frames / runtime_sec if runtime_sec > 0 else 0.0

    if pred_file is not None:
        pred_file.close()

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_fragmentations",
        ],
        name=seq_name,
    )
    row = summary.loc[seq_name]

    deta = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0

    assa_num = 0.0
    for (gk, pk), c in pair_tp.items():
        denom = gt_life[gk] + pr_life[pk] - c
        if denom > 0:
            assa_num += c * (c / denom)
    assa = assa_num / tp_total if tp_total > 0 else 0.0
    hota = float(np.sqrt(max(deta, 0.0) * max(assa, 0.0)))

    stats = SequenceStats(
        split=split_name,
        sequence=seq_name,
        frames=frames,
        runtime_sec=runtime_sec,
        fps=fps,
        mota=float(row["mota"]),
        motp=float(row["motp"]),
        idf1=float(row["idf1"]),
        idp=float(row["idp"]),
        idr=float(row["idr"]),
        num_switches=float(row["num_switches"]),
        num_false_positives=float(row["num_false_positives"]),
        num_misses=float(row["num_misses"]),
        mostly_tracked=float(row["mostly_tracked"]),
        partially_tracked=float(row["partially_tracked"]),
        mostly_lost=float(row["mostly_lost"]),
        num_fragmentations=float(row["num_fragmentations"]),
        deta_50=deta,
        assa_50=assa,
        hota_50_approx=hota,
    )
    return stats, acc


def split_roots(data_root: Path, split_arg: str) -> Iterable[Tuple[str, Path]]:
    mapping = {
        "val": data_root / "VisDrone2019-VID-val",
        "test-dev": data_root / "VisDrone2019-VID-test-dev",
    }
    if split_arg == "both":
        return [("val", mapping["val"]), ("test-dev", mapping["test-dev"])]
    return [(split_arg, mapping[split_arg])]


def main() -> None:
    args = parse_args()
    class_filter = parse_class_filter(args.classes)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(args.detector)

    all_stats: List[SequenceStats] = []
    accs: List[mm.MOTAccumulator] = []
    names: List[str] = []

    for split_name, split_root in split_roots(data_root, args.split):
        if not split_root.exists():
            print(f"[Warn] Split path not found, skipping: {split_root}")
            continue

        seqs = list_sequences(split_root, args.max_seqs)
        if not seqs:
            print(f"[Warn] No sequences found for split {split_name}")
            continue

        print(f"[Info] Evaluating split={split_name} with {len(seqs)} sequence(s)")

        split_pred_dir = out_dir / "preds" / split_name if args.save_preds else None

        for seq_name in seqs:
            print(f"[Info] Running {split_name}/{seq_name}")
            tracker = BotSort(model_path=args.model, device=args.device)

            stats, acc = evaluate_sequence(
                split_name=split_name,
                seq_name=seq_name,
                split_root=split_root,
                detector=detector,
                tracker=tracker,
                class_filter=class_filter,
                iou_thresh=args.iou_thresh,
                det_conf=args.conf,
                save_preds_dir=split_pred_dir,
            )

            all_stats.append(stats)
            accs.append(acc)
            names.append(f"{split_name}/{seq_name}")

            print(
                f"  MOTA={stats.mota:.4f} IDF1={stats.idf1:.4f} "
                f"MOTP={stats.motp:.4f} HOTA@0.5~={stats.hota_50_approx:.4f} FPS={stats.fps:.2f}"
            )

    if not all_stats:
        raise SystemExit("No sequences were evaluated.")

    mh = mm.metrics.create()
    overall_summary = mh.compute_many(
        accs,
        names=names,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "num_switches",
            "num_false_positives",
            "num_misses",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_fragmentations",
        ],
        generate_overall=True,
    )

    total_frames = sum(s.frames for s in all_stats)
    total_time = sum(s.runtime_sec for s in all_stats)
    overall_fps = total_frames / total_time if total_time > 0 else 0.0

    mean_hota = float(np.mean([s.hota_50_approx for s in all_stats]))
    mean_deta = float(np.mean([s.deta_50 for s in all_stats]))
    mean_assa = float(np.mean([s.assa_50 for s in all_stats]))

    print("\n========== Overall MOT Summary ==========")
    print(mm.io.render_summary(overall_summary, namemap=mm.io.motchallenge_metric_names))
    print(f"Overall FPS: {overall_fps:.2f}")
    print(f"Mean HOTA@0.5 (approx): {mean_hota:.4f}")
    print(f"Mean DetA@0.5: {mean_deta:.4f}")
    print(f"Mean AssA@0.5: {mean_assa:.4f}")

    # Determine next run ID by scanning existing files in out_dir.
    existing = [p.stem for p in out_dir.glob("overall_metrics_*.json")]
    used_ids = set()
    for stem in existing:
        try:
            used_ids.add(int(stem.split("_")[-1]))
        except ValueError:
            pass
    run_id = 1
    while run_id in used_ids:
        run_id += 1
    run_tag = f"{run_id:03d}"

    # Save machine-readable reports.
    per_sequence_json = [asdict(s) for s in all_stats]
    per_seq_path = out_dir / f"per_sequence_metrics_{run_tag}.json"
    per_seq_path.write_text(json.dumps(per_sequence_json, indent=2), encoding="utf-8")

    overall_payload = {
        "run_id": run_id,
        "overall_fps": overall_fps,
        "mean_hota_50_approx": mean_hota,
        "mean_deta_50": mean_deta,
        "mean_assa_50": mean_assa,
        "motmetrics_overall": overall_summary.loc["OVERALL"].to_dict(),
    }
    overall_path = out_dir / f"overall_metrics_{run_tag}.json"
    overall_path.write_text(json.dumps(overall_payload, indent=2), encoding="utf-8")

    print(f"\n[Info] Saved reports to: {out_dir} (run {run_tag})")


if __name__ == "__main__":
    main()
