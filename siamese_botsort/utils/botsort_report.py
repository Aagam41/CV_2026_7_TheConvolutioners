from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# Repo root is one level above this file (utils/)
_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV summaries and plots from eval_botsort run JSON files."
    )
    parser.add_argument(
        "--in-dir",
        default=str(_REPO_ROOT / "eval_botsort"),
        help="Directory containing overall_metrics_XXX.json and per_sequence_metrics_XXX.json",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_REPO_ROOT / "eval_botsort" / "reports"),
        help="Directory to save generated CSVs and figures",
    )
    return parser.parse_args()


def extract_run_id(path: Path) -> int | None:
    match = re.search(r"_(\d+)\.json$", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(in_dir: Path) -> List[Dict]:
    runs: List[Dict] = []
    for overall_path in sorted(in_dir.glob("overall_metrics_*.json")):
        run_id = extract_run_id(overall_path)
        if run_id is None:
            continue

        per_seq_path = in_dir / f"per_sequence_metrics_{run_id:03d}.json"
        if not per_seq_path.exists():
            print(f"[Warn] Missing per-sequence file for run {run_id:03d}, skipping")
            continue

        overall = load_json(overall_path)
        per_sequence = load_json(per_seq_path)

        runs.append(
            {
                "run_id": run_id,
                "overall": overall,
                "per_sequence": per_sequence,
            }
        )

    return sorted(runs, key=lambda x: x["run_id"])


def write_overall_csv(runs: List[Dict], out_path: Path) -> None:
    fieldnames = [
        "run_id",
        "overall_fps",
        "mean_hota_50_approx",
        "mean_deta_50",
        "mean_assa_50",
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
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            overall = run["overall"]
            mm_overall = overall.get("motmetrics_overall", {})
            writer.writerow(
                {
                    "run_id": run["run_id"],
                    "overall_fps": overall.get("overall_fps", 0.0),
                    "mean_hota_50_approx": overall.get("mean_hota_50_approx", 0.0),
                    "mean_deta_50": overall.get("mean_deta_50", 0.0),
                    "mean_assa_50": overall.get("mean_assa_50", 0.0),
                    "mota": mm_overall.get("mota", 0.0),
                    "motp": mm_overall.get("motp", 0.0),
                    "idf1": mm_overall.get("idf1", 0.0),
                    "idp": mm_overall.get("idp", 0.0),
                    "idr": mm_overall.get("idr", 0.0),
                    "num_switches": mm_overall.get("num_switches", 0.0),
                    "num_false_positives": mm_overall.get("num_false_positives", 0.0),
                    "num_misses": mm_overall.get("num_misses", 0.0),
                    "mostly_tracked": mm_overall.get("mostly_tracked", 0.0),
                    "partially_tracked": mm_overall.get("partially_tracked", 0.0),
                    "mostly_lost": mm_overall.get("mostly_lost", 0.0),
                    "num_fragmentations": mm_overall.get("num_fragmentations", 0.0),
                }
            )


def write_per_sequence_csv(runs: List[Dict], out_path: Path) -> None:
    fieldnames = [
        "run_id",
        "split",
        "sequence",
        "frames",
        "fps",
        "mota",
        "motp",
        "idf1",
        "idp",
        "idr",
        "deta_50",
        "assa_50",
        "hota_50_approx",
        "num_switches",
        "num_false_positives",
        "num_misses",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            run_id = run["run_id"]
            for row in run["per_sequence"]:
                writer.writerow(
                    {
                        "run_id": run_id,
                        "split": row.get("split", ""),
                        "sequence": row.get("sequence", ""),
                        "frames": row.get("frames", 0),
                        "fps": row.get("fps", 0.0),
                        "mota": row.get("mota", 0.0),
                        "motp": row.get("motp", 0.0),
                        "idf1": row.get("idf1", 0.0),
                        "idp": row.get("idp", 0.0),
                        "idr": row.get("idr", 0.0),
                        "deta_50": row.get("deta_50", 0.0),
                        "assa_50": row.get("assa_50", 0.0),
                        "hota_50_approx": row.get("hota_50_approx", 0.0),
                        "num_switches": row.get("num_switches", 0.0),
                        "num_false_positives": row.get("num_false_positives", 0.0),
                        "num_misses": row.get("num_misses", 0.0),
                    }
                )


def plot_overall_trends(runs: List[Dict], out_path: Path) -> None:
    run_ids = [r["run_id"] for r in runs]
    mota = [r["overall"].get("motmetrics_overall", {}).get("mota", 0.0) for r in runs]
    idf1 = [r["overall"].get("motmetrics_overall", {}).get("idf1", 0.0) for r in runs]
    hota = [r["overall"].get("mean_hota_50_approx", 0.0) for r in runs]
    deta = [r["overall"].get("mean_deta_50", 0.0) for r in runs]
    assa = [r["overall"].get("mean_assa_50", 0.0) for r in runs]

    plt.figure(figsize=(10, 6))
    plt.plot(run_ids, mota, marker="o", label="MOTA")
    plt.plot(run_ids, idf1, marker="o", label="IDF1")
    plt.plot(run_ids, hota, marker="o", label="HOTA@0.5~")
    plt.plot(run_ids, deta, marker="o", label="DetA@0.5")
    plt.plot(run_ids, assa, marker="o", label="AssA@0.5")

    plt.title("BoT-SORT Overall Metrics by Run")
    plt.xlabel("Run ID")
    plt.ylabel("Metric value")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_last_run_sequences(last_run: Dict, out_path: Path) -> None:
    rows = last_run["per_sequence"]
    labels = [f"{r.get('split', '')}/{r.get('sequence', '')}" for r in rows]
    mota = [r.get("mota", 0.0) for r in rows]
    idf1 = [r.get("idf1", 0.0) for r in rows]
    hota = [r.get("hota_50_approx", 0.0) for r in rows]

    x = list(range(len(labels)))
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar([i - width for i in x], mota, width=width, label="MOTA")
    plt.bar(x, idf1, width=width, label="IDF1")
    plt.bar([i + width for i in x], hota, width=width, label="HOTA@0.5~")

    plt.title(f"BoT-SORT Per-Sequence Metrics (Run {last_run['run_id']:03d})")
    plt.xlabel("Sequence")
    plt.ylabel("Metric value")
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylim(min(-0.1, min(mota) - 0.02), 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(in_dir)
    if not runs:
        raise SystemExit(f"No runs found in {in_dir}")

    overall_csv = out_dir / "overall_metrics_runs.csv"
    per_sequence_csv = out_dir / "per_sequence_metrics_runs.csv"
    trends_png = out_dir / "overall_trends.png"
    last_run_png = out_dir / "last_run_per_sequence.png"

    write_overall_csv(runs, overall_csv)
    write_per_sequence_csv(runs, per_sequence_csv)
    plot_overall_trends(runs, trends_png)
    plot_last_run_sequences(runs[-1], last_run_png)

    print(f"[Info] Loaded {len(runs)} run(s) from {in_dir}")
    print(f"[Info] Wrote CSV: {overall_csv}")
    print(f"[Info] Wrote CSV: {per_sequence_csv}")
    print(f"[Info] Wrote plot: {trends_png}")
    print(f"[Info] Wrote plot: {last_run_png}")


if __name__ == "__main__":
    main()
