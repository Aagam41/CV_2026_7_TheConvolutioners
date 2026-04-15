"""Benchmark several trackers on the same dataset and print one metrics table.

Per BoxMOT's recommended workflow:
    1. Generate detections + ReID embeddings ONCE per detector/reid combo.
    2. Run val() per tracker — it auto-reuses the cache for fast comparison.

Usage:
    python scripts/compare_trackers.py --config configs/compare.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import Boxmot  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare trackers via Boxmot API.")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    detector = cfg["detector"]
    reid = cfg.get("reid")
    device = cfg.get("device")
    half = cfg.get("half", False)
    benchmark = cfg["benchmark"]
    val_extras = cfg.get("val", {})

    # 1) Pre-compute detections + embeddings once.
    cache = Boxmot(detector=detector, reid=reid, device=device, half=half).generate(
        benchmark=benchmark
    )
    print(f"Cache ready at {cache.cache_dir} (elapsed {cache.timings.get('elapsed', 0):.1f}s)")

    # 2) Evaluate every tracker against the same cache.
    rows: list[dict] = []
    for tracker in cfg["trackers"]:
        bm = Boxmot(detector=detector, reid=reid, tracker=tracker,
                    device=device, half=half)
        result = bm.val(benchmark=benchmark, **val_extras)
        rows.append({"tracker": tracker, **result.metrics})
        print(f"\n[{tracker}] {result}")

    # 3) Pretty-print comparison.
    if not rows:
        return
    keys = ("HOTA", "MOTA", "IDF1", "AssA", "DetA", "IDs", "FP", "FN")
    headers = ["tracker"] + [k for k in keys if any(k in r for r in rows)]
    print("\n" + " | ".join(f"{h:>9}" for h in headers))
    print("-" * (12 * len(headers)))
    for r in rows:
        cells = [f"{r['tracker']:>9}"]
        for k in headers[1:]:
            cells.append(f"{r.get(k, float('nan')):>9.3f}")
        print(" | ".join(cells))


if __name__ == "__main__":
    main()
