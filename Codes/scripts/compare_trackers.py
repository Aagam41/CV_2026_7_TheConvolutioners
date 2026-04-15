"""Benchmark several trackers via BoxMOT 17's native Boxmot facade.

Per BoxMOT's recommended workflow:
    1. Generate detections + ReID embeddings ONCE per detector/reid combo.
    2. Run val() per tracker — auto-reuses the cache for fast comparison.

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
    device = cfg.get("device", "cuda:0")
    half = cfg.get("half", False)
    benchmark = cfg["benchmark"]
    val_extras = cfg.get("val", {})

    # 1) Pre-compute detections + embeddings once.
    cache = Boxmot(detector=detector, reid=reid).generate(
        benchmark=benchmark, device=device, half=half,
    )
    logging.info("Cache ready at %s", cache.cache_dir)

    # 2) Evaluate every tracker against the same cache.
    rows: list[dict] = []
    for tracker in cfg["trackers"]:
        bm = Boxmot(detector=detector, reid=reid, tracker=tracker)
        result = bm.val(benchmark=benchmark, device=device, half=half, **val_extras)
        rows.append({"tracker": tracker, "result": result})
        logging.info("[%s] %s", tracker, result)

    print("\n=== Comparison ===")
    for row in rows:
        print(f"{row['tracker']:>12}  {row['result']}")


if __name__ == "__main__":
    main()
