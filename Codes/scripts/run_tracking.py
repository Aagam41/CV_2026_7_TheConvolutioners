"""Run tracking on a video using the BoxMOT-consistent Boxmot API.

Every YAML key under `track:` is forwarded to BoxMOT's CLI, so you can use
ANY flag BoxMOT supports — see usage.md for the full list.

Usage:
    python scripts/run_tracking.py --config configs/single.yaml
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
    ap = argparse.ArgumentParser(description="Track a video via the Boxmot API.")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    boxmot = Boxmot(
        detector=cfg.get("detector"),
        reid=cfg.get("reid"),
        tracker=cfg.get("tracker"),
        device=cfg.get("device"),
        half=cfg.get("half", False),
    )

    track_kwargs = dict(cfg.get("track", {}))
    source = track_kwargs.pop("source")
    run = boxmot.track(source=source, **track_kwargs)
    print(run)
    if run.video:
        print(f"Video: {run.video}")
    if run.txt:
        print(f"MOT txt: {run.txt}")

    val_cfg = cfg.get("val")
    if val_cfg:
        result = boxmot.val(**val_cfg)
        print(result)
        for k, v in result.metrics.items():
            print(f"  {k:>6} = {v:.3f}")


if __name__ == "__main__":
    main()
