"""Single tracking run via the BoxMOT 17 native Boxmot facade.

Usage:
    python scripts/run_tracking.py --config configs/single.yaml

For features not exposed by Boxmot.track() (show-trajectories, save-crop,
target-id, per-class), use scripts/visdrone_run.py which shells out to the
BoxMOT CLI.
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
    ap = argparse.ArgumentParser(description="Track via the Boxmot API.")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    bm = Boxmot(
        detector=cfg.get("detector"),
        reid=cfg.get("reid"),
        tracker=cfg.get("tracker"),
        classes=cfg.get("classes"),
        project=cfg.get("project", "outputs/track"),
    )

    track_kwargs = dict(cfg.get("track", {}))
    source = track_kwargs.pop("source")
    run = bm.track(source=source, **track_kwargs)
    print(run)

    val_cfg = cfg.get("val")
    if val_cfg:
        result = bm.val(**val_cfg)
        print(result)


if __name__ == "__main__":
    main()
