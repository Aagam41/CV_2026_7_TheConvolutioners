"""Run scripts/visdrone_run.py from a YAML config (configs/visdrone_*.yaml)."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ds = cfg["dataset"]

    cmd = [sys.executable, str(ROOT / "scripts" / "visdrone_run.py"),
           "--root", str(ds["root"]),
           "--split", ds["split"],
           "--mode", cfg.get("mode", "boxmot"),
           "--tracker", cfg.get("tracker", "botsort"),
           "--device", cfg.get("device", "cuda:0"),
           "--conf", str(cfg.get("conf", 0.3)),
           "--iou", str(cfg.get("iou", 0.7)),
           "--imgsz", str(cfg.get("imgsz", 1280)),
           "--slice-size", str(cfg.get("slice_size", 640)),
           "--out", str(cfg.get("out", "outputs/visdrone")),
           "--gt-out", str(cfg.get("gt_out", "outputs/visdrone/gt")),
           "--limit", str(cfg.get("limit", 0))]

    if cfg.get("mode", "boxmot") == "boxmot":
        cmd += ["--detector", cfg.get("detector", "yolox_x_visdrone"),
                "--reid", cfg.get("reid", "lmbn_n_duke")]
    else:
        cmd += ["--detector", cfg.get("detector", "yolov8n.pt"),
                "--reid-weights", cfg.get("reid_weights", "osnet_x0_25_msmt17.pt")]

    if cfg.get("classes"):
        cmd += ["--classes", *[str(c) for c in cfg["classes"]]]
    for flag, key in [("--half", "half"), ("--per-class", "per_class"),
                      ("--save-video", "save_video"),
                      ("--save-crop", "save_crop"),
                      ("--save-trajectories", "save_trajectories"),
                      ("--evaluate", "evaluate")]:
        if cfg.get(key):
            cmd.append(flag)

    print("RUN:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
