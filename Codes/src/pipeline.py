"""End-to-end detect → track → render → log pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from .detectors import build_detector
from .sources import open_source
from .trackers import build_tracker
from .visualizer import VideoSink

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    mot_path: Optional[Path]
    video_path: Optional[Path]
    n_frames: int


class TrackingPipeline:
    """Plug-and-play pipeline: any BaseDetector + any BoxMOT tracker."""

    def __init__(self, detector_cfg: Dict[str, Any], tracker_cfg: Dict[str, Any]) -> None:
        log.info("Building detector: %s", detector_cfg.get("type"))
        self.detector = build_detector(detector_cfg)
        log.info("Building tracker:  %s", tracker_cfg.get("type"))
        self.tracker = build_tracker(tracker_cfg)

    def run(
        self,
        source: str | Path,
        output_dir: str | Path,
        save_video: bool = True,
        save_mot: bool = True,
        show_trajectories: bool = True,
        run_name: Optional[str] = None,
        fps: float = 30.0,
    ) -> RunResult:
        source_path = Path(source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = run_name or source_path.stem

        src = open_source(source_path, fps=fps)
        log.info("Source: %s  fps=%.2f size=%s frames=%d",
                 source_path, src.fps, src.size, src.length)

        sink: Optional[VideoSink] = None
        video_path: Optional[Path] = None
        if save_video:
            video_path = output_dir / f"{stem}.mp4"
            sink = VideoSink(video_path, src.fps, src.size)

        mot_lines: list[str] = []
        frame_idx = 0

        with tqdm(total=src.length or None, desc=f"Tracking[{stem}]", unit="f") as pbar:
            for frame_idx, frame in src:
                # 1. Detect.
                dets = self.detector.detect(frame)
                # 2. Track. BoxMOT returns (M, 8): x1,y1,x2,y2,id,conf,cls,det_ind.
                tracks = self.tracker.update(dets, frame)
                # 3. Log MOT-Challenge format lines.
                if save_mot and len(tracks):
                    for t in tracks:
                        x1, y1, x2, y2, tid, conf, cls, _ = t
                        mot_lines.append(
                            f"{frame_idx},{int(tid)},"
                            f"{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},"
                            f"{conf:.4f},-1,-1,-1"
                        )
                # 4. Draw and write frame using BoxMOT's renderer.
                if sink is not None:
                    self.tracker.plot_results(frame, show_trajectories=show_trajectories)
                    sink.write(frame)
                pbar.update(1)

        if sink is not None:
            sink.close()

        mot_path: Optional[Path] = None
        if save_mot:
            mot_path = output_dir / f"{stem}.txt"
            mot_path.write_text("\n".join(mot_lines))
            log.info("MOT results -> %s", mot_path)

        if video_path:
            log.info("Video       -> %s", video_path)

        return RunResult(mot_path=mot_path, video_path=video_path, n_frames=frame_idx)
