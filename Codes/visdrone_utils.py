"""
visdrone_utils.py
-----------------
Helpers for reading VisDrone-MOT sequences and writing MOT-format
tracking results compatible with TrackEval / official VisDrone devkit.

VisDrone-MOT folder layout
---------------------------
VisDrone2019-MOT-train/
    annotations/     ← <seq>.txt  (ground-truth, not used here)
    sequences/
        uav0000013_00000_v/
            0000001.jpg
            0000002.jpg
            ...
        ...

Output (MOT-format .txt)
------------------------
<frame>,<id>,<left>,<top>,<width>,<height>,<conf>,<x>,<y>,<z>
where x, y, z = -1 for 2-D tracking (MOT convention).
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterator, Tuple, List


# ---------------------------------------------------------------------------
# Sequence reader
# ---------------------------------------------------------------------------

class VisDroneSequence:
    """
    Iterates over frames of a single VisDrone-MOT sequence directory.

    Usage
    -----
    seq = VisDroneSequence('/path/to/sequences/uav0000013_00000_v')
    for frame_id, frame in seq:
        ...
    """

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

    def __init__(self, seq_dir: str | Path):
        self.seq_dir = Path(seq_dir)
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

        self.name = self.seq_dir.name
        self._frame_paths: List[Path] = sorted(
            p for p in self.seq_dir.iterdir()
            if p.suffix.lower() in self.IMAGE_EXTS
        )
        if not self._frame_paths:
            raise ValueError(f"No images found in {seq_dir}")

    # ------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return len(self._frame_paths)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_frames

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yields (1-based frame_id, BGR frame)."""
        for idx, path in enumerate(self._frame_paths, start=1):
            frame = cv2.imread(str(path))
            if frame is None:
                raise IOError(f"Could not read image: {path}")
            yield idx, frame

    # ------------------------------------------------------------------
    def frame_size(self) -> Tuple[int, int]:
        """Returns (width, height) of the first frame."""
        frame = cv2.imread(str(self._frame_paths[0]))
        h, w = frame.shape[:2]
        return w, h

    def __repr__(self) -> str:
        return f"VisDroneSequence('{self.name}', {self.n_frames} frames)"


# ---------------------------------------------------------------------------
# Dataset scanner
# ---------------------------------------------------------------------------

def list_sequences(split_dir: str | Path) -> List[Path]:
    """
    Return sorted list of sequence directories under
    <split_dir>/sequences/.
    """
    seq_root = Path(split_dir) / "sequences"
    if not seq_root.is_dir():
        raise FileNotFoundError(f"sequences/ not found under {split_dir}")
    return sorted(p for p in seq_root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# MOT result writer
# ---------------------------------------------------------------------------

class MOTResultWriter:
    """
    Accumulates per-frame tracking results and writes them to the
    MOT-format text file expected by TrackEval.

    MOT line format:
        <frame>,<id>,<left>,<top>,<width>,<height>,<conf>,-1,-1,-1
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lines: List[str] = []

    # ------------------------------------------------------------------
    def update(self, frame_id: int, tracks: np.ndarray) -> None:
        if tracks is None or len(tracks) == 0:
            return

        for t in tracks:
            x1, y1, x2, y2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            track_id = int(t[4])
            conf = float(t[5])
            w = x2 - x1
            h = y2 - y1
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            w = max(1.0, w)
            h = max(1.0, h)
            self._lines.append(
                f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                f"{conf:.4f},-1,-1,-1"
            )

    # ------------------------------------------------------------------
    def save(self) -> None:
        with open(self.output_path, "w") as f:
            f.write("\n".join(self._lines))
            if self._lines:
                f.write("\n")
        print(f"  [MOTResultWriter] Saved {len(self._lines)} track lines → {self.output_path}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._lines)


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def make_video_writer(
    output_path: str | Path,
    width: int,
    height: int,
    fps: float = 30.0,
) -> cv2.VideoWriter:
    """Create an mp4 VideoWriter."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter for {output_path}")
    return writer


# ---------------------------------------------------------------------------
# Trail buffer
# ---------------------------------------------------------------------------

class TrailBuffer:
    """
    Keeps the last `max_len` centre-points for every track ID so that
    draw_tracks() can render a fading motion trail.

    Usage
    -----
    trails = TrailBuffer(max_len=40)

    # inside your per-frame loop:
    trails.update(tracks)
    draw_tracks(vis_frame, tracks, trails)
    """

    def __init__(self, max_len: int = 40):
        self.max_len = max_len
        # dict[track_id → deque of (cx, cy)]
        self._history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_len))

    # ------------------------------------------------------------------
    def update(self, tracks: np.ndarray) -> None:
        """Push the current-frame centre of each active track."""
        if tracks is None or len(tracks) == 0:
            return
        for t in tracks:
            tid = int(t[4])
            cx = int((t[0] + t[2]) / 2)
            cy = int((t[1] + t[3]) / 2)
            self._history[tid].append((cx, cy))

    # ------------------------------------------------------------------
    def get(self, track_id: int) -> deque:
        return self._history.get(track_id, deque())

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Call at the start of each new sequence."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Colour palette & drawing helpers
# ---------------------------------------------------------------------------

# BGR colour palette – indexed by track_id % len(PALETTE)
PALETTE = [
    (255,  56,  56),   # red
    ( 56, 255,  56),   # green
    ( 56,  56, 255),   # blue
    (255, 157,  56),   # orange
    (255, 225,  56),   # yellow
    ( 56, 255, 225),   # cyan
    (255,  56, 225),   # magenta
    (128, 255,  56),   # lime
    ( 56, 128, 255),   # sky
    (200,  56, 255),   # violet
    (255, 128, 128),   # salmon
    (128, 255, 128),   # mint
]


def _color(track_id: int) -> Tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]


def draw_tracks(
    frame: np.ndarray,
    tracks: np.ndarray,
    trails: "TrailBuffer | None" = None,
    show_conf: bool = True,
    trail_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes, track ID labels, and (optionally) motion trails.

    Parameters
    ----------
    frame          : BGR image (H, W, 3)
    tracks         : BoxMOT output (N, 8)  [x1,y1,x2,y2,id,conf,cls,ind]
    trails         : TrailBuffer instance (pass None to skip trails)
    show_conf      : append confidence score to the label
    trail_thickness: pixel width of trail polyline
    """
    if tracks is None or len(tracks) == 0:
        return frame

    for t in tracks:
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        track_id = int(t[4])
        conf = float(t[5])
        color = _color(track_id)

        # ── Trail ─────────────────────────────────────────────────────
        if trails is not None:
            pts = list(trails.get(track_id))   # oldest → newest
            n = len(pts)
            for i in range(1, n):
                # Fade: alpha goes from ~30% (oldest) to 100% (newest)
                alpha = (i / n)
                faded = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(trail_thickness * alpha))
                cv2.line(frame, pts[i - 1], pts[i], faded, thickness, cv2.LINE_AA)

            # Solid dot at the current head
            if pts:
                cv2.circle(frame, pts[-1], trail_thickness + 1, color, -1, cv2.LINE_AA)

        # ── Bounding box ──────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # ── Label ─────────────────────────────────────────────────────
        label = f"ID:{track_id}"
        if show_conf:
            label += f" {conf:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        )
        lx, ly = x1, max(y1 - 4, th + 2)
        # Small filled rectangle behind text for readability
        cv2.rectangle(frame, (lx, ly - th - baseline), (lx + tw, ly + baseline),
                      color, cv2.FILLED)
        cv2.putText(
            frame, label,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return frame