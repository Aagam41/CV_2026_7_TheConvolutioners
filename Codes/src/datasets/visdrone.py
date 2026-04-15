"""VisDrone-MOT dataset support.

Folder layout the user is assumed to have on disk:

    <root>/
        VisDrone2019-MOT-train/
            sequences/<seq>/0000001.jpg ...
            annotations/<seq>.txt
        VisDrone2019-MOT-val/
            sequences/<seq>/...
            annotations/<seq>.txt
        VisDrone2019-MOT-test-dev/
            sequences/<seq>/...
            annotations/<seq>.txt          # available for test-dev too

Each annotation file is a CSV with one row per (frame, object):

    <frame>,<id>,<x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>

VisDrone categories (1-indexed; 0 + 11 are ignored):
    0: ignored-regions   1: pedestrian   2: people     3: bicycle
    4: car               5: van          6: truck      7: tricycle
    8: awning-tricycle   9: bus          10: motor     11: others
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

SPLIT_DIRS = {
    "train":    "VisDrone2019-MOT-train",
    "val":      "VisDrone2019-MOT-val",
    "test-dev": "VisDrone2019-MOT-test-dev",
    "testdev":  "VisDrone2019-MOT-test-dev",
}

# Categories worth tracking by default (drop ignored-regions=0 and others=11).
DEFAULT_KEEP_CATEGORIES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

CATEGORY_NAMES = {
    0: "ignored-regions", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle",
    8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}


# ---------------------------------------------------------------------------
@dataclass
class Sequence:
    name: str                # e.g. 'uav0000013_00000_v'
    img_dir: Path            # folder of jpgs
    ann_path: Optional[Path] # annotation txt (may be missing for blind splits)

    @property
    def n_frames(self) -> int:
        return sum(1 for _ in self.img_dir.glob("*.jpg"))


# ---------------------------------------------------------------------------
class VisDroneMOT:
    """Light handle on a VisDrone-MOT split."""

    def __init__(self, root: str | Path, split: str = "train") -> None:
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be one of {sorted(SPLIT_DIRS)}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / SPLIT_DIRS[split]
        self.seq_root = self.split_dir / "sequences"
        self.ann_root = self.split_dir / "annotations"
        if not self.seq_root.is_dir():
            raise FileNotFoundError(f"No sequences folder at {self.seq_root}")

    # --------------------------------------------------------- iteration
    def sequences(self) -> List[Sequence]:
        seqs: List[Sequence] = []
        for d in sorted(self.seq_root.iterdir()):
            if not d.is_dir():
                continue
            ann = self.ann_root / f"{d.name}.txt"
            seqs.append(Sequence(name=d.name, img_dir=d,
                                 ann_path=ann if ann.exists() else None))
        return seqs

    def __iter__(self) -> Iterator[Sequence]:
        return iter(self.sequences())

    def __len__(self) -> int:
        return len(self.sequences())

    # -------------------------------------------------- format conversion
    @staticmethod
    def visdrone_to_motchallenge(
        visdrone_txt: str | Path,
        out_path: str | Path,
        keep_categories: Sequence[int] = DEFAULT_KEEP_CATEGORIES,
        require_score_one: bool = True,
    ) -> Path:
        """Convert one VisDrone annotation file to MOT-Challenge ``gt.txt``.

        MOT-Challenge ``gt.txt`` columns:
            <frame>,<id>,<x>,<y>,<w>,<h>,<conf>,<class>,<visibility>

        - ``conf`` here is the "consider in eval" flag (1 or 0).
          We drop rows where the VisDrone score column is 0 by default.
        - ``visibility`` is derived from the VisDrone occlusion column
          (0 → 1.0, 1 → 0.5, 2 → 0.1).
        - ``class`` is mapped to 1 for any kept category (pedestrian-style),
          which is what TrackEval's default MOT loader expects.
        """
        in_path = Path(visdrone_txt)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise FileNotFoundError(in_path)

        rows = np.loadtxt(in_path, delimiter=",")
        if rows.size == 0:
            out_path.write_text("")
            return out_path
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)

        keep_set = set(keep_categories)
        vis_map = {0: 1.0, 1: 0.5, 2: 0.1}
        out_lines: List[str] = []
        for r in rows:
            frame, tid, x, y, w, h, score, cat, trunc, occ = r[:10]
            if require_score_one and int(score) != 1:
                continue
            if int(cat) not in keep_set:
                continue
            visibility = vis_map.get(int(occ), 1.0)
            out_lines.append(
                f"{int(frame)},{int(tid)},"
                f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"1,1,{visibility:.2f}"
            )
        out_path.write_text("\n".join(out_lines))
        return out_path

    # -------------------------------------- build TrackEval-style GT root
    def export_motchallenge_gt(
        self,
        out_root: str | Path,
        benchmark_name: str = "VisDrone-MOT",
        keep_categories: Sequence[int] = DEFAULT_KEEP_CATEGORIES,
        write_seqmap: bool = True,
        fps: int = 30,
    ) -> Path:
        """Materialise a MOTChallenge-style GT folder TrackEval can read.

        Output layout::

            <out_root>/<benchmark>-<split>/<seq>/gt/gt.txt
            <out_root>/<benchmark>-<split>/<seq>/seqinfo.ini
            <out_root>/seqmaps/<benchmark>-<split>.txt
        """
        out_root = Path(out_root)
        bench_split = f"{benchmark_name}-{self.split.replace('-', '')}"
        bench_dir = out_root / bench_split
        bench_dir.mkdir(parents=True, exist_ok=True)

        names: List[str] = []
        for seq in self.sequences():
            seq_out = bench_dir / seq.name
            (seq_out / "gt").mkdir(parents=True, exist_ok=True)

            if seq.ann_path is None:
                log.warning("No annotation for sequence %s — skipping GT export", seq.name)
            else:
                self.visdrone_to_motchallenge(
                    seq.ann_path, seq_out / "gt" / "gt.txt",
                    keep_categories=keep_categories,
                )

            # seqinfo.ini — TrackEval uses imWidth, imHeight, seqLength, frameRate
            n_frames = seq.n_frames
            w, h = _probe_image_size(seq.img_dir)
            (seq_out / "seqinfo.ini").write_text(
                "[Sequence]\n"
                f"name={seq.name}\n"
                f"imDir=img1\n"
                f"frameRate={fps}\n"
                f"seqLength={n_frames}\n"
                f"imWidth={w}\nimHeight={h}\n"
                "imExt=.jpg\n"
            )
            # symlink (or copy) frames into the conventional 'img1' folder.
            img1 = seq_out / "img1"
            if not img1.exists():
                try:
                    img1.symlink_to(seq.img_dir.resolve())
                except OSError:
                    import shutil
                    shutil.copytree(seq.img_dir, img1)
            names.append(seq.name)

        if write_seqmap:
            seqmaps = out_root / "seqmaps"
            seqmaps.mkdir(parents=True, exist_ok=True)
            (seqmaps / f"{bench_split}.txt").write_text(
                "name\n" + "\n".join(names) + "\n"
            )
        log.info("Exported %d sequences to %s", len(names), bench_dir)
        return bench_dir


# ---------------------------------------------------------------------------
def _probe_image_size(img_dir: Path) -> tuple[int, int]:
    """Return (width, height) of the first .jpg in ``img_dir``."""
    try:
        import cv2
        first = next(img_dir.glob("*.jpg"))
        im = cv2.imread(str(first))
        h, w = im.shape[:2]
        return w, h
    except Exception:
        return 1920, 1080


__all__ = [
    "VisDroneMOT", "Sequence",
    "DEFAULT_KEEP_CATEGORIES", "CATEGORY_NAMES", "SPLIT_DIRS",
]
