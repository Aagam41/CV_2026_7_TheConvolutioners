"""Register a BoxMOT 17 benchmark + dataset for VisDrone-MOT.

BoxMOT 17 looks up benchmarks by id in `boxmot/configs/benchmarks/<id>.yaml`
and datasets in `boxmot/configs/datasets/<id>.yaml`. We materialise both,
pointing the dataset at the MOT-Challenge-format GT we already converted
with VisDroneMOT.export_motchallenge_gt().
"""
from __future__ import annotations

import logging
from pathlib import Path

import boxmot
import yaml

log = logging.getLogger(__name__)


def _boxmot_configs_root() -> Path:
    return Path(boxmot.__file__).parent / "configs"


def register_visdrone_benchmark(
    split: str,
    gt_root: str | Path,
    detector: str = "yolox_x_visdrone",
    reid: str = "lmbn_n_duke",
    benchmark_name: str | None = None,
) -> str:
    """Write the dataset + benchmark YAMLs BoxMOT needs for `bm.val(...)`.

    Args:
        split: 'train' | 'val' | 'testdev' (no dash).
        gt_root: parent of `<benchmark>-<split>/<seq>/gt/gt.txt`. Typically
            the directory you passed as `--gt-out`.
        detector: short name BoxMOT recognises (or a .pt path).
        reid: short name BoxMOT recognises.
        benchmark_name: id used in YAML and bm.val(benchmark=...). Defaults
            to ``visdrone-mot-<split>``.

    Returns:
        The benchmark id you should pass to ``bm.val(benchmark=...)``.
    """
    split = split.replace("-", "")
    benchmark_id = benchmark_name or f"visdrone-mot-{split}"
    dataset_id = benchmark_id

    cfg_root = _boxmot_configs_root()
    ds_dir = cfg_root / "datasets"
    bm_dir = cfg_root / "benchmarks"
    ds_dir.mkdir(parents=True, exist_ok=True)
    bm_dir.mkdir(parents=True, exist_ok=True)

    # Path to the folder our converter created:
    # <gt_root>/VisDrone-MOT-<split>/<seq>/gt/gt.txt
    sequences_root = Path(gt_root).resolve() / f"VisDrone-MOT-{split}"
    if not sequences_root.is_dir():
        raise FileNotFoundError(
            f"Expected sequences at {sequences_root}. "
            "Run VisDroneMOT.export_motchallenge_gt() first."
        )

    dataset_yaml = {
        "id": dataset_id,
        "path": str(sequences_root.parent),  # the gt_root
        "split": f"VisDrone-MOT-{split}",     # subfolder under path
        "layout": "mot",
        "box_type": "aabb",
        "classes": ["pedestrian"],            # we mapped everything to class 1
    }
    benchmark_yaml = {
        "id": benchmark_id,
        "dataset": dataset_id,
        "detector": detector,
        "reid": reid,
    }

    (ds_dir / f"{dataset_id}.yaml").write_text(yaml.safe_dump(dataset_yaml))
    (bm_dir / f"{benchmark_id}.yaml").write_text(yaml.safe_dump(benchmark_yaml))
    log.info("Registered BoxMOT benchmark %r -> %s", benchmark_id, sequences_root)
    return benchmark_id
