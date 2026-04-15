"""Tracker factory built on top of BoxMOT's Python API.

Every tracker exposes the same interface:
    tracks = tracker.update(dets, frame)         # dets: (N, 6), tracks: (M, 8)
    tracker.plot_results(frame, show_trajectories=True)

Columns of `tracks`: [x1, y1, x2, y2, id, conf, cls, det_ind].
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from boxmot import (
    BoostTrack,
    BotSort,
    ByteTrack,
    DeepOcSort,
    ImprAssoc,
    OcSort,
    StrongSort,
)

# Map yaml-friendly names to BoxMOT classes.
_TRACKERS = {
    "botsort": BotSort,
    "bytetrack": ByteTrack,
    "deepocsort": DeepOcSort,
    "ocsort": OcSort,
    "strongsort": StrongSort,
    "imprassoc": ImprAssoc,
    "boosttrack": BoostTrack,
}

# Trackers that take a ReID model (Siamese / OSNet / CLIP / etc.).
_REID_TRACKERS = {"botsort", "deepocsort", "strongsort", "imprassoc", "boosttrack"}


def register_tracker(name: str, cls) -> None:
    """Register a custom tracker (e.g. your own subclass of BotSort)."""
    _TRACKERS[name.lower()] = cls
    if getattr(cls, "_is_reid", False):
        _REID_TRACKERS.add(name.lower())


def build_tracker(cfg: Dict[str, Any]):
    """Instantiate a BoxMOT tracker from a config dict.

    Recognised top-level keys: ``type``, ``reid_weights``, ``device``, ``half``,
    ``per_class``. Anything under ``params`` is passed verbatim to the tracker
    constructor, so you get full control of BoxMOT's hyperparameters from YAML.
    """
    cfg = dict(cfg)
    name = cfg.pop("type").lower()
    if name not in _TRACKERS:
        raise ValueError(
            f"Unknown tracker '{name}'. Available: {sorted(_TRACKERS)}."
        )

    cls = _TRACKERS[name]
    kwargs: Dict[str, Any] = dict(cfg.pop("params", {}))

    if name in _REID_TRACKERS:
        if "reid_weights" not in cfg:
            raise ValueError(f"Tracker '{name}' requires 'reid_weights' in config.")
        kwargs.setdefault("reid_weights", Path(cfg.pop("reid_weights")))
        kwargs.setdefault("device", cfg.pop("device", "cuda:0"))
        kwargs.setdefault("half", cfg.pop("half", False))
        if "per_class" in cfg:
            kwargs.setdefault("per_class", cfg.pop("per_class"))
    else:
        # Drop reid-specific keys silently if user left them in YAML.
        for k in ("reid_weights", "device", "half", "per_class"):
            cfg.pop(k, None)

    # Anything else in cfg is forwarded too (lets users tweak e.g. track_thresh).
    kwargs.update(cfg)

    return cls(**kwargs)


def list_trackers() -> list:
    return sorted(_TRACKERS)


__all__ = ["build_tracker", "register_tracker", "list_trackers"]
