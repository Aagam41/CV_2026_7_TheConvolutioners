"""Tracker factory wrapping BoxMOT 17's create_tracker(...).

In BoxMOT 17 the tracker classes are no longer top-level imports. The
canonical way to instantiate a tracker is `boxmot.trackers.tracker_zoo
.create_tracker(tracker_type, ...)`, which loads the matching default
config from boxmot/configs/trackers/<name>.yaml and applies overrides.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

log = logging.getLogger(__name__)

# Tracker types BoxMOT 17 ships (lowercase, the names you pass to --tracker
# and to create_tracker()).
KNOWN_TRACKERS = (
    "bytetrack",
    "botsort",
    "sfsort",
    "strongsort",
    "ocsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
)

# Custom (subclass) trackers registered at runtime go here. They take
# priority over BoxMOT's built-ins if names collide.
_CUSTOM: Dict[str, Any] = {}


def register_tracker(name: str, factory) -> None:
    """Register a custom tracker.

    `factory` is either a callable taking the same kwargs as create_tracker
    (reid_weights, device, half, per_class, ...) or a class to instantiate
    directly with those kwargs.
    """
    _CUSTOM[name.lower()] = factory


def build_tracker(cfg: Dict[str, Any]):
    """Instantiate a tracker from a config dict.

    Recognised keys (snake_case):
        type            (required) e.g. 'botsort', 'boosttrack', 'ocsort'
        tracker_config  optional path to a custom tracker yaml
        reid_weights    path to a ReID .pt (required for appearance trackers)
        device          'cpu' | 'cuda:0' | ...
        half            bool
        per_class       bool
        params          dict of overrides forwarded as evolve_param_dict
    """
    cfg = dict(cfg)
    name = cfg.pop("type").lower()

    # Custom subclass overrides built-in.
    if name in _CUSTOM:
        return _CUSTOM[name](**cfg)

    if name not in KNOWN_TRACKERS:
        raise ValueError(
            f"Unknown tracker '{name}'. "
            f"Built-in: {KNOWN_TRACKERS}. Custom: {sorted(_CUSTOM)}"
        )

    return create_tracker(
        tracker_type=name,
        tracker_config=cfg.get("tracker_config"),
        reid_weights=Path(cfg["reid_weights"]) if cfg.get("reid_weights") else None,
        device=cfg.get("device", "cuda:0"),
        half=cfg.get("half", False),
        per_class=cfg.get("per_class", False),
        evolve_param_dict=cfg.get("params"),
    )


def list_trackers() -> list:
    return sorted(set(KNOWN_TRACKERS) | set(_CUSTOM))


__all__ = ["build_tracker", "register_tracker", "list_trackers",
           "create_tracker", "get_tracker_config", "KNOWN_TRACKERS"]
