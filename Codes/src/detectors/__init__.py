"""Detector registry + factory."""
from __future__ import annotations

from typing import Dict, Any

from .base import BaseDetector
from .yolo import YoloDetector
from .sahi_yolo import SahiYoloDetector

_REGISTRY = {
    "yolo": YoloDetector,
    "sahi_yolo": SahiYoloDetector,
}


def register_detector(name: str, cls) -> None:
    """Register a custom detector class so it becomes selectable from config."""
    _REGISTRY[name.lower()] = cls


def build_detector(cfg: Dict[str, Any]) -> BaseDetector:
    """Build a detector from a config dict.

    Required: ``type``. Everything else is forwarded as kwargs (with a few
    aliases unpacked to keep YAML readable).
    """
    cfg = dict(cfg)
    name = cfg.pop("type").lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown detector '{name}'. Available: {sorted(_REGISTRY)}. "
            "Use register_detector() to add your own."
        )

    # Friendly aliases from YAML.
    if "confidence_threshold" in cfg:
        cfg["conf"] = cfg.pop("confidence_threshold")

    # Flatten optional 'sahi:' sub-block for sahi_yolo.
    if name == "sahi_yolo" and "sahi" in cfg:
        cfg.update(cfg.pop("sahi"))

    return _REGISTRY[name](**cfg)


__all__ = ["BaseDetector", "YoloDetector", "SahiYoloDetector",
           "build_detector", "register_detector"]
