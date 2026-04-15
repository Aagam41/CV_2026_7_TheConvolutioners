"""Abstract detector interface.

All detectors must return a numpy array of shape (N, 6) with columns:
    [x1, y1, x2, y2, conf, cls]
This is the canonical format consumed by every BoxMOT tracker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """Plug-and-play detector contract."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection on a BGR frame and return (N, 6) ndarray."""
        raise NotImplementedError
