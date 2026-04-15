"""Example: drop a custom Siamese ReID network into BoxMOT's BotSort.

BoxMOT's BotSort uses an internal `ReidAutoBackend` for appearance features.
The cleanest way to swap in a custom Siamese (or transformer-based) model is
to subclass BotSort and override the appearance-feature extractor.

This file is a *template* — wire your own torch.nn.Module + preprocessing
into `MySiameseReID.extract`. Then register the tracker so YAML can use it.

YAML usage after registration:
    tracker:
      type: botsort_siamese
      reid_weights: weights/my_siamese.pt
      device: cuda:0
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from boxmot import BotSort

from src import register_tracker


# ---------------------------------------------------------------------------
# 1. Your Siamese embedding network. Replace this stub with your real model.
# ---------------------------------------------------------------------------
class MySiameseBackbone(nn.Module):
    """Minimal placeholder. Swap with your trained Siamese / triplet net."""

    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return torch.nn.functional.normalize(z, dim=-1)


# ---------------------------------------------------------------------------
# 2. Feature extractor used by the tracker (one method: crops -> (N, D) array).
# ---------------------------------------------------------------------------
class MySiameseReID:
    def __init__(self, weights: str | Path, device: str = "cuda:0",
                 input_size=(128, 256)):
        self.device = torch.device(device)
        self.size = input_size
        self.model = MySiameseBackbone().to(self.device).eval()
        if Path(weights).exists():
            sd = torch.load(weights, map_location=self.device)
            self.model.load_state_dict(sd, strict=False)

    @torch.no_grad()
    def extract(self, frame: np.ndarray, xyxys: np.ndarray) -> np.ndarray:
        if len(xyxys) == 0:
            return np.empty((0, self.model.embed_dim), dtype=np.float32)

        crops: List[np.ndarray] = []
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in xyxys.astype(int):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((self.size[1], self.size[0], 3), np.uint8))
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], self.size)
            crops.append(crop)

        batch = np.stack(crops).astype(np.float32) / 255.0
        batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        feats = self.model(batch).cpu().numpy()
        return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Subclass BotSort and override appearance extraction.
# ---------------------------------------------------------------------------
class BotSortSiamese(BotSort):
    """BotSort with a user-supplied Siamese appearance model."""

    def __init__(self, reid_weights, device="cuda:0", half=False, **kw):
        super().__init__(reid_weights=reid_weights, device=device, half=half, **kw)
        self.model = MySiameseReID(weights=reid_weights, device=device)

    # BoxMOT calls self.model(...) internally; our class exposes `.extract`
    # via a callable interface to stay drop-in compatible.
    def __call__(self, *args, **kwargs):
        return self.model.extract(*args, **kwargs)


# ---------------------------------------------------------------------------
# 4. Register so it becomes available as `type: botsort_siamese` in YAML.
# ---------------------------------------------------------------------------
register_tracker("botsort_siamese", BotSortSiamese)


if __name__ == "__main__":
    print("Registered: botsort_siamese")
    print("Use it in any YAML by setting tracker.type: botsort_siamese")
