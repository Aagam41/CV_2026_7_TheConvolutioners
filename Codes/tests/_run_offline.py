"""Run smoke_test with optional deps stubbed.

Loads smoke_test.py by file path so we don't collide with Colab's system
`tests` package at /usr/local/lib/python3.12/dist-packages/tests.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


# Stub heavy optional deps so the package imports without installing them.
for name in ["ultralytics", "sahi", "boxmot", "boxmot.trackers",
             "boxmot.trackers.tracker_zoo", "cv2", "tqdm"]:
    if name not in sys.modules:
        _stub(name)
if "sahi.predict" not in sys.modules:
    _stub("sahi.predict")

sys.modules["ultralytics"].YOLO = lambda *a, **k: None
sys.modules["sahi"].AutoDetectionModel = type(
    "X", (), {"from_pretrained": staticmethod(lambda **k: None)}
)
sys.modules["sahi.predict"].get_sliced_prediction = lambda *a, **k: None
# Minimal Boxmot stub matching v17's signature
sys.modules["boxmot"].Boxmot = type("Boxmot", (), {
    "__init__": lambda self, detector=None, reid=None, tracker=None,
                       classes=None, project=None: None
})
sys.modules["boxmot"].track = lambda *a, **k: None
sys.modules["boxmot"].evaluate = lambda *a, **k: None
sys.modules["boxmot.trackers.tracker_zoo"].create_tracker = lambda **k: None
sys.modules["boxmot.trackers.tracker_zoo"].get_tracker_config = lambda *a, **k: None
sys.modules["cv2"].VideoCapture = lambda *a, **k: None
sys.modules["cv2"].VideoWriter = lambda *a, **k: None
sys.modules["cv2"].VideoWriter_fourcc = lambda *a: 0
sys.modules["cv2"].imread = lambda *a, **k: None
sys.modules["tqdm"].tqdm = lambda *a, **k: iter([])

# Load smoke_test by file path to avoid the system `tests` package.
spec = importlib.util.spec_from_file_location(
    "_local_smoke_test", Path(__file__).parent / "smoke_test.py"
)
smoke_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke_test)

print("(Offline mode: BoxMOT stubbed — for real validation, use real env)")
smoke_test.test_src_imports()
smoke_test.test_tracker_factory()
smoke_test.test_visdrone_dataset()
print("\nOffline smoke tests PASSED.")
