"""Run the smoke tests with optional deps stubbed out.

Loads ``smoke_test.py`` by file path so we never collide with Colab's
system-wide ``tests`` package at /usr/local/lib/python3.12/dist-packages/tests.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional deps so the package imports without installing them.
# We use real submodule objects to avoid the "X is not a package" error
# when something tries `from X.Y import Z` later.
def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []   # marks it as a package so submodule imports work
    sys.modules[name] = mod
    return mod

for name in ["ultralytics", "sahi", "boxmot", "cv2", "tqdm"]:
    if name not in sys.modules:
        _stub(name)
# Sub-modules of stubbed packages.
if "sahi.predict" not in sys.modules:
    _stub("sahi.predict")

sys.modules["ultralytics"].YOLO = lambda *a, **k: None
sys.modules["sahi"].AutoDetectionModel = type(
    "X", (), {"from_pretrained": staticmethod(lambda **k: None)}
)
sys.modules["sahi.predict"].get_sliced_prediction = lambda *a, **k: None
for cls in ["BotSort", "ByteTrack", "DeepOcSort", "OcSort",
            "StrongSort", "ImprAssoc", "BoostTrack"]:
    setattr(sys.modules["boxmot"], cls, type(cls, (), {}))
sys.modules["cv2"].VideoCapture = lambda *a, **k: None
sys.modules["cv2"].VideoWriter = lambda *a, **k: None
sys.modules["cv2"].VideoWriter_fourcc = lambda *a: 0
sys.modules["cv2"].imread = lambda *a, **k: None
sys.modules["tqdm"].tqdm = lambda *a, **k: iter([])

# Load smoke_test BY FILE PATH so we don't import the system `tests` package.
spec = importlib.util.spec_from_file_location(
    "_local_smoke_test", Path(__file__).parent / "smoke_test.py"
)
smoke_test = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke_test)

smoke_test.test_track_argv()
smoke_test.test_val_argv()
smoke_test.test_generate_argv()
smoke_test.test_extra_passthrough()
print("\nAll smoke tests PASSED.")
