"""Run the smoke test with optional deps stubbed out."""
import sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy optional deps so the package imports without installing them.
for name in ["ultralytics", "sahi", "sahi.predict", "boxmot", "cv2", "tqdm"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["ultralytics"].YOLO = lambda *a, **k: None
sys.modules["sahi"].AutoDetectionModel = type("X", (), {"from_pretrained": staticmethod(lambda **k: None)})
sys.modules["sahi.predict"].get_sliced_prediction = lambda *a, **k: None
for cls in ["BotSort", "ByteTrack", "DeepOcSort", "OcSort",
            "StrongSort", "ImprAssoc", "BoostTrack"]:
    setattr(sys.modules["boxmot"], cls, type(cls, (), {}))
sys.modules["cv2"].VideoCapture = lambda *a, **k: None
sys.modules["cv2"].VideoWriter = lambda *a, **k: None
sys.modules["cv2"].VideoWriter_fourcc = lambda *a: 0
sys.modules["cv2"].imread = lambda *a, **k: None
sys.modules["tqdm"].tqdm = lambda *a, **k: iter([])

# Run the actual tests.
from tests import smoke_test  # noqa: E402
smoke_test.test_track_argv()
smoke_test.test_val_argv()
smoke_test.test_generate_argv()
smoke_test.test_extra_passthrough()
print("\nAll smoke tests PASSED.")
