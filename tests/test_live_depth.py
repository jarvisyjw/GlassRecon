"""Exercise DA3's process boundary without requiring a CUDA model in unit tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

from glassrecon.live_depth import DA3Worker


@pytest.fixture
def model_python(tmp_path):
    # Keep the real worker process/IPC while substituting a lightweight predictor.
    path = tmp_path / "model-python"
    package_root = str(Path(__file__).resolve().parents[1])
    path.write_text(
        f"#!{sys.executable}\n"
        f"import sys, os\nsys.path.insert(0, {package_root!r})\n"
        """import numpy as np
from glassrecon import live_depth

class Predictor:
    frames = 0

    def predict(self, rgb, key):
        if key == "inference-error":
            raise ValueError("bad model input")
        if key == "crash":
            os._exit(9)
        print("model log on stdout", flush=True)
        self.frames += 1
        return rgb[:, :, 1].astype(np.float32) + self.frames

def load(model, model_id, device, *, process_res):
    assert model == "da3" and device == "cpu" and process_res == 252
    if model_id == "missing":
        raise FileNotFoundError("checkpoint missing")
    return Predictor()

assert sys.argv[1:3] == ["-m", "glassrecon.live_depth"]
live_depth.create_predictor = load
sys.argv = sys.argv[2:]
live_depth.main()
"""
    )
    path.chmod(0o755)
    return path


def test_worker_reuses_model_and_preserves_rgb_order(model_python):
    frame = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)[:, ::2]
    with DA3Worker(model_python, "checkpoint", "cpu", 252) as worker:
        process = worker._process
        first = worker.predict(frame, "frame_01")
        second = worker.predict(frame, "frame_02")
        assert first.dtype == np.float32 and first.shape == frame.shape[:2]
        np.testing.assert_array_equal(first, frame[:, :, 1].astype(np.float32) + 1)
        np.testing.assert_array_equal(second, first + 1)
    assert process.returncode == 0


def test_worker_propagates_model_initialization_error(model_python):
    with pytest.raises(RuntimeError, match="initializing the model.*checkpoint missing"):
        with DA3Worker(model_python, "missing", "cpu", 252):
            pytest.fail("A failed model must never report ready")


@pytest.mark.parametrize(
    ("key", "message"),
    [("inference-error", "predicting inference-error.*bad model input"), ("crash", "DA3 worker")],
)
def test_worker_reports_frame_failure_and_reaps_child(model_python, key, message):
    with pytest.raises(RuntimeError, match=message):
        with DA3Worker(model_python, "checkpoint", "cpu", 252) as worker:
            process = worker._process
            worker.predict(np.zeros((3, 4, 3), dtype=np.uint8), key)
    assert process.returncode not in (None, 0)
