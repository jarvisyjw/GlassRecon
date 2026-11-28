"""Exercise demand, segment boundaries and cancellation across the real worker protocol."""

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

from glassrecon.orb_stream import open_orb_poses


@pytest.fixture
def sequence(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "masks").mkdir()
    camera = dict(width=2, height=2, fx=10.0, fy=10.0, cx=0.5, cy=0.5)
    (dataset / "intrinsics.json").write_text(json.dumps(camera))
    for i in range(6):
        rgb = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
        Image.fromarray(rgb).save(dataset / f"frame{i}.png")
        np.save(dataset / f"frame{i}.npy", np.full((2, 2), 1250.0))
        Image.fromarray(np.array([[0, 255], [255, 255]], dtype=np.uint8)).save(
            dataset / "masks" / f"frame{i}.png"
        )
    for name, ext in (("rgb", "png"), ("depth", "npy")):
        (dataset / (name + ".txt")).write_text(
            "".join(f"{i / 10 + 100:.1f} frame{i}.{ext}\n" for i in range(6))
        )
    (tmp_path / "pyorbslam3.py").write_text(
        """
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
Sensor = SimpleNamespace(RGBD="RGBD")
class Slam:
    def __init__(self, vocab, settings, sensor, viewer):
        self.log = Path(vocab)
        self.i = 0
        self.log.write_text("started\\n")
        config = settings.read_text()
        assert "loopClosing: 0" in config and "Camera.fps: 10" in config
        assert sensor == "RGBD" and not viewer
    def track_rgbd(self, rgb, depth, timestamp):
        os.write(1, b"native diagnostic\\n")
        assert rgb[1,1].tolist() == [10,20,30] and rgb[0,0].tolist() == [0,0,0]
        assert depth[1,1] == 1.25 and depth[0,0] == 0
        state = ["NOT_INITIALIZED","OK","OK_KLT","RECENTLY_LOST","OK","OK"][self.i]
        self.i += 1
        if self.i == 2 and os.getenv("TEST_ORB_FAILURE") == "abrupt":
            os._exit(17)
        if self.i == 2 and os.getenv("TEST_ORB_FAILURE") == "error":
            raise ValueError("test tracking error")
        pose = np.eye(4)
        pose[0,3] = self.i
        with self.log.open("a") as f: f.write(str(timestamp) + "\\n")
        return SimpleNamespace(pose_valid=True, state=SimpleNamespace(name=state),
                               pose_wc=pose, pose_cw=np.linalg.inv(pose))
    def shutdown(self):
        with self.log.open("a") as f: f.write("shutdown\\n")
"""
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return dataset, tmp_path / "worker.log"


def test_pull_one_frame_and_separate_worlds(sequence):
    dataset, log = sequence
    with open_orb_poses(dataset, None, 1000, sys.executable, log) as poses:
        assert log.read_text().splitlines() == ["started"]
        records = [next(poses) for _ in range(5)]
        # Never request frame 5. Closing the consumer must release the native worker.
    assert log.read_text().splitlines() == [
        "started",
        "100.0",
        "100.1",
        "100.2",
        "100.3",
        "100.4",
        "shutdown",
    ]
    assert [r.map_id for r in records] == [None, 0, 0, None, 1]
    assert records[3].state == "RECENTLY_LOST" and records[3].pose is None
    assert records[4].key == "frame4" and records[4].timestamp == 100.4 and records[4].index == 4
    assert records[4].pose[0, 3] == 5


@pytest.mark.parametrize(
    "failure,detail", [("error", "test tracking error"), ("abrupt", "native diagnostic")]
)
def test_worker_failure_is_visible(sequence, monkeypatch, failure, detail):
    dataset, log = sequence
    monkeypatch.setenv("TEST_ORB_FAILURE", failure)
    with pytest.raises(RuntimeError, match=detail):
        with open_orb_poses(dataset, None, 1000, sys.executable, log) as poses:
            list(poses)
