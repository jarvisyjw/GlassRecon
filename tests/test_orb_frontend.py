"""Contracts between ORB-SLAM3 RGB-D and the reconstruction trajectory reader."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from glassrecon.data import PoseTrajectory
from glassrecon.orb_frontend import _final_poses, localize


def test_raw_rgbd_to_largest_final_map(tmp_path, monkeypatch):
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "masks").mkdir()
    (dataset / "intrinsics.json").write_text(
        '{"width": 2, "height": 2, "fx": 20, "fy": 21, "cx": 0.5, "cy": 0.5}'
    )
    stamps = [1723456789.1234567 + i / 6 for i in range(4)]
    for i in range(4):
        rgb = np.full((2, 2, 3), [20, 75, 124], dtype=np.uint8)
        Image.fromarray(rgb).save(dataset / f"{i}.png")
        np.save(dataset / f"{i}.npy", np.full((2, 2), 2500.0))
        mask = np.array([[0, 255], [255, 255]], dtype=np.uint8)
        Image.fromarray(mask).save(dataset / "masks" / f"{i}.png")
    for kind, ext in (("rgb", "png"), ("depth", "npy")):
        (dataset / (kind + ".txt")).write_text(
            "".join(f"{ts!r} {i}.{ext}\n" for i, ts in enumerate(stamps))
        )
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.touch()

    class Backend:
        def __init__(self, vocab, settings, sensor, use_viewer):
            assert vocab == vocabulary and not use_viewer and sensor == "RGBD"
            config = settings.read_text()
            assert "RGBD.DepthMapFactor: 1.0" in config and "Camera.RGB: 1" in config
            assert "Camera.fps: 6" in config and "Camera1.fx: 20.0" in config
            self.stopped = False

        def track_rgbd(self, rgb, z, ts):
            assert tuple(rgb[1, 1]) == (20, 75, 124)
            assert np.all(rgb[0, 0] == 0)
            np.testing.assert_array_equal(z, [[0, 2.5], [2.5, 2.5]])
            state = "RECENTLY_LOST" if ts == stamps[1] else "OK"
            return SimpleNamespace(pose_valid=True, state=SimpleNamespace(name=state))

        def shutdown(self):
            self.stopped = True

        def save_trajectory_euroc(self, filename):
            assert self.stopped
            # Index 1 is uncertain; index 2 belongs to a different atlas map.
            # ORB can also repeat the preceding row while recently lost.
            rows = [[stamps[i] * 1e9, i, 0, 0, 0, 0, 0, 1] for i in (0, 0, 1, 1, 3)]
            np.savetxt(filename, rows, fmt="%.6f")

    monkeypatch.setitem(
        sys.modules,
        "pyorbslam3",
        SimpleNamespace(Slam=Backend, Sensor=SimpleNamespace(RGBD="RGBD")),
    )
    output = tmp_path / "trajectory"
    stats = localize(dataset, vocabulary, 1000, output)
    assert stats["localized_frames"] == 2 and stats["coverage"] == 0.5
    assert json.loads((output / "summary.json").read_text()) == stats
    result = PoseTrajectory(output / "poses.txt")
    np.testing.assert_array_equal(result.times, [stamps[0], stamps[3]])
    assert result.at(stamps[3], 0)[0, 3] == 3

    # A failed retry should not publish a trajectory or replace earlier output.
    with pytest.raises(ValueError, match="empty output"):
        localize(dataset, vocabulary, 1000, output)
    (dataset / "masks/0.png").unlink()
    with pytest.raises(FileNotFoundError, match="0.png"):
        localize(dataset, vocabulary, 1000, tmp_path / "failed")
    assert not (tmp_path / "failed/poses.txt").exists()


def test_different_poses_at_same_exported_time_are_ambiguous(tmp_path):
    export = tmp_path / "euroc.txt"
    np.savetxt(export, [[1e9, 0, 0, 0, 0, 0, 0, 1], [1e9, 1, 0, 0, 0, 0, 0, 1]])
    with pytest.raises(ValueError, match="duplicate frame timestamps"):
        _final_poses(export, np.array([1.0, 2.0]), np.array([True, True]))
