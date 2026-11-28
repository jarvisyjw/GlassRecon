import numpy as np
import pytest

from glassrecon.data import CameraIntrinsics
from glassrecon.multiview import SlidingWindowRecovery, WindowConfig


def test_fixed_lag_order_flush_and_no_future_mutation():
    cam = CameraIntrinsics(32, 24, 30, 30, 15.5, 11.5)
    solver = SlidingWindowRecovery(cam, WindowConfig(size=3, sample_step=2))
    depth = np.full((24, 32), 2.0, np.float32)
    rgb = np.zeros((24, 32, 3), np.uint8)
    emitted = []
    for i in range(6):
        node = solver.push(str(i), float(i), rgb, depth, depth, np.eye(4))
        if i < 2:
            assert node is None
        if node is not None:
            emitted.append((node["key"], node["depth"].copy()))
            assert len(solver.pending) == 2
    emitted += [(n["key"], n["depth"].copy()) for n in solver.flush()]
    assert [k for k, _ in emitted] == list(map(str, range(6)))
    assert solver.emitted == 6 and list(solver.flush()) == []
    for _, z in emitted:
        np.testing.assert_allclose(z, 2, atol=1e-6)
    with pytest.raises(ValueError, match="increasing"):
        solver.push("bad", 3.0, rgb, depth, depth, np.eye(4))


def test_joint_window_reduces_affine_flicker_with_metric_sensor_anchors():
    cam = CameraIntrinsics(64, 48, 60, 60, 31.5, 23.5)
    yy, xx = np.indices((48, 64))
    true = 2 + 0.005 * xx + 0.003 * yy
    rgb = np.zeros((48, 64, 3), np.uint8)
    solver = SlidingWindowRecovery(cam, WindowConfig(size=5, sample_step=4, temporal_weight=2))
    results = []
    for i in range(16):
        bias = 0.025 * (-1) ** i
        sensor = true + bias
        base = true * (1 + 0.02 * np.sin(i)) + 0.02 * np.cos(i)
        node = solver.push(str(i), float(i), rgb, sensor, base, np.eye(4))
        if node:
            results.append(node)
    results += list(solver.flush())
    joint = np.stack([n["depth"] for n in results])
    separate = np.stack([n["anchor_depth"] for n in results])
    assert np.std(joint, axis=0).mean() < np.std(separate, axis=0).mean() * 0.8
    assert np.abs(joint - true).mean() < 0.02
    assert all(
        n["diagnostics"]["objective_after"] <= n["diagnostics"]["objective_before"] + 1e-8
        for n in results
    )


def test_sensor_occlusions_and_color_disagreement_exclude_pairs():
    cam = CameraIntrinsics(32, 24, 30, 30, 15.5, 11.5)
    rgb = np.zeros((24, 32, 3), np.uint8)
    z = np.full((24, 32), 2.0)
    solver = SlidingWindowRecovery(cam, WindowConfig(size=2, sample_step=2))
    solver.push("a", 0.0, rgb, z, z, np.eye(4))
    node = solver.push("b", 1.0, rgb, z + 2, z + 2, np.eye(4))
    assert node["diagnostics"]["correspondences"] == 0
    solver = SlidingWindowRecovery(cam, WindowConfig(size=2, sample_step=2))
    solver.push("a", 0.0, rgb, z, z, np.eye(4))
    node = solver.push("b", 1.0, rgb + 255, z, z, np.eye(4))
    assert node["diagnostics"]["correspondences"] == 0


def test_moving_camera_exact_plane_remains_metric():
    cam = CameraIntrinsics(48, 32, 45, 45, 23.5, 15.5)
    _, xx = np.indices((32, 48))
    rays_x = (xx - cam.cx) / cam.fx
    rgb = np.zeros((32, 48, 3), np.uint8)
    solver = SlidingWindowRecovery(cam, WindowConfig(size=3, sample_step=2))
    results = []
    expected = {}
    for i in range(6):
        pose = np.eye(4)
        pose[0, 3] = i * 0.03
        pose[2, 3] = i * 0.01
        depth = (2 + 0.2 * pose[0, 3] - pose[2, 3]) / (1 - 0.2 * rays_x)
        expected[str(i)] = depth
        node = solver.push(str(i), float(i), rgb, depth, depth, pose)
        if node:
            results.append(node)
    results += list(solver.flush())
    for node in results:
        assert np.abs(node["depth"] - expected[node["key"]]).mean() < 0.002


@pytest.mark.parametrize("missing_pose", [False, True])
def test_multiview_cli_flushes_every_selected_frame_once(tmp_path, missing_pose):
    import json

    pytest.importorskip("open3d")
    from PIL import Image

    from examples.make_synthetic_sequence import make_sequence
    from glassrecon.cli import main

    dataset = make_sequence(tmp_path / "sequence", frame_count=7, corrupt_sensor=False)
    pose_path = dataset / "groundtruth.txt"
    if missing_pose:
        # One dropout at 33 Hz is close enough for the default external-pose interpolation.
        poses = np.loadtxt(pose_path)
        np.savetxt(pose_path, np.delete(poses, 2, axis=0), fmt="%.9f")
    masks = dataset / "masks"
    masks.mkdir()
    for frame in (dataset / "rgb").glob("*.png"):
        with Image.open(frame) as image:
            mask = np.full((image.height, image.width), 255, dtype=np.uint8)
        mask[:, :8] = 0
        Image.fromarray(mask).save(masks / frame.name)
    output = tmp_path / "map"
    main(
        [
            "reconstruct",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--intrinsics",
            str(dataset / "intrinsics.json"),
            "--model",
            "cached",
            "--prediction-dir",
            str(dataset / "predictions"),
            "--valid-mask-dir",
            str(masks),
            "--pose-source",
            "file",
            "--pose-file",
            str(dataset / "groundtruth.txt"),
            "--fusion-policy",
            "multiview",
            "--window-size",
            "3",
            "--iterations",
            "3",
            "--save-depths",
        ]
        + (["--pose-max-difference", "0"] if missing_pose else [])
    )
    summary = json.loads((output / "summary.json").read_text())
    expected = 7 - missing_pose
    assert summary["integrated"] == expected and summary["skipped"] == int(missing_pose)
    trajectory = np.loadtxt(output / "trajectory.txt")
    assert trajectory.shape == (expected, 8) and np.all(np.diff(trajectory[:, 0]) > 0)
    depths = list((output / "metric_depths").glob("*.npy"))
    assert len(depths) == expected
    for path in depths:
        depth = np.load(path)
        assert not np.any(depth[:, :8])
        assert np.any(depth[:, 8:] > 0)
    rows = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    emitted = [r["emitted_frame"] for r in rows if r["status"] == "integrated"]
    assert len(emitted) == len(set(emitted)) == expected
    if missing_pose:
        assert "1.060000" not in emitted
        after_gap = next(r for r in rows if r.get("emitted_frame") == "1.090000")
        assert after_gap["multiview"]["boundary_frame"] is None
    else:
        assert sum(r["status"] == "buffered" for r in rows) == 2
    assert json.loads((output / "progress.json").read_text())["integrated"] == expected


def test_zero_temporal_weight_matches_sensor_anchor_ablation():
    cam = CameraIntrinsics(32, 24, 30, 30, 15.5, 11.5)
    yy, xx = np.indices((24, 32))
    true = 2 + 0.006 * xx + 0.004 * yy
    rgb = np.zeros((24, 32, 3), np.uint8)
    solver = SlidingWindowRecovery(cam, WindowConfig(size=3, sample_step=2, temporal_weight=0))
    results = []
    for i in range(6):
        node = solver.push(
            str(i), float(i), rgb, true + 0.01 * (-1) ** i, true * (1 + 0.01 * i), np.eye(4)
        )
        if node is not None:
            results.append(node)
    results += list(solver.flush())
    for node in results:
        np.testing.assert_allclose(node["depth"], node["anchor_depth"], atol=1e-5)
