import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from glassrecon.cli import build_parser
from glassrecon.data import CameraIntrinsics
from glassrecon.viewer import LiveReconstructionViewer, camera_pose, load_reconstruction


def test_camera_pose_reorders_quaternion_without_inverting_world_transform():
    row = np.array([12.0, 1.0, 2.0, 3.0, 0.0, 0.6, 0.0, 0.8])
    rotation, position = camera_pose(row)
    np.testing.assert_array_equal(rotation, [0.8, 0.0, 0.6, 0.0])
    np.testing.assert_array_equal(position, [1.0, 2.0, 3.0])


def test_load_reconstruction_preserves_color_scale_and_camera_calibration(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=[2, 4, 6])
    mesh.visual.vertex_colors = np.tile([10, 20, 30, 255], (len(mesh.vertices), 1))
    mesh.export(tmp_path / "mesh.ply")
    points = np.arange(90, dtype=np.float32).reshape(30, 3) / 10
    colors = np.tile([12, 34, 56, 255], (30, 1))
    trimesh.points.PointCloud(points, colors=colors).export(tmp_path / "pointcloud.ply")
    (tmp_path / "trajectory.txt").write_text("0 1 2 3 0 0 0 1\n")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "intrinsics_used": {
                    "width": 640,
                    "height": 480,
                    "fx": 500,
                    "fy": 500,
                    "cx": 320,
                    "cy": 240,
                }
            }
        )
    )
    data = load_reconstruction(tmp_path, max_points=7)
    assert data.total_points == 30 and len(data.points) == 7
    np.testing.assert_array_equal(data.colors, np.tile([12, 34, 56], (7, 1)))
    np.testing.assert_allclose(data.mesh.extents, [2, 4, 6])
    np.testing.assert_array_equal(data.mesh.visual.vertex_colors[0], [10, 20, 30, 255])
    assert data.camera_fov == pytest.approx(2 * np.arctan(480 / 1000))
    assert data.camera_aspect == pytest.approx(4 / 3)
    assert data.poses.shape == (1, 8)


def test_cloud_only_result_and_missing_geometry(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    with pytest.raises(ValueError, match="No nonempty"):
        load_reconstruction(tmp_path)
    trimesh.points.PointCloud([[0, 0, 1], [1, 1, 2]]).export(tmp_path / "pointcloud.ply")
    data = load_reconstruction(tmp_path)
    assert data.mesh is None and len(data.points) == 2
    assert data.poses.shape == (0, 8)
    np.testing.assert_allclose(data.center, [0.5, 0.5, 1.5])


def test_view_command_is_independent_of_model_and_mapping_options():
    args = build_parser().parse_args(["view", "--result", "example"])
    assert args.command == "view" and args.host == "127.0.0.1"
    assert args.port == 8080 and args.max_points == 500_000
    assert not hasattr(args, "model")


@pytest.fixture
def viewer():
    pytest.importorskip("viser")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    viewer = LiveReconstructionViewer(port=port, max_points=5)
    try:
        yield viewer
    finally:
        viewer.stop()


def test_live_viewer_updates_growing_geometry_and_trajectory(viewer):
    viewer.configure(CameraIntrinsics(96, 72, 85, 85, 47.5, 35.5), 3, "odometry", "aligned")
    for index in range(3):
        pose = np.eye(4)
        pose[0, 3] = index * 0.1
        viewer.on_frame(
            {
                "processed": index + 1,
                "selected": 3,
                "integrated": index + 1,
                "skipped": 0,
                "timestamp": float(index),
                "frame_status": "integrated",
            },
            pose,
        )
        points = np.arange((index + 3) * 3, dtype=np.float32).reshape(-1, 3)
        viewer.update_map(points, np.full(points.shape, 100, dtype=np.uint8))
    assert len(viewer.positions) == 3
    assert viewer.path.points.shape == viewer.path.colors.shape == (2, 2, 3)
    assert viewer.cloud.points.shape == viewer.cloud.colors.shape == (5, 3)
    np.testing.assert_allclose(viewer.camera.position, [0.2, 0, 0])
    assert "**3 / 3**" in viewer.status.content


@pytest.mark.parametrize("autoplay", [False, True])
def test_replay_button_keeps_map_until_clicked_and_starts_fresh_rounds(
    viewer, monkeypatch, autoplay
):
    armed = threading.Event()
    wait = viewer._start.wait

    def signal_ready():
        armed.set()
        return wait()

    monkeypatch.setattr(viewer._start, "wait", signal_ready)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            if not autoplay:
                first = pool.submit(viewer.wait_for_start)
                assert armed.wait(3)
                assert viewer._start_button.label == "Start replay" and not first.done()
                viewer._start_button.value = True
                first.result(timeout=3)
                assert viewer._start_button.disabled
            for _ in range(2):
                viewer.on_tracking(1, 1.0, "OK", np.eye(4))
                viewer.on_tracking(2, 2.0, "LOST", None)
                viewer.update_map(np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8), segment=0)
                viewer.on_fusion(1, timestamp=1.0, stage="Replay complete", segment=0)
                assert len(viewer.positions) == 1
                cloud, status, button = viewer.cloud, viewer.status.content, viewer._start_button
                armed.clear()
                replay = pool.submit(viewer.wait_for_replay)
                assert armed.wait(3)
                assert viewer._start_button.label == "Replay" and not viewer._start_button.disabled
                assert button is None or button is viewer._start_button
                assert (
                    not replay.done() and viewer.cloud is cloud and viewer.status.content == status
                )
                viewer._start_button.value = True
                replay.result(timeout=3)
                assert viewer._start_button.disabled and viewer._processed == viewer._missing == 0
                assert viewer._tracking_time is None and viewer._fusion_time is None
                assert viewer.cloud is None and viewer.camera is None and viewer.positions == []
        finally:
            viewer._start.set()  # Release a blocked worker even when an assertion fails.


def test_stream_keeps_tracking_and_fusion_separate_across_gaps_and_resets(viewer, monkeypatch):
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr("glassrecon.viewer.time", SimpleNamespace(monotonic=lambda: clock.now))
    viewer.configure(CameraIntrinsics(96, 72, 85, 85, 47.5, 35.5), 6, "ORB-SLAM3", "GlassRecon")
    points = np.arange(18, dtype=np.float32).reshape(-1, 3)
    colors = np.full(points.shape, 100, dtype=np.uint8)
    pose = np.eye(4)
    viewer.on_tracking(1, 1.0, "OK", pose)
    pose[0, 3] = 0.2
    clock.now = 1
    viewer.on_tracking(2, 2.0, "OK", pose)
    viewer.update_map(points, colors, segment=0)
    viewer.on_fusion(1, pending=1, timestamp=1.0, segment=0)
    assert "Latest input: 2.000000 s" in viewer.status.content
    assert "Latest fused frame: 1.000000 s" in viewer.status.content
    viewer.on_tracking(3, 3.0, "LOST", None)
    assert "LOST" in viewer.status.content and not viewer.camera.visible
    clock.now = 4
    pose[0, 3] = 8.0
    viewer.on_tracking(4, 4.0, "OK", pose)
    assert viewer.camera.visible
    np.testing.assert_allclose(viewer.path.points[:, :, 0], [[0.0, 0.2]])
    assert len(viewer.positions) == 3

    # A mapper already processing the old submap may finish after tracking resets.
    resume_mapping = threading.Event()

    def delayed_old_map():
        assert resume_mapping.wait(3)
        viewer.update_map(points, colors, segment=0)
        viewer.on_fusion(3, timestamp=3.0, stage="Old map complete", segment=0)

    with ThreadPoolExecutor(max_workers=1) as pool:
        stale = pool.submit(delayed_old_map)
        clock.now = 100
        viewer.on_tracking(5, 5.0, "OK", np.eye(4), segment=1)
        resume_mapping.set()
        stale.result(timeout=3)
    assert viewer.cloud is None and viewer.path is None and len(viewer.positions) == 1
    assert "Submap **1**" in viewer.status.content
    assert "Old map complete" not in viewer.status.content
    assert "Latest fused frame: —" in viewer.status.content
    clock.now = 102
    viewer.on_fusion(1, timestamp=5.0, segment=1)
    viewer.update_map(points + 5, colors, segment=1)
    assert viewer.cloud is not None
    assert "**1** integrated · 0.5 fps" in viewer.status.content


def test_growing_map_frames_trajectory_and_preserves_a_manual_view(viewer, monkeypatch):
    auto, manual = SimpleNamespace(), SimpleNamespace()
    clients = {0: SimpleNamespace(camera=auto), 1: SimpleNamespace(camera=manual)}
    monkeypatch.setattr(viewer.server, "get_clients", lambda: clients)
    pose = np.eye(4)
    pose[0, 3] = -4
    viewer.on_tracking(1, 0.0, "OK", pose)
    points = np.array([[0, 0, 2], [2, 2, 4]], dtype=np.float32)
    colors = np.full(points.shape, 100, dtype=np.uint8)
    viewer.update_map(points, colors)
    np.testing.assert_allclose(auto.look_at, [-1, 1, 2])
    manual.position = (10, 2, -1)
    viewer.update_map(points * 4, colors)
    np.testing.assert_allclose(auto.look_at, [2, 4, 8])
    np.testing.assert_allclose(manual.position, [10, 2, -1])
    np.testing.assert_allclose(manual.look_at, [-1, 1, 2])
