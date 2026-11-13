import json

import numpy as np
import pytest

from glassrecon.cli import build_parser
from glassrecon.viewer import camera_pose, load_reconstruction


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


def test_live_viewer_updates_growing_geometry_and_trajectory():
    pytest.importorskip("viser")
    import socket

    from glassrecon.data import CameraIntrinsics
    from glassrecon.viewer import LiveReconstructionViewer

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    viewer = LiveReconstructionViewer(port=port, max_points=5)
    try:
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
    finally:
        viewer.stop()
