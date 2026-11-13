import json

import numpy as np
import pytest

from glassrecon.cli import build_parser, main
from glassrecon.data import CameraIntrinsics
from glassrecon.mapping import RGBDMapper, TrackingError
from glassrecon.pipeline import reconstruct


@pytest.mark.mapping
def test_cached_sequence_reconstructs_metric_plane(tmp_path):
    o3d = pytest.importorskip("open3d")
    # Exercise the public CLI, file association, scale fitting, pose convention,
    # actual TSDF backend, and exported world geometry in one regression test.
    from examples.make_synthetic_sequence import make_sequence

    dataset = make_sequence(tmp_path / "sequence")
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
            "--pose-source",
            "file",
            "--pose-file",
            str(dataset / "groundtruth.txt"),
            "--iterations",
            "10",
            "--save-depths",
        ]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["integrated"] == 3 and summary["skipped"] == 0
    cloud = o3d.io.read_point_cloud(str(output / "pointcloud.ply"))
    points = np.asarray(cloud.points)
    assert len(points) > 100
    assert np.median(np.abs(points[:, 2] - 2 - 0.2 * points[:, 0])) < 0.015
    records = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    assert all(abs(row["alignment"]["scale"] - 1.7) < 0.01 for row in records)
    depth = np.load(output / "metric_depths" / "1.000000.npy")
    assert np.all(depth[10:20, 10:25] > 0)  # Sensor holes were completed.
    trajectory = np.loadtxt(output / "trajectory.txt")
    np.testing.assert_allclose(trajectory[:, 1], [0, 0.02, 0.04])


@pytest.mark.mapping
def test_real_odometry_stationary_pair_and_empty_depth(tmp_path):
    pytest.importorskip("open3d")
    camera = CameraIntrinsics(96, 72, 85, 85, 47.5, 35.5)
    mapper = RGBDMapper(camera)
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 255, (72, 96, 3), dtype=np.uint8)
    depth = np.full((72, 96), 2, dtype=np.float32)
    np.testing.assert_allclose(mapper.track(rgb, depth), np.eye(4))
    np.testing.assert_allclose(mapper.track(rgb, depth), np.eye(4), atol=1e-5)
    with pytest.raises(TrackingError):
        mapper.track(rgb, depth * 0)
    np.testing.assert_allclose(mapper.previous_c2w, np.eye(4), atol=1e-5)


def test_cli_rejects_ambiguous_pose_mode_and_invalid_parameters():
    base = ["reconstruct", "--dataset", "/unused", "--output", "/unused"]
    for flags in (
        ["--pose-source", "file"],
        ["--min-depth", "10", "--max-depth", "1"],
        ["--stride", "0"],
        ["--depth-units-per-meter", "nan"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(base + flags)
        assert exc.value.code == 2


@pytest.mark.mapping
def test_alignment_failure_is_logged_and_excluded_from_map(tmp_path):
    pytest.importorskip("open3d")
    from examples.make_synthetic_sequence import make_sequence

    dataset = make_sequence(tmp_path / "sequence")
    # A constant prediction cannot constrain an affine scale and shift.
    np.save(dataset / "predictions" / "1.030000.npy", np.ones((72, 96)))
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
            "--pose-source",
            "file",
            "--pose-file",
            str(dataset / "groundtruth.txt"),
            "--iterations",
            "2",
            "--on-failure",
            "skip",
        ]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["integrated"] == 2 and summary["skipped"] == 1
    records = [json.loads(row) for row in (output / "frames.jsonl").read_text().splitlines()]
    assert records[1]["status"] == "failed"
    assert "variation" in records[1]["reason"]
    assert np.loadtxt(output / "trajectory.txt").shape == (2, 8)


@pytest.mark.mapping
def test_raw_sensor_baseline_never_loads_prediction_model(tmp_path, monkeypatch):
    pytest.importorskip("open3d")
    from examples.make_synthetic_sequence import make_sequence

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("Sensor baseline must not load a model")

    monkeypatch.setattr("glassrecon.pipeline.create_predictor", unexpected_model_load)
    dataset = make_sequence(tmp_path / "sequence", frame_count=1)
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
            "--depth-source",
            "sensor",
        ]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["integrated"] == 1
    assert summary["depth_source"] == "sensor"


@pytest.mark.mapping
@pytest.mark.parametrize("tracking_depth", ["sensor", "aligned"])
def test_sequence_estimates_motion_and_reports_incremental_fusion(tmp_path, tracking_depth):
    pytest.importorskip("open3d")
    from examples.make_synthetic_sequence import make_sequence

    # Fixed image-space corruption is useful for recovery tests, but is not
    # consistent world geometry for evaluating raw-sensor camera motion.
    dataset = make_sequence(
        tmp_path / "sequence", frame_count=8, corrupt_sensor=tracking_depth == "aligned"
    )
    output = tmp_path / "map"
    args = build_parser().parse_args(
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
            "--iterations",
            "5",
            "--viewer-every",
            "3",
        ]
        + (["--tracking-depth", "aligned"] if tracking_depth == "aligned" else [])
    )

    class Observer:
        def __init__(self):
            self.progress = []
            self.cloud_sizes = []

        def configure(self, intrinsics, total, pose_source, depth_source, **options):
            assert total == 8 and pose_source == "odometry" and depth_source == "aligned"
            assert options["tracking_depth"] == tracking_depth

        def on_frame(self, progress, pose):
            assert pose is not None
            self.progress.append(progress.copy())

        def update_map(self, points, colors):
            assert points.shape == colors.shape and len(points) > 0
            self.cloud_sizes.append(len(points))

        def finish(self, summary):
            assert summary["integrated"] == 8

    observer = Observer()
    summary = reconstruct(args, observer)
    assert summary["processed"] == summary["integrated"] == 8
    assert summary["tracking_depth"] == tracking_depth
    assert summary["estimated_path_length_m"] > 0.05
    assert [row["integrated"] for row in observer.progress] == list(range(1, 9))
    assert len(observer.cloud_sizes) == 4  # Frames 1, 3, 6, then the final map.
    trajectory = np.loadtxt(output / "trajectory.txt")
    assert trajectory.shape == (8, 8)
    np.testing.assert_allclose(trajectory[-1, 1:4], [0.14, 0, 0], atol=0.05)
    assert json.loads((output / "progress.json").read_text())["integrated"] == 8
