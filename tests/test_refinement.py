import numpy as np

from glassrecon.alignment import AlignmentResult
from glassrecon.data import CameraIntrinsics
from glassrecon.refinement import ConservativeFusion, filter_sensor, refine_alignment, warp_depth


def test_reprojection_translation_and_occlusion():
    camera = CameraIntrinsics(9, 7, 4, 4, 4, 3)
    depth = np.full((7, 9), 2, np.float32)
    pose = np.eye(4)
    np.testing.assert_array_equal(warp_depth(depth, pose, pose, camera), depth)
    target = pose.copy()
    target[0, 3] = 0.5
    projected = warp_depth(depth, pose, target, camera)
    np.testing.assert_allclose(projected[:, :-1], 2)
    np.testing.assert_array_equal(projected[:, -1], 0)


def test_filter_reduces_plane_noise_without_blurring_depth_step():
    rng = np.random.default_rng(4)
    gt = np.ones((48, 64), np.float32)
    gt[:, 32:] = 2
    sensor = gt + rng.normal(0, 0.008, gt.shape).astype(np.float32)
    rgb = np.zeros((*gt.shape, 3), np.uint8)
    filtered, _ = filter_sensor(sensor, rgb)
    assert np.abs(filtered - gt).mean() < np.abs(sensor - gt).mean() * 0.6
    assert filtered[:, 31].max() < 1.04 and filtered[:, 32].min() > 1.96


def test_completion_requires_independent_previous_sensor_and_keeps_unsupported_holes():
    camera = CameraIntrinsics(32, 24, 30, 30, 15.5, 11.5)
    rgb = np.zeros((24, 32, 3), np.uint8)
    sensor = np.full((24, 32), 2, np.float32)
    predicted = np.full_like(sensor, 2.2)  # local residual corrects model shape offset
    missing = sensor.copy()
    missing[12, 16] = 0
    missing[3:9, 3:9] = 0
    fusion = ConservativeFusion(camera)
    first, _, fills, _ = fusion.fuse(missing, predicted, rgb, np.eye(4))
    assert not fills.any() and first[12, 16] == 0
    fusion.fuse(sensor, predicted, rgb, np.eye(4))
    result, filtered, fills, _ = fusion.fuse(missing, predicted, rgb, np.eye(4))
    assert fills[12, 16] and abs(result[12, 16] - 2) < 1e-5
    assert result[5, 5] == 0  # no spatial sensor support
    np.testing.assert_array_equal(result[filtered > 0], filtered[filtered > 0])
    assert fusion.previous_sensor[12, 16] == 0  # prediction never becomes temporal evidence


def test_refit_rejects_outliers_and_requires_validation_improvement():
    yy, xx = np.indices((96, 128))
    prediction = 1 + xx / 128 + yy / 192
    gt = 1.6 * prediction + 0.2
    rng = np.random.default_rng(9)
    sensor = gt + rng.normal(0, 0.005, gt.shape)
    sensor[20:30, 20:30] += 1
    initial = AlignmentResult(1.5, 0.3, gt.size, 0.1, 1600, False, "depth")
    refined, record = refine_alignment(prediction, sensor, initial)
    assert record["accepted"]
    assert np.abs(refined - gt).mean() < 0.02
    assert record["sensor_validation_mae_after_m"] < record["sensor_validation_mae_before_m"]


def test_conservative_pipeline_keeps_sensor_frame_when_scale_is_unidentifiable(tmp_path):
    import json

    import pytest

    pytest.importorskip("open3d")
    from examples.make_synthetic_sequence import make_sequence
    from glassrecon.cli import main

    dataset = make_sequence(tmp_path / "sequence", frame_count=1, corrupt_sensor=False)
    np.save(dataset / "predictions" / "1.000000.npy", np.ones((72, 96)))
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
            "--fusion-policy",
            "conservative",
            "--iterations",
            "2",
        ]
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["integrated"] == 1
    record = json.loads((output / "frames.jsonl").read_text().strip())
    assert "recovery_fallback" in record and record["fusion"]["temporally_supported_fills"] == 0
