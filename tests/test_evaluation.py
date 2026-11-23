import json

import numpy as np
import pytest

from glassrecon.data import CameraIntrinsics
from glassrecon.evaluation import DepthMetrics, MeshSurface, sample_mesh, summarize_distances


def test_depth_coverage_and_errors_do_not_hide_holes():
    metric = DepthMetrics()
    metric.add(np.array([1.0, 0.0, 4.0, np.nan]), np.array([1.0, 2.0, 2.0, 0.0]))
    result = metric.summary()
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["mae_m"] == 1
    assert result["rmse_m"] == pytest.approx(np.sqrt(2))
    assert result["abs_rel"] == 0.5
    assert DepthMetrics().summary()["mae_m"] is None


@pytest.mark.mapping
def test_surface_evaluation_uses_metric_distance_and_z_depth():
    o3d = pytest.importorskip("open3d")
    plane = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[-10, -10, 2], [10, -10, 2], [10, 10, 2], [-10, 10, 2]]),
        o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]),
    )
    surface = MeshSurface(plane)
    camera = CameraIntrinsics(8, 6, 4, 4, 3.5, 2.5)
    depth = surface.render(camera, np.eye(4))
    np.testing.assert_allclose(depth, 2, atol=1e-6)
    points = sample_mesh(plane, 1000)
    assert points.shape == (1000, 3)
    np.testing.assert_allclose(surface.distances(points), 0, atol=1e-5)
    points[:, 2] += 0.1
    distances = surface.distances(points)
    np.testing.assert_allclose(distances, 0.1, atol=1e-5)
    metrics = summarize_distances(distances, distances)
    assert metrics["accuracy_mean_m"] == pytest.approx(0.1, abs=1e-5)
    assert metrics["at_0.05m"]["fscore"] == 0  # No ICP/scale fit hides this offset.


def test_icl_coordinate_conversion_preserves_upright_image_rays():
    from experiments.prepare_icl import convert_pose

    theta = 0.3
    rotation = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    native = np.column_stack([rotation, [1.0, 2.0, 3.0]])
    pose = convert_pose(native)
    native_ray = np.array([0.2, -0.3, 1.0])  # Negative fy for a pixel below center.
    cv_ray = np.array([0.2, 0.3, 1.0])
    reflected_world = (rotation @ native_ray + native[:, 3]) * [1, -1, 1]
    np.testing.assert_allclose(pose[:3, :3] @ cv_ray + pose[:3, 3], reflected_world)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1)


@pytest.mark.mapping
@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("enhance", [False, True])
def test_comparison_fuses_identical_frames_and_excludes_heldout(tmp_path, with_reference, enhance):
    o3d = pytest.importorskip("open3d")
    from examples.make_synthetic_sequence import make_sequence
    from experiments.run_comparison import parser, run

    dataset = make_sequence(tmp_path / "input", frame_count=10)
    output = tmp_path / "comparison"
    reference_args = []
    if with_reference:
        clean = make_sequence(tmp_path / "clean", frame_count=10, corrupt_sensor=False)
        plane = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector([[-2, -2, 1.6], [2, -2, 2.4], [2, 2, 2.4], [-2, 2, 1.6]]),
            o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]),
        )
        mesh_path = tmp_path / "reference.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), plane)
        reference_args = [
            "--reference-depth",
            str(clean),
            "--reference-mesh",
            str(mesh_path),
            "--surface-samples",
            "1000",
        ]
    args = parser().parse_args(
        [
            "--dataset",
            str(dataset),
            "--poses",
            str(dataset / "groundtruth.txt"),
            "--intrinsics",
            str(dataset / "intrinsics.json"),
            "--model",
            "cached",
            "--prediction-dir",
            str(dataset / "predictions"),
            "--iterations",
            "5",
            "--output",
            str(output),
        ]
        + reference_args
        + (["--enhance"] if enhance else [])
    )
    report = run(args)
    if enhance:
        assert {"refined", "filtered", "conservative"} <= report["results"].keys()
        assert "conservative" in report["protocol"]["common_support_variants"]
    selection = json.loads((output / "selection.json").read_text())
    assert len(selection["fused"]) == 8 and len(selection["heldout"]) == 2
    assert not set(selection["fused"]) & set(selection["heldout"])
    trajectories = [np.loadtxt(output / name / "trajectory.txt") for name in report["results"]]
    for trajectory in trajectories:
        np.testing.assert_array_equal(trajectory, trajectories[0])
        assert trajectory.shape == (8, 8)
    baseline = report["results"]["global"]["heldout_render_common_support"]["mae_m"]
    regional = report["results"]["regional"]["heldout_render_common_support"]["mae_m"]
    assert regional < baseline
    if with_reference:
        for name in ["regional", "clean_oracle"]:
            assert report["results"][name]["surface"]["accuracy_mean_m"] < 0.02
        assert (output / "visible_reference.ply").exists()
