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
