import json

import numpy as np
import pytest

from glassrecon.cli import main
from glassrecon.data import CameraIntrinsics
from glassrecon.mapping import TrackingError, create_mapper


@pytest.fixture
def cuda_nvblox():
    pytest.importorskip("nvblox_torch")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("nvblox needs CUDA")


@pytest.mark.nvblox
def test_nvblox_world_coordinates_color_and_invalid_depth(tmp_path, cuda_nvblox):
    camera = CameraIntrinsics(96, 72, 85, 85, 47.5, 35.5)
    mapper = create_mapper(camera, backend="nvblox")
    rgb = np.full((72, 96, 3), [210, 70, 30], dtype=np.uint8)
    for z in [0.5, 0.7]:
        pose = np.eye(4)
        pose[:3, 3] = [0.4, 0, z]
        depth = np.full((72, 96), 3 - z, dtype=np.float32)
        depth[:4] = np.nan
        depth[-4:] = np.inf
        depth[4:8] = 100
        original = depth.copy()
        mapper.integrate(rgb, depth, pose)
        np.testing.assert_array_equal(depth, original)
    points, colors = mapper.point_cloud()
    assert len(points) > 100
    assert np.quantile(np.abs(points[:, 2] - 3), 0.95) < 0.02
    np.testing.assert_allclose(np.median(colors, axis=0), [210, 70, 30], atol=3)
    assert mapper.export(tmp_path)["mesh_triangles"] > 100
    assert mapper.metadata()["truncation_distance_vox"] == 4
    for invalid in [np.full((72, 96), np.nan), np.zeros((72, 96))]:
        with pytest.raises(TrackingError):
            mapper.integrate(rgb, invalid, np.eye(4))
    with pytest.raises(ValueError, match="dimensions"):
        mapper.integrate(rgb, np.ones((2, 2)), np.eye(4))
    with pytest.raises(TrackingError, match="rigid"):
        mapper.integrate(rgb, np.ones((72, 96)), np.eye(4) * 2)
    assert mapper.integrated == 2


@pytest.mark.nvblox
@pytest.mark.parametrize("policy", ["replace", "multiview"])
def test_nvblox_cached_cli_and_window_flush(tmp_path, cuda_nvblox, policy):
    import open3d as o3d

    from examples.make_synthetic_sequence import make_sequence

    dataset = make_sequence(tmp_path / "sequence", frame_count=7)
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
            "--backend",
            "nvblox",
            "--fusion-policy",
            policy,
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
    assert summary["integrated"] == 7 and summary["skipped"] == 0
    points = np.asarray(o3d.io.read_point_cloud(str(output / "pointcloud.ply")).points)
    assert np.median(np.abs(points[:, 2] - 2 - 0.2 * points[:, 0])) < 0.02
    assert len(list((output / "metric_depths").glob("*.npy"))) == 7
    config = json.loads((output / "config.json").read_text())
    assert config["reconstruction_backend"]["name"] == "nvblox"
    if policy == "multiview":
        assert config["integration_latency_frames"] == 4
