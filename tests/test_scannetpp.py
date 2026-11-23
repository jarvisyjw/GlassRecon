"""Checks for the ScanNet++ coordinate and image conversion boundary."""

import numpy as np
import pytest

pytest.importorskip("cv2")
from experiments.prepare_scannetpp import read_colmap


def test_colmap_inversion_preserves_camera_coordinates(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "# cameras\n1 OPENCV 1920 1440 1400 1400 960 720 0.06 -0.07 0 0\n"
    )
    # 90-degree rotation around z, plus world-to-camera translation.
    q = np.sqrt(0.5)
    (tmp_path / "images.txt").write_text(
        f"# images\n0 {q} 0 0 {q} 1 2 3 1 frame_000000.jpg\n\n"
        "1 1 0 0 0 0 0 0 1 frame_000010.jpg\n\n"
    )
    cameras, frames = read_colmap(tmp_path)
    assert cameras[1][:2] == (1920, 1440)
    assert len(frames) == 2
    pose = frames[0][2]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([1, 2, 3])
    camera_point = np.array([0.2, -0.5, 2])
    world = pose[:3, :3] @ camera_point + pose[:3, 3]
    np.testing.assert_allclose(rotation @ world + translation, camera_point, atol=1e-12)
    np.testing.assert_allclose(pose[:3, 3], [-2, 1, -3], atol=1e-12)


@pytest.mark.mapping
def test_comparison_excludes_masked_pixels_from_every_fusion_variant(tmp_path, monkeypatch):
    pytest.importorskip("open3d")
    from PIL import Image

    from examples.make_synthetic_sequence import make_sequence
    from experiments import run_comparison

    dataset = make_sequence(tmp_path / "input", frame_count=5)
    masks = tmp_path / "masks"
    masks.mkdir()
    for image in (dataset / "rgb").glob("*.png"):
        shape = np.array(Image.open(image)).shape[:2]
        mask = np.full(shape, 255, dtype=np.uint8)
        mask[:, -8:] = 0
        Image.fromarray(mask).save(masks / image.name)
    original = run_comparison.create_mapper
    integrated = []

    def checked_mapper(*args, **kwargs):
        mapper = original(*args, **kwargs)
        integrate = mapper.integrate

        def checked_integrate(rgb, depth, pose):
            assert not np.any(depth[:, -8:])
            assert np.any(depth[:, :-8] > 0)
            integrated.append(True)
            return integrate(rgb, depth, pose)

        mapper.integrate = checked_integrate
        return mapper

    monkeypatch.setattr(run_comparison, "create_mapper", checked_mapper)
    args = run_comparison.parser().parse_args(
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
            "--valid-mask-dir",
            str(masks),
            "--iterations",
            "5",
            "--enhance",
            "--output",
            str(tmp_path / "output"),
        ]
    )
    result = run_comparison.run(args)
    assert len(integrated) == 4 * len(result["results"])


@pytest.mark.mapping
def test_preparation_keeps_millimeters_and_masks_reference(tmp_path):
    import json

    o3d = pytest.importorskip("open3d")
    from PIL import Image

    from experiments.prepare_scannetpp import prepare
    from glassrecon.data import CameraIntrinsics, load_sequence

    scene = tmp_path / "source/data/test"
    phone = scene / "iphone"
    for folder in ["rgb", "depth", "rgb_masks", "colmap"]:
        (phone / folder).mkdir(parents=True)
    (scene / "scans").mkdir()
    (phone / "colmap/cameras.txt").write_text("1 OPENCV 1920 1440 1400 1400 960 720 0 0 0 0\n")
    (phone / "colmap/images.txt").write_text("0 1 0 0 0 0 0 0 1 frame_000000.jpg\n\n")
    (phone / "pose_intrinsic_imu.json").write_text(json.dumps({"frame_000000": {"timestamp": 1.0}}))
    Image.new("RGB", (1920, 1440), (100, 150, 200)).save(phone / "rgb/frame_000000.jpg")
    Image.fromarray(np.full((192, 256), 2000, dtype=np.uint16)).save(
        phone / "depth/frame_000000.png"
    )
    mask = np.full((1440, 1920), 255, dtype=np.uint8)
    mask[:, :240] = 0
    Image.fromarray(mask).save(phone / "rgb_masks/frame_000000.png")
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[-10, -10, 2], [10, -10, 2], [10, 10, 2], [-10, 10, 2]]),
        o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]),
    )
    o3d.io.write_triangle_mesh(str(scene / "scans/mesh_aligned_0.05.ply"), mesh)
    (scene / "scans/mesh_aligned_0.05_mask.txt").write_text("")
    output = prepare(tmp_path / "source", "test", tmp_path / "prepared", width=64)
    camera = CameraIntrinsics.load(output / "intrinsics.json")
    assert (camera.width, camera.height) == (64, 48)
    assert camera.fx == pytest.approx(1400 / 30)
    _, sensor = load_sequence(output)[0][0].load(1000, camera)
    _, reference = load_sequence(output / "reference")[0][0].load(1000, camera)
    assert not np.any(sensor[:, :8])
    assert not np.any(reference[:, :8])
    np.testing.assert_allclose(sensor[:, 8:], 2)
    np.testing.assert_allclose(reference[:, 8:], 2, atol=1e-6)


@pytest.mark.mapping
def test_failed_global_fit_abstains_without_dropping_other_methods(tmp_path, monkeypatch):
    import json

    pytest.importorskip("open3d")
    from examples.make_synthetic_sequence import make_sequence
    from experiments import run_comparison
    from glassrecon.alignment import AlignmentError

    dataset = make_sequence(tmp_path / "input", frame_count=5)
    original = run_comparison.recover_metric_depth
    failures = []

    def recover(prediction, sensor, representation, config, rng):
        if config.method == "global" and not failures:
            failures.append(True)
            raise AlignmentError("nonpositive scale test")
        return original(prediction, sensor, representation, config, rng)

    monkeypatch.setattr(run_comparison, "recover_metric_depth", recover)
    output = tmp_path / "output"
    args = run_comparison.parser().parse_args(
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
            "--global-failure",
            "empty",
            "--iterations",
            "5",
            "--output",
            str(output),
        ]
    )
    result = run_comparison.run(args)
    logs = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    assert len(logs) == 4
    assert logs[0]["alignment"]["global"]["policy"] == "empty_depth"
    assert logs[0]["alignment"]["regional"]["scale"] > 0
    assert result["protocol"]["fused_frames"] == 4
    for method in result["results"]:
        assert json.loads((output / method / "summary.json").read_text())["integrated"] == (
            3 if method == "global" else 4
        )


@pytest.mark.parametrize("prefix", ["", "video/", "iphone/"])
def test_colmap_normalizes_supported_prefix_without_changing_pose(tmp_path, prefix):
    (tmp_path / "cameras.txt").write_text("2 OPENCV 1920 1440 1400 1400 960 720 0 0 0 0\n")
    (tmp_path / "images.txt").write_text(f"1 1 0 0 0 1 2 3 2 {prefix}frame_000010.jpg\n1 2 -1\n")
    cameras, frames = read_colmap(tmp_path)
    assert 2 in cameras
    assert frames[0][:2] == ("frame_000010.jpg", 2)
    np.testing.assert_array_equal(frames[0][2][:3, 3], [-1, -2, -3])


def test_colmap_rejects_normalized_filename_collision(tmp_path):
    (tmp_path / "cameras.txt").write_text("1 OPENCV 1920 1440 1400 1400 960 720 0 0 0 0\n")
    (tmp_path / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame_000010.jpg\n\n2 1 0 0 0 0 0 0 1 video/frame_000010.jpg\n\n"
    )
    with pytest.raises(ValueError, match="Duplicate normalized"):
        read_colmap(tmp_path)
