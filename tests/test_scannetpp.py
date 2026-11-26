"""Checks for the ScanNet++ coordinate and image conversion boundary."""

import json

import numpy as np
import pytest
from PIL import Image

from glassrecon.data import CameraIntrinsics, load_sequence

pytest.importorskip("cv2")
from examples.prepare_scannetpp import prepare, read_colmap


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


def test_preparation_preserves_registered_sensor_depth_timestamps_and_masks(tmp_path):
    phone = tmp_path / "source" / "iphone"
    for folder in ("rgb", "depth", "rgb_masks", "colmap"):
        (phone / folder).mkdir(parents=True)
    (phone / "colmap/cameras.txt").write_text("1 OPENCV 640 480 500 500 320 240 0 0 0 0\n")
    (phone / "colmap/images.txt").write_text("1 1 0 0 0 1 2 3 1 frame_000010.jpg\n\n")
    (phone / "pose_intrinsic_imu.json").write_text(
        json.dumps({"frame_000010": {"timestamp": 1.25}})
    )
    Image.new("RGB", (640, 480), (100, 150, 200)).save(phone / "rgb/frame_000010.jpg")
    mask = np.full((480, 640), 255, dtype=np.uint8)
    mask[:, :80] = 0
    Image.fromarray(mask).save(phone / "rgb_masks/frame_000010.png")

    output = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError):
        prepare(phone.parent, output)
    assert not output.exists()

    Image.fromarray(np.full((192, 256), 2000, dtype=np.uint16)).save(
        phone / "depth/frame_000010.png"
    )
    assert prepare(phone.parent, output) == 1
    camera = CameraIntrinsics.load(output / "intrinsics.json")
    assert (camera.width, camera.height, camera.fx, camera.fy) == (512, 384, 400, 400)
    frame = load_sequence(output)[0][0]
    assert frame.timestamp == 1.25
    rgb, sensor = frame.load(1000, camera)
    with Image.open(output / "masks/frame_000010.png") as image:
        valid = np.array(image) == 255
    assert not valid[:, :64].any()
    assert valid[10:-10, 65:-10].all()
    assert not rgb[~valid].any() and not sensor[~valid].any()
    np.testing.assert_allclose(sensor[10:-10, 65:-10], 2)
    np.testing.assert_allclose(np.loadtxt(output / "poses.txt"), [1.25, -1, -2, -3, 0, 0, 0, 1])
    with pytest.raises(FileExistsError):
        prepare(phone.parent, output)


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
