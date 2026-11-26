"""Convert a decoded ScanNet++ iPhone scene into inputs for GlassRecon mapping."""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from glassrecon.data import CameraIntrinsics, matrix_quaternion, quaternion_matrix


def read_colmap(folder):
    cameras = {}
    for line in (folder / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 12 or fields[1] != "OPENCV":
            raise ValueError("Expected an OPENCV camera with four distortion parameters")
        cameras[int(fields[0])] = (int(fields[2]), int(fields[3]), np.array(fields[4:], float))

    lines = iter((folder / "images.txt").read_text().splitlines())
    frames, names = [], set()
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10 or not re.fullmatch(
            r"(?:(?:video|iphone)/)?frame_[0-9]+\.jpg", fields[9]
        ):
            raise ValueError(f"Invalid COLMAP iPhone image row: {line}")
        name = Path(fields[9]).name
        if name in names:
            raise ValueError(f"Duplicate normalized COLMAP image name: {name}")
        names.add(name)
        # COLMAP stores world-to-camera rotations with a scalar-first quaternion.
        rotation = quaternion_matrix(np.array(fields[1:5], float)[[1, 2, 3, 0]])
        pose = np.eye(4)
        pose[:3, :3] = rotation.T
        pose[:3, 3] = -rotation.T @ np.array(fields[5:8], float)
        frames.append((name, int(fields[8]), pose))
        next(lines, None)  # POINTS2D row, possibly empty
    return cameras, sorted(frames)


def prepare(scene_dir, output):
    phone = Path(scene_dir) / "iphone"
    cameras, frames = read_colmap(phone / "colmap")
    if len(cameras) != 1 or not frames:
        raise ValueError("Expected one calibrated iPhone camera and at least one COLMAP frame")
    camera_id, (width, height, params) = next(iter(cameras.items()))
    calibration = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
    target_width = 512
    target_height = round(target_width * height / width)
    target = calibration.copy()
    target[0] *= target_width / width
    target[1] *= target_height / height
    camera = CameraIntrinsics(
        target_width, target_height, target[0, 0], target[1, 1], target[0, 2], target[1, 2]
    )
    rgb_x, rgb_y = cv2.initUndistortRectifyMap(
        calibration, params[4:], None, target, (target_width, target_height), cv2.CV_32FC1
    )
    depth_x, depth_y = rgb_x * 256 / width, rgb_y * 192 / height
    timing = json.loads((phone / "pose_intrinsic_imu.json").read_text())
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new scene directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        scene = Path(tmp) / "scene"
        for folder in ("rgb", "depth", "masks"):
            (scene / folder).mkdir(parents=True)
        rgb_rows, depth_rows, poses = [], [], []
        for name, frame_camera, pose in frames:
            if frame_camera != camera_id:
                raise ValueError(f"Unknown camera for {name}")
            stem = Path(name).stem
            with Image.open(phone / "rgb" / name) as image:
                rgb = np.array(image.convert("RGB"))
            with Image.open(phone / "depth" / f"{stem}.png") as image:
                depth = np.array(image)
            with Image.open(phone / "rgb_masks" / f"{stem}.png") as image:
                mask = np.array(image.convert("L"))
            if (
                rgb.shape[:2] != (height, width)
                or mask.shape != (height, width)
                or depth.shape != (192, 256)
            ):
                raise ValueError(f"Unexpected decoded image dimensions for {name}")
            rgb = cv2.remap(rgb, rgb_x, rgb_y, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, depth_x, depth_y, cv2.INTER_NEAREST)
            valid = cv2.remap(mask, rgb_x, rgb_y, cv2.INTER_NEAREST) == 255
            # Keep pixels whose RGB interpolation also uses only valid source pixels.
            valid &= cv2.remap(mask.astype(np.float32), rgb_x, rgb_y, cv2.INTER_LINEAR) >= 254.99
            rgb[~valid] = 0
            depth[~valid] = 0
            Image.fromarray(rgb).save(scene / "rgb" / f"{stem}.png")
            Image.fromarray(depth).save(scene / "depth" / f"{stem}.png")
            Image.fromarray(valid.astype(np.uint8) * 255).save(scene / "masks" / f"{stem}.png")
            stamp = float(timing[stem]["timestamp"])
            rgb_rows.append(f"{stamp:.9f} rgb/{stem}.png\n")
            depth_rows.append(f"{stamp:.9f} depth/{stem}.png\n")
            poses.append([stamp, *pose[:3, 3], *matrix_quaternion(pose[:3, :3])])
        (scene / "rgb.txt").write_text("".join(rgb_rows))
        (scene / "depth.txt").write_text("".join(depth_rows))
        np.savetxt(scene / "poses.txt", poses, fmt="%.12f")
        (scene / "intrinsics.json").write_text(json.dumps(asdict(camera), indent=2) + "\n")
        scene.rename(output)
    return len(frames)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scene-dir", required=True, type=Path, help="Scene containing iphone/")
    cli.add_argument(
        "--output", required=True, type=Path, help="New directory for the RGB-D sequence"
    )
    args = cli.parse_args()
    print(f"Prepared {prepare(args.scene_dir, args.output)} frames in {args.output}")
