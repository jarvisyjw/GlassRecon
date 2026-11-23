"""Rectify decoded ScanNet++ iPhone frames and render evaluation-only laser depth."""

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from experiments.run_comparison import save_json
from glassrecon.data import CameraIntrinsics, matrix_quaternion, quaternion_matrix
from glassrecon.evaluation import MeshSurface
from glassrecon.pipeline import prepare_output


def read_colmap(folder):
    cameras = {}
    for line in (folder / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if p[1] != "OPENCV" or len(p) != 12:
            raise ValueError("Expected an OPENCV camera with four distortion parameters")
        cameras[int(p[0])] = (int(p[2]), int(p[3]), np.array(p[4:], dtype=float))
    lines = iter((folder / "images.txt").read_text().splitlines())
    frames = []
    names = set()
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) != 10 or not re.fullmatch(r"(?:(?:video|iphone)/)?frame_[0-9]+\.jpg", p[9]):
            raise ValueError("Invalid COLMAP image row or unsupported image name")
        name = Path(p[9]).name
        if name in names:
            raise ValueError(f"Duplicate normalized COLMAP image name: {name}")
        names.add(name)
        q = np.array(p[1:5], dtype=float)
        rotation = quaternion_matrix(q[[1, 2, 3, 0]])
        pose = np.eye(4)
        pose[:3, :3] = rotation.T
        pose[:3, 3] = -rotation.T @ np.array(p[5:8], dtype=float)
        frames.append((name, int(p[8]), pose))
        next(lines)  # POINTS2D, including an empty line.
    return cameras, sorted(frames)


def prepare(root, scene, output, width=512, max_frames=None):
    import open3d as o3d

    source = Path(root) / "data" / scene
    iphone = source / "iphone"
    cameras, frames = read_colmap(iphone / "colmap")
    if len(cameras) != 1:
        raise ValueError("This adapter requires a single COLMAP camera")
    camera_id, (w, h, params) = next(iter(cameras.items()))
    height = round(width * h / w)
    k = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
    target = k.copy()
    target[0] *= width / w
    target[1] *= height / h
    camera = CameraIntrinsics(width, height, target[0, 0], target[1, 1], target[0, 2], target[1, 2])
    mx, my = cv2.initUndistortRectifyMap(k, params[4:], None, target, (width, height), cv2.CV_32FC1)
    timing = json.loads((iphone / "pose_intrinsic_imu.json").read_text())
    available = len(frames)
    if max_frames:
        frames = frames[:max_frames]
    output = prepare_output(output)
    for name in ["rgb", "depth", "masks", "reference/depth"]:
        (output / name).mkdir(parents=True)
    mesh_path = source / "scans/mesh_aligned_0.05.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mask_path = source / "scans/mesh_aligned_0.05_mask.txt"
    masked = [int(x) for x in mask_path.read_text().split()]
    if masked:
        remove = np.zeros(len(mesh.vertices), dtype=bool)
        remove[masked] = True
        mesh.remove_vertices_by_mask(remove)
    o3d.io.write_triangle_mesh(str(output / "reference/mesh.ply"), mesh)
    surface = MeshSurface(mesh)
    rgb_rows, depth_rows, ref_rows, poses, checks = [], [], [], [], []
    for i, (name, cid, pose) in enumerate(frames):
        if cid != camera_id:
            raise ValueError("Unknown camera ID")
        stem = Path(name).stem
        stamp = float(timing[stem]["timestamp"])
        rgb = np.array(Image.open(iphone / "rgb" / name).convert("RGB"))
        depth = np.array(Image.open(iphone / "depth" / (stem + ".png")))
        mask = np.array(Image.open(iphone / "rgb_masks" / (stem + ".png")).convert("L"))
        if rgb.shape[:2] != (h, w) or mask.shape != (h, w) or depth.shape != (192, 256):
            raise ValueError("Unexpected source image dimensions")
        rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR)
        valid = cv2.remap(mask, mx, my, cv2.INTER_NEAREST) == 255
        # Remove interpolation footprints touching anonymized pixels or image borders.
        valid &= cv2.remap(mask.astype(np.float32), mx, my, cv2.INTER_LINEAR) >= 254.99
        depth = cv2.remap(depth, mx * 256 / w, my * 192 / h, cv2.INTER_NEAREST)
        depth[~valid] = 0
        rgb[~valid] = 0
        reference = surface.render(camera, pose)
        reference[~valid | (reference < 0.1) | (reference > 10)] = 0
        Image.fromarray(rgb).save(output / "rgb" / (stem + ".png"))
        Image.fromarray(depth).save(output / "depth" / (stem + ".png"))
        Image.fromarray(valid.astype(np.uint8) * 255).save(output / "masks" / (stem + ".png"))
        np.save(output / "reference/depth" / (stem + ".npy"), reference * 1000)
        rgb_rows.append(f"{stamp:.9f} rgb/{stem}.png\n")
        depth_rows.append(f"{stamp:.9f} depth/{stem}.png\n")
        ref_rows.append(f"{stamp:.9f} depth/{stem}.npy\n")
        poses.append([stamp, *pose[:3, 3], *matrix_quaternion(pose[:3, :3])])
        good = valid & (depth > 100) & (depth < 10000) & (reference > 0)
        error = np.abs(depth[good] / 1000 - reference[good])
        checks.append(
            {
                "frame": stem,
                "pixels": int(good.sum()),
                "median_m": float(np.median(error)) if error.size else None,
            }
        )
        if i % 100 == 0:
            print(scene, i + 1, "/", len(frames), flush=True)
    (output / "rgb.txt").write_text("".join(rgb_rows))
    (output / "depth.txt").write_text("".join(depth_rows))
    (output / "reference/rgb").symlink_to("../rgb", target_is_directory=True)
    (output / "reference/rgb.txt").write_text("".join(rgb_rows))
    (output / "reference/depth.txt").write_text("".join(ref_rows))
    np.savetxt(output / "poses.txt", poses, fmt="%.12f")
    save_json(output / "intrinsics.json", asdict(camera))
    save_json(
        output / "preparation.json",
        {
            "scene": scene,
            "source": str(source),
            "available_colmap_frames": available,
            "prepared_frames": len(frames),
            "width": width,
            "height": height,
            "depth_units_per_meter": 1000,
            "depth_interpolation": "nearest",
            "filename_mapping": "bare names or video/iphone prefix; reject collisions",
            "poses": "released mesh-aligned COLMAP w2c inverted to OpenCV c2w",
            "selection_caveat": "released COLMAP frames prefiltered using sensor/laser agreement",
            "reference": "laser mesh rendered only for evaluation and labeled oracle",
            "anonymized_mesh_vertices_removed": len(masked),
            "source_sha256": {
                str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [mesh_path, iphone / "colmap/cameras.txt", iphone / "colmap/images.txt"]
            },
            "sensor_laser_checks": checks,
        },
    )
    return output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--root", type=Path, required=True, help="Directory containing data/, metadata/, splits/"
    )
    cli.add_argument("--scene", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--width", type=int, default=512)
    cli.add_argument("--max-frames", type=int)
    args = cli.parse_args()
    if args.width < 1 or (args.max_frames is not None and args.max_frames < 1):
        cli.error("Width and frame cap must be positive")
    prepare(args.root, args.scene, args.output, args.width, args.max_frames)


if __name__ == "__main__":
    main()
