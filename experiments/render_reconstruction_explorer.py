"""Render the project page from the repository root.

Point clouds and RGB (viewer extra): python -m experiments.render_reconstruction_explorer
Input depths and metric scale (experiments extra):
    python -m experiments.render_reconstruction_explorer --depth-only

Office frame_002450 was held out of fusion. Before running --depth-only, use the
DA3 frontend environment to compute its five-frame metric prediction:
    python -m experiments.prepare_explorer_depth 8a20d62ac0 frame_002450
The supplemental cache defaults to outputs/reconstruction_explorer. Pass
--supplemental-depths for another directory with the same <scene>/<frame>.npy layout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from glassrecon.data import CameraIntrinsics, PoseTrajectory, read_index

WIDTH, HEIGHT = 1200, 800
BACKGROUND = (23, 35, 41)
POINT_RADIUS_M = 0.009
DEPTH_SCALE_M = 6.0
# Overview azimuth/elevation (degrees) and a prepared camera facing the glazing.
VIEWS = {
    "c0f5742640": (95, 4, "frame_003100"),
    "791a5c253d": (103, 15, "frame_002230"),
    "8a20d62ac0": (-4, 9, "frame_002450"),
}


def load_cloud(path):
    import trimesh

    cloud = trimesh.load(path, process=False)
    if not isinstance(cloud, trimesh.points.PointCloud) or not len(cloud.vertices):
        raise ValueError(f"Expected a nonempty point cloud in {path}")
    points = np.asarray(cloud.vertices, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.uint8)
    if colors.shape not in ((len(points), 3), (len(points), 4)) or not np.isfinite(points).all():
        raise ValueError(f"Expected finite XYZ coordinates and vertex colors in {path}")
    return points, colors[:, :3]


def rasterize(xy, depth, radii, colors, width, height):
    """Draw circular RGB splats with a nearest-point z-buffer at every pixel."""
    margin = int(radii.max(initial=0))
    x, y = np.rint(xy).astype(np.int32).T
    inside = (x >= -margin) & (x < width + margin) & (y >= -margin) & (y < height + margin)
    order = np.flatnonzero(inside)
    order = order[np.argsort(depth[order], kind="stable")]
    empty = len(order)
    centers = np.full((height + 2 * margin, width + 2 * margin), empty, dtype=np.int32)
    # The nearer of two points with a shared center pixel also has at least
    # as large a disk (constant in the overview, inverse-depth in the detail).
    np.minimum.at(centers, (y[order] + margin, x[order] + margin), np.arange(empty))
    radius_squared = np.append(radii[order] ** 2, 0)[centers]
    visible = np.full((height, width), empty, dtype=np.int32)
    for dy in range(-margin, margin + 1):
        for dx in range(-margin, margin + 1):
            distance = dx * dx + dy * dy
            if distance > margin * margin:
                continue
            ys = slice(margin - dy, margin - dy + height)
            xs = slice(margin - dx, margin - dx + width)
            covers_pixel = radius_squared[ys, xs] >= distance
            np.minimum(visible, centers[ys, xs], out=visible, where=covers_pixel)
    palette = np.vstack((colors[order], np.array(BACKGROUND, dtype=np.uint8)))
    return Image.fromarray(palette[visible])


def render_full(clouds, azimuth, elevation, width=WIDTH, height=HEIGHT):
    az, el = np.deg2rad([azimuth, elevation])
    toward_eye = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    right = [-np.sin(az), np.cos(az), 0]
    down = -np.cross(toward_eye, right)
    axes = np.array([right, down, -toward_eye], dtype=np.float32)
    projected = [points @ axes.T for points, _ in clouds]
    low = np.min([p[:, :2].min(axis=0) for p in projected], axis=0)
    high = np.max([p[:, :2].max(axis=0) for p in projected], axis=0)
    scale = 0.95 * min(width / (high[0] - low[0]), height / (high[1] - low[1]))
    radius = max(1, round(POINT_RADIUS_M * scale))
    for p, (_, colors) in zip(projected, clouds):
        xy = (p[:, :2] - (low + high) / 2) * scale + [(width - 1) / 2, (height - 1) / 2]
        yield rasterize(xy, p[:, 2], np.full(len(p), radius), colors, width, height)


def render_glass(clouds, camera, pose, width=WIDTH, height=HEIGHT):
    scale = max(width / camera.width, height / camera.height)
    # Pixel centers after the same centered aspect-ratio crop as the RGB reference.
    principal_offset = np.array([camera.cx + 0.5, camera.cy + 0.5]) - [
        camera.width / 2, camera.height / 2
    ]
    center = principal_offset * scale + [(width - 1) / 2, (height - 1) / 2]
    focal = np.array([camera.fx, camera.fy]) * scale
    for points, colors in clouds:
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        front = local[:, 2] > 0.05
        z = local[front, 2]
        xy = local[front, :2] * focal / z[:, None] + center
        radii = np.clip(np.rint(POINT_RADIUS_M * focal[0] / z), 1, 7).astype(np.int32)
        yield rasterize(xy, z, radii, colors[front], width, height)


def render_scene(scene, dataset, results, output):
    azimuth, elevation, frame_id = VIEWS[scene]
    prepared = dataset / scene
    run = results / scene / "multiview"
    clouds = [load_cloud(run / method / "pointcloud.ply") for method in ("sensor", "multiview")]
    frames = {Path(p).stem: (t, p) for t, p in read_index(prepared / "rgb.txt")}
    stamp, path = frames[frame_id]
    pose = PoseTrajectory(prepared / "poses.txt").at(stamp, 0.05)
    if pose is None:
        raise ValueError(f"Missing prepared pose for {scene}/{frame_id}")
    camera = CameraIntrinsics(**json.loads((prepared / "intrinsics.json").read_text()))
    target = output / scene
    target.mkdir(parents=True, exist_ok=True)
    for name, panels in (
        ("full", render_full(clouds, azimuth, elevation)),
        ("glass", render_glass(clouds, camera, pose)),
    ):
        for method, image in zip(("raw", "ours"), panels):
            image.save(target / f"{name}-{method}.webp", quality=90, method=6)
    with Image.open(prepared / path) as rgb:
        if rgb.size != (camera.width, camera.height):
            raise ValueError(f"RGB size differs from camera intrinsics for {scene}/{frame_id}")
        # Use the largest 3:2 crop at the native resolution.
        unit = min(camera.width // 3, camera.height // 2)
        reference = ImageOps.fit(
            rgb.convert("RGB"), (3 * unit, 2 * unit), method=Image.Resampling.LANCZOS
        )
        reference.save(target / "rgb.webp", quality=90, method=6)


def depth_palette():
    from matplotlib import colormaps

    return np.rint(255 * colormaps["viridis"](np.arange(256))[:, :3]).astype(np.uint8)


def refined_depth_path(scene, results, supplemental):
    frame = VIEWS[scene][2]
    cached = results / scene / "multiview_depths" / "multiview" / f"{frame}.npy"
    extra = supplemental / scene / f"{frame}.npy"
    for path in (cached, extra):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing metric input depth for {scene}/{frame}: {cached} or {extra}")


def render_input_depths(scene, dataset, results, refined_path, output, palette):
    frame = VIEWS[scene][2]
    prepared = dataset / scene
    camera = CameraIntrinsics.load(prepared / "intrinsics.json")
    expected = (camera.height, camera.width)
    config = json.loads((results / scene / "multiview" / "config.json").read_text())
    units = config["depth_units_per_meter"]
    if not np.isfinite(units) or units <= 0:
        raise ValueError(f"Invalid sensor depth units in the {scene} fusion configuration")
    with Image.open(prepared / "depth" / f"{frame}.png") as im:
        sensor = np.asarray(im, dtype=np.float32) / units
    with Image.open(prepared / "masks" / f"{frame}.png") as im:
        valid_pixels = np.asarray(im) == 255
    refined = np.load(refined_path, allow_pickle=False)
    if not all(array.shape == expected for array in (sensor, refined, valid_pixels)):
        raise ValueError(f"Input depths and mask must match the calibration for {scene}/{frame}")
    target = output / scene
    target.mkdir(parents=True, exist_ok=True)
    for name, depth in (("raw", sensor), ("ours", refined)):
        valid = valid_pixels & np.isfinite(depth)
        valid &= (depth >= config["min_depth"]) & (depth <= config["max_depth"])
        fraction = np.where(valid, np.minimum(depth / DEPTH_SCALE_M, 1), 0)
        indices = np.minimum((fraction * len(palette)).astype(np.int32), len(palette) - 1)
        colors = palette[indices]
        colors[~valid] = 0
        view = ImageOps.fit(Image.fromarray(colors), (WIDTH, HEIGHT), Image.Resampling.NEAREST)
        view.save(target / f"depth-{name}.webp", lossless=True, method=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/scannetpp"))
    parser.add_argument("--results", type=Path, default=Path("outputs/scannetpp"))
    parser.add_argument("--output", type=Path, default=Path("pics/reconstructions"))
    parser.add_argument("--scenes", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--depth-only", action="store_true", help="Render input depths and scale")
    parser.add_argument(
        "--supplemental-depths",
        type=Path,
        default=Path("outputs/reconstruction_explorer"),
        help="Metric NPY files at <root>/<scene>/<frame>.npy for frames outside the fusion cache",
    )
    args = parser.parse_args()
    if not args.depth_only:
        for scene in args.scenes:
            render_scene(scene, args.dataset, args.results, args.output)
        return
    paths = {
        scene: refined_depth_path(scene, args.results, args.supplemental_depths)
        for scene in args.scenes
    }
    palette = depth_palette()
    for scene, path in paths.items():
        render_input_depths(scene, args.dataset, args.results, path, args.output, palette)
    Image.fromarray(palette[None, :, :]).save(args.output / "depth-scale.png")


if __name__ == "__main__":
    main()
