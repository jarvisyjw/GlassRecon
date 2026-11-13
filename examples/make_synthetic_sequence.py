"""Create a tiny calibrated sequence for a download-free reconstruction smoke test."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_sequence(output, frame_count=3, corrupt_sensor=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    for folder in ("rgb", "depth", "predictions"):
        (output / folder).mkdir()
    width, height, focal = 96, 72, 85.0
    camera = dict(
        width=width, height=height, fx=focal, fy=focal, cx=(width - 1) / 2, cy=(height - 1) / 2
    )
    (output / "intrinsics.json").write_text(json.dumps(camera, indent=2) + "\n")
    (output / "predictions" / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "model_id": "synthetic-affine", "representation": "depth"})
        + "\n"
    )
    yy, xx = np.mgrid[:height, :width]
    ray_x, ray_y = (xx - camera["cx"]) / focal, (yy - camera["cy"]) / focal
    rgb_rows, depth_rows, poses = [], [], []
    for i in range(frame_count):
        stamp = 1 + i * 0.03
        key = f"{stamp:.6f}"
        tx = i * 0.02
        # World plane z = 2 + 0.2*x, with camera motion along world x.
        depth = (2 + 0.2 * tx) / (1 - 0.2 * ray_x)
        world_x, world_y = ray_x * depth + tx, ray_y * depth
        # Mixed frequencies remain distinctive through the odometry pyramid;
        # a single repeating pattern can alias at this tiny image resolution.
        rgb = np.stack(
            [
                128
                + 45 * np.sin(world_x * 6)
                + 40 * np.cos(world_y * 8)
                + 25 * np.sin((world_x + world_y) * 14),
                128
                + 45 * np.sin(world_x * 9 + 0.7)
                + 40 * np.cos(world_y * 5 + 0.2)
                + 25 * np.cos((world_x - 2 * world_y) * 10),
                128
                + 45 * np.cos(world_x * 4 - 0.4)
                + 40 * np.sin(world_y * 12)
                + 25 * np.sin((2 * world_x + world_y) * 7),
            ],
            axis=-1,
        ).astype(np.uint8)
        sensor = depth.copy()
        if corrupt_sensor:
            sensor[10:20, 10:25] = 0  # Missing measurements.
            sensor[30:35, 30:45] += 1.5  # Local sensor corruption.
        Image.fromarray(rgb).save(output / "rgb" / f"{key}.png")
        Image.fromarray(np.rint(sensor * 5000).astype(np.uint16)).save(
            output / "depth" / f"{key}.png"
        )
        np.save(output / "predictions" / f"{key}.npy", ((depth - 0.3) / 1.7).astype(np.float32))
        rgb_rows.append(f"{stamp:.6f} rgb/{key}.png\n")
        depth_rows.append(f"{stamp + 0.002:.6f} depth/{key}.png\n")
        poses.append(f"{stamp:.6f} {tx:.6f} 0 0 0 0 0 1\n")
    (output / "rgb.txt").write_text("".join(rgb_rows))
    (output / "depth.txt").write_text("".join(depth_rows))
    (output / "groundtruth.txt").write_text("".join(poses))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(make_sequence(args.output))
