"""Package a completed ScanNet++ multiview depth cache for direct nvblox mapping."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--depth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in (args.depth_cache / "frames.jsonl").read_text().splitlines()
    ]
    rgb = {}
    for line in (args.dataset / "rgb.txt").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            stamp, name = line.split(maxsplit=1)
            rgb[Path(name).stem] = (float(stamp), args.dataset / name)
    if not rows or len({row["frame"] for row in rows}) != len(rows):
        parser.error("Depth cache must contain unique frames")
    for row in rows:
        key = row["frame"]
        if key not in rgb or not rgb[key][1].is_file():
            parser.error(f"Missing RGB frame {key}")
        if not (args.depth_cache / "multiview" / f"{key}.npy").is_file():
            parser.error(f"Missing metric depth {key}")
    if args.output.exists():
        parser.error("Choose a new output directory")
    import numpy as np
    from PIL import Image

    from glassrecon.data import matrix_quaternion

    (args.output / "rgb").mkdir(parents=True)
    (args.output / "depth").mkdir()
    rgb_rows, depth_rows, poses = [], [], []
    for row in rows:
        key = row["frame"]
        stamp, image = rgb[key]
        shutil.copy2(image, args.output / "rgb" / image.name)
        depth = np.load(args.depth_cache / "multiview" / f"{key}.npy", allow_pickle=False)
        with Image.open(args.dataset / "masks" / f"{key}.png") as mask:
            valid = np.asarray(mask) == 255
        depth = np.where(valid, depth, 0).astype(np.float32)
        np.save(args.output / "depth" / f"{key}.npy", depth)
        rgb_rows.append(f"{stamp:.12f} rgb/{image.name}\n")
        depth_rows.append(f"{stamp:.12f} depth/{key}.npy\n")
        pose = np.asarray(row["pose"])
        poses.append([stamp, *pose[:3, 3], *matrix_quaternion(pose[:3, :3])])
    (args.output / "rgb.txt").write_text("".join(rgb_rows))
    (args.output / "depth.txt").write_text("".join(depth_rows))
    np.savetxt(args.output / "poses.txt", poses, fmt="%.12f")
    shutil.copy2(args.dataset / "intrinsics.json", args.output)
    shutil.copy2(args.depth_cache / "config.json", args.output / "source_config.json")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "depth_unit": "meters",
                "model": "depth-anything/DA3-LARGE",
                "pose_source": "supplied COLMAP poses",
                "frames": [r["frame"] for r in rows],
                "expected_integrated_frames": len(rows),
                "backend": "nvblox",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Packaged {len(rows)} frames in {args.output}")


if __name__ == "__main__":
    main()
