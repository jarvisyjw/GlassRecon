"""Run DA3-LARGE prediction and nvblox reconstruction in separate environments."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rgb-folder", type=Path, help="Defaults to DATASET/rgb")
    parser.add_argument("--intrinsics", type=Path, help="Defaults to DATASET/intrinsics.json")
    parser.add_argument("--poses", type=Path)
    parser.add_argument("--model-python", required=True)
    parser.add_argument(
        "--model-id",
        default="depth-anything/DA3-LARGE",
        help="DA3 model ID or local checkpoint directory",
    )
    parser.add_argument("--mapping-python", default=sys.executable)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-units-per-meter", type=float, default=5000)
    parser.add_argument("--live-view", action="store_true")
    args = parser.parse_args()
    rgb_folder = args.rgb_folder or args.dataset / "rgb"
    intrinsics = args.intrinsics or args.dataset / "intrinsics.json"
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new or empty output directory")
    for path in [args.dataset / "rgb.txt", args.dataset / "depth.txt", intrinsics]:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if not rgb_folder.is_dir() or (args.poses and not args.poses.is_file()):
        parser.error("RGB folder and supplied poses must exist")
    if args.depth_units_per_meter <= 0:
        parser.error("Depth units per meter must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    predictions = args.output.resolve() / "predictions"
    reconstruction = args.output.resolve() / "reconstruction"
    subprocess.run(
        [
            args.model_python,
            "-m",
            "glassrecon",
            "predict",
            "--image-folder",
            str(rgb_folder.resolve()),
            "--model",
            "da3",
            "--model-id",
            args.model_id,
            "--device",
            "cuda",
            "--output",
            str(predictions),
        ],
        cwd=root,
        check=True,
    )
    command = [
        args.mapping_python,
        "-m",
        "glassrecon",
        "reconstruct",
        "--dataset",
        str(args.dataset.resolve()),
        "--intrinsics",
        str(intrinsics.resolve()),
        "--model",
        "cached",
        "--prediction-dir",
        str(predictions),
        "--backend",
        "nvblox",
        "--alignment",
        "local",
        "--fusion-policy",
        "multiview",
        "--window-size",
        "5",
        "--depth-units-per-meter",
        str(args.depth_units_per_meter),
        "--save-depths",
        "--output",
        str(reconstruction),
    ]
    if (args.dataset / "masks").is_dir():
        command += ["--valid-mask-dir", str((args.dataset / "masks").resolve())]
    if args.poses:
        command += ["--pose-source", "file", "--pose-file", str(args.poses.resolve())]
    else:
        command += ["--pose-source", "odometry", "--tracking-depth", "sensor"]
    if args.live_view:
        command += ["--live-view", "--keep-viewer"]
    subprocess.run(command, cwd=root, check=True)
    print(f"View with: {args.mapping_python} -m glassrecon view --result {reconstruction}")


if __name__ == "__main__":
    main()
