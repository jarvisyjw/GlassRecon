"""Run optional ORB-SLAM3 localization, DA3-LARGE prediction and nvblox reconstruction."""

import argparse
import math
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--rgb-folder", type=Path, help="Defaults to DATASET/rgb")
    parser.add_argument("--intrinsics", type=Path, help="Defaults to DATASET/intrinsics.json")
    localization = parser.add_mutually_exclusive_group()
    localization.add_argument("--poses", type=Path, help="Supplied camera-to-world TUM trajectory")
    localization.add_argument(
        "--orb-python", help="Use this ORB-SLAM3 environment to estimate poses"
    )
    parser.add_argument("--orb-vocabulary", type=Path, help="ORBvoc.txt for RGB-D localization")
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
    if bool(args.orb_python) != (args.orb_vocabulary is not None):
        parser.error("Use --orb-python and --orb-vocabulary together")
    if args.orb_vocabulary is not None and not args.orb_vocabulary.is_file():
        parser.error(f"Missing ORB vocabulary: {args.orb_vocabulary}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Choose a new or empty output directory")
    for path in [args.dataset / "rgb.txt", args.dataset / "depth.txt", intrinsics]:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if not rgb_folder.is_dir() or (args.poses and not args.poses.is_file()):
        parser.error("RGB folder and supplied poses must exist")
    if not math.isfinite(args.depth_units_per_meter) or args.depth_units_per_meter <= 0:
        parser.error("Depth units per meter must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    predictions = args.output.resolve() / "predictions"
    reconstruction = args.output.resolve() / "reconstruction"
    pose_file = args.poses
    if args.orb_python:
        localization_dir = args.output.resolve() / "localization"
        subprocess.run(
            [
                args.orb_python,
                "-m",
                "glassrecon",
                "localize",
                "--dataset",
                str(args.dataset.resolve()),
                "--intrinsics",
                str(intrinsics.resolve()),
                "--depth-units-per-meter",
                str(args.depth_units_per_meter),
                "--vocabulary",
                str(args.orb_vocabulary.resolve()),
                "--output",
                str(localization_dir),
            ],
            cwd=root,
            check=True,
        )
        pose_file = localization_dir / "poses.txt"
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
    if pose_file:
        command += ["--pose-source", "file", "--pose-file", str(pose_file.resolve())]
        if args.orb_python:
            command += ["--pose-max-difference", "0"]
    else:
        command += ["--pose-source", "odometry", "--tracking-depth", "sensor"]
    if args.live_view:
        command += ["--live-view", "--keep-viewer"]
    subprocess.run(command, cwd=root, check=True)
    print(f"View with: {args.mapping_python} -m glassrecon view --result {reconstruction}")


if __name__ == "__main__":
    main()
