"""Command line entry points; run `python -m glassrecon --help`."""

import argparse
import json
import math


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def model_options(parser, cached=True):
    parser.add_argument(
        "--model", choices=["da3", "da2", "cached"] if cached else ["da3", "da2"], default="da3"
    )
    parser.add_argument("--model-id", help="Hugging Face model ID or local model directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--process-res", type=positive_int, default=504, help="DA3 input resolution"
    )
    parser.add_argument("--prediction-dir", help="Folder containing cached <RGB-stem>.npy files")
    parser.add_argument(
        "--prediction-space",
        choices=["auto", "depth", "inverse_depth"],
        default="auto",
        help="Cached prediction convention; auto reads metadata.json",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Metric scale recovery and RGB-D reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict relative depth for an RGB folder")
    predict.add_argument("--image-folder", required=True)
    predict.add_argument("--output", required=True, help="New or empty output directory")
    model_options(predict, cached=False)

    recon = commands.add_parser("reconstruct", help="Reconstruct a TUM-format RGB-D sequence")
    recon.add_argument("--dataset", required=True, help="Directory with rgb.txt and depth.txt")
    recon.add_argument("--output", required=True, help="New or empty output directory")
    recon.add_argument("--intrinsics", help="JSON containing width, height, fx, fy, cx, cy")
    recon.add_argument(
        "--depth-units-per-meter",
        type=positive_float,
        default=5000.0,
        help="Divide sensor values by this number; TUM PNG=5000, metric NPY=1",
    )
    recon.add_argument("--max-time-difference", type=positive_float, default=0.02)
    recon.add_argument(
        "--start", type=int, default=0, help="Start index after RGB/depth association"
    )
    recon.add_argument("--stride", type=positive_int, default=1)
    recon.add_argument("--max-frames", type=positive_int)
    model_options(recon)
    recon.add_argument(
        "--depth-source",
        choices=["aligned", "sensor"],
        default="aligned",
        help="Fuse recovered model depth, or run a raw-sensor baseline",
    )
    recon.add_argument("--alignment", choices=["local", "global"], default="local")
    recon.add_argument(
        "--fusion-policy",
        choices=["replace", "conservative", "multiview"],
        default="replace",
        help="Per-frame replacement, conservative completion, or fixed-lag multi-view recovery",
    )
    recon.add_argument("--window-size", type=positive_int, default=5)
    recon.add_argument("--temporal-weight", type=positive_float, default=1.0)
    recon.add_argument("--rows", type=positive_int, default=4)
    recon.add_argument("--cols", type=positive_int, default=4)
    recon.add_argument("--samples", type=positive_int, default=10)
    recon.add_argument("--iterations", type=positive_int, default=100)
    recon.add_argument("--seed", type=int, default=0)
    recon.add_argument("--min-depth", type=positive_float, default=0.1)
    recon.add_argument("--max-depth", type=positive_float, default=10.0)
    recon.add_argument("--pose-source", choices=["odometry", "file"], default="odometry")
    recon.add_argument(
        "--pose-file", help="TUM timestamp tx ty tz qx qy qz qw camera-to-world poses"
    )
    recon.add_argument(
        "--pose-max-difference",
        type=positive_float,
        default=0.05,
        help="Maximum distance to each bracketing pose timestamp in seconds",
    )
    recon.add_argument(
        "--tracking-depth",
        choices=["aligned", "sensor"],
        default="sensor",
        help="Depth used for odometry; fusion still follows --depth-source",
    )
    recon.add_argument("--odometry-depth-difference", type=positive_float, default=0.07)
    recon.add_argument(
        "--backend",
        choices=["open3d", "nvblox"],
        default="open3d",
        help="TSDF fusion backend; nvblox requires its CUDA environment",
    )
    recon.add_argument(
        "--voxel-size", type=positive_float, default=0.02, help="TSDF voxel size, meters"
    )
    recon.add_argument(
        "--sdf-trunc", type=positive_float, default=0.08, help="TSDF truncation, meters"
    )
    recon.add_argument("--on-failure", choices=["stop", "skip"], default="stop")
    recon.add_argument("--save-predictions", action="store_true")
    recon.add_argument("--save-depths", action="store_true", help="Save metric float32 NPY maps")
    recon.add_argument("--live-view", action="store_true", help="Show cumulative fusion in Viser")
    recon.add_argument("--viewer-host", default="127.0.0.1")
    recon.add_argument("--viewer-port", type=positive_int, default=8080)
    recon.add_argument(
        "--viewer-every",
        type=positive_int,
        default=10,
        help="Refresh the live map every N integrated frames",
    )
    recon.add_argument("--viewer-max-points", type=positive_int, default=200_000)
    recon.add_argument(
        "--keep-viewer",
        action="store_true",
        help="Keep Viser running after reconstruction until Ctrl+C",
    )

    view = commands.add_parser("view", help="Explore a saved reconstruction in Viser")
    view.add_argument("--result", required=True, help="Reconstruction output directory")
    view.add_argument("--host", default="127.0.0.1", help="Viewer bind address")
    view.add_argument("--port", type=positive_int, default=8080)
    view.add_argument(
        "--max-points",
        type=positive_int,
        default=500_000,
        help="Maximum displayed cloud points (does not change saved geometry)",
    )
    view.add_argument(
        "--max-cameras",
        type=positive_int,
        default=100,
        help="Maximum displayed trajectory camera frustums",
    )
    view.add_argument(
        "--point-size", type=positive_float, default=0.005, help="Initial point size in meters"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "view" and args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.command == "reconstruct":
        if args.start < 0 or args.seed < 0:
            parser.error("--start and --seed must be nonnegative")
        if args.samples < 2 or args.min_depth >= args.max_depth:
            parser.error("Require --samples >= 2 and --min-depth < --max-depth")
        if (args.pose_source == "file") != (args.pose_file is not None):
            parser.error("Use --pose-source file and --pose-file together")
        if args.sdf_trunc <= args.voxel_size:
            parser.error("--sdf-trunc must exceed --voxel-size")
        if args.viewer_port > 65535:
            parser.error("--viewer-port must be between 1 and 65535")
        if args.keep_viewer and not args.live_view:
            parser.error("--keep-viewer requires --live-view")
    live_viewer = None
    try:
        if args.command == "view":
            from .viewer import serve

            serve(args)
            return
        from .pipeline import predict_folder, reconstruct

        if args.command == "predict":
            result = predict_folder(args)
        else:
            if args.live_view:
                from .viewer import LiveReconstructionViewer

                live_viewer = LiveReconstructionViewer(
                    args.viewer_host, args.viewer_port, args.viewer_max_points
                )
            result = reconstruct(args, observer=live_viewer)
        print(json.dumps(result, indent=2), flush=True)
        if live_viewer is not None and args.keep_viewer:
            print("Reconstruction complete. Viser remains available; Ctrl+C to close.", flush=True)
            live_viewer.wait()
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"glassrecon: {exc}\n")
    finally:
        if live_viewer is not None:
            live_viewer.stop()
