"""Streaming prediction, scale recovery, tracking, and integration orchestration."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .alignment import AlignmentConfig, AlignmentError, recover_metric_depth
from .data import TUM_INTRINSICS, CameraIntrinsics, PoseTrajectory, load_sequence, matrix_quaternion
from .mapping import TrackingError, create_mapper
from .multiview import SlidingWindowRecovery, WindowConfig
from .predictors import create_predictor, write_prediction_metadata
from .refinement import ConservativeFusion, refine_alignment


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}; choose a new run directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _predictor(args):
    return create_predictor(
        args.model,
        args.model_id,
        args.device,
        args.prediction_dir,
        args.prediction_space,
        args.process_res,
    )


def predict_folder(args):
    paths = sorted(
        path
        for path in Path(args.image_folder).iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise ValueError("No RGB images found")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("RGB filenames must have unique stems")
    output = prepare_output(args.output)
    predictor = _predictor(args)
    write_prediction_metadata(output, predictor, args.model)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        depth = predictor.predict(rgb, path.stem)
        np.save(output / f"{path.stem}.npy", depth)
        print(f"Predicted {index + 1}/{len(paths)}: {path.name}", flush=True)
    return {
        "predictions": len(paths),
        "representation": predictor.representation,
        "output": str(output),
    }


def reconstruct(args, observer=None):
    intrinsics = CameraIntrinsics.load(args.intrinsics) if args.intrinsics else TUM_INTRINSICS
    frames, counts = load_sequence(args.dataset, args.max_time_difference)
    frames = frames[args.start :: args.stride]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError("No frames selected")
    print(
        f"Sequential reconstruction: {len(frames)} RGB-D frames, stride={args.stride}, "
        f"poses={args.pose_source}",
        flush=True,
    )
    if len(frames) == 1:
        print(
            "Single-frame input: geometry will be fused, but no camera motion can be estimated.",
            flush=True,
        )
    if observer is not None:
        observer.configure(
            intrinsics,
            len(frames),
            args.pose_source,
            args.depth_source,
            tracking_depth=args.tracking_depth,
        )
    trajectory = PoseTrajectory(args.pose_file) if args.pose_source == "file" else None
    config = AlignmentConfig(
        args.alignment,
        args.rows,
        args.cols,
        args.samples,
        args.iterations,
        args.min_depth,
        args.max_depth,
    )
    output = prepare_output(args.output)
    # Create mapper before downloading model weights, to fail early on setup errors.
    mapper = create_mapper(
        intrinsics,
        args.voxel_size,
        args.sdf_trunc,
        args.min_depth,
        args.max_depth,
        args.odometry_depth_difference,
        backend=getattr(args, "backend", "open3d"),
    )
    predictor = _predictor(args) if args.depth_source == "aligned" else None
    window = None
    if predictor is not None and args.fusion_policy == "multiview":
        if predictor.representation != "depth":
            raise ValueError("Multi-view recovery currently requires DA3 relative depth")
        window = SlidingWindowRecovery(
            intrinsics,
            WindowConfig(
                size=args.window_size,
                temporal_weight=args.temporal_weight,
                minimum=args.min_depth,
                maximum=args.max_depth,
            ),
        )
    if args.save_predictions and predictor is not None:
        write_prediction_metadata(output / "predictions", predictor, args.model)
    if args.save_depths:
        (output / "metric_depths").mkdir()
    settings = vars(args).copy()
    settings.update(
        {
            "intrinsics_used": asdict(intrinsics),
            "alignment_config": asdict(config),
            "pose_convention": "camera_to_world",
            "metric_depth_unit": "meters",
            "reconstruction_backend": mapper.metadata(),
        }
    )
    if predictor is not None:
        settings.update(model_id=predictor.model_id, prediction_space=predictor.representation)
    if window is not None:
        settings["multiview"] = window.settings()
        settings["integration_latency_frames"] = args.window_size - 1
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    rng = np.random.default_rng(args.seed)
    conservative = ConservativeFusion(intrinsics, args.min_depth, args.max_depth)
    skipped = 0
    camera_positions = []
    start_time = time.perf_counter()
    with (output / "frames.jsonl").open("w") as log, (output / "trajectory.txt").open("w") as poses:
        poses.write("# camera-to-world: timestamp tx ty tz qx qy qz qw\n")

        def commit(rgb, depth, pose, key, timestamp):
            mapper.integrate(rgb, depth, pose)
            camera_positions.append(pose[:3, 3].copy())
            if args.save_depths:
                np.save(output / "metric_depths" / f"{key}.npy", depth)
            values = [timestamp, *pose[:3, 3], *matrix_quaternion(pose[:3, :3])]
            poses.write(" ".join(f"{value:.9f}" for value in values) + "\n")
            poses.flush()
            return dict(
                status="integrated",
                valid_depth_pixels=int((depth > 0).sum()),
                camera_to_world=pose.tolist(),
                emitted_frame=key,
                emitted_timestamp=timestamp,
            )

        for index, frame in enumerate(frames):
            record = {
                "frame": frame.key,
                "timestamp": frame.timestamp,
                "depth_timestamp": frame.depth_timestamp,
            }
            started = time.perf_counter()
            pose = trajectory.at(frame.timestamp, args.pose_max_difference) if trajectory else None
            try:
                if trajectory is not None and pose is None:
                    # Unmatched trajectory endpoints/gaps are normal association exclusions.
                    record.update(status="skipped", reason="No pose within interpolation tolerance")
                    skipped += 1
                    continue
                rgb, sensor = frame.load(args.depth_units_per_meter, intrinsics)
                sensor[(sensor < args.min_depth) | (sensor > args.max_depth)] = 0
                if predictor is not None:
                    prediction = predictor.predict(rgb, frame.key)
                    if args.save_predictions:
                        np.save(output / "predictions" / f"{frame.key}.npy", prediction)
                    try:
                        depth, alignment = recover_metric_depth(
                            prediction, sensor, predictor.representation, config, rng
                        )
                        record["alignment"] = asdict(alignment)
                        if args.fusion_policy == "conservative":
                            depth, record["refinement"] = refine_alignment(
                                prediction, sensor, alignment, args.min_depth, args.max_depth
                            )
                    except AlignmentError as exc:
                        if args.fusion_policy != "conservative":
                            raise
                        depth = np.zeros_like(sensor)
                        record["recovery_fallback"] = str(exc)
                else:
                    depth = sensor
                if pose is None:
                    tracking_depth = (
                        sensor
                        if (args.tracking_depth == "sensor" or "recovery_fallback" in record)
                        else depth
                    )
                    pose = mapper.track(rgb, tracking_depth)
                if predictor is not None and args.fusion_policy == "conservative":
                    depth, _, _, record["fusion"] = conservative.fuse(sensor, depth, rgb, pose)
                key, timestamp = frame.key, frame.timestamp
                if window is not None:
                    node = window.push(key, timestamp, rgb, sensor, depth, pose)
                    if node is None:
                        record["status"] = "buffered"
                        continue
                    rgb, depth, pose = node["rgb"], node["depth"], node["pose"]
                    key, timestamp = node["key"], node["timestamp"]
                    record["multiview"] = node["diagnostics"]
                record.update(commit(rgb, depth, pose, key, timestamp))
            except (AlignmentError, TrackingError) as exc:
                record.update(status="failed", reason=str(exc))
                if args.on_failure == "stop":
                    raise RuntimeError(f"Frame {frame.key}: {exc}") from exc
                skipped += 1
            finally:
                record["seconds"] = time.perf_counter() - started
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                progress = {
                    "processed": index + 1,
                    "selected": len(frames),
                    "integrated": mapper.integrated,
                    "skipped": skipped,
                    "timestamp": frame.timestamp,
                    "frame_status": record.get("status", "error"),
                }
                pending = output / ".progress.json.tmp"
                pending.write_text(json.dumps(progress, indent=2) + "\n")
                pending.replace(output / "progress.json")
                if observer is not None:
                    observer.on_frame(
                        progress, pose if record.get("status") == "integrated" else None
                    )
                    if record.get("status") == "integrated" and (
                        mapper.integrated == 1 or mapper.integrated % args.viewer_every == 0
                    ):
                        observer.update_map(*mapper.point_cloud())
                print(
                    f"[{index + 1}/{len(frames)}] {frame.key}: {record.get('status', 'error')}",
                    flush=True,
                )
        if window is not None:
            for node in window.flush():
                record = commit(
                    node["rgb"], node["depth"], node["pose"], node["key"], node["timestamp"]
                )
                record.update(
                    frame=node["key"],
                    timestamp=node["timestamp"],
                    phase="window_flush",
                    multiview=node["diagnostics"],
                )
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                progress.update(
                    integrated=mapper.integrated,
                    timestamp=node["timestamp"],
                    frame_status="integrated",
                )
                (output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                if observer is not None:
                    observer.on_frame(progress, node["pose"])
    geometry = mapper.export(output)
    positions = np.asarray(camera_positions)
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    summary = {
        **counts,
        "selected": len(frames),
        "processed": len(frames),
        "integrated": mapper.integrated,
        "skipped": skipped,
        "seconds": time.perf_counter() - start_time,
        "pose_source": args.pose_source,
        "depth_source": args.depth_source,
        "tracking_depth": args.tracking_depth,
        "reconstruction_backend": mapper.metadata(),
        "first_timestamp": frames[0].timestamp,
        "last_timestamp": frames[-1].timestamp,
        "sequence_duration_seconds": frames[-1].timestamp - frames[0].timestamp,
        "estimated_path_length_m": path_length,
        **geometry,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if observer is not None:
        observer.update_map(*mapper.point_cloud())
        observer.finish(summary)
    return summary
