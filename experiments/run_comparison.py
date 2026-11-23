"""Paired raw/global/regional-depth fusion with identical supplied camera poses.

Run from the repository root: python -m experiments.run_comparison --help.
Every fifth associated frame is held out of both prediction/alignment and fusion.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from glassrecon.alignment import (
    AlignmentConfig,
    AlignmentError,
    AlignmentResult,
    recover_metric_depth,
)
from glassrecon.data import (
    TUM_INTRINSICS,
    CameraIntrinsics,
    PoseTrajectory,
    load_sequence,
    matrix_quaternion,
)
from glassrecon.evaluation import DepthMetrics, MeshSurface, sample_mesh, summarize_distances
from glassrecon.mapping import create_mapper
from glassrecon.pipeline import prepare_output
from glassrecon.predictors import create_predictor, write_prediction_metadata
from glassrecon.refinement import ConservativeFusion, metric_transform, refine_alignment


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args):
    import open3d as o3d

    if (
        args.holdout_every < 2
        or min(args.eval_pixel_step, args.reference_every, args.surface_samples, args.iterations)
        < 1
    ):
        raise ValueError("Require holdout-every >= 2 and positive sampling counts/strides")
    positive = [args.depth_units_per_meter, args.pose_max_difference, args.reference_voxel_size]
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("Depth divisor, pose tolerance and reference voxel size must be positive")
    if not 0 < args.voxel_size < args.sdf_trunc < float("inf"):
        raise ValueError("Require 0 < voxel-size < sdf-trunc < infinity")
    if not 0 < args.min_depth < args.max_depth < float("inf"):
        raise ValueError("Require 0 < min-depth < max-depth < infinity")
    if args.seed < 0 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError("Seed must be nonnegative and max-frames positive")
    if (args.model == "cached") != (args.prediction_dir is not None):
        raise ValueError("Use --model cached and --prediction-dir together")
    if args.reference_mesh and not args.reference_depth:
        raise ValueError("--reference-mesh requires independent --reference-depth")
    camera = CameraIntrinsics.load(args.intrinsics) if args.intrinsics else TUM_INTRINSICS
    frames, counts = load_sequence(args.dataset)
    if args.max_frames:
        frames = frames[: args.max_frames]
    trajectory = PoseTrajectory(args.poses)
    selected, excluded = [], []
    for index, frame in enumerate(frames):
        pose = trajectory.at(frame.timestamp, args.pose_max_difference)
        if pose is None:
            excluded.append(frame.key)
        else:
            selected.append((index, frame, pose))
    train = [row for row in selected if row[0] % args.holdout_every != 0]
    heldout = [row for row in selected if row[0] % args.holdout_every == 0]
    if not train or not heldout:
        raise ValueError("Need at least one training and held-out frame with valid poses")
    clean_frames = {}
    if args.reference_depth:
        clean_frames = {frame.key: frame for frame in load_sequence(args.reference_depth)[0]}
        if any(frame.key not in clean_frames for _, frame, _ in selected):
            raise ValueError("Reference depth must have the same RGB frame stems")
    output = prepare_output(args.output)
    settings = vars(args).copy()
    settings.update(
        intrinsics_used=asdict(camera),
        counts=counts,
        selected_frames=len(selected),
        fused_frames=len(train),
        heldout_frames=len(heldout),
        excluded_no_pose=excluded,
        pose_source="supplied",
        pose_convention="camera_to_world",
        metric_depth_unit="meters",
        registration="none; no ICP or scale fit",
        heldout_rule="associated frame index modulo holdout_every equals zero",
    )
    save_json(output / "config.json", settings)
    save_json(
        output / "selection.json",
        {
            "fused": [f.key for _, f, _ in train],
            "heldout": [f.key for _, f, _ in heldout],
            "excluded_no_pose": excluded,
        },
    )
    predictor = None
    cache = Path(args.prediction_dir) if args.prediction_dir else output / "predictions"
    if args.model == "cached":
        predictor = create_predictor("cached", prediction_dir=cache)
    previous_fits = None
    if getattr(args, "reuse_recovery", None):
        baseline = Path(args.reuse_recovery)
        if args.model != "cached" or cache.resolve() != (baseline / "predictions").resolve():
            raise ValueError("Reused fits require the baseline's exact prediction cache")
        old = json.loads((baseline / "config.json").read_text())
        for key in [
            "dataset",
            "poses",
            "intrinsics_used",
            "seed",
            "iterations",
            "min_depth",
            "max_depth",
            "depth_units_per_meter",
            "pose_max_difference",
        ]:
            if old[key] != settings[key]:
                raise ValueError(f"Reused recovery configuration differs: {key}")
        if json.loads((baseline / "selection.json").read_text()) != json.loads(
            (output / "selection.json").read_text()
        ):
            raise ValueError("Reused recovery frame selection differs")
        if old.get("valid_mask_dir") != settings.get("valid_mask_dir"):
            raise ValueError("Reused recovery validity masks differ")
        previous_fits = {
            r["frame"]: r["alignment"]
            for r in map(json.loads, (baseline / "frames.jsonl").read_text().splitlines())
        }
    variants = ["sensor", "global", "regional"]
    enhanced = getattr(args, "enhance", False)
    if enhanced:
        variants += ["refined", "filtered", "conservative"]
    window_dir = getattr(args, "multiview_dir", None)
    if window_dir:
        if previous_fits is None:
            raise ValueError("Multi-view cache requires --reuse-recovery and its prediction cache")
        window_dir = Path(window_dir)
        window_config = json.loads((window_dir / "config.json").read_text())
        if Path(window_config["baseline"]).resolve() != Path(args.reuse_recovery).resolve():
            raise ValueError("Multi-view cache must use the same verified baseline")
        if json.loads((window_dir / "selection.json").read_text()) != json.loads(
            (output / "selection.json").read_text()
        ):
            raise ValueError("Multi-view frame selection differs")
        variants += ["window_anchor", "multiview"]
        settings["multiview"] = window_config["window"]
        settings["multiview_latency_frames"] = window_config["latency_frames"]
    fusion = ConservativeFusion(camera, args.min_depth, args.max_depth)
    settings["common_support_variants"] = list(variants)
    save_json(output / "config.json", settings)
    if clean_frames:
        variants.append("clean_oracle")
    mappers = {
        name: create_mapper(
            camera,
            args.voxel_size,
            args.sdf_trunc,
            args.min_depth,
            args.max_depth,
            backend=getattr(args, "backend", "open3d"),
        )
        for name in variants
    }
    metric_accumulators = {name: DepthMetrics() for name in variants}
    settings["reconstruction_backend"] = next(iter(mappers.values())).metadata()
    save_json(output / "config.json", settings)
    configs = {
        name: AlignmentConfig(
            method=name,
            iterations=args.iterations,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        for name in ["global", "local"]
    }
    pose_rows, visible_points = [], []
    started = time.perf_counter()
    with (output / "frames.jsonl").open("w") as log:
        for ordinal, (index, frame, pose) in enumerate(train):
            rgb, sensor = frame.load(args.depth_units_per_meter, camera)
            sensor[(sensor < args.min_depth) | (sensor > args.max_depth)] = 0
            if predictor is None:
                predictor = create_predictor(args.model, args.model_id, args.device)
                write_prediction_metadata(cache, predictor, args.model)
            prediction = predictor.predict(rgb, frame.key)
            if args.model != "cached":
                np.save(cache / f"{frame.key}.npy", prediction)
            depths, alignments = {"sensor": sensor}, {}
            for name, method in [("global", "global"), ("regional", "local")]:
                # Frame-specific seeds are stable if the split or execution order changes.
                if previous_fits is not None:
                    recorded = previous_fits[frame.key][name]
                    if "failure" in recorded:
                        if name != "global" or getattr(args, "global_failure", "error") != "empty":
                            raise ValueError(
                                "Reused failed fit requires explicit global empty policy"
                            )
                        depths[name] = np.zeros_like(sensor)
                        alignments[name] = recorded
                        continue
                    fit = AlignmentResult(**recorded)
                    depths[name] = metric_transform(
                        prediction,
                        fit.scale,
                        fit.shift,
                        fit.representation,
                        args.min_depth,
                        args.max_depth,
                    )
                else:
                    try:
                        depths[name], fit = recover_metric_depth(
                            prediction,
                            sensor,
                            predictor.representation,
                            configs[method],
                            np.random.default_rng(np.random.SeedSequence([args.seed, index])),
                        )
                    except AlignmentError as exc:
                        if name != "global" or getattr(args, "global_failure", "error") != "empty":
                            raise
                        depths[name] = np.zeros_like(sensor)
                        alignments[name] = {"failure": str(exc), "policy": "empty_depth"}
                        print(
                            f"{frame.key}: global fit failed; empty global depth: {exc}", flush=True
                        )
                        continue
                alignments[name] = asdict(fit)
            if enhanced:
                depths["refined"], alignments["refinement"] = refine_alignment(
                    prediction,
                    sensor,
                    AlignmentResult(**alignments["regional"]),
                    args.min_depth,
                    args.max_depth,
                )
                depths["conservative"], depths["filtered"], _, alignments["fusion"] = fusion.fuse(
                    sensor,
                    depths["refined"],
                    rgb,
                    pose,
                )
            if window_dir:
                for name in ["window_anchor", "multiview"]:
                    depths[name] = np.load(window_dir / name / f"{frame.key}.npy")
                    if depths[name].shape != sensor.shape or not np.isfinite(depths[name]).all():
                        raise ValueError("Invalid multi-view depth cache")
            if clean_frames:
                _, clean = clean_frames[frame.key].load(args.depth_units_per_meter, camera)
                clean[(clean < args.min_depth) | (clean > args.max_depth)] = 0
                depths["clean_oracle"] = clean
                # One common, method-independent observed reference support.
                if ordinal % args.reference_every == 0:
                    from glassrecon.evaluation import camera_rays

                    rays = camera_rays(camera, pose, 4)
                    z = clean[::4, ::4]
                    points = rays[..., :3] + rays[..., 3:] * z[..., None]
                    visible_points.append(points[z > 0])
                for name in variants:
                    metric_accumulators[name].add(depths[name], clean)
            if getattr(args, "valid_mask_dir", None):
                with Image.open(Path(args.valid_mask_dir) / f"{frame.key}.png") as image:
                    valid_mask = np.array(image) == 255
                if valid_mask.shape != sensor.shape:
                    raise ValueError("Validity mask dimensions differ from depth")
                for depth in depths.values():
                    depth[~valid_mask] = 0
            for name, mapper in mappers.items():
                if alignments.get(name, {}).get("policy") == "empty_depth":
                    continue
                mapper.integrate(rgb, depths[name], pose)
            pose_rows.append([frame.timestamp, *pose[:3, 3], *matrix_quaternion(pose[:3, :3])])
            log.write(
                json.dumps(
                    {
                        "frame": frame.key,
                        "source_index": index,
                        "alignment": alignments,
                        "pose": pose.tolist(),
                    }
                )
                + "\n"
            )
            log.flush()
            progress = {
                "fused": ordinal + 1,
                "total_fusion_frames": len(train),
                "heldout": len(heldout),
                "seconds": time.perf_counter() - started,
            }
            save_json(output / "progress.json", progress)
            if ordinal % 20 == 0 or ordinal + 1 == len(train):
                print(
                    f"{output.name}: {ordinal + 1}/{len(train)} "
                    f"fused in all {len(variants)} variants",
                    flush=True,
                )
    results = {}
    meshes = {}
    for name in variants:
        mapper = mappers.pop(name)
        folder = output / name
        folder.mkdir()
        geometry = mapper.export(folder)
        np.savetxt(folder / "trajectory.txt", pose_rows, fmt="%.9f")
        save_json(folder / "config.json", {**settings, "variant": name})
        save_json(folder / "summary.json", {"integrated": mapper.integrated, **geometry})
        meshes[name] = o3d.io.read_triangle_mesh(str(folder / "mesh.ply"))
        results[name] = {"geometry": geometry}
        if clean_frames:
            results[name]["input_depth_against_clean"] = metric_accumulators[name].summary()
        # Export allocates a GPU mesh; release each volume before exporting the next.
        del mapper
    del mappers
    surfaces = {name: MeshSurface(mesh) for name, mesh in meshes.items()}
    rendered_metrics = {name: DepthMetrics() for name in variants}
    common_metrics = {name: DepthMetrics() for name in variants}
    for _, frame, pose in heldout:
        reference_frame = clean_frames[frame.key] if clean_frames else frame
        _, reference = reference_frame.load(args.depth_units_per_meter, camera)
        reference = reference[:: args.eval_pixel_step, :: args.eval_pixel_step]
        reference[(reference < args.min_depth) | (reference > args.max_depth)] = 0
        rendered = {
            name: surface.render(camera, pose, args.eval_pixel_step)
            for name, surface in surfaces.items()
        }
        # Oracle does not determine the common support of the competing methods.
        common = np.logical_and.reduce(
            [
                np.isfinite(rendered[name])
                & (rendered[name] >= args.min_depth)
                & (rendered[name] <= args.max_depth)
                for name in settings["common_support_variants"]
            ]
        )
        for name in variants:
            rendered[name][
                (rendered[name] < args.min_depth) | (rendered[name] > args.max_depth)
            ] = 0
            rendered_metrics[name].add(rendered[name], reference)
            common_metrics[name].add(rendered[name], reference, common)
    for name in variants:
        results[name]["heldout_render"] = rendered_metrics[name].summary()
        results[name]["heldout_render_common_support"] = common_metrics[name].summary()
    if args.reference_mesh:
        if not visible_points:
            raise ValueError("Surface completeness requires independent clean reference depths")
        reference_mesh = o3d.io.read_triangle_mesh(args.reference_mesh)
        reference_surface = MeshSurface(reference_mesh)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(visible_points)))
        visible = np.asarray(cloud.voxel_down_sample(args.reference_voxel_size).points)
        # Check coordinate conversion against clean depth before trusting any score.
        reference_check = reference_surface.distances(visible)
        validation = {
            "mean_m": float(reference_check.mean()),
            "median_m": float(np.median(reference_check)),
            "p95_m": float(np.percentile(reference_check, 95)),
        }
        save_json(output / "reference_coordinate_check.json", validation)
        if validation["median_m"] > 0.01 or validation["p95_m"] > 0.05:
            raise ValueError(f"Reference geometry/pose/depth mismatch: {validation}")
        o3d.io.write_point_cloud(
            str(output / "visible_reference.ply"),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(visible)),
        )
        for name in variants:
            sampled = sample_mesh(meshes[name], args.surface_samples, args.seed)
            accuracy = reference_surface.distances(sampled)
            completeness = surfaces[name].distances(visible)
            results[name]["surface"] = summarize_distances(accuracy, completeness)
            np.savez_compressed(
                output / name / "surface_errors.npz",
                points=sampled,
                accuracy=accuracy,
                completeness=completeness,
            )
    report = {
        "protocol": settings,
        "model_id": predictor.model_id,
        "prediction_space": predictor.representation,
        "heldout_reference": "independent clean rendered depth"
        if clean_frames
        else "held-out noisy TUM sensor depth; consistency, not surface ground truth",
        "seconds": time.perf_counter() - started,
        "results": results,
    }
    save_json(output / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--poses", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--intrinsics")
    result.add_argument(
        "--valid-mask-dir", help="Rectified validity masks applied to every fused depth"
    )
    result.add_argument("--reference-depth")
    result.add_argument("--reference-mesh")
    result.add_argument("--model", choices=["da3", "da2", "cached"], default="da3")
    result.add_argument("--model-id")
    result.add_argument("--device", default="cpu")
    result.add_argument("--prediction-dir")
    result.add_argument("--multiview-dir", help="Verified fixed-lag recovery cache")
    result.add_argument(
        "--enhance", action="store_true", help="Evaluate robust refit and conservative fusion"
    )
    result.add_argument(
        "--reuse-recovery", help="Reuse verified baseline fits with its exact cache"
    )
    result.add_argument(
        "--global-failure",
        choices=["error", "empty"],
        default="error",
        help="Explicitly allow a failed global baseline to contribute no geometry",
    )
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--iterations", type=int, default=100)
    result.add_argument("--holdout-every", type=int, default=5)
    result.add_argument("--max-frames", type=int)
    result.add_argument("--pose-max-difference", type=float, default=0.05)
    result.add_argument("--depth-units-per-meter", type=float, default=5000)
    result.add_argument("--voxel-size", type=float, default=0.02)
    result.add_argument("--backend", choices=["open3d", "nvblox"], default="open3d")
    result.add_argument("--sdf-trunc", type=float, default=0.08)
    result.add_argument("--min-depth", type=float, default=0.1)
    result.add_argument("--max-depth", type=float, default=10)
    result.add_argument("--eval-pixel-step", type=int, default=4)
    result.add_argument("--reference-every", type=int, default=5)
    result.add_argument("--reference-voxel-size", type=float, default=0.01)
    result.add_argument("--surface-samples", type=int, default=100_000)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if (
        arguments.holdout_every < 2
        or arguments.eval_pixel_step < 1
        or arguments.reference_every < 1
    ):
        raise SystemExit("Need holdout-every >= 2 and positive evaluation sampling strides")
    run(arguments)
