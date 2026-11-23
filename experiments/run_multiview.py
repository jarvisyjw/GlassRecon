"""Fixed-lag multi-view ablation using exactly a completed run's DA3 prediction cache."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from experiments.run_comparison import parser, run, save_json
from glassrecon.data import CameraIntrinsics, load_sequence
from glassrecon.multiview import SlidingWindowRecovery, WindowConfig
from glassrecon.pipeline import prepare_output
from glassrecon.refinement import metric_transform


def prepare(baseline, destination, config):
    baseline, destination = Path(baseline), prepare_output(destination)
    old = json.loads((baseline / "config.json").read_text())
    selected = json.loads((baseline / "selection.json").read_text())
    rows = [json.loads(line) for line in (baseline / "frames.jsonl").read_text().splitlines()]
    if [row["frame"] for row in rows] != selected["fused"]:
        raise ValueError("Baseline log order differs from the fused selection")
    camera = CameraIntrinsics(**old["intrinsics_used"])
    frames = {f.key: f for f in load_sequence(old["dataset"])[0]}
    model = json.loads((baseline / "predictions/metadata.json").read_text())
    if model["representation"] != "depth":
        raise ValueError("Multi-view affine recovery currently requires relative depth (DA3)")
    solver = SlidingWindowRecovery(camera, config)
    for method in ["multiview", "window_anchor"]:
        (destination / method).mkdir()
    save_json(destination / "selection.json", selected)
    save_json(
        destination / "config.json",
        {
            "baseline": str(baseline),
            "baseline_protocol": old,
            "window": solver.settings(),
            "prediction_metadata": model,
            "latency_frames": config.size - 1,
            "correspondence_source": "observed sensor depth and RGB; no clean GT",
        },
    )
    with (destination / "frames.jsonl").open("w") as log:

        def emit(node):
            for name, depth in [
                ("multiview", node["depth"]),
                ("window_anchor", node["anchor_depth"]),
            ]:
                np.save(destination / name / (node["key"] + ".npy"), depth)
            log.write(
                json.dumps(
                    {
                        "frame": node["key"],
                        "pose": node["pose"].tolist(),
                        "timestamp": node["timestamp"],
                        **node["diagnostics"],
                    }
                )
                + "\n"
            )
            log.flush()

        for i, row in enumerate(rows):
            frame = frames[row["frame"]]
            rgb, sensor = frame.load(old["depth_units_per_meter"], camera)
            prediction = np.load(baseline / "predictions" / f"{frame.key}.npy")
            fit = row["alignment"]["regional"]
            base = metric_transform(
                prediction,
                fit["scale"],
                fit["shift"],
                minimum=old["min_depth"],
                maximum=old["max_depth"],
            )
            node = solver.push(frame.key, frame.timestamp, rgb, sensor, base, np.array(row["pose"]))
            if node is not None:
                emit(node)
            save_json(
                destination / "progress.json",
                {"processed": i + 1, "total": len(rows), "emitted": solver.emitted},
            )
            if i % 40 == 0:
                print(destination.name, i + 1, "/", len(rows), flush=True)
        for node in solver.flush():
            emit(node)
    if solver.emitted != len(rows):
        raise RuntimeError("Not every frame was emitted")
    save_json(
        destination / "progress.json",
        {"processed": len(rows), "total": len(rows), "emitted": solver.emitted, "complete": True},
    )
    return old


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("baseline", type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--window-size", type=int, default=5)
    cli.add_argument("--temporal-weight", type=float, default=1.0)
    cli.add_argument("--backend", choices=["open3d", "nvblox"], default="open3d")
    args = cli.parse_args()
    old = json.loads((args.baseline / "config.json").read_text())
    depth_dir = args.output.with_name(args.output.name + "_depths")
    config = WindowConfig(
        size=args.window_size,
        temporal_weight=args.temporal_weight,
        minimum=old["min_depth"],
        maximum=old["max_depth"],
    )
    old = prepare(args.baseline, depth_dir, config)
    accepted = {a.dest for a in parser()._actions} - {"help"}
    settings = {k: v for k, v in old.items() if k in accepted}
    settings.update(
        output=str(args.output),
        model="cached",
        model_id=None,
        device="cpu",
        prediction_dir=str(args.baseline / "predictions"),
        reuse_recovery=str(args.baseline),
        enhance=False,
        multiview_dir=str(depth_dir),
        backend=args.backend,
    )
    run(argparse.Namespace(**settings))
    source = args.baseline / "model_environment.json"
    if source.exists():
        environment = json.loads(source.read_text())
        environment["prediction_reuse"] = str(args.baseline / "predictions")
        environment["recovery_runtime"] = {
            name: importlib.metadata.version(name) for name in ["numpy", "open3d"]
        }
        environment["note"] = (
            "Original DA3 runtime recorded above; recovery reuses exact predictions."
        )
        save_json(args.output / "model_environment.json", environment)


if __name__ == "__main__":
    main()
