"""Predict one held-out RGB frame for display and align it with the original five-frame solver.

The office exhibit used ByteDance-Seed/Depth-Anything-3 commit
3d835ec1a5802d64a8b8b15f817a1ab54809bfe4. Run from the repository root in the
DA3 frontend environment with

    python -m experiments.prepare_explorer_depth 8a20d62ac0 frame_002450

The supplemental frame enters a separate replay of the original fusion sequence.
Only its pre-fusion metric depth is exported. The published evaluation uses the
original frame split and reconstructions.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from glassrecon.alignment import AlignmentConfig, recover_metric_depth
from glassrecon.data import CameraIntrinsics, PoseTrajectory, load_sequence
from glassrecon.multiview import SlidingWindowRecovery, WindowConfig
from glassrecon.predictors import create_predictor
from glassrecon.refinement import metric_transform

MODEL = "depth-anything/DA3-LARGE"
REVISION = "c54c26b16ec04d218e8d584ecf4bce082a9fcc20"
WEIGHT_SHA256 = "eaf2ae06df55889ad23eb245c82e2dd2a30c0cbf7e3d873a118fa5ed27a3e421"


def load_model(checkpoint, device):
    if checkpoint is None:
        from huggingface_hub import snapshot_download

        checkpoint = Path(
            snapshot_download(
                MODEL, revision=REVISION, allow_patterns=["config.json", "model.safetensors"]
            )
        )
    with (checkpoint / "model.safetensors").open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("Checkpoint differs from the original ScanNet++ DA3-LARGE run")
    return create_predictor("da3", str(checkpoint), device)


def supplemental_depth(dataset, run, key, model):
    config = json.loads((run / "multiview_depths/config.json").read_text())
    baseline = config["baseline_protocol"]
    split = json.loads((run / "baseline/selection.json").read_text())
    if key not in split["heldout"]:
        raise ValueError(f"{key} is not held out in this scene")
    with (run / "baseline/frames.jsonl").open() as log:
        rows = {row["frame"]: row for row in map(json.loads, log)}
    camera = CameraIntrinsics.load(dataset / "intrinsics.json")
    trajectory = PoseTrajectory(dataset / "poses.txt")
    solver = SlidingWindowRecovery(camera, WindowConfig(**config["window"]))
    alignment = AlignmentConfig(
        method="local",
        iterations=baseline["iterations"],
        min_depth=baseline["min_depth"],
        max_depth=baseline["max_depth"],
    )
    frames, _ = load_sequence(dataset)
    receipt = {}
    for index, frame in enumerate(frames):
        if frame.key != key and frame.key not in rows:
            continue
        rgb, sensor = frame.load(baseline["depth_units_per_meter"], camera)
        if frame.key == key:
            prediction = model.predict(rgb, key)
            _, fit = recover_metric_depth(
                prediction,
                sensor,
                "depth",
                alignment,
                np.random.default_rng(np.random.SeedSequence([baseline["seed"], index])),
            )
            scale, shift = fit.scale, fit.shift
            pose = trajectory.at(frame.timestamp, baseline["pose_max_difference"])
            if pose is None:
                raise ValueError(f"Missing pose for {key}")
            receipt = {"frame": key, "source_index": index, "regional_alignment": asdict(fit)}
        else:
            row = rows[frame.key]
            prediction = np.load(run / "predictions" / f"{frame.key}.npy", allow_pickle=False)
            cached_fit = row["alignment"]["regional"]
            scale, shift = cached_fit["scale"], cached_fit["shift"]
            pose = np.asarray(row["pose"])
        base = metric_transform(
            prediction,
            scale,
            shift,
            minimum=alignment.min_depth,
            maximum=alignment.max_depth,
        )
        node = solver.push(frame.key, frame.timestamp, rgb, sensor, base, pose)
        if node is not None and node["key"] == key:
            return node["depth"], {**receipt, "window": node["diagnostics"]}
    for node in solver.flush():
        if node["key"] == key:
            return node["depth"], {**receipt, "window": node["diagnostics"]}
    raise ValueError(f"Missing RGB-D frame {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene")
    parser.add_argument("frame")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/scannetpp"))
    parser.add_argument("--run-root", type=Path, default=Path("outputs/scannetpp"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reconstruction_explorer"))
    parser.add_argument("--checkpoint", type=Path, help="Local copy of the pinned model snapshot")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    dataset, run = args.dataset_root / args.scene, args.run_root / args.scene
    output = args.output_root / args.scene
    if any(
        output.resolve().is_relative_to(p.resolve()) for p in (args.dataset_root, args.run_root)
    ):
        parser.error("Write supplemental predictions to a separate output root")
    model = load_model(args.checkpoint, args.device)
    result, receipt = supplemental_depth(dataset, run, args.frame, model)
    path = output / f"{args.frame}.npy"
    output.mkdir(parents=True, exist_ok=True)
    np.save(path, result, allow_pickle=False)
    original = json.loads((run / "predictions/environment.json").read_text())
    receipt.update(
        scene=args.scene,
        model=MODEL,
        revision=REVISION,
        weight_sha256=WEIGHT_SHA256,
        runtime={
            "torch": model.torch.__version__,
            "cuda": model.torch.version.cuda,
            "numpy": np.__version__,
            "device": str(next(model.model.parameters()).device),
        },
        original_prediction_versions=original["versions"],
        depth_unit="meters",
        evaluation_split="heldout",
        use="display only",
    )
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
