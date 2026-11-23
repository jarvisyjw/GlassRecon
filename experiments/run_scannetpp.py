"""Run ScanNet++ preparation, held-in DA3 inference, and paired nvblox comparisons."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.run_comparison import save_json

SCENES = ["0a76e06478", "fd361ab85f", "c0f5742640"]


def predict(dataset, output):
    import numpy as np
    from huggingface_hub import try_to_load_from_cache

    from glassrecon.data import CameraIntrinsics, load_sequence
    from glassrecon.pipeline import prepare_output
    from glassrecon.predictors import create_predictor, write_prediction_metadata

    prepare_output(output)
    camera = CameraIntrinsics.load(dataset / "intrinsics.json")
    model = create_predictor("da3", "depth-anything/DA3-LARGE", "cuda")
    write_prediction_metadata(output, model, "da3")
    frames = load_sequence(dataset)[0]
    for i, frame in enumerate(frames):
        if i % 5 == 0:
            continue
        rgb, _ = frame.load(1000, camera)
        np.save(output / (frame.key + ".npy"), model.predict(rgb, frame.key))
        if i % 20 == 1:
            print("prediction", i + 1, "/", len(frames), flush=True)
    save_json(
        output / "environment.json",
        {
            "versions": {
                x: importlib.metadata.version(x) for x in ["torch", "numpy", "depth-anything-3"]
            },
            "model": model.model_id,
            "heldout_every": 5,
            "checkpoint_config_cache": str(try_to_load_from_cache(model.model_id, "config.json")),
            "frames": [frame.key for i, frame in enumerate(frames) if i % 5 != 0],
        },
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--root", type=Path, default=Path("/media/jingwen-orchtech/JWDATA2/scannetppv2/data")
    )
    cli.add_argument("--scenes", nargs="+", default=SCENES)
    cli.add_argument("--prepared-root", type=Path, default=Path("data/scannetpp"))
    cli.add_argument("--output-root", type=Path, default=Path("outputs/scannetpp"))
    cli.add_argument("--model-python", default="/tmp/glassrecon-model-cuda/bin/python")
    cli.add_argument("--max-frames", type=int)
    cli.add_argument("--predict-only", action="store_true")
    cli.add_argument("--dataset", type=Path)
    cli.add_argument("--prediction-output", type=Path)
    args = cli.parse_args()
    if args.predict_only:
        predict(args.dataset, args.prediction_output)
        return
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        HF_HOME=os.environ.get("HF_HOME", "/tmp/glassrecon-hf-cache"),
        HF_HUB_OFFLINE="1",
    )

    def execute(command, log):
        print("Running:", " ".join(map(str, command)), flush=True)
        with log.open("w") as stream:
            subprocess.run(
                list(map(str, command)),
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        dataset = args.prepared_root / scene
        base = args.output_root / scene
        base.mkdir(exist_ok=True)
        if not (dataset / "preparation.json").exists():
            command = [
                sys.executable,
                "-m",
                "experiments.prepare_scannetpp",
                "--root",
                args.root,
                "--scene",
                scene,
                "--output",
                dataset,
            ]
            if args.max_frames:
                command += ["--max-frames", args.max_frames]
            execute(command, base / "prepare.log")
        receipt = json.loads((dataset / "preparation.json").read_text())
        expected = min(
            args.max_frames or receipt["available_colmap_frames"],
            receipt["available_colmap_frames"],
        )
        if (
            receipt["prepared_frames"] != expected
            or Path(receipt["source"]).resolve() != (args.root / "data" / scene).resolve()
        ):
            raise ValueError("Prepared dataset differs from requested source/frame selection")
        cache = base / "predictions"
        if not (cache / "environment.json").exists():
            execute(
                [
                    args.model_python,
                    "-m",
                    "experiments.run_scannetpp",
                    "--predict-only",
                    "--dataset",
                    dataset,
                    "--prediction-output",
                    cache,
                ],
                base / "predict.log",
            )
        baseline = base / "baseline"
        if not (baseline / "comparison.json").exists():
            execute(
                [
                    sys.executable,
                    "-m",
                    "experiments.run_comparison",
                    "--dataset",
                    dataset,
                    "--poses",
                    dataset / "poses.txt",
                    "--intrinsics",
                    dataset / "intrinsics.json",
                    "--depth-units-per-meter",
                    1000,
                    "--reference-depth",
                    dataset / "reference",
                    "--reference-mesh",
                    dataset / "reference/mesh.ply",
                    "--valid-mask-dir",
                    dataset / "masks",
                    "--model",
                    "cached",
                    "--prediction-dir",
                    cache,
                    "--global-failure",
                    "empty",
                    "--backend",
                    "nvblox",
                    "--output",
                    baseline,
                ],
                base / "baseline.log",
            )
        if not (baseline / "predictions").exists():
            (baseline / "predictions").symlink_to("../predictions", target_is_directory=True)
        save_json(
            baseline / "model_environment.json",
            json.loads((cache / "environment.json").read_text()),
        )
        multi = base / "multiview"
        if not (multi / "comparison.json").exists():
            execute(
                [
                    sys.executable,
                    "-m",
                    "experiments.run_multiview",
                    baseline,
                    "--backend",
                    "nvblox",
                    "--output",
                    multi,
                ],
                base / "multiview.log",
            )
        print("Completed", scene, flush=True)


if __name__ == "__main__":
    main()
