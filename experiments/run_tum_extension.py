"""Run the fixed additional-TUM protocol, preserving completed experiments.

Run from the repository root in the DA3 environment. Download the selected datasets
first with experiments.download_data. No frame cap or per-sequence tuning is used.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from experiments.prepare_tum_poses import prepare_poses
from experiments.run_comparison import parser, run, save_json

MANIFEST = Path(__file__).with_name("tum_extension.json")


def environment(manifest, device):
    import torch
    from huggingface_hub import try_to_load_from_cache

    checkpoint = try_to_load_from_cache(manifest["model_id"], "model.safetensors")
    if not isinstance(checkpoint, str):
        raise RuntimeError("Download the pinned DA3-LARGE checkpoint into HF_HOME first")
    with open(checkpoint, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    revision = Path(checkpoint).parent.name
    if digest != manifest["checkpoint_sha256"] or revision != manifest["model_revision"]:
        raise RuntimeError("Cached model differs from the fixed experiment manifest")
    return {
        "model_id": manifest["model_id"],
        "model_revision": revision,
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "numpy", "open3d", "transformers"]
        },
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "da3_autocast": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "process_res": 504,
        "inference_mode": "single RGB image; cached predictions shared by all methods",
        "manifest": str(MANIFEST.relative_to(Path.cwd())),
    }


def main():
    manifest = json.loads(MANIFEST.read_text())
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sequences", nargs="+", choices=list(manifest["sequences"]))
    cli.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = cli.parse_args()
    provenance = environment(manifest, args.device)
    for name in args.sequences or manifest["sequences"]:
        entry = manifest["sequences"][name]
        dataset = Path(entry["dataset"])
        if not all((dataset / f).exists() for f in ["rgb.txt", "depth.txt", "groundtruth.txt"]):
            raise FileNotFoundError(f"Download {entry['download']} before running {name}")
        baseline = Path("outputs") / f"comparison_large_{name}"
        window = Path("outputs") / f"comparison_multiview_{name}"
        poses = prepare_poses(dataset / "groundtruth.txt", Path("data/prepared") / name)
        command = [
            "--dataset",
            str(dataset),
            "--poses",
            str(poses),
            "--output",
            str(baseline),
            "--model",
            "da3",
            "--model-id",
            manifest["model_id"],
            "--device",
            args.device,
            *manifest["comparison_arguments"],
        ]
        if entry["intrinsics"]:
            command += ["--intrinsics", entry["intrinsics"]]
        settings = parser().parse_args(command)
        if (baseline / "comparison.json").exists():
            old = json.loads((baseline / "config.json").read_text())
            if any(
                old.get(k, "open3d" if k == "backend" else None) != v
                for k, v in vars(settings).items()
            ):
                raise ValueError(f"Completed baseline settings differ: {baseline}")
            saved = json.loads((baseline / "model_environment.json").read_text())
            if saved["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
                raise ValueError("Completed baseline has a different checkpoint")
            print(f"Reusing completed baseline {baseline}", flush=True)
        else:
            run(settings)
            save_json(baseline / "model_environment.json", provenance)
            gc.collect()
            if args.device == "cuda":
                import torch

                torch.cuda.empty_cache()
        if (window / "comparison.json").exists():
            old = json.loads((window / "config.json").read_text())
            if (
                old["reuse_recovery"] != str(baseline)
                or old["multiview"]["size"] != manifest["window_size"]
                or old["multiview"]["temporal_weight"] != manifest["temporal_weight"]
            ):
                raise ValueError(f"Completed window settings differ: {window}")
            print(f"Reusing completed window {window}", flush=True)
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.run_multiview",
                    str(baseline),
                    "--output",
                    str(window),
                    "--window-size",
                    str(manifest["window_size"]),
                    "--temporal-weight",
                    str(manifest["temporal_weight"]),
                ],
                check=True,
            )
        print(f"Completed baseline and sliding window: {name}", flush=True)


if __name__ == "__main__":
    main()
