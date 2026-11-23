"""Repeat an existing comparison with a different predictor, preserving its protocol."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from experiments.run_comparison import parser, run, save_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("baseline", type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--model-id", default="depth-anything/DA3-LARGE")
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    old = json.loads((args.baseline / "config.json").read_text())
    accepted = {action.dest for action in parser()._actions} - {"help"}
    settings = {key: value for key, value in old.items() if key in accepted}
    settings.update(
        output=str(args.output),
        model="da3",
        model_id=args.model_id,
        device=args.device,
        prediction_dir=None,
    )
    report = run(argparse.Namespace(**settings))
    selection = json.loads((args.output / "selection.json").read_text())
    if selection != json.loads((args.baseline / "selection.json").read_text()):
        raise RuntimeError("Frame selection differs from baseline")
    import torch
    from huggingface_hub import try_to_load_from_cache

    local_model = Path(args.model_id)
    checkpoint = (
        str(local_model / "model.safetensors")
        if local_model.is_dir()
        else try_to_load_from_cache(args.model_id, "model.safetensors")
    )
    provenance = {
        "baseline": str(args.baseline),
        "identical_frame_selection": True,
        "model_id": report["model_id"],
        "checkpoint": str(checkpoint),
        "model_revision": Path(checkpoint).parent.name
        if isinstance(checkpoint, str) and not local_model.is_dir()
        else None,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "torchvision", "numpy", "open3d", "transformers"]
        },
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "da3_autocast": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "note": "Original Small runs used CPU; hardware, precision and runtime may differ.",
    }
    save_json(args.output / "model_environment.json", provenance)


if __name__ == "__main__":
    main()
