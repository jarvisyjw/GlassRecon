"""Re-fuse a completed comparison with another backend and exactly the same depths/poses."""

import argparse
import json
from pathlib import Path

from experiments.run_comparison import parser, run, save_json


def settings_for(source, output, backend):
    source = Path(source)
    if not (source / "comparison.json").exists():
        raise ValueError("Source must be a completed comparison")
    old = json.loads((source / "config.json").read_text())
    # Legacy completed runs predate backend selection. Fill new arguments with
    # parser defaults, then preserve every recorded experimental setting.
    settings = {a.dest: a.default for a in parser()._actions if a.dest != "help"}
    settings.update({k: v for k, v in old.items() if k in settings})
    baseline = Path(old.get("reuse_recovery") or source)
    settings.update(
        output=str(output),
        backend=backend,
        model="cached",
        model_id=None,
        device="cpu",
        prediction_dir=str(baseline / "predictions"),
        reuse_recovery=str(baseline),
    )
    return argparse.Namespace(**settings)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--backend", choices=["open3d", "nvblox"], required=True)
    args = cli.parse_args()
    settings = settings_for(args.source, args.output, args.backend)
    report = run(settings)
    original = json.loads((args.source / "selection.json").read_text())
    if original != json.loads((args.output / "selection.json").read_text()):
        raise RuntimeError("Backend rerun changed frame selection")
    source_env = args.source / "model_environment.json"
    save_json(
        args.output / "backend_rerun.json",
        {
            "source": str(args.source),
            "prediction_reuse": settings.prediction_dir,
            "recovery_reuse": settings.reuse_recovery,
            "multiview_reuse": settings.multiview_dir,
            "reconstruction_backend": report["protocol"]["reconstruction_backend"],
            "source_model_environment": json.loads(source_env.read_text())
            if source_env.exists()
            else None,
            "note": "Same split, poses, predictions and recovery; fusion implementation changes.",
        },
    )


if __name__ == "__main__":
    main()
