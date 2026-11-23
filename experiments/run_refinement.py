"""Repeat a DA3 comparison with cached predictions and recovery/fusion ablations."""

import argparse
import json
from pathlib import Path

from experiments.run_comparison import parser, run, save_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("baseline", type=Path)
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    baseline = args.baseline
    old = json.loads((baseline / "config.json").read_text())
    accepted = {action.dest for action in parser()._actions} - {"help"}
    settings = {key: value for key, value in old.items() if key in accepted}
    settings.update(
        output=args.output,
        model="cached",
        model_id=None,
        device="cpu",
        prediction_dir=str(baseline / "predictions"),
        enhance=True,
        reuse_recovery=str(baseline),
    )
    run(argparse.Namespace(**settings))
    source = baseline / "model_environment.json"
    if source.exists():
        environment = json.loads(source.read_text())
        environment["prediction_reuse"] = str(baseline / "predictions")
        environment["note"] = (
            "Original prediction runtime below; this run reuses exact cached arrays."
        )
        save_json(Path(args.output) / "model_environment.json", environment)


if __name__ == "__main__":
    main()
