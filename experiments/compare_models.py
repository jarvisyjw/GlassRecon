"""Compare paired model-size experiments without overwriting their original reports."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def compare(baselines, candidates, output):
    if len(baselines) != len(candidates):
        raise ValueError("Supply equal numbers of baseline and candidate runs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for baseline, candidate in zip(baselines, candidates, strict=True):
        roots = [Path(baseline), Path(candidate)]
        reports = [json.loads((root / "comparison.json").read_text()) for root in roots]
        selections = [json.loads((root / "selection.json").read_text()) for root in roots]
        if selections[0] != selections[1]:
            raise ValueError(f"Frame selection mismatch: {roots}")
        exempt = {"model", "model_id", "device", "prediction_dir", "output", "reuse_recovery"}
        protocols = [{k: v for k, v in r["protocol"].items() if k not in exempt} for r in reports]
        for protocol in protocols:
            protocol.setdefault("enhance", False)
            protocol.setdefault("common_support_variants", ["sensor", "global", "regional"])
        if protocols[0] != protocols[1]:
            raise ValueError(f"Protocol mismatch: {roots}")
        trajectories = [np.loadtxt(root / "sensor" / "trajectory.txt") for root in roots]
        if not np.array_equal(*trajectories):
            raise ValueError(f"Camera trajectory mismatch: {roots}")
        for variant in reports[0]["results"]:
            values = [report["results"][variant] for report in reports]
            metrics = {
                "heldout_depth_mae_cm": [
                    v["heldout_render_common_support"]["mae_m"] * 100 for v in values
                ],
                "render_coverage_percent": [v["heldout_render"]["coverage"] * 100 for v in values],
            }
            if "surface" in values[0]:
                for key in [
                    "accuracy_mean_m",
                    "accuracy_median_m",
                    "accuracy_p95_m",
                    "completeness_mean_m",
                ]:
                    metrics[key.removesuffix("_m") + "_cm"] = [
                        v["surface"][key] * 100 for v in values
                    ]
                metrics["fscore_5cm_percent"] = [
                    v["surface"]["at_0.05m"]["fscore"] * 100 for v in values
                ]
            for metric, pair in metrics.items():
                rows.append(
                    {
                        "sequence": Path(reports[0]["protocol"]["dataset"]).name,
                        "variant": variant,
                        "metric": metric,
                        "baseline_model": reports[0]["model_id"],
                        "candidate_model": reports[1]["model_id"],
                        "baseline": pair[0],
                        "candidate": pair[1],
                        "candidate_minus_baseline": pair[1] - pair[0],
                    }
                )
    with (output / "model_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "model_comparison.json").write_text(
        json.dumps(
            {
                "notes": [
                    "Frame selections, camera trajectories and reconstruction/evaluation "
                    "parameters verified identical.",
                    "Depth common support is shared across fusion methods within each run, "
                    "not across models; inspect coverage.",
                    "Consult model_environment.json for hardware, precision "
                    "and runtime differences.",
                    "TUM measures held-out sensor consistency, not independent surface accuracy.",
                ],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.output)
