"""Audit and preserve compact results from the preselected additional TUM sequences."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.download_data import SOURCES
from experiments.run_comparison import save_json
from experiments.run_tum_extension import MANIFEST


def read(path):
    return json.loads(Path(path).read_text())


def summarize(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = read(MANIFEST)
    shutil.copy2(MANIFEST, output / "manifest.json")
    audits, rows = {}, []
    for name, entry in manifest["sequences"].items():
        baseline = Path("outputs") / f"comparison_large_{name}"
        window = Path("outputs") / f"comparison_multiview_{name}"
        report = read(window / "comparison.json")
        old = read(baseline / "config.json")
        config = report["protocol"]
        selection = read(baseline / "selection.json")
        cache = Path(config["multiview_dir"])
        assert selection == read(window / "selection.json") == read(cache / "selection.json")
        assert set(selection["fused"]).isdisjoint(selection["heldout"])
        for key in [
            "dataset",
            "poses",
            "intrinsics_used",
            "seed",
            "iterations",
            "holdout_every",
            "voxel_size",
            "sdf_trunc",
            "min_depth",
            "max_depth",
            "depth_units_per_meter",
        ]:
            assert config[key] == old[key], (name, key)
        assert old["max_frames"] is None and old["reference_depth"] is None
        assert report["model_id"] == manifest["model_id"]
        assert config["multiview"]["size"] == manifest["window_size"]
        assert config["multiview"]["temporal_weight"] == manifest["temporal_weight"]
        cached_keys = {path.stem for path in (baseline / "predictions").glob("*.npy")}
        assert cached_keys == set(selection["fused"])
        emitted = [json.loads(line) for line in (cache / "frames.jsonl").read_text().splitlines()]
        assert [row["frame"] for row in emitted] == selection["fused"]
        for i, row in enumerate(emitted):
            assert row["latest_available_frame"] == selection["fused"][min(i + 4, len(emitted) - 1)]
            assert row["boundary_frame"] == (selection["fused"][i - 1] if i else None)
        trajectory = (baseline / "sensor/trajectory.txt").read_bytes()
        for method in report["results"]:
            assert (window / method / "trajectory.txt").read_bytes() == trajectory
        runtime = read(baseline / "model_environment.json")
        assert runtime["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        audits[name] = {
            "fused": len(selection["fused"]),
            "heldout": len(selection["heldout"]),
            "excluded_no_pose": len(selection["excluded_no_pose"]),
            "associated": old["counts"]["associated"],
            "identical_selections_and_trajectories": True,
            "prediction_cache_contains_only_fused_frames": True,
            "fixed_lag_and_boundary_verified": True,
            "rejected_windows": sum(not row["accepted"] for row in emitted),
            "mean_correspondences_per_window": float(
                np.mean([r["correspondences"] for r in emitted])
            ),
            "intrinsics": config["intrinsics_used"],
        }
        row = {"sequence": name}
        for method in ["sensor", "regional", "window_anchor", "multiview"]:
            result = report["results"][method]
            row[method + "_mae_cm"] = result["heldout_render_common_support"]["mae_m"] * 100
            row[method + "_coverage_percent"] = result["heldout_render"]["coverage"] * 100
        row["window_vs_regional_percent"] = (
            row["multiview_mae_cm"] / row["regional_mae_cm"] - 1
        ) * 100
        rows.append(row)
        for source, filename in [
            (window / "comparison.json", name + ".json"),
            (window / "model_environment.json", name + "_environment.json"),
            (cache / "config.json", name + "_window_config.json"),
            (window / "selection.json", name + "_selection.json"),
        ]:
            shutil.copy2(source, output / filename)
        url = SOURCES[entry["download"]][0]
        receipt = Path("data/downloads") / (url.rsplit("/", 1)[1] + ".json")
        shutil.copy2(receipt, output / (name + "_download.json"))
        pose_preparation = Path("data/prepared") / name / "pose_preparation.json"
        if pose_preparation.exists():
            shutil.copy2(pose_preparation, output / (name + "_pose_preparation.json"))
        diagnostic = window / "raw_heldout_diagnostic.json"
        if diagnostic.exists():
            shutil.copy2(diagnostic, output / (name + "_raw_heldout_diagnostic.json"))
    summary = {
        "scope": "Four preselected additional sequences, single seed, fixed settings",
        "metric": "Held-out noisy TUM sensor depth consistency on common rendered pixels",
        "rows": rows,
        "fused_total": sum(row["fused"] for row in audits.values()),
        "heldout_total": sum(row["heldout"] for row in audits.values()),
        "sequence_macro_mae_cm": {
            method: float(np.mean([row[method + "_mae_cm"] for row in rows]))
            for method in ["sensor", "regional", "window_anchor", "multiview"]
        },
    }
    save_json(output / "audit.json", audits)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", default="experiments/results/da3_tum_extension")
    summarize(cli.parse_args().output)
