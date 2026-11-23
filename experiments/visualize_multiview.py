"""Depth-map and temporal diagnostics for completed paired multi-view experiments."""

import argparse
import html
import json
import os
import shutil
from pathlib import Path

import numpy as np

from experiments.make_report import make_report
from experiments.run_comparison import save_json
from glassrecon.data import CameraIntrinsics, load_sequence
from glassrecon.evaluation import DepthMetrics
from glassrecon.refinement import interior_mask, metric_transform, warp_depth


def analyze(root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, output = Path(root), Path(output)
    report = json.loads((root / "comparison.json").read_text())
    config = report["protocol"]
    baseline = Path(config["reuse_recovery"])
    window = Path(config["multiview_dir"])
    rows = [json.loads(line) for line in (baseline / "frames.jsonl").read_text().splitlines()]
    window_rows = [json.loads(line) for line in (window / "frames.jsonl").read_text().splitlines()]
    if [r["frame"] for r in rows] != [r["frame"] for r in window_rows]:
        raise ValueError("Window output order differs from baseline")
    camera = CameraIntrinsics(**config["intrinsics_used"])
    frames = {f.key: f for f in load_sequence(config["dataset"])[0]}
    clean = (
        {f.key: f for f in load_sequence(config["reference_depth"])[0]}
        if config["reference_depth"]
        else None
    )

    def load(i):
        row = rows[i]
        frame = frames[row["frame"]]
        rgb, sensor = frame.load(config["depth_units_per_meter"], camera)
        prediction = np.load(baseline / "predictions" / f"{frame.key}.npy")
        fit = row["alignment"]["regional"]
        depths = {
            "sensor": sensor,
            "regional": metric_transform(prediction, fit["scale"], fit["shift"]),
        }
        for name in ["window_anchor", "multiview"]:
            depths[name] = np.load(window / name / f"{frame.key}.npy")
        return rgb, prediction, depths, np.array(row["pose"])

    lags = [lag for lag in [1, 10, 30] if lag < len(rows)]
    start = max(lags, default=0)
    indices = np.linspace(start, len(rows) - 1, min(24, len(rows) - start), dtype=int)
    temporal = {}
    for lag in lags:
        accumulators = {
            name: DepthMetrics() for name in ["sensor", "regional", "window_anchor", "multiview"]
        }
        for i in indices:
            _, _, depths, pose = load(i)
            _, _, previous, previous_pose = load(i - lag)
            warped = {
                name: warp_depth(z, previous_pose, pose, camera) for name, z in previous.items()
            }
            if clean:
                _, reference = clean[rows[i]["frame"]].load(config["depth_units_per_meter"], camera)
                _, prev_reference = clean[rows[i - lag]["frame"]].load(
                    config["depth_units_per_meter"], camera
                )
                warped_reference = warp_depth(prev_reference, previous_pose, pose, camera)
                mask = (
                    interior_mask(reference)
                    & (warped_reference > 0)
                    & (abs(warped_reference - reference) < 0.02)
                )
            else:
                mask = (
                    interior_mask(depths["sensor"])
                    & (warped["sensor"] > 0)
                    & (abs(warped["sensor"] - depths["sensor"]) < 0.1)
                )
            common = mask & np.logical_and.reduce(
                [(depths[name] > 0) & (warped[name] > 0) for name in depths]
            )
            for name in depths:
                accumulators[name].add(depths[name], warped[name], common)
        temporal[str(lag)] = {name: value.summary() for name, value in accumulators.items()}
    chosen = {
        len(rows) // 2: "middle fused frame",
        max(
            range(len(rows)), key=lambda i: rows[i]["alignment"]["regional"]["mean_absolute_error"]
        ): "worst per-frame sensor fit (selection does not use clean GT)",
    }
    examples = []
    depth_cmap = plt.get_cmap("viridis").copy()
    depth_cmap.set_bad("#dddddd")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#dddddd")
    for i, rule in chosen.items():
        rgb, prediction, depths, pose = load(i)
        key = rows[i]["frame"]
        reference = depths["sensor"]
        if clean:
            _, reference = clean[key].load(config["depth_units_per_meter"], camera)
        upper = float(np.percentile(reference[reference > 0], 99.5))
        fig, axes = plt.subplots(3, 4, figsize=(17, 10), layout="constrained")
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("RGB")
        rel = axes[0, 1].imshow(np.ma.masked_less_equal(prediction, 0), cmap=depth_cmap)
        axes[0, 1].set_title("DA3-LARGE before metric recovery")
        fig.colorbar(
            rel, ax=axes[0, 1], shrink=0.6, label="Relative depth: arbitrary units, NOT meters"
        )
        panels = [
            (axes[0, 2], depths["sensor"], "Raw sensor"),
            (
                axes[0, 3],
                reference,
                "Clean GT (evaluation only)" if clean else "Sensor reference (not GT)",
            ),
            (axes[1, 0], depths["regional"], "Per-frame regional recovery"),
            (axes[1, 1], depths["window_anchor"], "Sensor-anchor refit only"),
            (axes[1, 2], depths["multiview"], "Sliding-window recovery"),
        ]
        for axis, z, title in panels:
            im = axis.imshow(np.ma.masked_less_equal(z, 0), cmap=depth_cmap, vmin=0, vmax=upper)
            axis.set_title(title)
        fig.colorbar(
            im,
            ax=[axes[0, 2], axes[0, 3]],
            shrink=0.7,
            location="top",
            orientation="horizontal",
            label="Metric depth (m), common to all metric panels",
        )
        difference = np.where(
            (depths["regional"] > 0) & (depths["multiview"] > 0),
            (depths["multiview"] - depths["regional"]) * 100,
            np.nan,
        )
        change = axes[1, 3].imshow(difference, cmap="coolwarm", vmin=-10, vmax=10)
        axes[1, 3].set_title("Window minus per-frame correction")
        fig.colorbar(change, ax=axes[1, 3], shrink=0.6, label="Signed change (cm), clipped ±10")
        for col, name in enumerate(["regional", "window_anchor", "multiview"]):
            z = depths[name]
            valid = (z > 0) & (reference > 0)
            error = np.where(valid, abs(z - reference) * 100, np.nan)
            err = axes[2, col].imshow(error, cmap=error_cmap, vmin=0, vmax=20)
            axes[2, col].set_title(f"Error: {np.nanmean(error):.2f} cm MAE", fontsize=11)
        fig.colorbar(
            err, ax=list(axes[2, :3]), shrink=0.6, label="Absolute error (cm), clipped at 20"
        )
        diag = window_rows[i]
        axes[2, 3].text(
            0,
            0.95,
            f"Fixed-lag window: {diag['window_frames']} frames\n"
            f"Output correction: z = {diag['alpha']:.4f} z_perframe {diag['beta_m']:+.4f} m\n"
            f"Supported pairs: {diag['pairs']}\nSensor correspondences: {diag['correspondences']}\n"
            f"Latest input: {diag['latest_available_frame']}\nFinalized output: {key}\n"
            "Gray = missing\nSame RGB, model cache, poses and TSDF\n"
            "No clean depth used for optimization",
            va="top",
            fontsize=10,
            linespacing=1.6,
        )
        for axis in axes.ravel():
            axis.axis("off")
        fig.suptitle(f"{root.name} · frame {key}\n{rule}", fontsize=13, y=1.08)
        filename = f"{root.name}_{key}.png"
        fig.savefig(output / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        examples.append({"frame": key, "selection": rule, "image": filename})
    result = {
        "sequence": root.name,
        "examples": examples,
        "temporal": temporal,
        "target_frame_indices": indices.tolist(),
        "visibility_mask": "independent clean visible interiors, tolerance 2cm"
        if clean
        else "sensor-stable interiors, sensor warp agreement <10cm; favors sensor stability",
        "temporal_note": "Up to 24 uniformly spaced target frames; common valid support across all "
        "methods at each lag; disagreement is not accuracy.",
    }
    save_json(output / (root.name + "_diagnostics.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = [analyze(path, output) for path in args.runs]
    make_report(args.runs, output)
    # Keep every web-served report link within this narrowly scoped directory.
    document = (output / "report.html").read_text()
    (output / "data").mkdir(exist_ok=True)
    for path in args.runs:
        root = Path(path)
        shutil.copy2(root / "comparison.json", output / "data" / (root.name + ".json"))
        document = document.replace(
            html.escape(os.path.relpath(root / "comparison.json", output)), f"data/{root.name}.json"
        )
    (output / "report.html").write_text(document)
    document = """<!doctype html><html><meta charset="utf-8">
<title>Sliding-window depth recovery</title>
<style>body{font:16px/1.5 system-ui;max-width:1600px;margin:30px auto;padding:20px}
img{width:100%}a{color:#17697d}details{margin:20px 0}td,th{padding:8px;text-align:left}</style>
<h1>DA3-LARGE: per-frame versus sliding-window recovery</h1>
<p>Five active frames, four-frame latency. Same predictions and supplied GT poses.
Each reconstruction run records its TSDF backend and settings in the linked report.
Sensor depth and RGB provide optimization constraints; clean depth is used only for evaluation.
The sensor-anchor-only ablation omits cross-view terms. These are diagnostic fused-frame images;
held-out reconstruction scores are in the report.</p>
<p><a href="report.html">Reconstruction metrics and ablations</a></p>"""
    document += (
        "<nav>Jump to sequence: "
        + " · ".join(
            f"<a href='#{html.escape(result['sequence'])}'>"
            f"{html.escape(result['sequence'].removeprefix('comparison_multiview_'))}</a>"
            for result in results
        )
        + "</nav>"
    )
    for result in results:
        name = html.escape(result["sequence"])
        document += f"<h2 id='{name}'>{name}</h2>"
        document += "<p>" + html.escape(result["visibility_mask"]) + "</p>"
        document += (
            "<table><tr><th>Frame gap</th><th>Raw RMSE (cm)</th>"
            "<th>Per-frame</th><th>Anchor only</th><th>Window</th></tr>"
        )
        for lag, metrics in result["temporal"].items():
            cells = [lag] + [
                f"{metrics[n]['rmse_m'] * 100:.2f}"
                for n in ["sensor", "regional", "window_anchor", "multiview"]
            ]
            document += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        document += "</table><p>" + html.escape(result["temporal_note"]) + "</p>"
        for example in result["examples"]:
            filename = html.escape(example["image"])
            label = html.escape(example["frame"] + " — " + example["selection"])
            document += f"<details open><summary>{label}</summary>"
            document += (
                f"<a href='{filename}'><img loading='lazy' src='{filename}' "
                "alt='Per-frame and multi-view metric depth comparison'></a></details>"
            )
    (output / "index.html").write_text(document + "</html>")
    print(output / "index.html", flush=True)


if __name__ == "__main__":
    main()
