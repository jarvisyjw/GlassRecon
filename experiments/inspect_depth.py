"""Before/after depth plots and failure diagnostics using cached, fused-frame predictions.

No clean reference is used for refinement, filtering, frame selection, or fusion.
These are diagnostic input frames, separate from held-out reconstruction evaluation.
"""

import argparse
import html
import json
from pathlib import Path

import numpy as np

from glassrecon.alignment import AlignmentResult
from glassrecon.data import CameraIntrinsics, load_sequence
from glassrecon.evaluation import DepthMetrics
from glassrecon.refinement import (
    ConservativeFusion,
    interior_mask,
    metric_transform,
    refine_alignment,
    warp_depth,
)


def inspect(root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "config.json").read_text())
    camera = CameraIntrinsics(**config["intrinsics_used"])
    frames = {f.key: f for f in load_sequence(config["dataset"])[0]}
    clean = (
        {f.key: f for f in load_sequence(config["reference_depth"])[0]}
        if config["reference_depth"]
        else None
    )
    records = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    selected = {
        len(records) // 2: "middle fused frame",
        max(
            range(len(records)),
            key=lambda i: records[i]["alignment"]["regional"]["mean_absolute_error"],
        ): "worst regional fit to input sensor (not selected using clean depth)",
    }
    sample_indices = sorted(
        set(np.linspace(1, len(records) - 1, min(24, len(records) - 1), dtype=int)) | set(selected)
    )
    accumulators, temporal, entries, diagnostics = {}, {}, [], []
    metric_cmap = plt.get_cmap("viridis").copy()
    metric_cmap.set_bad("#dddddd")
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#dddddd")

    def load(i):
        row = records[i]
        rgb, sensor = frames[row["frame"]].load(config["depth_units_per_meter"], camera)
        sensor[(sensor < config["min_depth"]) | (sensor > config["max_depth"])] = 0
        prediction = np.load(root / "predictions" / f"{row['frame']}.npy")
        fit = AlignmentResult(**row["alignment"]["regional"])
        recovered = metric_transform(
            prediction,
            fit.scale,
            fit.shift,
            fit.representation,
            config["min_depth"],
            config["max_depth"],
        )
        refined, stats = refine_alignment(
            prediction, sensor, fit, config["min_depth"], config["max_depth"]
        )
        return row, rgb, sensor, prediction, recovered, refined, stats

    for index in sample_indices:
        row, rgb, sensor, prediction, recovered, refined, stats = load(index)
        previous, prev_rgb, prev_sensor, _, prev_recovered, _, _ = load(index - 1)
        fusion = ConservativeFusion(camera, config["min_depth"], config["max_depth"])
        fusion.fuse(prev_sensor, prev_recovered, prev_rgb, np.array(previous["pose"]))
        fused, filtered, fills, fusion_stats = fusion.fuse(
            sensor, refined, rgb, np.array(row["pose"])
        )
        if clean:
            _, reference = clean[row["frame"]].load(config["depth_units_per_meter"], camera)
        else:
            reference = sensor
        masks = {
            "all": reference > 0,
            "sensor_valid": sensor > 0,
            "sensor_hole": sensor == 0,
            "sensor_interior": interior_mask(sensor),
            "near": reference < 2,
            "far": reference >= 2,
        }
        depths = {
            "sensor": sensor,
            "regional": recovered,
            "refined": refined,
            "filtered": filtered,
            "conservative": fused,
        }
        for group, mask in masks.items():
            for name, depth in depths.items():
                accumulators.setdefault(group + "/" + name, DepthMetrics()).add(
                    depth, reference, mask
                )
        # Stability is diagnostic, not accuracy: an incorrectly constant map can also be stable.
        warped_sensor = warp_depth(
            prev_sensor, np.array(previous["pose"]), np.array(row["pose"]), camera
        )
        overlap = (
            interior_mask(sensor) & (warped_sensor > 0) & (np.abs(warped_sensor - sensor) < 0.1)
        )
        warped_recovered = warp_depth(
            prev_recovered, np.array(previous["pose"]), np.array(row["pose"]), camera
        )
        overlap &= (warped_recovered > 0) & (recovered > 0)
        for name, depth, warped in [
            ("sensor", sensor, warped_sensor),
            ("regional", recovered, warped_recovered),
        ]:
            temporal.setdefault(name, DepthMetrics()).add(depth, warped, overlap)
        diagnostics.append({"frame": row["frame"], "refinement": stats, "fusion": fusion_stats})
        if index not in selected:
            continue
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), layout="constrained")
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("RGB")
        rel = axes[0, 1].imshow(np.ma.masked_less_equal(prediction, 0), cmap=metric_cmap)
        axes[0, 1].set_title("Before recovery: DA3-LARGE")
        fig.colorbar(
            rel, ax=axes[0, 1], shrink=0.65, label="Relative depth — arbitrary units, NOT meters"
        )
        upper = float(np.percentile(reference[reference > 0], 99.5))
        metric_panels = [
            (axes[0, 2], sensor, "Raw sensor"),
            (
                axes[0, 3],
                reference,
                "Independent clean reference" if clean else "Sensor reference (not ground truth)",
            ),
            (axes[1, 0], recovered, "After regional scale + shift"),
            (axes[1, 1], refined, "After robust refinement"),
            (axes[1, 2], filtered, "Filtered sensor ablation"),
            (axes[1, 3], fused, "Conservative fusion input"),
        ]
        for axis, depth, title in metric_panels:
            metric = axis.imshow(
                np.ma.masked_less_equal(depth, 0), cmap=metric_cmap, vmin=0, vmax=upper
            )
            axis.set_title(title)
        fig.colorbar(
            metric,
            ax=[p[0] for p in metric_panels],
            shrink=0.6,
            label="Depth (meters), common scale",
        )
        for axis, (name, depth) in zip(
            axes[2],
            [
                ("Raw sensor", sensor),
                ("Regional", recovered),
                ("Refined", refined),
                ("Conservative", fused),
            ],
            strict=True,
        ):
            valid = (reference > 0) & (depth > 0)
            error = np.where(valid, np.abs(depth - reference) * 100, np.nan)
            im = axis.imshow(error, cmap=error_cmap, vmin=0, vmax=20)
            mae = float(np.nanmean(error))
            coverage = float(valid.sum() / max(1, (reference > 0).sum()))
            axis.set_title(f"{name}: MAE {mae:.2f} cm / coverage {coverage:.1%}", fontsize=10)
        fig.colorbar(
            im, ax=list(axes[2]), shrink=0.65, label="Absolute error (cm), display clipped at 20"
        )
        for axis in axes.ravel():
            axis.axis("off")
        fig.suptitle(
            f"{root.name} · {row['frame']}\n{selected[index]}\n"
            "Gray = missing; clean reference used only for diagnostics",
            fontsize=12,
        )
        name = f"{root.name}_{row['frame']}"
        fig.savefig(output / (name + ".png"), dpi=150)
        plt.close(fig)
        # Separate diagnostic figure makes the actual affine correction and admission mask explicit.
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        valid = (sensor > 0) & (prediction > 0)
        x, y = prediction[valid][::50], sensor[valid][::50]
        axes[0].scatter(x, y, s=2, alpha=0.15, label="Input sensor correspondences")
        line = np.linspace(float(x.min()), float(x.max()), 100)
        fit = row["alignment"]["regional"]
        axes[0].plot(
            line,
            fit["scale"] * line + fit["shift"],
            label=f"Regional: z={fit['scale']:.3f}d+{fit['shift']:.3f}",
        )
        axes[0].plot(
            line,
            stats["scale"] * line + stats["shift"],
            label=f"Refined: z={stats['scale']:.3f}d+{stats['shift']:.3f}",
        )
        axes[0].set(xlabel="DA3 relative depth", ylabel="Sensor depth (m)")
        axes[0].legend(fontsize=8)
        source = np.zeros((*sensor.shape, 3), np.uint8)
        source[filtered > 0] = [90, 140, 190]
        source[fills] = [240, 160, 40]
        axes[1].imshow(source)
        axes[1].set_title("Blue: sensor; orange: admitted model fill; black: missing")
        axes[1].axis("off")
        fig.savefig(output / (name + "_support.png"), dpi=150)
        plt.close(fig)
        entries.append(
            {
                "sequence": root.name,
                "frame": row["frame"],
                "selection": selected[index],
                "depth_plot": name + ".png",
                "support_plot": name + "_support.png",
                "refinement": stats,
                "fusion": fusion_stats,
            }
        )
    result = {
        "sequence": root.name,
        "reference": "independent clean depth" if clean else "input sensor consistency only",
        "sample_rule": "24 uniform fused frames plus middle and worst sensor-fit examples; "
        "exploratory diagnostics",
        "metrics": {k: v.summary() for k, v in accumulators.items()},
        "temporal_sensor_overlap": {k: v.summary() for k, v in temporal.items()},
        "frames": diagnostics,
        "examples": entries,
    }
    (output / (root.name + "_diagnostics.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [inspect(root, args.output) for root in args.runs]
    output = Path(args.output)
    document = """<!doctype html><html><meta charset="utf-8">
<title>Depth recovery diagnostics</title>
<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}
img{width:100%}details{margin:25px 0}a{color:#17697d}</style>
<h1>DA3-LARGE: before and after scale recovery</h1>
<p>Relative model depth has arbitrary units. All metric panels within an example share
one meter scale. Errors are saturated only for display; MAE uses unclipped values.
TUM comparisons use noisy sensor consistency, not independent ground truth.
ICL clean depth is used only for diagnostics. These are fused-frame examples,
not the held-out reconstruction test.</p>"""
    for result in results:
        document += f"<h2>{html.escape(result['sequence'])}</h2>"
        for entry in result["examples"]:
            document += (
                f"<details open><summary>{html.escape(entry['frame'] + ' — ' + entry['selection'])}"
                f"</summary><a href='{entry['depth_plot']}'>"
                f"<img src='{entry['depth_plot']}'></a>"
                f"<img src='{entry['support_plot']}'></details>"
            )
    (output / "index.html").write_text(document + "</html>")
    print(output / "index.html")
