"""Export comparison JSONs as CSV, scientific plots, and a standalone HTML report."""

import argparse
import csv
import html
import json
import os
from pathlib import Path

import numpy as np

from experiments.view_comparison import LABELS


def make_report(paths, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports = [
        (Path(path), json.loads((Path(path) / "comparison.json").read_text())) for path in paths
    ]
    rows = []
    for root, report in reports:
        for variant, result in report["results"].items():
            depth = result["heldout_render_common_support"]
            surface = result.get("surface", {})
            rows.append(
                {
                    "sequence": root.name,
                    "variant": variant,
                    "model_id": report["model_id"],
                    "backend": report["protocol"].get("backend", "open3d"),
                    "fused_frames": report["protocol"]["fused_frames"],
                    "heldout_frames": report["protocol"]["heldout_frames"],
                    "heldout_reference": report["heldout_reference"],
                    "common_depth_mae_cm": depth["mae_m"] * 100,
                    "common_depth_rmse_cm": depth["rmse_m"] * 100,
                    "render_coverage_percent": result["heldout_render"]["coverage"] * 100,
                    "surface_accuracy_cm": surface.get("accuracy_mean_m", np.nan) * 100,
                    "surface_accuracy_median_cm": surface.get("accuracy_median_m", np.nan) * 100,
                    "surface_accuracy_p95_cm": surface.get("accuracy_p95_m", np.nan) * 100,
                    "surface_accuracy_rmse_cm": surface.get("accuracy_rmse_m", np.nan) * 100,
                    "surface_completeness_cm": surface.get("completeness_mean_m", np.nan) * 100,
                    "surface_fscore_5cm_percent": surface.get("at_0.05m", {}).get("fscore", np.nan)
                    * 100,
                    "surface_fscore_2cm_percent": surface.get("at_0.02m", {}).get("fscore", np.nan)
                    * 100,
                    "surface_precision_5cm_percent": surface.get("at_0.05m", {}).get(
                        "precision", np.nan
                    )
                    * 100,
                    "surface_recall_5cm_percent": surface.get("at_0.05m", {}).get("recall", np.nan)
                    * 100,
                }
            )
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    colors = {
        "sensor": "#64748b",
        "global": "#d99a32",
        "regional": "#287c91",
        "clean_oracle": "#70a865",
        "refined": "#9567ba",
        "filtered": "#9ba6af",
        "conservative": "#d95062",
        "window_anchor": "#b990b9",
        "multiview": "#c54e46",
    }
    available = [name for name in LABELS if all(name in r["results"] for _, r in reports)]
    displayed = [name for name in LABELS if any(name in r["results"] for _, r in reports)]
    tum_names = [root.name for root, report in reports if not report["protocol"]["reference_mesh"]]
    icl_names = [root.name for root, report in reports if report["protocol"]["reference_mesh"]]
    backends = {root.name: report["protocol"].get("backend", "open3d") for root, report in reports}
    mixed_backends = len(set(backends.values())) > 1
    if len(tum_names) > 4:
        fig = plt.figure(figsize=(18, 10), layout="constrained")
        grid = fig.add_gridspec(2, 3)
        tum_axis = fig.add_subplot(grid[0, :])
        icl_axes = [fig.add_subplot(grid[1, column]) for column in range(3)]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
        tum_axis = axes[0, 0]
        icl_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]

    def bars(axis, names, field, title, ylabel, variants):
        width = 0.8 / len(variants)
        for i, variant in enumerate(variants):
            values = [
                next(
                    row[field]
                    for row in rows
                    if row["sequence"] == name and row["variant"] == variant
                )
                for name in names
            ]
            x = np.arange(len(names)) + (i - (len(variants) - 1) / 2) * width
            containers = axis.bar(x, values, width, label=LABELS[variant], color=colors[variant])
            axis.bar_label(containers, fmt="%.1f", fontsize=8, padding=3)
        axis.set_xticks(
            np.arange(len(names)),
            [
                name.removeprefix("comparison_").removeprefix("multiview_").removeprefix("nvblox_")
                + (f"\n({backends[name]})" if mixed_backends else "")
                for name in names
            ],
        )
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_ylabel(ylabel)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.margins(y=0.18)
        axis.set_ylim(bottom=0)

    bars(
        tum_axis,
        tum_names,
        "common_depth_mae_cm",
        "TUM: held-out sensor consistency ↓",
        "Depth MAE (cm)",
        [name for name in available if name != "clean_oracle"],
    )
    for axis, field, title, unit in [
        (icl_axes[0], "surface_accuracy_cm", "ICL-NUIM: surface accuracy ↓", "Mean distance (cm)"),
        (
            icl_axes[1],
            "surface_completeness_cm",
            "ICL-NUIM: visible-surface completeness ↓",
            "Mean distance (cm)",
        ),
        (icl_axes[2], "surface_fscore_5cm_percent", "ICL-NUIM: F-score at 5 cm ↑", "F-score (%)"),
    ]:
        icl_variants = [
            name
            for name in LABELS
            if all(name in report["results"] for root, report in reports if root.name in icl_names)
        ]
        bars(axis, icl_names, field, title, unit, icl_variants)
    icl_axes[2].set_ylim(0, 110)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[name]) for name in displayed]
    fig.legend(
        handles,
        [LABELS[name] for name in displayed],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Depth recovery ablation · supplied poses · recorded fusion settings", fontsize=15)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)
    table_rows = []
    for row in rows:

        def number(key):
            return f"{row[key]:.2f}" if np.isfinite(row[key]) else "—"

        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in [
                    row["sequence"].removeprefix("comparison_"),
                    row["backend"],
                    LABELS[row["variant"]],
                    row["fused_frames"],
                    row["heldout_frames"],
                    number("common_depth_mae_cm"),
                    number("render_coverage_percent"),
                    number("surface_accuracy_cm"),
                    number("surface_completeness_cm"),
                    number("surface_fscore_5cm_percent"),
                ]
            )
            + "</tr>"
        )
    sources = "".join(
        f'<li><a href="{html.escape(os.path.relpath(root / "comparison.json", output))}">'
        f"{html.escape(root.name)}: full configuration and metrics</a></li>"
        for root, _ in reports
    )
    configurations = "".join(
        "<li>"
        + html.escape(
            f"{root.name}: {report['model_id']}; seed {report['protocol']['seed']}; "
            f"backend {report['protocol'].get('backend', 'open3d')}; "
            f"voxel {report['protocol']['voxel_size']} m; "
            f"truncation {report['protocol']['sdf_trunc']} m; "
            f"hold out every {report['protocol']['holdout_every']}th associated frame."
        )
        + "</li>"
        for root, report in reports
    )
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>GlassRecon depth recovery comparison</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1250px;margin:40px auto;
padding:0 24px;color:#203039}
h1{font-size:30px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:9px;text-align:left;border-bottom:1px solid #dce3e7}th{background:#edf3f5}
a{color:#17697d}.scroll{overflow:auto}</style>
<h1>Depth recovery comparison</h1>
<p>Raw sensor depth, global scale/shift fitting, and regional RANSAC recovery use identical
frames, supplied ground-truth poses, and per-run TSDF settings. The periodic test split
is held out of alignment and fusion (see each linked configuration). These experiments isolate depth
and fusion quality; they do not evaluate estimated tracking or loop closure.</p>
<p>TUM errors measure consistency with held-out sensor depth, which is noisy and is not
independent surface ground truth. ICL-NUIM uses official noisy input, independent clean
depth for evaluation, and the author's renderer mesh. The clean-depth oracle is a control,
not a deployable method. No fitted scale or ICP correction is applied to the maps.</p>
<img src="comparison.png" alt="Depth consistency and surface accuracy comparison">
<div class="scroll"><table><thead><tr><th>Sequence</th><th>Backend</th><th>Fusion depth</th>
<th>Fused</th><th>Held out</th>
<th>Depth MAE, cm ↓</th><th>Coverage, % ↑</th><th>Surface accuracy, cm ↓</th>
<th>Completeness distance, cm ↓</th><th>F-score at 5 cm, % ↑</th></tr></thead><tbody>"""
    document += (
        "".join(table_rows)
        + """</tbody></table></div>
<p>Depth errors use the intersection of valid rendered pixels across all non-oracle
methods. Coverage is reported separately over reference-valid pixels. Surface accuracy
and completeness use the same reference geometry across backends; rendered common-pixel
support is computed within each run, not jointly across backend runs. Surface accuracy
uses 100,000 area-uniform reconstructed-mesh samples and exact point-to-reference-triangle
distance. Completeness uses a common observed clean-reference point set (1 cm voxels)
and point-to-reconstruction-triangle distance; unobserved room surfaces are excluded.
These are individual runs; no multi-seed confidence intervals are claimed.</p>
<p><a href="metrics.csv">Download CSV</a> · <a href="comparison.pdf">Download PDF figure</a></p>
<ul>"""
        + sources
        + "</ul><ul>"
        + configurations
        + "</ul>"
    )
    paired_path = output / "model_comparison.json"
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        selected_metrics = {
            "heldout_depth_mae_cm": "Held-out depth MAE (cm) ↓",
            "accuracy_mean_cm": "Mean surface error (cm) ↓",
            "completeness_mean_cm": "Completeness distance (cm) ↓",
            "fscore_5cm_percent": "F-score at 5 cm (%) ↑",
        }
        document += (
            "<h2>Paired model comparison: regional recovery</h2>"
            "<p>Hardware, precision and package versions may differ; consult each run's "
            "model_environment.json before attributing differences solely to model size.</p>"
            '<p><a href="model_comparison.csv">All methods and paired metrics (CSV)</a></p>'
            "<table><tr><th>Sequence</th><th>Metric</th><th>Baseline</th><th>Candidate</th></tr>"
        )
        for row in paired["rows"]:
            if row["variant"] != "regional" or row["metric"] not in selected_metrics:
                continue
            cells = [
                row["sequence"],
                selected_metrics[row["metric"]],
                f"{row['baseline']:.2f} ({row['baseline_model']})",
                f"{row['candidate']:.2f} ({row['candidate_model']})",
            ]
            document += "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in cells) + "</tr>"
        document += "</table><ul>"
        document += "".join(f"<li>{html.escape(note)}</li>" for note in paired["notes"])
        document += "</ul>"
    if "conservative" in displayed:
        document += (
            "<h2>Recovery and fusion ablations</h2><p>Robust scale refinement performs an "
            "edge-filtered IRLS refit and accepts it only when disjoint input-sensor pixels "
            "improve. Filtered sensor only isolates the effect of depth/RGB-aware filtering. "
            "Conservative model completion retains filtered measurements and admits locally "
            "corrected model fills only with spatial sensor support and agreement with the "
            "previous observed sensor frame. Predictions never enter that history buffer.</p>"
            "<p>Common rendered support includes every non-oracle method in the current run; "
            "compare methods within this report and inspect coverage. Fixed thresholds are "
            "heuristics, not calibrated confidence probabilities. Clean depth is evaluation-only. "
            "These sequences have been inspected during development; they are not an untouched "
            "final benchmark. A separate sequence and repeated seeds are needed "
            "for paper claims.</p>"
        )
    if "multiview" in displayed:
        document += (
            "<h2>Sliding-window ablation</h2><p>All methods reuse identical DA3-LARGE predictions. "
            "The sensor-anchor refit controls for the new robust fitting objective without "
            "cross-view terms. Sliding-window recovery adds pose-aware 3D correspondence "
            "constraints, sensor-derived visibility and a fixed committed boundary frame. "
            "Five active frames give four-frame latency; each frame is finalized and fused "
            "once. Neither held-out frames nor clean reference depths enter optimization. "
            "These are affine corrections to per-frame recovered depth, not joint DA3 inference "
            "or nonrigid depth refinement. Common rendered support includes all non-oracle "
            "methods in this run; inspect coverage separately.</p>"
        )
    if (output / "heldout_examples.png").exists():
        document += (
            "<h2>Held-out examples</h2><p>The middle held-out frame of each sequence "
            "is chosen deterministically. Display clipping does not affect metrics.</p>"
            '<img src="heldout_examples.png" alt="Held-out depth error maps">'
        )
    document += "</html>"
    (output / "report.html").write_text(document)
    print(output / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="outputs/comparison_report")
    args = parser.parse_args()
    make_report(args.results, args.output)
