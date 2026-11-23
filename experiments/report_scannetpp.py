"""Audit and visualize the ScanNet++ laser-reference comparison."""

import argparse
import csv
import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.run_comparison import save_json
from experiments.run_scannetpp import SCENES
from glassrecon.data import CameraIntrinsics, load_sequence
from glassrecon.refinement import metric_transform

LABELS = {
    "sensor": "Raw sensor",
    "global": "Global fit",
    "regional": "Per-frame regional",
    "window_anchor": "Sensor-anchor refit",
    "multiview": "Sliding window",
}


SCENE_NAMES = {
    "0a76e06478": "Bedroom",
    "fd361ab85f": "Office",
    "c0f5742640": "Apartment",
    "08bd80ce2a": "Lounge A",
    "1b379f1114": "Lounge B",
    "33f78338cd": "Apartment B",
    "7f68c514bd": "Lounge C",
    "511061232": "Winter garden",
    "791a5c253d": "Conference",
    "45ba9ea301": "Kitchen",
    "8a20d62ac0": "Office B",
}


def report(root, output, compact, scenes=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = list(SCENES if scenes is None else scenes)
    output.mkdir(parents=True, exist_ok=True)
    compact.mkdir(parents=True, exist_ok=True)
    rows, audits, pictures = [], {}, []
    for scene in scenes:
        base, multi = root / scene / "baseline", root / scene / "multiview"
        result = json.loads((multi / "comparison.json").read_text())
        config = result["protocol"]
        selection = json.loads((multi / "selection.json").read_text())
        assert selection == json.loads((base / "selection.json").read_text())
        logs = [json.loads(line) for line in (base / "frames.jsonl").read_text().splitlines()]
        window = Path(config["multiview_dir"])
        window_logs = [
            json.loads(line) for line in (window / "frames.jsonl").read_text().splitlines()
        ]
        assert [x["frame"] for x in window_logs] == selection["fused"]
        assert set(selection["fused"]).isdisjoint(selection["heldout"])
        assert {p.stem for p in (base / "predictions").glob("*.npy")} == set(selection["fused"])
        failures = {
            row["frame"]: row["alignment"]["global"]
            for row in logs
            if "failure" in row["alignment"]["global"]
        }
        save_json(compact / f"{scene}_global_failures.json", failures)
        for method in result["results"]:
            trajectory = np.loadtxt(multi / method / "trajectory.txt")
            np.testing.assert_array_equal(trajectory, np.loadtxt(multi / "sensor/trajectory.txt"))
        np.testing.assert_array_equal(
            np.loadtxt(base / "sensor/trajectory.txt"), np.loadtxt(multi / "sensor/trajectory.txt")
        )
        preparation = json.loads((Path(config["dataset"]) / "preparation.json").read_text())
        assert preparation["prepared_frames"] == preparation["available_colmap_frames"]
        assert len(selection["fused"]) + len(selection["heldout"]) == preparation["prepared_frames"]
        audits[scene] = {
            "global_fit_failures": len(failures),
            "identical_selection_poses": True,
            "heldout_not_predicted": True,
            "each_window_frame_emitted_once_in_order": True,
            "all_released_colmap_frames": preparation["prepared_frames"],
            "fused": len(selection["fused"]),
            "heldout": len(selection["heldout"]),
            "sensor_laser_median_of_frame_medians_m": float(
                np.median(
                    [
                        x["median_m"]
                        for x in preparation["sensor_laser_checks"]
                        if x["median_m"] is not None
                    ]
                )
            ),
        }
        for filename in [
            "comparison.json",
            "selection.json",
            "reference_coordinate_check.json",
            "model_environment.json",
        ]:
            shutil.copy2(multi / filename, compact / f"{scene}_{filename}")
        save_json(compact / f"{scene}_preparation.json", preparation)
        for method, value in result["results"].items():
            surface = value["surface"]
            integrated = json.loads((multi / method / "summary.json").read_text())["integrated"]
            assert integrated == len(selection["fused"]) - (
                len(failures) if method == "global" else 0
            )
            rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "integrated": integrated,
                    "fused": len(selection["fused"]),
                    "heldout": len(selection["heldout"]),
                    "global_fit_failures": len(failures),
                    "depth_mae_cm": value["heldout_render_common_support"]["mae_m"] * 100,
                    "coverage_percent": value["heldout_render"]["coverage"] * 100,
                    "accuracy_cm": surface["accuracy_mean_m"] * 100,
                    "accuracy_median_cm": surface["accuracy_median_m"] * 100,
                    "accuracy_p95_cm": surface["accuracy_p95_m"] * 100,
                    "surface_over_50cm_percent": float(
                        (np.load(multi / method / "surface_errors.npz")["accuracy"] > 0.5).mean()
                        * 100
                    ),
                    "completeness_cm": surface["completeness_mean_m"] * 100,
                    "fscore_5cm_percent": surface["at_0.05m"]["fscore"] * 100,
                }
            )
        camera = CameraIntrinsics(**config["intrinsics_used"])
        frames = {f.key: f for f in load_sequence(config["dataset"])[0]}
        reference = {f.key: f for f in load_sequence(config["reference_depth"])[0]}
        indices = [
            len(logs) // 2,
            int(np.argmax([r["alignment"]["regional"]["mean_absolute_error"] for r in logs])),
        ]
        for label, i in zip(["middle", "worst_sensor_fit"], indices):
            row = logs[i]
            frame = frames[row["frame"]]
            rgb, sensor = frame.load(1000, camera)
            _, laser = reference[frame.key].load(1000, camera)
            prediction = np.load(base / "predictions" / (frame.key + ".npy"))
            fit = row["alignment"]["regional"]
            regional = metric_transform(prediction, fit["scale"], fit["shift"])
            window_depth = np.load(window / "multiview" / (frame.key + ".npy"))
            mask = (
                np.array(Image.open(Path(config["valid_mask_dir"]) / (frame.key + ".png"))) == 255
            )
            regional[~mask] = 0
            window_depth[~mask] = 0
            maximum = float(np.percentile(laser[laser > 0], 99))
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title("RGB")
            for ax, depth, title in zip(
                axes[0, 1:],
                [sensor, regional, window_depth],
                ["Raw sensor", "Per-frame regional", "Sliding window"],
            ):
                im = ax.imshow(
                    np.ma.masked_less_equal(depth, 0), vmin=0, vmax=maximum, cmap="viridis"
                )
                ax.set_title(title + " (m)")
            fig.colorbar(im, ax=list(axes[0, 1:]), shrink=0.7, label="Depth (m)")
            axes[1, 0].imshow(
                np.ma.masked_less_equal(laser, 0), vmin=0, vmax=maximum, cmap="viridis"
            )
            axes[1, 0].set_title("Laser reference (evaluation only)")
            common = (laser > 0) & (sensor > 0) & (regional > 0) & (window_depth > 0) & mask
            for ax, depth, title in zip(
                axes[1, 1:], [sensor, regional, window_depth], ["Sensor", "Regional", "Window"]
            ):
                error = np.ma.array(np.abs(depth - laser), mask=~common)
                im = ax.imshow(error, vmin=0, vmax=0.3, cmap="magma")
                ax.set_title(f"{title} error; MAE {error.mean() * 100:.1f} cm")
            fig.colorbar(
                im, ax=list(axes[1, 1:]), shrink=0.7, label="Absolute error (m), clipped at 0.3"
            )
            for ax in axes.flat:
                ax.axis("off")
            fig.suptitle(f"{scene} · {frame.key} · {label}; same valid pixels for error maps")
            filename = f"{scene}_{label}.png"
            fig.savefig(output / filename, dpi=120)
            plt.close(fig)
            pictures.append(filename)
    save_json(compact / "audit.json", audits)
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(output / "metrics.csv", compact / "metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(max(16, len(scenes) * 3), 5), layout="constrained")
    for ax, field, title in zip(
        axes,
        ["accuracy_cm", "completeness_cm", "fscore_5cm_percent"],
        ["Surface accuracy ↓ (cm)", "Visible-surface completeness ↓ (cm)", "F-score at 5 cm ↑ (%)"],
    ):
        for j, (method, label) in enumerate(LABELS.items()):
            values = [
                next(r[field] for r in rows if r["scene"] == s and r["method"] == method)
                for s in scenes
            ]
            ax.bar(np.arange(len(scenes)) + (j - 2) * 0.15, values, 0.15, label=label)
        ax.set_xticks(
            range(len(scenes)),
            [SCENE_NAMES.get(s, s) for s in scenes],
        )
        ax.tick_params(axis="x", labelrotation=25 if len(scenes) > 3 else 0, labelsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.legend(
        *axes[-1].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=5,
        fontsize=9,
    )
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(compact / "comparison.pdf")
    plt.close(fig)
    body = [
        "<h1>ScanNet++ v2: laser-reference reconstruction comparison</h1>",
        (
            "<p>DA3-LARGE, seed 0, 512×384 rectified RGB-D, nvblox 2 cm voxels, 8 c"
            "m truncation, five-frame window. All released COLMAP frames; every fif"
            "th held out. Supplied mesh-aligned poses, not estimated tracking.</p>"
        ),
        "<p>Scenes: "
        + "; ".join(f"{html.escape(SCENE_NAMES.get(scene, scene))} ({scene})" for scene in scenes)
        + ". All are in nvs_sem_train. These are development comparisons using "
        "the released COLMAP subset, not an untouched benchmark. Follow-up "
        "scenes were selected by RGB inspection before their reconstruction outcomes.</p>",
        (
            "<p>Laser depth is evaluation-only except the separately labeled oracle"
            ". Surface accuracy samples the full estimated mesh; completeness uses "
            "visible laser points sampled from fusion views. No ICP or scale correc"
            "tion. Coverage uses each method’s rendered support; depth MAE uses sha"
            "red rendered support. Privacy masks are excluded.</p>"
        ),
        "<p>Full-mesh distance also penalizes geometry outside laser coverage, including "
        "possible views through windows. Large mean distances alone do not prove that "
        "all distant surfaces are false. Read median/tail error, completeness, coverage, "
        "and held-out laser depth error together; no outlier surfaces are cropped.</p>",
        "<p>Invalid nonpositive global fits contribute empty depth only to the global "
        "baseline and are recorded in the audit; the frame stays in every other method. "
        "No identity scale or sensor-depth fallback is substituted.</p>",
        '<p><a href="metrics.csv">Download metrics CSV</a></p><img src="comparison.png">',
        "<table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in rows[0]) + "</tr>",
    ]
    if (output / "glass_candidates/index.html").exists():
        body.insert(
            1, '<p><a href="glass_candidates/">Visually inspected glass-scene candidates</a></p>'
        )
    for row in rows:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{v:.3f}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>"
                for v in row.values()
            )
            + "</tr>"
        )
    body.append(
        (
            "</table><h2>Depth diagnostics</h2><p>Frames selected by temporal midpo"
            "int and worst sensor-fit residual, without laser-error ranking. Depth "
            "colors share a scale within each figure; error maps share common valid"
            " support.</p>"
        )
    )
    body.extend(f'<img src="{p}">' for p in pictures)
    (output / "index.html").write_text(
        (
            '<!doctype html><meta charset="utf-8"><title>ScanNet++ comparison</titl'
            "e><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;pa"
            "dding:20px;color:#182432}img{width:100%}table{border-collapse:collapse"
            ";font-size:13px}td,th{padding:8px;border:1px solid #ddd}th{background:"
            "#eef2f6}</style>"
        )
        + "\n".join(body)
    )
    save_json(output / "audit.json", audits)
    print(json.dumps(audits, indent=2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=Path("outputs/scannetpp"))
    cli.add_argument("--output", type=Path, default=Path("outputs/scannetpp_report"))
    cli.add_argument("--compact", type=Path, default=Path("experiments/results/scannetpp_v2"))
    cli.add_argument("--scenes", nargs="+", default=SCENES)
    args = cli.parse_args()
    report(args.root, args.output, args.compact, args.scenes)


if __name__ == "__main__":
    main()
