"""Plot deterministic middle held-out views, with one error scale for all methods."""

import argparse
import json
from pathlib import Path

import numpy as np

from glassrecon.data import CameraIntrinsics, PoseTrajectory, load_sequence
from glassrecon.evaluation import MeshSurface


def render_examples(paths, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    fig, axes = plt.subplots(
        len(paths), 4, figsize=(14, 3 * len(paths)), squeeze=False, layout="constrained"
    )
    details = []
    color_map = plt.get_cmap("magma").copy()
    color_map.set_bad("#d1d5db")
    for row, path in enumerate(paths):
        root = Path(path)
        report = json.loads((root / "comparison.json").read_text())
        config = report["protocol"]
        selection = json.loads((root / "selection.json").read_text())
        key = selection["heldout"][len(selection["heldout"]) // 2]
        frames = {frame.key: frame for frame in load_sequence(config["dataset"])[0]}
        frame = frames[key]
        camera = CameraIntrinsics(**config["intrinsics_used"])
        pose = PoseTrajectory(config["poses"]).at(frame.timestamp, config["pose_max_difference"])
        rgb, reference = frame.load(config["depth_units_per_meter"], camera)
        if config["reference_depth"]:
            reference_frames = {f.key: f for f in load_sequence(config["reference_depth"])[0]}
            _, reference = reference_frames[key].load(config["depth_units_per_meter"], camera)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(
            root.name.removeprefix("comparison_") + f" · frame {key}", fontsize=9
        )
        for column, (name, label) in enumerate(
            [("sensor", "Raw sensor"), ("global", "Global fit"), ("regional", "Regional RANSAC")], 1
        ):
            surface = MeshSurface(o3d.io.read_triangle_mesh(str(root / name / "mesh.ply")))
            depth = surface.render(camera, pose)
            valid = (reference >= config["min_depth"]) & (reference <= config["max_depth"])
            valid &= (depth >= config["min_depth"]) & (depth <= config["max_depth"])
            error_cm = np.where(valid, np.abs(depth - reference) * 100, np.nan)
            heatmap = axes[row, column].imshow(error_cm, cmap=color_map, vmin=0, vmax=20)
            axes[row, column].set_title(label, fontsize=10)
        for axis in axes[row]:
            axis.axis("off")
        details.append(
            {
                "sequence": root.name,
                "frame": key,
                "timestamp": frame.timestamp,
                "selection_rule": "middle element of held-out frame list",
                "reference": report["heldout_reference"],
            }
        )
    fig.colorbar(
        heatmap,
        ax=axes[:, 1:].ravel().tolist(),
        shrink=0.5,
        aspect=35,
        label="Absolute depth error (cm); saturated at 20 cm for display",
    )
    fig.suptitle(
        "Held-out reconstructed depth: TUM sensor consistency / ICL clean-depth accuracy\n"
        "Gray: no valid reference or rendered depth",
        fontsize=13,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "heldout_examples.png", dpi=180)
    fig.savefig(output / "heldout_examples.pdf")
    plt.close(fig)
    (output / "example_frames.json").write_text(json.dumps(details, indent=2) + "\n")
    print(output / "heldout_examples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="outputs/comparison_report")
    args = parser.parse_args()
    render_examples(args.results, args.output)
