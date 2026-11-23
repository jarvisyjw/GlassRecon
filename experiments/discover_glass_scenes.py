"""Screen local ScanNet++ geometry labels for architectural glazing candidates.

Vertex fractions are a retrieval proxy, not glass area or RGB pixel coverage.
Transparent surfaces can be absent from laser geometry. Inspect RGB before selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np

GLAZING = {
    "window",
    "windows",
    "glass wall",
    "window pane",
    "glass pane",
    "shower glass",
    "glass",
    "door window",
    "french window",
    "glass door",
    "sliding glass door",
    "sliding window",
    "window door",
}


def scan(root, output, limit=200, exclude_screening=None):
    previous = json.loads(exclude_screening.read_text()) if exclude_screening else {}
    skipped = {r["scene"] for r in previous.get("ranked_scenes", [])}
    skipped.update(previous.get("excluded_missing_inputs_or_nvs_test", []))
    classes = (root / "metadata/semantic_classes.txt").read_text().splitlines()
    wanted = {i: name for i, name in enumerate(classes) if name in GLAZING}
    types = json.loads((root / "metadata/scene_types.json").read_text())
    splits = {p.stem: set(p.read_text().split()) for p in (root / "splits").glob("*.txt")}
    rows, excluded = [], []
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "<i4"),
        ]
    )
    for i, folder in enumerate(sorted((root / "data").iterdir())):
        if limit and len(rows) >= limit:
            break
        if not folder.is_dir():
            continue
        scene = folder.name
        if scene in skipped:
            continue
        required = [
            folder / "scans/mesh_aligned_0.05_semantic.ply",
            folder / "scans/mesh_aligned_0.05.ply",
            folder / "iphone/colmap/images.txt",
            folder / "iphone/rgb",
            folder / "iphone/depth",
            folder / "iphone/rgb_masks",
        ]
        if any(not p.exists() for p in required) or any(
            scene in v for k, v in splits.items() if k.startswith("nvs_test")
        ):
            excluded.append(scene)
            continue
        with required[0].open("rb") as stream:
            header = []
            while True:
                line = stream.readline().decode("ascii").strip()
                header.append(line)
                if line == "end_header":
                    break
                if not line:
                    raise ValueError(f"Incomplete header: {required[0]}")
            offset = stream.tell()
        count = int(next(x for x in header if x.startswith("element vertex ")).split()[-1])
        vertex_start = header.index(f"element vertex {count}") + 1
        properties = header[vertex_start : vertex_start + 7]
        expected = [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int label",
        ]
        if "format binary_little_endian 1.0" not in header or properties != expected:
            raise ValueError(f"Unexpected vertex format: {required[0]}")
        vertices = np.memmap(required[0], mode="r", dtype=dtype, offset=offset, shape=(count,))
        ids, counts = np.unique(vertices["label"], return_counts=True)
        labels = {wanted[int(k)]: int(n) for k, n in zip(ids, counts) if k in wanted}
        total = sum(labels.values())
        with required[2].open() as stream:
            frame_count = sum(
                1
                for line in stream
                if len(parts := line.split(maxsplit=9)) == 10 and parts[-1].strip().endswith(".jpg")
            )
        rows.append(
            {
                "scene": scene,
                "scene_type": types.get(scene),
                "split": [k for k, v in splits.items() if scene in v],
                "vertices": count,
                "glazing_vertices": total,
                "glazing_vertex_fraction": total / count,
                "glazing_labels": labels,
                "colmap_frames": frame_count,
            }
        )
        del vertices
        if i % 50 == 0:
            print("Screened", i + 1, flush=True)
    rows.sort(key=lambda x: x["glazing_vertex_fraction"], reverse=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "root": str(root),
                "proxy": (
                    "fraction of semantic mesh vertices labeled architectural glazing; "
                    "not RGB glass coverage"
                ),
                "screening_limit": limit,
                "excluded_previous_screening": str(exclude_screening)
                if exclude_screening
                else None,
                "selection": "first eligible scenes by sorted scene ID; not exhaustive",
                "available_scene_directories": len(list((root / "data").iterdir())),
                "target_labels": sorted(GLAZING),
                "excluded_missing_inputs_or_nvs_test": excluded,
                "ranked_scenes": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print("Eligible scenes", len(rows), "excluded", len(excluded), flush=True)
    for row in rows[:30]:
        print(row, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--root", type=Path, default=Path("/media/jingwen-orchtech/JWDATA2/scannetppv2/data")
    )
    cli.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/scannetpp_v2/glass_scene_screening.json"),
    )
    cli.add_argument("--limit", type=int, default=200, help="Maximum eligible scenes; 0 scans all")
    cli.add_argument("--exclude-screening", type=Path)
    args = cli.parse_args()
    if args.limit < 0:
        cli.error("Limit must be nonnegative")
    scan(args.root, args.output, args.limit, args.exclude_screening)
