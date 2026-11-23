"""Render fixed, uniformly spaced RGB views for glass-candidate visual review."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_gallery(screening, output, top=22, include_earlier=True):
    manifest = json.loads(screening.read_text())
    root = Path(manifest["root"])
    rows = manifest["ranked_scenes"]
    tested = {"c0f5742640", "fd361ab85f", "0a76e06478"}
    candidates = [r for r in rows if r["scene"] not in tested and r["glazing_vertices"] > 0][:top]
    types = json.loads((root / "metadata/scene_types.json").read_text())
    for scene in ["0a7cc12c0e", "8a35ef3cfe", "4c5c60fa76"] if include_earlier else []:
        if scene not in {r["scene"] for r in candidates}:
            candidates.append(
                {
                    "scene": scene,
                    "scene_type": types.get(scene),
                    "reason": "earlier expansion candidate; independent of semantic ranking",
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    pages = []
    for start in range(0, len(candidates), 5):
        group = candidates[start : start + 5]
        sheet = Image.new("RGB", (1344, len(group) * 216), "#f0f3f5")
        draw = ImageDraw.Draw(sheet)
        for row, record in enumerate(group):
            phone = root / "data" / record["scene"] / "iphone"
            frames = []
            with (phone / "colmap/images.txt").open() as stream:
                for line in stream:
                    parts = line.split(maxsplit=9)
                    if len(parts) == 10:
                        name = Path(parts[-1].strip()).name
                        if name.startswith("frame_") and name.endswith(".jpg"):
                            frames.append((name,))
            frames = sorted(set(frames))
            indices = np.linspace(0, len(frames) - 1, 6, dtype=int)
            record["review_frames"] = [frames[i][0] for i in indices]
            caption = f"{record['scene']} | {record['scene_type']}"
            if "glazing_vertex_fraction" in record:
                caption += (
                    f" | glazing mesh-vertex proxy {record['glazing_vertex_fraction'] * 100:.1f}%"
                    " (NOT image area)"
                )
            draw.text((6, row * 216 + 3), caption, font=font, fill="#101820")
            for col, index in enumerate(indices):
                name = frames[index][0]
                image = Image.open(phone / "rgb" / name).convert("RGB")
                mask = Image.open(phone / "rgb_masks" / (Path(name).stem + ".png")).convert("L")
                image = Image.composite(image, Image.new("RGB", image.size), mask)
                image = image.resize((224, 168), Image.Resampling.LANCZOS)
                sheet.paste(image, (col * 224, row * 216 + 26))
                draw.text(
                    (col * 224 + 3, row * 216 + 196), Path(name).stem, font=font, fill="#101820"
                )
        filename = f"contact_{start // 5 + 1:02d}.jpg"
        sheet.save(output / filename, quality=92)
        pages.append(filename)
    (output / "review_manifest.json").write_text(json.dumps(candidates, indent=2) + "\n")
    body = "".join(
        f'<h2>Contact sheet {i + 1}</h2><a href="{p}"><img src="{p}"></a>'
        for i, p in enumerate(pages)
    )
    if (output / "batch2/index.html").exists():
        body = (
            '<p><a href="batch2/">New search: 300 additional scenes '
            "and five recommendations</a></p>" + body
        )
    shortlist_path = output / "shortlist.json"
    if shortlist_path.exists():
        shortlist = json.loads(shortlist_path.read_text())
        selection = (
            "<h2>Recommended scenes</h2><p>Candidates selected before reconstruction. "
            'Glass pixel percentages were not measured.</p><img src="shortlist.jpg"><ul>'
        )
        for record in shortlist["recommended"]:
            scene = record["scene"]
            selection += (
                f'<li><a href="focus_{scene}.jpg">{scene}</a> '
                f"({record['priority']}): {html.escape(record['visual_observation'])} "
                f"<small>{html.escape(record['adapter_note'])}</small></li>"
            )
        status_path = output / "run_status.json"
        status_html = ""
        if status_path.exists():
            status = json.loads(status_path.read_text())
            report_link = "../../" if output.name == "batch2" else "../"
            status_html = (
                f'<h2>Reconstruction runs</h2><p><a href="{report_link}">'
                "Full comparison report</a></p><ul>"
            )
            for scene, result in status.items():
                status_html += f"<li>{scene}: {html.escape(result['status'])}</li>"
            status_html += (
                "</ul><p>Preparation now handles both iphone/ and video/ filename prefixes. "
                "Notes in the original selection manifest describe discovery-time checks.</p>"
            )
        body = (
            selection
            + '</ul><p><a href="shortlist.json">Selection manifest</a></p>'
            + status_html
            + body
        )
    (output / "index.html").write_text(
        (
            '<!doctype html><meta charset="utf-8"><title>Glass scene candidates</ti'
            "tle><style>body{font:16px system-ui;max-width:1400px;margin:24px auto}"
            "img{width:100%}</style><h1>Glass scene candidates</h1><p>"
            f"{len(rows)} eligible scenes screened by semantic mesh vertex fraction. "
            "Six uniformly spaced released COLMAP RGB frames p"
            "er scene; anonymization masks applied. Mesh labels are a retrieval pro"
            "xy, not glass pixel coverage. No reconstruction results used for selec"
            "tion.</p>"
        )
        + body
    )
    print(json.dumps({"scenes": len(candidates), "pages": pages}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--screening",
        type=Path,
        default=Path("experiments/results/scannetpp_v2/glass_scene_screening.json"),
    )
    cli.add_argument(
        "--output", type=Path, default=Path("outputs/scannetpp_report/glass_candidates")
    )
    cli.add_argument("--top", type=int, default=22)
    cli.add_argument("--no-earlier", action="store_true")
    args = cli.parse_args()
    make_gallery(args.screening, args.output, args.top, not args.no_earlier)
