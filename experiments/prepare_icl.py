"""Convert ICL-NUIM kt2 global poses and renderer geometry to OpenCV coordinates.

Images stay upright and unmodified. Native fy=-480 is represented by reflecting
both camera and world y: R_cv=S R S, t_cv=S t, with S=diag(1,-1,1).
The author's POV-Ray geometry already includes the OBJ-to-renderer x reflection
and triangulation; using it avoids dropping quads or incorrectly filling ngons.
"""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from glassrecon.data import matrix_quaternion


def renderer_mesh(text):
    """Read explicit vertex/triangle lists from the published static mesh2 objects."""
    vertices, faces, offset = [], [], 0
    for block in text.split("=mesh2{")[1:]:
        vertex_text = re.search(r"vertex_vectors\s*\{([^}]+)", block)[1]
        face_text = re.search(r"face_indices\s*\{([^}]+)", block)[1]
        v = np.array(
            [[float(x) for x in row.split(",")] for row in re.findall(r"<([^>]+)>", vertex_text)]
        )
        f = np.array(
            [[int(x) for x in row.split(",")] for row in re.findall(r"<([^>]+)>", face_text)]
        )
        if len(v) != int(vertex_text.split(",")[0]) or len(f) != int(face_text.split(",")[0]):
            raise ValueError("Renderer geometry count mismatch")
        if v.shape[1] != 3 or f.shape[1] != 3 or f.min() < 0 or f.max() >= len(v):
            raise ValueError("Unsupported renderer geometry")
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    if not vertices:
        raise ValueError("No renderer meshes found")
    return np.concatenate(vertices), np.concatenate(faces)


def convert_pose(native):
    s = np.diag([1.0, -1.0, 1.0])
    pose = np.eye(4)
    # Source rotations are rounded to six decimals; project to SO(3) before
    # quaternion export, without fitting to any reconstruction or depth data.
    u, _, vt = np.linalg.svd(native[:, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) <= 0:
        raise ValueError("Native pose is not a proper rotation")
    pose[:3, :3] = s @ rotation @ s
    pose[:3, 3] = s @ native[:, 3]
    return pose


def prepare(sequences=("data/icl_kt2_clean", "data/icl_kt2_noisy")):
    import open3d as o3d

    root = Path("data/icl_reference")
    root.mkdir(parents=True, exist_ok=True)
    sources = {
        "livingRoom2n.gt.sim": "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom2n.gt.sim",
        "living_room_POV_geom.inc": "https://raw.githubusercontent.com/ankurhanda/"
        "living_room_iclnuim/master/living_room_POV_geom.inc",
    }
    receipts = {}
    for name, url in sources.items():
        path = root / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        receipts[name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    native = np.loadtxt(root / "livingRoom2n.gt.sim").reshape(-1, 3, 4)
    vertices, faces = renderer_mesh((root / "living_room_POV_geom.inc").read_text())
    vertices[:, 1] *= -1
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces[:, ::-1].copy())
    )
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(root / "living_room_opencv.ply"), mesh)
    camera = {"width": 640, "height": 480, "fx": 481.2, "fy": 480.0, "cx": 319.5, "cy": 239.5}
    (root / "intrinsics.json").write_text(json.dumps(camera, indent=2) + "\n")
    for sequence in map(Path, sequences):
        rgb_ids = {int(p.stem) for p in (sequence / "rgb").glob("*.png")}
        depth_ids = {int(p.stem) for p in (sequence / "depth").glob("*.png")}
        if not rgb_ids or rgb_ids != depth_ids:
            raise ValueError(f"Incomplete RGB-D sequence: {sequence}")
        posed_ids = set(range(1, len(native) + 1))
        if rgb_ids - {0} != posed_ids:
            raise ValueError("Expected one-based frame IDs matching published global poses")
        rows = []
        for index in sorted(posed_ids):
            pose = convert_pose(native[index - 1])
            rows.append([index / 30, *pose[:3, 3], *matrix_quaternion(pose[:3, :3])])
        np.savetxt(
            sequence / "groundtruth_opencv.txt",
            rows,
            fmt="%.10f",
            header="seconds tx ty tz qx qy qz qw; reflected native world y",
        )
        # Retain the archive's original index files for provenance.
        for kind in ["rgb", "depth"]:
            index_file = sequence / f"{kind}.txt"
            original = sequence / f"{kind}_original.txt"
            if index_file.exists() and not original.exists():
                original.write_bytes(index_file.read_bytes())
            index_file.write_text(
                "".join(f"{i / 30:.10f} {kind}/{i}.png\n" for i in sorted(rgb_ids))
            )
    receipts["conversion"] = {
        "camera_and_world_reflection": [1, -1, 1],
        "pose_rotation": "S R S",
        "pose_translation": "S t",
        "frame_to_pose": "one-based image id minus one",
        "unposed_frames": "Frame 0 has no published pose; retained but excluded from fusion",
        "frames_per_second": 30,
        "geometry": "author's explicit POV-Ray triangles (OBJ x reflection already applied)",
        "vertices": len(vertices),
        "triangles": len(faces),
    }
    (root / "provenance.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(f"Prepared {len(native)} frames per ICL kt2 sequence, {len(faces)} reference triangles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences", nargs="+", default=["data/icl_kt2_clean", "data/icl_kt2_noisy"]
    )
    prepare(parser.parse_args().sequences)
