"""Switch paired reconstructions in Viser while preserving the viewing camera."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from glassrecon.data import quaternion_matrix
from glassrecon.viewer import load_reconstruction

LABELS = {
    "sensor": "Raw sensor",
    "global": "Global scale/shift",
    "regional": "Regional RANSAC",
    "clean_oracle": "Clean-depth oracle",
    "refined": "Robust scale refinement",
    "filtered": "Filtered sensor only",
    "conservative": "Conservative model completion",
    "window_anchor": "Sensor-anchor refit only",
    "multiview": "Sliding-window multi-view",
}


def serve(paths, host="127.0.0.1", port=8082):
    import viser

    roots = {Path(path).name: Path(path) for path in paths}
    reports = {
        name: json.loads((root / "comparison.json").read_text()) for name, root in roots.items()
    }
    server = viser.ViserServer(host=host, port=port, label="GlassRecon · controlled comparison")
    sequence = server.gui.add_dropdown("Sequence", options=tuple(roots))
    method = server.gui.add_dropdown(
        "Fusion depth", options=tuple(LABELS.values())[:3], initial_value=LABELS["sensor"]
    )
    metrics = server.gui.add_markdown("")
    home = server.gui.add_button("Frame scene")
    first = server.gui.add_button("First camera")
    show_reference = server.gui.add_checkbox("Observed reference points", initial_value=False)
    state = {"handles": {}, "data": {}, "reference": None, "first": None}

    def camera_overview(camera):
        data = state["data"]["sensor"]
        rotation = quaternion_matrix(data.poses[0, 4:])
        up, forward, right = -rotation[:, 1], rotation[:, 2], rotation[:, 0]
        camera.position = tuple(
            data.center + data.radius * (-2.3 * forward + 0.3 * up + 0.2 * right)
        )
        camera.look_at = tuple(data.center)
        camera.up_direction = tuple(up)
        camera.fov = float(np.deg2rad(60))

    def change_method(_=None):
        name = next(k for k, v in LABELS.items() if v == method.value)
        for variant, handle in state["handles"].items():
            handle.visible = variant == name
        report = reports[sequence.value]
        value = report["results"][name]
        protocol = report["protocol"]
        depth = value["heldout_render_common_support"]
        coverage = value["heldout_render"]["coverage"]
        content = (
            f"**{LABELS[name]}**\n\n"
            f"Depth predictor: `{report['model_id']}`\n\n"
            f"Fusion backend: `{protocol.get('backend', 'open3d')}`\n\n"
            f"Identical supplied poses · {protocol['fused_frames']} fused frames · "
            f"{protocol['heldout_frames']} held out\n\n"
            f"Held-out depth MAE: **{depth['mae_m'] * 100:.2f} cm** "
            f"on common pixels\n\nRender coverage: **{coverage * 100:.2f}%**\n\n"
        )
        if "surface" in value:
            surface = value["surface"]
            content += (
                f"Surface accuracy: **{surface['accuracy_mean_m'] * 100:.2f} cm**\n\n"
                f"Completeness distance: **{surface['completeness_mean_m'] * 100:.2f} cm**\n\n"
                f"F-score at 5 cm: **{surface['at_0.05m']['fscore'] * 100:.1f}%**\n\n"
            )
        else:
            content += "TUM depth consistency; no independent reference mesh.\n\n"
        metrics.content = content + "Switch fusion depth to compare from the same camera."

    def change_sequence(_=None):
        for handle in state["handles"].values():
            handle.remove()
        if state["reference"] is not None:
            state["reference"].remove()
        state["handles"], state["data"], state["reference"] = {}, {}, None
        root = roots[sequence.value]
        report = reports[sequence.value]
        for name in report["results"]:
            data = load_reconstruction(root / name)
            state["data"][name] = data
            state["handles"][name] = server.scene.add_mesh_trimesh(
                f"/comparison/{name}", data.mesh, visible=False
            )
        reference_path = root / "visible_reference.ply"
        if reference_path.exists():
            import trimesh

            cloud = trimesh.load(reference_path, process=False)
            points = np.asarray(cloud.vertices, dtype=np.float32)[::3]
            state["reference"] = server.scene.add_point_cloud(
                "/comparison/reference",
                points=points,
                colors=(60, 170, 240),
                point_size=0.008,
                visible=show_reference.value,
            )
        show_reference.disabled = state["reference"] is None
        method.options = tuple(LABELS[name] for name in report["results"])
        method.value = LABELS["sensor"]
        camera_overview(server.initial_camera)
        rotation = quaternion_matrix(state["data"]["sensor"].poses[0, 4:])
        server.scene.set_up_direction(tuple(-rotation[:, 1]))
        for client in server.get_clients().values():
            camera_overview(client.camera)
        change_method()

    @home.on_click
    def frame_scene(event):
        if event.client:
            camera_overview(event.client.camera)

    @first.on_click
    def first_camera(event):
        if event.client:
            data = state["data"]["sensor"]
            pose = data.poses[0]
            rotation = quaternion_matrix(pose[4:])
            event.client.camera.position = tuple(pose[1:4])
            event.client.camera.look_at = tuple(pose[1:4] + rotation[:, 2])
            event.client.camera.up_direction = tuple(-rotation[:, 1])
            event.client.camera.fov = float(data.camera_fov)

    @show_reference.on_update
    def toggle_reference(_):
        if state["reference"] is not None:
            state["reference"].visible = show_reference.value

    change_sequence()
    sequence.on_update(change_sequence)
    method.on_update(change_method)
    print(f"Comparison viewer: http://{host}:{server.get_port()}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    serve(args.results, args.host, args.port)
