"""Interactive Viser visualization of exported reconstruction results."""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import CameraIntrinsics, PoseTrajectory, matrix_quaternion, quaternion_matrix


@dataclass
class Reconstruction:
    mesh: object | None
    points: np.ndarray
    colors: np.ndarray
    total_points: int
    poses: np.ndarray  # TUM rows: timestamp, translation, quaternion xyzw.
    camera_fov: float
    camera_aspect: float
    center: np.ndarray
    radius: float
    summary: dict = field(default_factory=dict)


def load_reconstruction(result, max_points=500_000):
    """Load PLY artifacts without requiring Open3D or model dependencies."""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("Install viewer dependencies: pip install -e '.[viewer]'") from exc
    if max_points < 1:
        raise ValueError("max_points must be positive")
    result = Path(result)
    mesh = None
    mesh_path, cloud_path = result / "mesh.ply", result / "pointcloud.ply"
    if mesh_path.exists():
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f"No triangle mesh in {mesh_path}")
        if not np.isfinite(mesh.vertices).all():
            raise ValueError(f"Mesh has nonfinite vertices: {mesh_path}")
    points, colors = np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if cloud_path.exists():
        cloud = trimesh.load(cloud_path, process=False)
        if not isinstance(cloud, trimesh.points.PointCloud):
            raise ValueError(f"Expected a point cloud in {cloud_path}")
        points = np.asarray(cloud.vertices, dtype=np.float32)
        colors = np.asarray(cloud.colors, dtype=np.uint8)
        if colors.shape != (len(points), 4) and colors.shape != (len(points), 3):
            colors = np.full((len(points), 3), 180, dtype=np.uint8)
        colors = colors[:, :3]
        finite = np.isfinite(points).all(axis=1)
        points, colors = points[finite], colors[finite]
    total_points = len(points)
    if mesh is None and not total_points:
        raise ValueError(f"No nonempty mesh.ply or pointcloud.ply found in {result}")
    # Bounds use all geometry before the display-only point budget is applied.
    vertices = mesh.vertices if mesh is not None else points
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.linalg.norm(upper - lower)) / 2, 0.1)
    if total_points > max_points:
        indices = np.random.default_rng(0).choice(total_points, max_points, replace=False)
        points, colors = points[indices], colors[indices]

    trajectory = result / "trajectory.txt"
    poses = PoseTrajectory(trajectory).rows if trajectory.exists() else np.empty((0, 8))
    fov, aspect = np.deg2rad(60), 4 / 3
    config_path = result / "config.json"
    if config_path.exists():
        calibration = json.loads(config_path.read_text()).get("intrinsics_used")
        if calibration is not None:
            camera = CameraIntrinsics(**calibration)
            fov = 2 * np.arctan(camera.height / (2 * camera.fy))
            aspect = camera.width / camera.height
    summary_path = result / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return Reconstruction(
        mesh, points, colors, total_points, poses, fov, aspect, center, radius, summary
    )


def camera_pose(row):
    """TUM c2w xyzw → Viser c2w wxyz; both use OpenCV camera axes."""
    return row[[7, 4, 5, 6]], row[1:4]


def create_viewer(
    result, host="127.0.0.1", port=8080, max_points=500_000, max_cameras=100, point_size=0.005
):
    try:
        import viser
    except ImportError as exc:
        raise ImportError("Install viewer dependencies: pip install -e '.[viewer]'") from exc
    if max_cameras < 1 or not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("Camera budget and point size must be positive")
    data = load_reconstruction(result, max_points)
    server = viser.ViserServer(host=host, port=port, label=f"GlassRecon · {Path(result).name}")
    try:
        _populate_scene(server, data, max_cameras, point_size)
    except Exception:
        server.stop()
        raise
    return server


def _populate_scene(server, data, max_cameras, point_size):
    rotation = quaternion_matrix(data.poses[0, 4:]) if len(data.poses) else np.eye(3)
    up, forward, right = -rotation[:, 1], rotation[:, 2], rotation[:, 0]
    position = data.center - forward * data.radius * 2.3 + up * data.radius * 0.3
    position += right * data.radius * 0.2
    server.scene.set_up_direction(tuple(up))
    server.initial_camera.position = tuple(position)
    server.initial_camera.look_at = tuple(data.center)
    server.initial_camera.up_direction = tuple(up)
    overview_fov = float(np.deg2rad(60))
    server.initial_camera.fov = overview_fov
    server.scene.world_axes.visible = True

    mesh_handle = None
    if data.mesh is not None:
        mesh_handle = server.scene.add_mesh_trimesh("/reconstruction/mesh", data.mesh)
    cloud_handle = None
    if len(data.points):
        cloud_handle = server.scene.add_point_cloud(
            "/reconstruction/points",
            points=data.points,
            colors=data.colors,
            point_size=point_size,
            point_shape="circle",
            precision="float32",
            visible=mesh_handle is None,
        )

    trajectory_handles = []
    if len(data.poses) > 1:
        segments = np.stack([data.poses[:-1, 1:4], data.poses[1:, 1:4]], axis=1)
        trajectory_handles.append(
            server.scene.add_line_segments(
                "/trajectory/path",
                points=segments.astype(np.float32),
                colors=(255, 180, 50),
                line_width=3.0,
            )
        )

    def use_camera(client, row):
        wxyz, translation = camera_pose(row)
        with client.atomic():
            client.camera.up_direction = tuple(-quaternion_matrix(row[4:])[:, 1])
            client.camera.wxyz = tuple(wxyz)
            client.camera.position = tuple(translation)
            client.camera.fov = float(data.camera_fov)
            client.camera.look_at = tuple(
                translation + quaternion_matrix(row[4:])[:, 2] * data.radius
            )

    if len(data.poses):
        indices = np.linspace(0, len(data.poses) - 1, min(max_cameras, len(data.poses)), dtype=int)
        for index in indices:
            row = data.poses[index]
            wxyz, translation = camera_pose(row)
            camera = server.scene.add_camera_frustum(
                f"/trajectory/camera_{index}",
                fov=float(data.camera_fov),
                aspect=float(data.camera_aspect),
                scale=min(data.radius * 0.07, 0.2),
                color=(255, 180, 50),
                wxyz=wxyz,
                position=translation,
            )
            trajectory_handles.append(camera)

            @camera.on_click
            def jump_to_camera(event, row=row):
                use_camera(event.client, row)

    face_count = len(data.mesh.faces) if data.mesh is not None else 0
    frame_info = ""
    if "integrated" in data.summary:
        frame_info = f"**Frames fused: {data.summary['integrated']}**\n\n"
        if data.summary["integrated"] == 1:
            frame_info += "Single-frame result; no sequential camera tracking.\n\n"
    server.gui.add_markdown(
        "**Reconstruction** · units: meters\n\n"
        + frame_info
        + f"{face_count:,} triangles · {len(data.points):,}/{data.total_points:,} cloud points"
        f" · {len(data.poses):,} camera poses\n\n"
        "Drag to orbit, right-drag to pan, and scroll to zoom. Click a camera to view from it."
    )
    modes = [
        name
        for name, available in [
            ("Mesh", mesh_handle),
            ("Point cloud", cloud_handle),
            ("Both", mesh_handle and cloud_handle),
        ]
        if available
    ]
    mode = server.gui.add_dropdown("Geometry", options=tuple(modes), initial_value=modes[0])

    @mode.on_update
    def update_geometry(_):
        if mesh_handle is not None:
            mesh_handle.visible = mode.value in {"Mesh", "Both"}
        if cloud_handle is not None:
            cloud_handle.visible = mode.value in {"Point cloud", "Both"}

    if cloud_handle is not None:
        size = server.gui.add_slider(
            "Point size (m)",
            min=min(0.001, point_size),
            max=max(0.05, point_size),
            step=min(0.001, point_size),
            initial_value=point_size,
        )

        @size.on_update
        def update_size(_):
            cloud_handle.point_size = size.value

    if trajectory_handles:
        show_trajectory = server.gui.add_checkbox("Camera trajectory", initial_value=True)

        @show_trajectory.on_update
        def update_trajectory(_):
            for handle in trajectory_handles:
                handle.visible = show_trajectory.value

    reset = server.gui.add_button("Frame scene")

    @reset.on_click
    def reset_view(event):
        if event.client is not None:
            with event.client.atomic():
                event.client.camera.position = tuple(position)
                event.client.camera.look_at = tuple(data.center)
                event.client.camera.up_direction = tuple(up)
                event.client.camera.fov = overview_fov

    if len(data.poses):
        first = server.gui.add_button("First camera")

        @first.on_click
        def first_camera(event):
            if event.client is not None:
                use_camera(event.client, data.poses[0])


class LiveReconstructionViewer:
    """Observe sequential fusion without changing tracking or integration decisions."""

    def __init__(self, host="127.0.0.1", port=8080, max_points=200_000):
        try:
            import viser
        except ImportError as exc:
            raise ImportError("Install viewer dependencies: pip install -e '.[viewer]'") from exc
        if max_points < 1:
            raise ValueError("Viewer point budget must be positive")
        self.server = viser.ViserServer(
            host=host, port=port, label="GlassRecon · sequential fusion"
        )
        panel = getattr(self.server.gui, "main_panel", None)
        if panel is None:  # Earlier Viser 1.x versions use the theme API for docking.
            self.server.gui.configure_theme(control_layout="fixed")
        else:
            panel.dock_right()
        self.max_points = max_points
        self._lock = threading.RLock()
        self._start = threading.Event()
        self._stopped = threading.Event()
        self._start_button = None
        self.positions = []
        self._segments = []
        self._completed_paths = []
        self.cloud = self.path = self.camera = None
        self.fov, self.aspect = float(np.deg2rad(60)), 4 / 3
        self.mode = ""
        self._total = 0
        self._reset_replay()
        self.server.scene.set_up_direction("-y")
        self.server.scene.world_axes.visible = True
        self.status = self.server.gui.add_markdown("Preparing sequential reconstruction…")
        self.show_map = self.server.gui.add_checkbox("Cumulative map", initial_value=True)
        self.show_path = self.server.gui.add_checkbox("Camera trajectory", initial_value=True)
        self.follow = self.server.gui.add_checkbox("Follow current camera", initial_value=False)
        self.home = self.server.gui.add_button("Frame scene")

        @self.show_map.on_update
        def toggle_map(_):
            with self._lock:
                if self.cloud is not None:
                    self.cloud.visible = self.show_map.value

        @self.show_path.on_update
        def toggle_path(_):
            with self._lock:
                for path in self._completed_paths:
                    path.visible = self.show_path.value
                if self.path is not None:
                    self.path.visible = self.show_path.value
                if self.camera is not None:
                    self.camera.visible = self.show_path.value and self._previous is not None

        @self.home.on_click
        def frame_scene(event):
            if event.client is not None:
                with self._lock:
                    self.follow.value = False
                    self._set_overview(event.client.camera)

        print(f"Live Viser: http://{self.server.get_host()}:{self.server.get_port()}", flush=True)

    def configure(self, intrinsics, total, pose_source, depth_source, tracking_depth="sensor"):
        self._total = total
        self.fov = float(2 * np.arctan(intrinsics.height / (2 * intrinsics.fy)))
        self.aspect = intrinsics.width / intrinsics.height
        self.mode = f"Poses: **{pose_source}** · Fusion depth: **{depth_source}**"
        if pose_source == "odometry":
            self.mode += f"\n\nTracking depth: **{tracking_depth}**"
        self.status.content = (
            f"**Sequential reconstruction: {total} RGB-D frames**\n\n"
            f"{self.mode}\n\nLoading model and preparing fusion…"
        )

    def wait_for_start(self):
        """Gate the first frame until a connected viewer starts the replay."""
        self.status.content = (
            f"**Ready to replay {self._total} RGB-D frames**\n\n{self.mode}\n\n"
            "Click **Start replay** to begin tracking and mapping."
        )
        self._wait_for_button("Start replay")

    def wait_for_replay(self):
        """Keep the finished scene interactive until any connected client starts a new run."""
        self._wait_for_button("Replay")
        with self._lock, self.server.atomic():
            self._reset_replay()
            self.show_map.value = self.show_path.value = True
            self.follow.value = False
            self._update_status()

    def _wait_for_button(self, label):
        with self._lock, self.server.atomic():
            if self._stopped.is_set():
                raise RuntimeError("Viewer stopped before replay started")
            self._start.clear()
            if self._start_button is None:
                self._start_button = self.server.gui.add_button(label)

                @self._start_button.on_click
                def start(_):
                    with self._lock:
                        if not self._start_button.disabled:
                            self._start_button.disabled = True
                            self._start.set()
            else:
                self._start_button.label = label
                self._start_button.disabled = False

        self._start.wait()
        if self._stopped.is_set():
            raise RuntimeError("Viewer stopped before replay started")

    def _set_overview(self, camera):
        up, forward, right = -self.rotation[:, 1], self.rotation[:, 2], self.rotation[:, 0]
        camera.position = tuple(
            self.center + self.radius * (-2.3 * forward + 0.3 * up + 0.2 * right)
        )
        camera.look_at = tuple(self.center)
        camera.up_direction = tuple(up)
        camera.fov = float(np.deg2rad(60))

    def on_frame(self, progress, pose):
        with self._lock, self.server.atomic():
            self.status.content = (
                f"**Sequential reconstruction**\n\n{self.mode}\n\n"
                f"Processed: **{progress['processed']} / {progress['selected']}**\n\n"
                f"Integrated: **{progress['integrated']}** · Skipped: **{progress['skipped']}**\n\n"
                f"Timestamp: {progress['timestamp']:.6f}\n\nLast frame: {progress['frame_status']}"
            )
            if pose is not None:
                self._draw_pose(pose)

    def on_tracking(self, index, timestamp, state, pose, segment=0):
        """Publish a 1-based input frame and its online camera-to-world pose, if tracked."""
        with self._lock, self.server.atomic():
            if index <= self._processed:
                return
            now = time.monotonic()
            if self._started_at is None:
                self._started_at = now
                self._stage = "Waiting for depth"
            elif now > self._started_at:
                self._tracking_rate = f"{(index - 1) / (now - self._started_at):.1f} fps"
            if self._segment is not None and segment != self._segment:
                self._clear_submap()
            if self._segment_started_at is None or segment != self._segment:
                self._segment_started_at = now
            self._segment = segment
            self._processed, self._tracking_time, self._state = index, timestamp, state
            if pose is None:
                # Relocalization can resume in this map. Never draw a line across the gap.
                self._missing += 1
                self._previous = None
                if self.camera is not None:
                    self.camera.visible = False
            else:
                self._draw_pose(pose)
            self._update_status()

    def on_fusion(self, integrated, *, pending=0, timestamp=None, stage="Mapping", segment=None):
        """Update mapping progress independently of tracking. Timestamps refer to fused frames."""
        with self._lock:
            if segment is not None and segment != self._segment:
                return
            if self._segment_started_at is not None and integrated > self._integrated:
                elapsed = time.monotonic() - self._segment_started_at
                if elapsed > 0:
                    self._fusion_rate = f"{integrated / elapsed:.1f} fps"
            self._integrated, self._pending, self._stage = integrated, pending, stage
            if timestamp is not None:
                self._fusion_time = timestamp
            self._update_status()

    def _update_status(self):
        tracking_at = "—" if self._tracking_time is None else f"{self._tracking_time:.6f} s"
        fused_at = "—" if self._fusion_time is None else f"{self._fusion_time:.6f} s"
        self.status.content = (
            f"**RGB-D replay** · Submap **{self._segment if self._segment is not None else '—'}**"
            f"\n\n{self.mode}\n\n"
            f"Tracking: **{self._state}** · **{self._processed} / {self._total}** frames"
            f" · {self._tracking_rate}\n\nLatest input: {tracking_at}"
            f" · {self._missing} untracked\n\n"
            f"Fusion: **{self._stage}** · **{self._integrated}** integrated"
            f" · {self._fusion_rate}\n\nLatest fused frame: {fused_at} "
            f"· {self._pending} pending in depth window"
        )

    def _clear_submap(self):
        # Treat each tracking segment as a separate coordinate system.
        for handle in (*self._completed_paths, self.path, self.camera, self.cloud):
            if handle is not None:
                handle.remove()
        self.cloud = self.path = self.camera = self._previous = None
        self.positions.clear()
        self._segments.clear()
        self._completed_paths.clear()
        self.center, self.radius, self.rotation = np.zeros(3), 1.0, np.eye(3)
        self._fusion_time = None
        self._fusion_rate = "—"
        self._integrated = self._pending = 0
        self._stage = "Waiting for new submap"

    def _reset_replay(self):
        self._clear_submap()
        self._processed = self._missing = 0
        self._started_at = self._tracking_time = self._segment_started_at = None
        self._tracking_rate = "—"
        self._segment = None
        self._state, self._stage = "Waiting", "Waiting for poses"

    def _draw_pose(self, pose):
        position = pose[:3, 3].copy()
        if not self.positions:
            self.rotation = pose[:3, :3].copy()
            self.server.scene.set_up_direction(tuple(-self.rotation[:, 1]))
            self.center = position
            self._set_overview(self.server.initial_camera)
            if not self.follow.value:
                for client in self.server.get_clients().values():
                    self._set_overview(client.camera)
        self.positions.append(position)
        xyzw = matrix_quaternion(pose[:3, :3])
        wxyz = xyzw[[3, 0, 1, 2]]
        if self.camera is None:
            self.camera = self.server.scene.add_camera_frustum(
                "/live/current_camera",
                fov=self.fov,
                aspect=self.aspect,
                scale=0.12,
                color=(255, 180, 50),
                wxyz=wxyz,
                position=position,
                visible=self.show_path.value,
            )
        else:
            self.camera.wxyz, self.camera.position = wxyz, position
            self.camera.visible = self.show_path.value
        if self._previous is not None:
            # Keep historical chunks on the client; only retransmit the changing tail.
            if len(self._segments) == 128:
                self._completed_paths.append(self.path)
                self.path = None
                self._segments.clear()
            self._segments.append((self._previous, position))
            segments = np.asarray(self._segments, dtype=np.float32)
            if self.path is None:
                self.path = self.server.scene.add_line_segments(
                    f"/live/trajectory/{len(self._completed_paths)}",
                    points=segments,
                    colors=(255, 180, 50),
                    line_width=3,
                    visible=self.show_path.value,
                )
            else:
                self.path.points = segments
                self.path.colors = np.broadcast_to((255, 180, 50), segments.shape).astype(np.uint8)
        self._previous = position
        if self.follow.value:
            for client in self.server.get_clients().values():
                client.camera.up_direction = tuple(-pose[:3, 1])
                client.camera.position = tuple(position)
                client.camera.look_at = tuple(position + pose[:3, 2] * self.radius)
                client.camera.fov = self.fov

    def update_map(self, points, colors, *, segment=None):
        if not len(points):
            return
        lower, upper = points.min(axis=0), points.max(axis=0)
        if len(points) > self.max_points:
            indices = np.random.default_rng(0).choice(len(points), self.max_points, replace=False)
            points, colors = points[indices], colors[indices]
        with self._lock, self.server.atomic():
            if segment is not None and segment != self._segment:
                return
            initial = self.server.initial_camera
            # Reframe clients still in the overview as the map expands. An orbit,
            # pan or zoom gives that client control until they click Frame scene.
            overview_clients = []
            if not self.follow.value:
                tolerance = max(self.radius * 0.005, 0.001)
                for client in self.server.get_clients().values():
                    if self.cloud is None or (
                        np.allclose(client.camera.position, initial.position, atol=tolerance)
                        and np.allclose(client.camera.look_at, initial.look_at, atol=tolerance)
                    ):
                        overview_clients.append(client)
            if self.positions:
                lower = np.minimum(lower, np.min(self.positions, axis=0))
                upper = np.maximum(upper, np.max(self.positions, axis=0))
            self.center = (lower + upper) / 2
            self.radius = max(float(np.linalg.norm(upper - lower)) / 2, 0.1)
            self._set_overview(initial)
            for client in overview_clients:
                self._set_overview(client.camera)
            if self.cloud is None:
                self.cloud = self.server.scene.add_point_cloud(
                    "/live/map",
                    points=points,
                    colors=colors,
                    point_size=0.008,
                    point_shape="circle",
                    precision="float32",
                    visible=self.show_map.value,
                )
            else:
                self.cloud.points = points
                self.cloud.colors = colors

    def finish(self, summary):
        self.status.content = (
            f"**Sequential reconstruction complete**\n\n{self.mode}\n\n"
            f"Processed: **{summary['processed']} / {summary['selected']}**\n\n"
            f"Integrated: **{summary['integrated']}** · Skipped: **{summary['skipped']}**\n\n"
            f"Sequence duration: {summary['sequence_duration_seconds']:.2f} s\n\n"
            f"Camera path: {summary['estimated_path_length_m']:.2f} m\n\n"
            "Mesh, point cloud, trajectory, and per-frame diagnostics saved "
            "to the output directory."
        )

    def wait(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def stop(self):
        with self._lock:
            self._stopped.set()
            self._start.set()
        self.server.stop()


def serve(args):
    server = create_viewer(
        args.result, args.host, args.port, args.max_points, args.max_cameras, args.point_size
    )
    print(
        f"Viser viewer: http://{server.get_host()}:{server.get_port()} · Ctrl+C to stop", flush=True
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
