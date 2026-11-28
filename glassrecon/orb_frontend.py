"""Estimate a metric camera trajectory from registered, rectified RGB-D with ORB-SLAM3."""

import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from .data import CameraIntrinsics, load_sequence


def _camera_settings(camera, fps):
    # ORB's RGB-D mode uses a virtual stereo baseline. These extractor and
    # close-depth parameters match the upstream RGB-D example. Its settings
    # reader requires the viewer fields even when the viewer is disabled.
    return f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {float(camera.fx)}
Camera1.fy: {float(camera.fy)}
Camera1.cx: {float(camera.cx)}
Camera1.cy: {float(camera.cy)}
Camera.width: {camera.width}
Camera.height: {camera.height}
Camera.fps: {fps}
Camera.RGB: 1
RGBD.DepthMapFactor: 1.0
Stereo.b: 0.07732
Stereo.ThDepth: 40.0
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""


def _final_poses(exported, timestamps, tracked):
    """Match the nanosecond EuRoC export to the exact timestamps given to ORB."""
    if not exported.is_file() or not exported.stat().st_size:
        raise RuntimeError("ORB-SLAM3 exported an empty trajectory")
    rows = np.loadtxt(exported, ndmin=2)
    if rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise ValueError("Invalid ORB-SLAM3 trajectory export")

    # While recently lost, upstream Tracking can append the preceding time,
    # relative pose and keyframe again. This is one observation, not a new pose.
    repeated = np.all(rows[1:] == rows[:-1], axis=1)
    rows = rows[np.r_[True, ~repeated]]

    # The upstream exporter prints double(timestamp * 1e9) to six decimal
    # places. Matching in those same units also works for Unix epoch timestamps.
    source_ns = timestamps * 1e9
    indices = np.searchsorted(source_ns, rows[:, 0])
    right = np.minimum(indices, len(timestamps) - 1)
    left = np.maximum(indices - 1, 0)
    use_left = abs(source_ns[left] - rows[:, 0]) < abs(source_ns[right] - rows[:, 0])
    indices = np.where(use_left, left, right)
    if np.any(abs(source_ns[indices] - rows[:, 0]) > 1e-6) or np.any(np.diff(indices) <= 0):
        raise ValueError("ORB-SLAM3 exported unknown or duplicate frame timestamps")
    rows[:, 0] = timestamps[indices]
    rows = rows[tracked[indices]]
    if not len(rows):
        raise RuntimeError("ORB-SLAM3 produced no tracked poses in its final map")
    norms = np.linalg.norm(rows[:, 4:], axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-4):
        raise ValueError("Invalid rotation in ORB-SLAM3 export")
    rows[:, 4:] /= norms[:, None]
    return rows


def _camera_fps(timestamps, fps):
    if fps is None:
        if len(timestamps) < 2:
            raise ValueError("Pass --fps to specify the camera rate for a single frame")
        fps = max(1, round(1.0 / np.median(np.diff(timestamps))))
    if not isinstance(fps, int) or fps < 1:
        raise ValueError("ORB camera fps must be a positive integer")
    return fps


def _track_raw_rgbd(slam, frame, camera, units, masks):
    rgb, depth = frame.load(units, camera)
    if masks is not None:
        with Image.open(masks / f"{frame.key}.png") as image:
            valid = np.asarray(image) == 255
        if valid.shape != depth.shape:
            raise ValueError(f"Invalid camera mask dimensions for {frame.key}")
        depth[~valid] = 0
        rgb[~valid] = 0
    result = slam.track_rgbd(rgb, depth, frame.timestamp)
    state = result.state.name
    return state, result if result.pose_valid and state in ("OK", "OK_KLT") else None


def localize(dataset, vocabulary, depth_units_per_meter, output, intrinsics=None, fps=None):
    """Track raw sensor depth, keeping successful frames in the largest final ORB map."""
    try:
        import pyorbslam3 as orb
    except ImportError as exc:
        raise ImportError(
            "Install GlassRecon's orbslam3 extra in the ORB Python environment"
        ) from exc

    dataset, output, vocabulary = Path(dataset), Path(output), Path(vocabulary)
    if not vocabulary.is_file():
        raise FileNotFoundError(f"Missing ORB-SLAM3 text vocabulary: {vocabulary}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose a new or empty output directory: {output}")
    if not np.isfinite(depth_units_per_meter) or depth_units_per_meter <= 0:
        raise ValueError("Depth units per meter must be finite and positive")
    camera = CameraIntrinsics.load(intrinsics or dataset / "intrinsics.json")
    frames, counts = load_sequence(dataset)
    timestamps = np.array([frame.timestamp for frame in frames])
    fps = _camera_fps(timestamps, fps)
    masks = dataset / "masks" if (dataset / "masks").is_dir() else None
    tracked = np.zeros(len(frames), dtype=bool)
    states = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "localization"
        stage.mkdir()
        settings = stage / "settings.yaml"
        settings.write_text(_camera_settings(camera, fps))
        slam = orb.Slam(vocabulary, settings, orb.Sensor.RGBD, False)
        try:
            for i, frame in enumerate(frames):
                state, result = _track_raw_rgbd(slam, frame, camera, depth_units_per_meter, masks)
                states[state] += 1
                tracked[i] = result is not None
        finally:
            slam.shutdown()
        if not tracked.any():
            raise RuntimeError("ORB-SLAM3 could not localize any RGB-D frames")
        exported = Path(tmp) / "euroc.txt"
        slam.save_trajectory_euroc(str(exported))
        rows = _final_poses(exported, timestamps, tracked)
        np.savetxt(
            stage / "poses.txt",
            rows,
            fmt="%.17g",
            header="camera-to-world: timestamp tx ty tz qx qy qz qw",
        )
        summary = dict(
            **counts,
            camera_fps=fps,
            depth_units_per_meter=depth_units_per_meter,
            tracking_states=states,
            localized_frames=len(rows),
            coverage=len(rows) / len(frames),
            trajectory="Largest optimized ORB map; camera-to-world, meters, xyzw",
        )
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(output)
    return summary
