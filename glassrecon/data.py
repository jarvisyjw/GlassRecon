"""Timestamp association and calibrated, registered RGB-D input in TUM format."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("Camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera focal lengths must be positive")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


# TUM recommends the ROS default for the pre-registered, unrectified images.
TUM_INTRINSICS = CameraIntrinsics(640, 480, 525.0, 525.0, 319.5, 239.5)


@dataclass(frozen=True)
class RGBDFrame:
    timestamp: float
    depth_timestamp: float
    rgb_path: Path
    depth_path: Path

    @property
    def key(self):
        return self.rgb_path.stem

    def load(self, depth_units_per_meter, intrinsics):
        with Image.open(self.rgb_path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        if self.depth_path.suffix.lower() == ".npy":
            raw = np.load(self.depth_path, allow_pickle=False)
        else:
            with Image.open(self.depth_path) as image:
                raw = np.array(image)
        expected = (intrinsics.height, intrinsics.width)
        if raw.shape != expected or rgb.shape[:2] != expected:
            raise ValueError(
                f"{self.key}: RGB {rgb.shape[:2]} and depth {raw.shape} must match "
                f"calibration {expected}; register depth to RGB before reconstruction"
            )
        depth = raw.astype(np.float32) / depth_units_per_meter
        depth[~np.isfinite(depth) | (depth <= 0)] = 0
        return rgb, depth


def read_index(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{number}: expected timestamp and image path")
        stamp = float(parts[0])
        if not np.isfinite(stamp):
            raise ValueError(f"{path}:{number}: nonfinite timestamp")
        rows.append((stamp, parts[1]))
    rows.sort()
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate timestamps in {path}")
    return rows


def associate(rgb, depth, max_difference=0.02):
    """Greedy closest-time, one-to-one pairing, ordered by RGB timestamp.

    Uses a bounded search window rather than constructing all N*M combinations.
    """
    if not np.isfinite(max_difference) or max_difference <= 0:
        raise ValueError("Association tolerance must be finite and positive")
    depth_times = np.array([row[0] for row in depth])
    candidates = []
    for i, (stamp, _) in enumerate(rgb):
        lo, hi = np.searchsorted(depth_times, [stamp - max_difference, stamp + max_difference])
        for j in range(lo, hi):
            candidates.append((abs(stamp - depth_times[j]), i, j))
    used_rgb, used_depth, matches = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            matches.append((i, j))
            used_rgb.add(i)
            used_depth.add(j)
    return sorted(matches)


def load_sequence(root, max_difference=0.02):
    root = Path(root)
    rgb, depth = read_index(root / "rgb.txt"), read_index(root / "depth.txt")
    frames = [
        RGBDFrame(rgb[i][0], depth[j][0], root / rgb[i][1], root / depth[j][1])
        for i, j in associate(rgb, depth, max_difference)
    ]
    if not frames:
        raise ValueError("No RGB/depth pairs found within the association tolerance")
    if len({frame.key for frame in frames}) != len(frames):
        raise ValueError("RGB filenames must have unique stems for prediction/depth output")
    return frames, {"rgb_frames": len(rgb), "depth_frames": len(depth), "associated": len(frames)}


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Expected a nonzero finite quaternion in qx qy qz qw order")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quaternion(rotation):
    # Symmetric eigenproblem is stable even for rotations close to 180 degrees.
    r = np.asarray(rotation)
    k = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[0, 1] + r[1, 0],
                    r[0, 2] + r[2, 0],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[0, 1] + r[1, 0],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[1, 2] + r[2, 1],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[0, 2] + r[2, 0],
                    r[1, 2] + r[2, 1],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        / 3
    )
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


class PoseTrajectory:
    """TUM camera-to-world poses; translation interpolation and quaternion SLERP."""

    def __init__(self, path):
        rows = []
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            values = np.array([float(v) for v in line.split()])
            if values.shape != (8,) or not np.isfinite(values).all():
                raise ValueError(f"{path}:{number}: expected 8 finite TUM pose values")
            quaternion_matrix(values[4:])  # Validate before normalizing.
            values[4:] /= np.linalg.norm(values[4:])
            rows.append(values)
        if not rows:
            raise ValueError(f"Empty pose trajectory: {path}")
        self.rows = np.array(sorted(rows, key=lambda row: row[0]))
        self.times = self.rows[:, 0]
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Pose timestamps must be unique")

    def at(self, timestamp, max_difference=0.05):
        """Return c2w or None; never extrapolate or interpolate across a large gap."""
        right = int(np.searchsorted(self.times, timestamp))
        if right < len(self.times) and abs(self.times[right] - timestamp) < 1e-9:
            translation, q = self.rows[right, 1:4], self.rows[right, 4:]
        elif right == 0 or right == len(self.times):
            return None
        else:
            a, b = self.rows[right - 1], self.rows[right]
            if max(timestamp - a[0], b[0] - timestamp) > max_difference:
                return None
            fraction = (timestamp - a[0]) / (b[0] - a[0])
            translation = (1 - fraction) * a[1:4] + fraction * b[1:4]
            qa, qb = a[4:], b[4:].copy()
            dot = float(np.dot(qa, qb))
            if dot < 0:
                qb, dot = -qb, -dot
            if dot > 0.9995:
                q = (1 - fraction) * qa + fraction * qb
            else:
                angle = np.arccos(np.clip(dot, -1, 1))
                q = (np.sin((1 - fraction) * angle) * qa + np.sin(fraction * angle) * qb) / np.sin(
                    angle
                )
        pose = np.eye(4)
        pose[:3, :3] = quaternion_matrix(q)
        pose[:3, 3] = translation
        return pose
