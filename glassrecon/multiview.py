"""Fixed-lag joint affine depth recovery, anchored by observed sensor measurements.

Only the active window is optimized. The oldest frame is emitted exactly once,
with W-1 frame latency. One committed boundary frame anchors the next window.
Correspondences/visibility come from observed depth and RGB, never clean GT.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .mapping import validate_pose
from .refinement import interior_mask, noise_scale


@dataclass(frozen=True)
class WindowConfig:
    size: int = 5
    sample_step: int = 8
    temporal_weight: float = 1.0
    iterations: int = 5
    huber_delta: float = 1.5
    correction_sigma: float = 0.15
    min_matches: int = 30
    minimum: float = 0.1
    maximum: float = 10.0

    def __post_init__(self):
        if not 1 <= self.size <= 20 or min(self.sample_step, self.iterations, self.min_matches) < 1:
            raise ValueError("Window size must be 1..20; sampling/iterations/matches positive")
        values = [
            self.temporal_weight,
            self.huber_delta,
            self.correction_sigma,
            self.minimum,
            self.maximum,
        ]
        if not np.isfinite(values).all() or self.temporal_weight < 0:
            raise ValueError("Window parameters must be finite; temporal weight nonnegative")
        if (
            min(self.huber_delta, self.correction_sigma, self.minimum) <= 0
            or self.maximum <= self.minimum
        ):
            raise ValueError("Invalid window tolerances/depth limits")


class SlidingWindowRecovery:
    def __init__(self, camera, config=None):
        self.camera = camera
        self.config = config or WindowConfig()
        self.pending = []
        self.boundary = None
        self.pairs = {}
        self.next_id = 0
        self.last_time = None
        self.emitted = 0

    def _sensor_system(self, node):
        cfg = self.config
        mask = node["interior"].copy()
        yy, xx = np.indices(mask.shape)
        mask &= (yy % cfg.sample_step == 0) & (xx % cfg.sample_step == 0)
        x, y = node["base"][mask], node["sensor"][mask]
        sigma = 0.03 + 0.01 * y
        return np.column_stack([x, np.ones_like(x)]) / sigma[:, None], y / sigma

    def _matches(self, source, target):
        """Sensor-projected nearest-pixel matches with z-buffer and two-view visibility."""
        key = (source["id"], target["id"])
        if key in self.pairs:
            return self.pairs[key]
        cam, cfg = self.camera, self.config
        yy, xx = np.nonzero(source["interior"])
        take = (yy % cfg.sample_step == 0) & (xx % cfg.sample_step == 0)
        yy, xx = yy[take], xx[take]
        rays = np.column_stack([(xx - cam.cx) / cam.fx, (yy - cam.cy) / cam.fy, np.ones(len(xx))])
        transform = np.linalg.inv(target["pose"]) @ source["pose"]
        rotated = rays @ transform[:3, :3].T
        xyz = rotated * source["sensor"][yy, xx, None] + transform[:3, 3]
        z = xyz[:, 2]
        u = np.rint(cam.fx * xyz[:, 0] / np.maximum(z, 1e-6) + cam.cx).astype(int)
        v = np.rint(cam.fy * xyz[:, 1] / np.maximum(z, 1e-6) + cam.cy).astype(int)
        keep = (z > cfg.minimum) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        yy, xx, u, v, z, rotated = [a[keep] for a in [yy, xx, u, v, z, rotated]]
        buffer = np.full((cam.height, cam.width), np.inf)
        np.minimum.at(buffer, (v, u), z)
        measured = target["sensor"][v, u]
        keep = target["interior"][v, u] & (z <= buffer[v, u] + 0.001)
        keep &= np.abs(measured - z) < 3 * noise_scale(measured)
        color = (
            np.mean(np.abs(source["rgb"][yy, xx].astype(float) - target["rgb"][v, u]), axis=1) / 255
        )
        keep &= color < 0.12
        yy, xx, u, v, z, rotated = [a[keep] for a in [yy, xx, u, v, z, rotated]]
        target_rays = np.column_stack(
            [(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones(len(u))]
        )
        # R ray_i (a_i z_i+b_i) + t = ray_j (a_j z_j+b_j).
        left = np.stack([rotated * source["base"][yy, xx, None], rotated], axis=-1)
        right = -np.stack([target_rays * target["base"][v, u, None], target_rays], axis=-1)
        sigma = 0.03 + 0.01 * target["sensor"][v, u]
        left = (left / sigma[:, None, None]).reshape(-1, 2)
        right = (right / sigma[:, None, None]).reshape(-1, 2)
        rhs = np.broadcast_to(-transform[:3, 3], (len(u), 3)) / sigma[:, None]
        result = (left, right, rhs.ravel(), len(u))
        self.pairs[key] = result
        return result

    def _solve(self, groups, initial):
        if not groups:
            return initial.copy(), 0.0, 0.0
        matrix = np.concatenate([g[0] for g in groups])
        target = np.concatenate([g[1] for g in groups])
        weights = np.concatenate([np.full(len(g[1]), g[2] / max(1, len(g[1]))) for g in groups])

        def cost(x):
            r = np.abs(matrix @ x - target)
            d = self.config.huber_delta
            return float(np.sum(weights * np.where(r <= d, 0.5 * r * r, d * (r - 0.5 * d))))

        x = initial.copy()
        before = cost(x)
        for _ in range(self.config.iterations):
            residual = np.abs(matrix @ x - target)
            w = weights * np.minimum(1, self.config.huber_delta / np.maximum(residual, 1e-12))
            lhs = matrix.T @ (w[:, None] * matrix)
            rhs = matrix.T @ (w * target)
            try:
                proposal = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(proposal).all():
                break
            if np.max(np.abs(proposal - x)) < 1e-7:
                x = proposal
                break
            x = proposal
        after = cost(x)
        if after > before + 1e-9:
            return initial.copy(), before, before
        return x, before, after

    def push(self, key, timestamp, rgb, sensor, base_depth, pose):
        cfg = self.config
        shape = (self.camera.height, self.camera.width)
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Sliding-window frames must have increasing timestamps")
        if (
            not np.isfinite(timestamp)
            or sensor.shape != shape
            or base_depth.shape != shape
            or rgb.shape != (*shape, 3)
        ):
            raise ValueError("Invalid timestamp or window image dimensions")
        pose = validate_pose(pose).copy()
        base = np.where(
            np.isfinite(base_depth) & (base_depth >= cfg.minimum) & (base_depth <= cfg.maximum),
            base_depth,
            0,
        ).astype(float)
        observed = np.where(
            np.isfinite(sensor) & (sensor >= cfg.minimum) & (sensor <= cfg.maximum), sensor, 0
        ).astype(float)
        node = {
            "id": self.next_id,
            "key": key,
            "timestamp": timestamp,
            "rgb": rgb.copy(),
            "sensor": observed,
            "base": base,
            "pose": pose,
            "coeff": np.array([1.0, 0.0]),
            "interior": interior_mask(observed) & (base > 0),
        }
        a, b = self._sensor_system(node)
        prior = np.eye(2) / cfg.correction_sigma
        node["anchors"] = (a, b)
        node["anchor_coeff"], _, _ = self._solve(
            [(a, b, 1.0), (prior, np.array([1.0, 0.0]) / cfg.correction_sigma, 0.05)],
            np.array([1.0, 0.0]),
        )
        self.next_id += 1
        self.last_time = timestamp
        self.pending.append(node)
        return self._emit() if len(self.pending) >= cfg.size else None

    def _emit(self):
        nodes = self.pending
        cfg = self.config
        n = len(nodes)
        groups = []
        for i, node in enumerate(nodes):
            a, b = node["anchors"]
            mat = np.zeros((len(b), 2 * n))
            mat[:, 2 * i : 2 * i + 2] = a
            groups.append((mat, b, 1 / n))
            prior = np.zeros((2, 2 * n))
            prior[:, 2 * i : 2 * i + 2] = np.eye(2) / cfg.correction_sigma
            groups.append((prior, np.array([1.0, 0.0]) / cfg.correction_sigma, 0.05 / n))
        pairs = []
        for j in range(n):
            sources = [(i, nodes[i]) for i in range(j)]
            if self.boundary is not None:
                sources.append((-1, self.boundary))
            for i, source in sources:
                left, right, rhs, count = self._matches(source, nodes[j])
                if count < cfg.min_matches:
                    continue
                mat = np.zeros((len(rhs), 2 * n))
                mat[:, 2 * j : 2 * j + 2] = right
                if i < 0:
                    rhs = rhs - left @ source["coeff"]
                else:
                    mat[:, 2 * i : 2 * i + 2] = left
                pairs.append((mat, rhs, count))
        groups += [(a, b, cfg.temporal_weight / max(1, len(pairs))) for a, b, _ in pairs]
        initial = np.concatenate([node["coeff"] for node in nodes])
        solution, before, after = self._solve(groups, initial)
        accepted = bool(np.isfinite(solution).all())
        for i, node in enumerate(nodes):
            a, b = solution[2 * i : 2 * i + 2]
            valid = node["base"] > 0
            if not 0.5 <= a <= 2 or abs(b) > 1 or np.any(a * node["base"][valid] + b <= 0):
                accepted = False
        if accepted:
            for i, node in enumerate(nodes):
                node["coeff"] = solution[2 * i : 2 * i + 2].copy()
        output = nodes.pop(0)

        def depth(coeff):
            z = coeff[0] * output["base"] + coeff[1]
            return np.where(
                (output["base"] > 0) & (z >= cfg.minimum) & (z <= cfg.maximum), z, 0
            ).astype(np.float32)

        output["depth"] = depth(output["coeff"])
        output["anchor_depth"] = depth(output["anchor_coeff"])
        output["diagnostics"] = {
            "alpha": float(output["coeff"][0]),
            "beta_m": float(output["coeff"][1]),
            "anchor_alpha": float(output["anchor_coeff"][0]),
            "anchor_beta_m": float(output["anchor_coeff"][1]),
            "window_frames": n,
            "boundary_frame": self.boundary["key"] if self.boundary else None,
            "latest_available_frame": nodes[-1]["key"] if nodes else output["key"],
            "pairs": len(pairs),
            "correspondences": sum(p[2] for p in pairs),
            "objective_before": before,
            "objective_after": after,
            "accepted": accepted,
        }
        self.boundary = output
        self.emitted += 1
        retained = {node["id"] for node in nodes} | {output["id"]}
        self.pairs = {k: v for k, v in self.pairs.items() if set(k) <= retained}
        return output

    def flush(self):
        while self.pending:
            yield self._emit()

    def settings(self):
        return asdict(self.config)
