"""Metric reconstruction evaluation, without ICP or fitted scale correction."""

import numpy as np


def camera_rays(intrinsics, pose, pixel_step=1):
    """World rays whose hit parameter is camera z depth, using integer pixel centers."""
    if pixel_step < 1:
        raise ValueError("pixel_step must be positive")
    yy, xx = np.mgrid[: intrinsics.height : pixel_step, : intrinsics.width : pixel_step]
    directions = (
        np.stack(
            [
                (xx - intrinsics.cx) / intrinsics.fx,
                (yy - intrinsics.cy) / intrinsics.fy,
                np.ones_like(xx),
            ],
            axis=-1,
        )
        @ pose[:3, :3].T
    )
    origins = np.broadcast_to(pose[:3, 3], directions.shape)
    return np.concatenate([origins, directions], axis=-1).astype(np.float32)


class MeshSurface:
    def __init__(self, mesh):
        import open3d as o3d

        if not len(mesh.vertices) or not len(mesh.triangles):
            raise ValueError("Evaluation requires a nonempty triangle mesh")
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def render(self, intrinsics, pose, pixel_step=1):
        rays = camera_rays(intrinsics, pose, pixel_step)
        depth = self.scene.cast_rays(self.o3d.core.Tensor(rays), nthreads=2)["t_hit"].numpy()
        depth[~np.isfinite(depth)] = 0
        return depth

    def distances(self, points):
        points = np.asarray(points, dtype=np.float32)
        return np.concatenate(
            [
                self.scene.compute_distance(self.o3d.core.Tensor(part), nthreads=2).numpy()
                for part in np.array_split(points, max(1, (len(points) + 99999) // 100000))
            ]
        )


def sample_mesh(mesh, count=100_000, seed=0):
    """Area-uniform triangle samples; independent of vertex density."""
    vertices = np.asarray(mesh.vertices)
    triangles = vertices[np.asarray(mesh.triangles)]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if count < 1 or not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Need positive sample count and finite nondegenerate mesh")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    return (
        (1 - root) * selected[:, 0]
        + root * (1 - uv[:, 1:]) * selected[:, 1]
        + root * uv[:, 1:] * selected[:, 2]
    ).astype(np.float32)


def summarize_distances(accuracy, completeness, thresholds=(0.02, 0.05, 0.10)):
    """Accuracy: estimate→reference; completeness: visible reference→estimate."""
    a, c = np.asarray(accuracy), np.asarray(completeness)
    if not len(a) or not len(c) or not np.isfinite(a).all() or not np.isfinite(c).all():
        raise ValueError("Surface distances must be nonempty and finite")
    result = {
        "accuracy_mean_m": float(a.mean()),
        "accuracy_median_m": float(np.median(a)),
        "accuracy_p95_m": float(np.percentile(a, 95)),
        "accuracy_p99_m": float(np.percentile(a, 99)),
        "accuracy_rmse_m": float(np.sqrt(np.mean(a**2))),
        "completeness_mean_m": float(c.mean()),
        "chamfer_l1_m": float((a.mean() + c.mean()) / 2),
        "estimated_surface_samples": len(a),
        "visible_reference_samples": len(c),
    }
    for threshold in thresholds:
        precision, recall = float((a <= threshold).mean()), float((c <= threshold).mean())
        result[f"at_{threshold:g}m"] = {
            "precision": precision,
            "recall": recall,
            "fscore": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        }
    return result


class DepthMetrics:
    """Pixel-weighted depth errors; coverage prevents holes hiding missing geometry."""

    def __init__(self):
        self.reference_pixels = self.compared_pixels = 0
        self.absolute = self.squared = self.relative = 0.0

    def add(self, estimate, reference, mask=None):
        estimate, reference = np.asarray(estimate), np.asarray(reference)
        if estimate.shape != reference.shape:
            raise ValueError("Depth images must have equal shapes")
        valid = np.isfinite(reference) & (reference > 0)
        if mask is not None:
            valid &= mask
        matched = valid & np.isfinite(estimate) & (estimate > 0)
        error = estimate[matched].astype(np.float64) - reference[matched]
        self.reference_pixels += int(valid.sum())
        self.compared_pixels += int(matched.sum())
        self.absolute += float(np.abs(error).sum())
        self.squared += float(np.square(error).sum())
        self.relative += float((np.abs(error) / reference[matched]).sum())

    def summary(self):
        n = self.compared_pixels
        return {
            "reference_pixels": self.reference_pixels,
            "compared_pixels": n,
            "coverage": n / self.reference_pixels if self.reference_pixels else None,
            "mae_m": self.absolute / n if n else None,
            "rmse_m": float(np.sqrt(self.squared / n)) if n else None,
            "abs_rel": self.relative / n if n else None,
        }
