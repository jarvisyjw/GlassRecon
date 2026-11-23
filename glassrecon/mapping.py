"""Open3D odometry and TSDF fusion. All public poses are camera-to-world."""

from pathlib import Path

import numpy as np


class TrackingError(RuntimeError):
    pass


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise TrackingError("Pose must be a finite 4 x 4 matrix")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)
    ):
        raise TrackingError("Pose must be a rigid camera-to-world transform")
    return pose


def accumulate_pose(previous_c2w, previous_to_current):
    """Odometry maps previous-camera points into current-camera coordinates."""
    return validate_pose(previous_c2w) @ np.linalg.inv(validate_pose(previous_to_current))


class RGBDTracker:
    def __init__(
        self,
        intrinsics,
        voxel_size=0.02,
        sdf_trunc=0.08,
        min_depth=0.1,
        max_depth=10.0,
        depth_diff_max=0.07,
    ):
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("Install mapping dependencies: pip install -e '.[mapping]'") from exc
        self.o3d = o3d
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )
        self.min_depth, self.max_depth = min_depth, max_depth
        self.odometry_options = o3d.pipelines.odometry.OdometryOption()
        self.odometry_options.depth_min = min_depth
        self.odometry_options.depth_max = max_depth
        self.odometry_options.depth_diff_max = depth_diff_max
        self.previous_rgbd = None
        self.previous_c2w = np.eye(4)
        self.integrated = 0

    def rgbd(self, rgb, depth, intensity=False):
        depth = np.asarray(depth, dtype=np.float32).copy()
        depth[~np.isfinite(depth) | (depth < self.min_depth) | (depth > self.max_depth)] = 0
        if not np.any(depth > 0):
            raise TrackingError("Frame has no valid depth for tracking/fusion")
        return self.o3d.geometry.RGBDImage.create_from_color_and_depth(
            self.o3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8)),
            self.o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=intensity,
        )

    def track(self, rgb, depth):
        current = self.rgbd(rgb, depth, intensity=True)
        if self.previous_rgbd is None:
            pose = np.eye(4)
        else:
            success, transform, _ = self.o3d.pipelines.odometry.compute_rgbd_odometry(
                self.previous_rgbd,
                current,
                self.intrinsics,
                np.eye(4),
                self.o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                self.odometry_options,
            )
            if not success:
                raise TrackingError("RGB-D odometry failed; frame was not integrated")
            pose = accumulate_pose(self.previous_c2w, transform)
        # Failed estimates never become the reference for subsequent frames.
        self.previous_rgbd, self.previous_c2w = current, pose
        return pose.copy()


class RGBDMapper(RGBDTracker):
    def __init__(
        self,
        intrinsics,
        voxel_size=0.02,
        sdf_trunc=0.08,
        min_depth=0.1,
        max_depth=10.0,
        depth_diff_max=0.07,
    ):
        super().__init__(intrinsics, voxel_size, sdf_trunc, min_depth, max_depth, depth_diff_max)
        self.volume = self.o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=self.o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def metadata(self):
        return {"name": "open3d", "version": self.o3d.__version__}

    def integrate(self, rgb, depth, c2w):
        pose = validate_pose(c2w)
        self.volume.integrate(self.rgbd(rgb, depth), self.intrinsics, np.linalg.inv(pose))
        self.integrated += 1

    def point_cloud(self):
        """Return a snapshot of the cumulative TSDF surface for live visualization."""
        cloud = self.volume.extract_point_cloud()
        points = np.asarray(cloud.points, dtype=np.float32)
        colors = np.rint(np.clip(np.asarray(cloud.colors), 0, 1) * 255).astype(np.uint8)
        return points, colors

    def export(self, output):
        if not self.integrated:
            raise RuntimeError("No frames were integrated")
        output = Path(output)
        mesh = self.volume.extract_triangle_mesh()
        cloud = self.volume.extract_point_cloud()
        if len(mesh.vertices) == 0 or len(cloud.points) == 0:
            raise RuntimeError(
                "TSDF contains no extractable surface; check depth, poses and voxels"
            )
        mesh.compute_vertex_normals()
        if not self.o3d.io.write_triangle_mesh(str(output / "mesh.ply"), mesh):
            raise OSError("Failed to write mesh.ply")
        if not self.o3d.io.write_point_cloud(str(output / "pointcloud.ply"), cloud):
            raise OSError("Failed to write pointcloud.ply")
        return {
            "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles),
            "pointcloud_points": len(cloud.points),
        }


def create_mapper(
    intrinsics,
    voxel_size=0.02,
    sdf_trunc=0.08,
    min_depth=0.1,
    max_depth=10.0,
    depth_diff_max=0.07,
    backend="open3d",
):
    if backend == "open3d":
        cls = RGBDMapper
    elif backend == "nvblox":
        from .nvblox_mapping import NvbloxMapper

        cls = NvbloxMapper
    else:
        raise ValueError(f"Unknown reconstruction backend: {backend}")
    return cls(intrinsics, voxel_size, sdf_trunc, min_depth, max_depth, depth_diff_max)
