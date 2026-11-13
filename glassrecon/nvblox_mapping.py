"""CUDA TSDF/color fusion through NVIDIA's official nvblox_torch interface.

Depth is optical-axis Z in meters. nvblox accepts camera-to-world poses directly;
Open3D is used only for optional odometry and exporting the extracted geometry.
"""

import ctypes
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import numpy as np

from .mapping import RGBDTracker, TrackingError, validate_pose


def _load_pip_npp(cuda_major):
    """Resolve optional pip CUDA NPP libraries without changing system loader paths.

    The nvblox wheel links NPP, which PyTorch itself does not ship. A toolkit
    installation can supply it; on runtime-only hosts use NVIDIA's NPP wheel.
    """
    try:
        package = distribution(f"nvidia-npp-cu{cuda_major}")
    except PackageNotFoundError:
        return []
    libraries = []
    for name in ["nppc", "nppitc", "nppim", "nppidei"]:
        filename = f"lib{name}.so.{cuda_major}"
        entry = next((p for p in package.files or [] if p.name == filename), None)
        if entry is None:
            raise ImportError(f"NVIDIA NPP package is missing {filename}")
        libraries.append(ctypes.CDLL(str(package.locate_file(entry)), mode=ctypes.RTLD_GLOBAL))
    return libraries


class NvbloxMapper(RGBDTracker):
    def __init__(
        self,
        intrinsics,
        voxel_size=0.02,
        sdf_trunc=0.08,
        min_depth=0.1,
        max_depth=10.0,
        depth_diff_max=0.07,
    ):
        if not 0 < voxel_size < sdf_trunc < float("inf"):
            raise ValueError("Require 0 < voxel-size < sdf-trunc < infinity")
        if not 0 < min_depth < max_depth < float("inf"):
            raise ValueError("Require 0 < min-depth < max-depth < infinity")
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "nvblox requires its CUDA/PyTorch environment; see docs/nvblox.md"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("nvblox requires a CUDA GPU; no CPU fallback is performed")
        try:
            self._cuda_libraries = _load_pip_npp(torch.version.cuda.split(".")[0])
            from nvblox_torch import __git_sha__
            from nvblox_torch.mapper import Mapper
            from nvblox_torch.mapper_params import MapperParams
            from nvblox_torch.sensor import Sensor
        except (ImportError, OSError, RuntimeError) as exc:
            raise ImportError(
                "Cannot load nvblox_torch. Use the matching NVIDIA wheel and PyTorch runtime; "
                "see docs/nvblox.md"
            ) from exc
        super().__init__(intrinsics, voxel_size, sdf_trunc, min_depth, max_depth, depth_diff_max)
        self.torch = torch
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.shape = (intrinsics.height, intrinsics.width)
        self.camera = Sensor.from_camera(
            fu=intrinsics.fx,
            fv=intrinsics.fy,
            cu=intrinsics.cx,
            cv=intrinsics.cy,
            width=intrinsics.width,
            height=intrinsics.height,
        )
        params = MapperParams()
        projective = params.get_projective_integrator_params()
        projective.projective_integrator_max_integration_distance_m = float(max_depth)
        projective.projective_integrator_truncation_distance_vox = float(sdf_trunc / voxel_size)
        params.set_projective_integrator_params(projective)
        self.mapper = Mapper(voxel_sizes_m=float(voxel_size), mapper_parameters=params)
        self._metadata = {
            "name": "nvblox",
            "version": version("nvblox_torch"),
            "git_sha": __git_sha__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device),
            "voxel_size_m": float(voxel_size),
            "truncation_distance_vox": float(sdf_trunc / voxel_size),
            "max_integration_distance_m": float(max_depth),
            "weighting_mode": projective.projective_integrator_weighting_mode,
            "max_weight": projective.projective_integrator_max_weight,
            "pose_convention": "camera_to_world",
            "depth_convention": "optical_axis_z_meters",
            "pointcloud_source": "extracted mesh vertices",
        }

    def metadata(self):
        return dict(self._metadata)

    def integrate(self, rgb, depth, c2w):
        pose = validate_pose(c2w)
        depth = np.array(depth, dtype=np.float32, order="C", copy=True)
        rgb = np.asarray(rgb)
        if depth.shape != self.shape or rgb.shape != (*self.shape, 3):
            raise ValueError("Depth/RGB dimensions must match the camera intrinsics")
        if rgb.dtype != np.uint8:
            raise ValueError("RGB must be uint8 in RGB channel order")
        valid = np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
        if not valid.any():
            raise TrackingError("Frame has no valid depth for tracking/fusion")
        depth[~valid] = 0
        torch = self.torch
        depth_gpu = torch.from_numpy(depth).to(self.device)
        rgb_gpu = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        mask_gpu = torch.from_numpy(valid.astype(np.uint8)).to(self.device)
        pose_cpu = torch.from_numpy(np.array(pose, dtype=np.float32, order="C", copy=True))
        # Inputs stay alive until the nvblox CUDA stream finishes consuming them.
        torch.cuda.synchronize(self.device)
        self.mapper.add_depth_frame(depth_gpu, pose_cpu, self.camera)
        self.mapper.add_color_frame(rgb_gpu, pose_cpu, self.camera, mask_frame=mask_gpu)
        torch.cuda.synchronize(self.device)
        self.integrated += 1

    def _mesh(self):
        self.mapper.update_color_mesh()
        self.torch.cuda.synchronize(self.device)
        return self.mapper.get_color_mesh().to_open3d()

    def point_cloud(self):
        """Return cumulative mesh vertices and RGB for the existing live Viser UI."""
        mesh = self._mesh()
        points = np.asarray(mesh.vertices, dtype=np.float32)
        colors = np.rint(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
        return points, colors

    def export(self, output):
        if not self.integrated:
            raise RuntimeError("No frames were integrated")
        mesh = self._mesh()
        if not len(mesh.vertices) or not len(mesh.triangles):
            raise RuntimeError("nvblox TSDF contains no surface; check depth, poses and voxels")
        mesh.compute_vertex_normals()
        cloud = self.o3d.geometry.PointCloud()
        cloud.points, cloud.colors, cloud.normals = (
            mesh.vertices,
            mesh.vertex_colors,
            mesh.vertex_normals,
        )
        output = Path(output)
        if not self.o3d.io.write_triangle_mesh(str(output / "mesh.ply"), mesh):
            raise OSError("Failed to write mesh.ply")
        if not self.o3d.io.write_point_cloud(str(output / "pointcloud.ply"), cloud):
            raise OSError("Failed to write pointcloud.ply")
        return {
            "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles),
            "pointcloud_points": len(cloud.points),
        }
