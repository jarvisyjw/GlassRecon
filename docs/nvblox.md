# nvblox setup

Quick Start and the complete pipeline use NVIDIA's nvblox TSDF and color integrators on a CUDA-capable GPU.

## Environment

The supplied wheel targets **Ubuntu 24.04, Linux x86_64 and CUDA 12**. Use the separate Python 3.12 mapping environment from the [README](../README.md#environment-setup). For another supported platform, select its wheel from the [nvblox installation guide](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/installation.html).

The mapping requirements install the nvblox v0.0.10 wheel, a compatible PyTorch, the NVIDIA NPP runtime and the viewer. This release asset uses a different version in its filename and package metadata, so the installation command sets `UV_SKIP_WHEEL_FILENAME_CHECK=1` for that command only.

## Inputs and outputs

- nvblox consumes RGB, optical-axis depth in meters and a camera-to-world pose for every frame. The reconstruction command converts stored depth to meters. Use `--depth-units-per-meter 1` for the [prepared mapping demo](../README.md#quick-start) and `1000` for the [ScanNet++ sensor PNGs](scannetpp.md).
- Supply camera poses via `--poses` in the complete pipeline. To estimate them from RGB and sensor depth, omit that option. In both modes nvblox does the fusion.
- The final map contains `mesh.ply` from the nvblox color mesh, `pointcloud.ply` from its colored vertices, and `trajectory.txt` with the integrated camera poses. Open a finished map with the viewer command in the [Quick Start](../README.md#quick-start).
