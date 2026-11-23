# nvblox reconstruction backend

`--backend nvblox` selects NVIDIA's CUDA TSDF and color integrators and mesh extractor.
It works with raw sensor depth, per-frame recovery, conservative fusion, and fixed-lag
multi-view recovery. The old `open3d` backend remains the default for existing commands
and CPU-only environments. There is no silent fallback when nvblox fails to load.

The backend uses the official [`nvblox_torch` API](https://github.com/nvidia-isaac/nvblox).
It does not estimate poses: use supplied camera-to-world poses for controlled experiments
or the existing Open3D hybrid RGB-D odometry for a sequence without poses. Neither mode
adds loop closure, relocalization, or pose-graph optimization.

## Environment

Use a separate environment because the official 0.0.10 wheel requires PyTorch <= 2.9.1
and NumPy < 1.27. The original DA3 prediction environment can keep its current packages.
The following targets Linux x86_64, Ubuntu 24.04, and a CUDA-capable NVIDIA GPU:

```bash
uv venv --python /usr/bin/python3.12 .venv-nvblox
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install \
  --python .venv-nvblox/bin/python -r requirements-nvblox-cu12.txt
```

For another platform choose its wheel using the
[official installation instructions](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/installation.html).
The requirements include NVIDIA's NPP runtime wheel for hosts without a system CUDA
toolkit. The adapter loads these libraries before nvblox; no system CUDA installation
or global linker configuration is modified.
The official release asset is named `0.0.10+cu12ubuntu24`, but its internal package
version is `0.0.10.dev1+cu12ubuntu24`. The scoped uv variable above permits that upstream
filename mismatch. Run metadata records the actual version and source commit.
This install supplies the reconstruction runtime; it does not install DA3. To reuse
DA3-LARGE predictions, select `--model cached --prediction-dir ...`. Fresh inference
requires installing DA3 in a compatible runtime as described in the main README.

## Sequential mapping

```bash
.venv-nvblox/bin/python -m glassrecon reconstruct \
  --dataset DATASET --output NEW_MAP \
  --backend nvblox --model cached --prediction-dir PREDICTIONS \
  --pose-source file --pose-file DATASET/groundtruth.txt \
  --fusion-policy multiview --window-size 5 \
  --voxel-size 0.02 --sdf-trunc 0.08 \
  --live-view --keep-viewer --viewer-host 0.0.0.0 --viewer-port 8088
```

Provide `--intrinsics CAMERA.json` when the sequence differs from the default TUM
camera. Cached predictions must cover every selected RGB frame. Experiment caches omit
held-out frames; use the comparison rerun command below for those caches.

## Repeat an existing comparison

This reuses the exact original DA3 arrays, RANSAC fits, optional window-depth cache,
training/held-out split, and ground-truth poses, and writes a new output directory:

```bash
.venv-nvblox/bin/python -m experiments.rerun_backend \
  outputs/comparison_multiview_fr1_xyz \
  --backend nvblox --output outputs/comparison_nvblox_fr1_xyz
.venv-nvblox/bin/python -m experiments.rerun_backend \
  outputs/comparison_multiview_icl_kt2 \
  --backend nvblox --output outputs/comparison_nvblox_icl_kt2
.venv-nvblox/bin/python -m experiments.view_comparison \
  outputs/comparison_nvblox_fr1_xyz outputs/comparison_nvblox_icl_kt2 \
  outputs/comparison_multiview_fr1_xyz outputs/comparison_multiview_icl_kt2 \
  --host 0.0.0.0 --port 8088
```

New `experiments.run_comparison` runs also accept `--backend nvblox`.
`config.json` records backend/library/runtime versions, GPU, voxel size, truncation in
voxels, integration distance, weight mode, and maximum weight. `backend_rerun.json`
records the reused sources. Compare ICL surface metrics against its independent mesh;
TUM held-out depth errors are sensor consistency, not independent geometry accuracy.
Rendered common support is shared by methods within a run, not across backend runs.

## Conventions and limits

- Depth input is registered optical-axis Z in **meters**, not ray distance. Nonfinite
  or out-of-range values become zero; all-invalid frames fail before integration.
- nvblox consumes camera-to-world matrices directly. Only Open3D integration inverts
  these poses. The adapter validates rigid transforms and matching image dimensions.
- `sdf_trunc / voxel_size` converts the physical TSDF truncation distance to nvblox
  voxels. The backend retains and records nvblox's default weight mode and weight cap;
  matching physical voxel/truncation sizes does not make Open3D's estimator identical.
- Registered uint8 RGB is fused with a valid-depth color mask. CUDA operations finish
  before frame buffers are released. No asynchronous throughput benchmark is claimed.
- `mesh.ply` comes from nvblox's color mesh. `pointcloud.ply` and live Viser updates use
  that mesh's vertices and colors. Open3D handles mesh/point-cloud serialization and
  evaluation, but performs no TSDF integration in this backend.
- The GPU stores the cumulative scene volume. A sliding depth-recovery window does not
  bound map memory. This adapter does not add ESDF generation or dynamic-object removal.

Run `.venv-nvblox/bin/python -m pytest -q tests/test_nvblox.py` to check real CUDA fusion,
world-coordinate geometry, RGB, invalid depth, CLI integration, and window flushing.
