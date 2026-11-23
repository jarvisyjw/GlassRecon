# Reproducible reconstruction demo

This demo generates a small RGB-D sequence, recovers metric depth from cached
relative predictions, reconstructs it with Open3D, and opens the result in Viser.
An optional second run uses nvblox with the same inputs and poses.

The sequence contains 12 frames of a textured, slanted plane, sensor holes, and
localized depth corruption. The relative predictions are generated analytically:
**this is a pipeline correctness demo, not DA3 inference or a real-scene accuracy
benchmark**. Supplied synthetic poses isolate recovery and fusion from tracking.
No dataset, model weights, or download token is needed; installing packages requires
network access. Run all commands from the repository root in Bash.

## 1. Install

Prerequisites: Python 3.12 and the `uv` package manager installed on your PATH.
Start from a checkout containing `glassrecon/` and `examples/`. The pipeline was
introduced in commit `d6e70a0`; ensure that commit or a later revision is present.

```bash
uv venv --python python3.12 .venv-demo
source .venv-demo/bin/activate
uv pip install -e '.[mapping,viewer]'
```

Open3D runs on the CPU. Record the exact checkout and installed versions with the
outputs, since dependency ranges may resolve differently over time:

```bash
mkdir -p outputs/demo
git rev-parse HEAD > outputs/demo/code_revision.txt
uv pip freeze > outputs/demo/environment.txt
```

Use new output paths for repeat runs. The generator and reconstruction commands
refuse nonempty destinations; do not mix artifacts from different runs.

## 2. Generate RGB, sensor depth, relative predictions, and poses

```bash
python - <<'PY'
from examples.make_synthetic_sequence import make_sequence
make_sequence("outputs/demo/input", frame_count=12)
PY
```

The input includes `rgb.txt`, `depth.txt`, camera intrinsics, camera-to-world
`groundtruth.txt`, and `predictions/metadata.json`. Images are 96 × 72 pixels.
Sensor PNGs use 5000 units/m; cached predictions use arbitrary relative-depth units.
The known transform is `metric_depth = 1.7 * prediction + 0.3` meters.

## 3. Reconstruct the sensor baseline and recovered depths

```bash
python -m glassrecon reconstruct \
  --dataset outputs/demo/input --intrinsics outputs/demo/input/intrinsics.json \
  --depth-source sensor --backend open3d \
  --pose-source file --pose-file outputs/demo/input/groundtruth.txt \
  --output outputs/demo/open3d_sensor

for policy in replace multiview; do
  python -m glassrecon reconstruct \
    --dataset outputs/demo/input --intrinsics outputs/demo/input/intrinsics.json \
    --model cached --prediction-dir outputs/demo/input/predictions \
    --backend open3d --fusion-policy "$policy" --window-size 5 \
    --pose-source file --pose-file outputs/demo/input/groundtruth.txt \
    --seed 0 --iterations 100 --voxel-size 0.02 --sdf-trunc 0.08 \
    --save-depths --output "outputs/demo/open3d_$policy"
done
```

Each map integrates all 12 frames into one cumulative volume. `replace` uses
per-frame regional recovery. `multiview` finalizes each frame once using a five-frame
window with four-frame latency; the remaining frames are flushed at the end.

Each result contains `mesh.ply`, `pointcloud.ply`, `trajectory.txt`, `config.json`,
`summary.json`, and `frames.jsonl`. Recovered runs also save `metric_depths/*.npy`
in meters. Backend and geometry conventions are recorded in `config.json`.

## 4. Check the result

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import open3d as o3d

for name in ["open3d_sensor", "open3d_replace", "open3d_multiview"]:
    root = Path("outputs/demo") / name
    summary = json.loads((root / "summary.json").read_text())
    assert summary["integrated"] == 12 and summary["skipped"] == 0
    mesh = o3d.io.read_triangle_mesh(str(root / "mesh.ply"))
    assert len(mesh.triangles) > 0
    if name != "open3d_sensor":
        points = np.asarray(mesh.vertices)
        residual = np.median(np.abs(points[:, 2] - 2 - 0.2 * points[:, 0]))
        assert residual < 0.02, (name, residual)
        assert len(list((root / "metric_depths").glob("*.npy"))) == 12
        depth = np.load(root / "metric_depths/1.000000.npy")
        assert np.all(depth[10:20, 10:25] > 0)
        print(name, "median plane residual (m):", residual)
    print(name, "OK: 12 integrated frames and a nonempty mesh")
PY
```

The plane is `z = 2 + 0.2*x` in world coordinates. The check requires a median
algebraic plane residual below 2 cm and completion of the known sensor hole.
This is a tolerance check, not a byte-for-byte mesh comparison or a real-data score.

## 5. Inspect the maps in Viser

```bash
python -m glassrecon view --result outputs/demo/open3d_replace --port 8090
```

Open `http://127.0.0.1:8090` on the reconstruction machine. Use the geometry selector
to inspect the mesh and point cloud. Stop with Ctrl+C, then change `--result` to
`outputs/demo/open3d_sensor` or `outputs/demo/open3d_multiview` to inspect the other maps.

For a remote machine, add `--host 0.0.0.0` and open `http://<server-ip>:8090` on
the same network, or forward the port over SSH. If the port is occupied, choose
another port. A saved-result viewer must remain running while you use its page.

## 6. Optional: repeat with nvblox

Install the separate environment using [the nvblox instructions](nvblox.md).
The documented wheel targets Ubuntu 24.04 x86_64 and requires an NVIDIA CUDA GPU.
Keep the synthetic input generated above; it does not depend on the Python environment.

```bash
for policy in replace multiview; do
  .venv-nvblox/bin/python -m glassrecon reconstruct \
    --dataset outputs/demo/input --intrinsics outputs/demo/input/intrinsics.json \
    --model cached --prediction-dir outputs/demo/input/predictions \
    --backend nvblox --fusion-policy "$policy" --window-size 5 \
    --pose-source file --pose-file outputs/demo/input/groundtruth.txt \
    --seed 0 --iterations 100 --voxel-size 0.02 --sdf-trunc 0.08 \
    --save-depths --output "outputs/demo/nvblox_$policy"
done

.venv-nvblox/bin/python -m glassrecon view \
  --result outputs/demo/nvblox_multiview --port 8091
```

Repeat the step 4 checks with `nvblox_replace` and `nvblox_multiview` as the names
and the nvblox Python interpreter. Both should integrate 12 frames and pass the
same plane/hole checks. The backends use different TSDF weighting and mesh extraction
implementations, so vertex counts and exact surfaces need not match.

## Moving to real data

For DA3-LARGE inference, install DA3 as described in the [environment setup](../README.md#environment-setup), generate
predictions with `python -m glassrecon predict`, and use registered real RGB-D data
with its calibration. Do not treat the synthetic predictions as DA3 output.
The controlled TUM/ICL experiments and their measured limitations are recorded in
[`experiments/results`](../experiments/results). A fresh DA3 installation and a
ScanNet++ adapter are separate from this demo.

## Validation record

The generation, reconstruction, and verification blocks above were executed against
a temporary clean checkout of `d6e70a0`. The Open3D path used a new Python 3.12
environment installed from cached packages; nvblox reused the existing CUDA
environment. All five maps integrated 12 frames without skips. The four recovered
maps passed the plane and hole checks. Both documented viewer commands passed
HTTP and WebSocket scene-delivery checks; browser interaction was not inspected.
See the [validation receipt](../experiments/results/synthetic_demo/validation.json).
This validation does not establish fresh-install portability of the CUDA environment.
