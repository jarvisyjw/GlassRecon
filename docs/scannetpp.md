# Reconstruct the three ScanNet++ v2 scenes

Use the registered iPhone RGB images, sensor depth and COLMAP camera poses to run DA3-LARGE, metric alignment and nvblox. Each row counts all iPhone frames with a released COLMAP pose. [Interactive scene comparisons](https://jarvisyjw.github.io/GlassRecon/#reconstructions)

| Scene | ScanNet++ v2 ID | Frames in the complete pipeline |
| --- | --- | ---: |
| Apartment | `c0f5742640` | 586 |
| Conference room | `791a5c253d` | 694 |
| Office | `8a20d62ac0` | 369 |

## Get the inputs

Set up the DA3 and nvblox environments and run the commands from the GlassRecon repository root. [Environment setup](../README.md#environment-setup)

**Prepared scene archive.** If you received `scannetpp_rgbd_scenes.tar.gz` through authorized ScanNet++ access, save it under `data/` and unpack it. It contains all three scenes (422 MiB). The separate [Quick Start archive](../README.md#quick-start) provides the final metric depths for the 468-frame apartment demo.

```bash
mkdir -p data
echo '8a2a8117f472328dc8082e40cad5c1be09f919a2cc7d55848c2930f3d32f924a  data/scannetpp_rgbd_scenes.tar.gz' | sha256sum -c -
tar -xzf data/scannetpp_rgbd_scenes.tar.gz -C data
```

**Official ScanNet++ release.** Request access to v2 and download the three scene IDs above. For each scene, get `iphone/rgb.mkv`, `iphone/rgb_mask.mkv`, `iphone/depth.bin`, `iphone/pose_intrinsic_imu.json` and `iphone/colmap/{cameras,images}.txt`. The official downloader extracts these under `data/<scene-id>/iphone/`. [ScanNet++ access](https://scannetpp.mlsg.cit.tum.de/scannetpp/) · [Data documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)

Set `SCANNETPP_ROOT` to the directory containing the official `data/` folder. For Conda, activate `glassrecon-nvblox` to use the installed `uv` command.

```bash
export SCANNETPP_ROOT=/absolute/path/to/scannetpp
```

If the scene already contains decoded `iphone/rgb/`, `iphone/rgb_masks/` and `iphone/depth/`, go directly to the conversion step. Otherwise install FFmpeg on your PATH and decode the three streams using the [official toolbox](https://github.com/scannetpp/scannetpp/tree/21ba1dc724e0d09c6a1f72b0d63944f5363eb675/iphone).

```bash
mkdir -p data
git clone https://github.com/scannetpp/scannetpp.git data/scannetpp-toolbox
(
  cd data/scannetpp-toolbox
  git checkout --quiet 21ba1dc724e0d09c6a1f72b0d63944f5363eb675
  cp iphone/configs/prepare_iphone_data.yml.template iphone/configs/prepare_iphone_data.yaml
  uv pip install --python "$NVBLOX_PYTHON" hydra-core munch lz4 imageio
  "$NVBLOX_PYTHON" -m iphone.prepare_iphone_data \
    data_root="$SCANNETPP_ROOT" 'scene_ids=[c0f5742640,791a5c253d,8a20d62ac0]' \
    +extract_only_rgb_in_colmap=false +output_root=null
)
```

Convert the decoded frames into registered 512 × 384 RGB-D sequences. The converter applies the official validity masks, preserves the sensor depth in millimeters and writes camera-to-world poses.

```bash
uv pip install --python "$NVBLOX_PYTHON" -e '.[scannetpp]'
for scene in c0f5742640 791a5c253d 8a20d62ac0; do
  "$NVBLOX_PYTHON" examples/prepare_scannetpp.py \
    --scene-dir "$SCANNETPP_ROOT/data/$scene" --output "data/scannetpp/$scene"
done
```

## Run a scene

Both input routes produce the same layout. Under `data/scannetpp/<scene-id>/`, `rgb/`, `depth/` and `masks/` share frame names. `rgb.txt` and `depth.txt` associate those files with timestamps. `intrinsics.json` describes the rectified camera, and `poses.txt` gives its camera-to-world trajectory.

```bash
scene=c0f5742640
sequence="data/scannetpp/$scene"
"$NVBLOX_PYTHON" examples/run_full_pipeline.py \
  --dataset "$sequence" --poses "$sequence/poses.txt" \
  --depth-units-per-meter 1000 --model-python "$DA3_PYTHON" \
  --output "outputs/$scene"
"$NVBLOX_PYTHON" -m glassrecon view --result "outputs/$scene/reconstruction" --port 8090
```

Change `scene` to run the conference room or office. The runner saves per-image DA3 depth in `predictions/`, the refined metric inputs in `reconstruction/metric_depths/`, and the colored point cloud and mesh in `reconstruction/{pointcloud,mesh}.ply`. The viewer opens at [localhost:8090](http://127.0.0.1:8090).
