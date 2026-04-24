# GlassRecon
Dataset with detailed annotated glass depth maps in indoor scenes.
🚀 The dataset and related code will be released soon~

## Dataset Structure
The dataset is organized as follows:
```
GlassRecon/
├── images/                  # RGB images (PNG)
├── intrinsics/              # JSON files with camera intrinsics & depth scale
├── masks/                   # Binary masks for glass regions (PNG)
├── sensor_depths/           # Raw sensor depth maps (PNG)
├── completed_depths/        # Completed depth maps (NPY)
└── evaluation_depths/
    ├── depths_npy/          # Filtered depth maps (NPY) – glass regions that could not be completed are masked out
    ├── depths_vis/          # Visualizations (PNG)
    └── pointclouds/         # 3D point clouds (PLY)
```

## Evaluation
Use `eval.py` to compute metrics (AbsRel, δ < 1.25) between predicted depth maps and ground truth.
```bash
python eval.py --image-folder IMAGE_PATH \
               --pred-folder PREDICTED_DEPTH_PATH \
               --sensor-depth-folder SENSOR_DEPTH_PATH \
               --gt-depth-folder GT_DEPTH_PATH \
               --depth-scale DEPTH_SCALE \
               --outdir OUTPUT_PATH
```

If the predicted depth maps represent inverse depth, add the `--inverse-depth` flag:
```
python eval.py ... --inverse-depth
```
