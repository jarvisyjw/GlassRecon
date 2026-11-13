# nvblox backend validation

Completed on 2026-11-13 using the official nvblox release wheel, actual internal version
`0.0.10.dev1+cu12ubuntu24`, commit `e4a2b00f45ecf989682d72a33a2c29defe39fd2d`.
Runtime: PyTorch 2.9.1/CUDA 12.8, NVIDIA RTX PRO 5000 Blackwell.

The TUM Freiburg1 XYZ run fuses 630 frames and holds out 158. ICL-NUIM kt2 fuses 704
frames and holds out 176. Each rerun preserves the corresponding completed Open3D run's
DA3-LARGE predictions, RANSAC fits, window-depth cache, frame selection, intrinsics,
and supplied ground-truth poses. `validation.json` records checks of every frame's
recovery fits and pose, every exported trajectory, and identical pre-fusion clean-depth
metrics on ICL. TUM has no independent clean-depth reference for that check.

Both backends use 2 cm voxels and 8 cm truncation. nvblox retains its default inverse
square weighting and maximum weight of 5; this is a comparison of configured backends,
not identical TSDF update equations. Open3D handles odometry when requested and mesh
I/O/evaluation; the new runs' TSDF/color fusion and mesh extraction are performed by
nvblox on the GPU. No loop closure or new pose estimator is introduced.

ICL kt2 independent mesh evaluation (lower surface error is better):

| Fusion depth | Open3D surface error, cm | nvblox surface error, cm | Open3D F-score at 5 cm, % | nvblox F-score at 5 cm, % |
|---|---:|---:|---:|---:|
| Noisy sensor | 1.115 | 1.198 | 99.74 | 99.62 |
| Regional RANSAC | 3.043 | 3.104 | 90.12 | 89.13 |
| Sliding-window recovery | 3.316 | 3.314 | 86.29 | 85.61 |
| Clean-depth oracle | 0.240 | 0.418 | 99.70 | 99.69 |

Switching fusion backends alone does not close the recovered-depth accuracy gap on
this sequence. These are individual diagnostic runs, not a multi-seed paper benchmark
or a controlled runtime benchmark. TUM depth errors measure held-out sensor consistency.
Rendered common support is computed within each backend run; it is not the same pixel
intersection across backends. ICL surface metrics use the same independent mesh and
observed reference points, without ICP or fitted scale correction.

Validation: 3 real-CUDA tests (world coordinates, RGB, invalid depths, per-frame CLI,
and sliding-window flushing), plus all 56 existing tests in the new environment.
Ruff lint and formatting checks pass. Interactive browser inspection was unavailable
on this host; HTTP/WebSocket health and saved scientific figures were checked separately.

Run instructions: [nvblox setup and backend reruns](../../../docs/nvblox.md).
The generated depth gallery and quantitative report are in `outputs/nvblox_visuals`.
Use `experiments.view_comparison` with the two `comparison_nvblox_*` directories and
their two `comparison_multiview_*` baselines to inspect the meshes in Viser.
