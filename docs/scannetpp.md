# ScanNet++ v2 reconstruction experiment

The local dataset root is `/media/jingwen-orchtech/JWDATA2/scannetppv2/data`.
It contains another `data/` directory with scene folders. Source files are read only;
prepared inputs go under `data/scannetpp/` and results under `outputs/scannetpp/`.

The starting scenes are `0a76e06478` (bedroom / hotel), `fd361ab85f` (office), and
`c0f5742640` (apartment). The last scene was previously described as a kitchen;
the released `metadata/scene_types.json` labels it apartment. All three belong to
`nvs_sem_train`. The optional expansion scenes are not part of this run.

## Protocol

- Use all released iPhone COLMAP images, sorted by frame name: 962, 624, and 586
  frames respectively. These are a subsampled, filtered subset of the original
  60 Hz capture. The official release filters COLMAP frames using sensor/laser
  agreement; this selection must be acknowledged in interpretation.
- Invert COLMAP world-to-camera extrinsics to OpenCV camera-to-world transforms.
  Use its mesh-aligned metric coordinate system. Do not mix raw ARKit poses with
  COLMAP calibration. No ICP, fitted scene scale, or ground-truth pose optimization
  is performed by this experiment. This evaluates depth/fusion with supplied poses,
  not estimated odometry or SLAM.
- Rectify RGB and depth with the supplied OPENCV distortion parameters to a
  512×384 pinhole camera. RGB is bilinearly sampled; measured 256×192 depth uses
  nearest-neighbor sampling in its corresponding scaled RGB coordinates. It stays
  in millimeters (divisor 1000). This resampling does not create new depth detail.
- Exclude anonymized image pixels and rectification borders from every fused depth,
  including model completion. Remove anonymized vertices from the reference mesh.
- Hold out every fifth prepared frame from inference, alignment, and fusion.
  Predict DA3-LARGE relative depth independently for the remaining RGB images,
  without providing poses or sensor depth to the network.
- Compare raw sensor, global fit, regional recovery, sensor-anchor window refit,
  and five-frame multiview recovery. Use fixed seed 0, 100 trials per partition,
  4×4 regions, 10 samples, depth limits 0.1–10 m, nvblox voxel size 2 cm, and
  truncation 8 cm. A separately labeled laser-depth oracle is also reconstructed.
- If the global least-squares baseline produces a nonpositive or otherwise invalid
  scale, its depth is empty for that frame (`--global-failure empty`). The failure
  is recorded in `frames.jsonl`; raw and regional/window methods retain the frame.
  This is an explicit abstention, not an identity-scale or sensor-depth fallback.
  Regional recovery failures still stop the run. The initial office attempt stopped
  at such a global failure and is retained as `baseline_failed_global_scale/`.
- Laser-mesh rendered depth is used only for evaluation and that oracle, never
  as scale-recovery or window-optimization input. Report held-out depth errors,
  rendering coverage, area-uniform mesh accuracy, and visible-reference completeness.
  Completeness is over laser points visible from sampled fusion views, not the entire
  laser scan. Surface sampling uses 100,000 points and seed 0.

Camera validation includes direct sensor-versus-laser depth checks. The mesh-rendered
reference coordinate check is also recorded, but is a consistency check of the
rendering implementation, not independent proof of correct calibration.

Full-mesh distances also penalize geometry outside the laser mesh's coverage, including
possible views through windows. Large mean distance alone does not establish that all
distant surfaces are erroneous. The report preserves the full mesh and includes median,
95th-percentile, and >50 cm tail statistics alongside held-out laser depth error,
completeness, and coverage. These are single-seed development results.

## Commands

Run from the repository root. Preparation needs OpenCV, Open3D, Pillow, and NumPy.
Install the adapter extra with `pip install -e '.[scannetpp,experiments]'` in the
nvblox environment; see [nvblox setup](nvblox.md). OpenCV is kept below 4.12 for
compatibility with the nvblox runtime’s NumPy 1.x constraint.
Use a separate DA3 environment if its PyTorch requirements differ from nvblox.
The model checkpoint must already be cached because the runner enables offline mode.

```bash
HF_HOME=/tmp/glassrecon-hf-cache \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv-nvblox/bin/python -u -m experiments.run_scannetpp \
  --root /media/jingwen-orchtech/JWDATA2/scannetppv2/data \
  --model-python /tmp/glassrecon-model-cuda/bin/python

# Small contiguous smoke test, in separate destinations:
.venv-nvblox/bin/python -m experiments.run_scannetpp \
  --scenes 0a76e06478 --max-frames 12 \
  --prepared-root data/scannetpp_smoke --output-root outputs/scannetpp_smoke

.venv-nvblox/bin/python -m experiments.report_scannetpp
```

The local `/tmp` model environment and cache paths are machine-specific; substitute
your own DA3 Python and `HF_HOME`. Completed stages can be reused. Partial output
folders are preserved and rejected rather than overwritten; choose fresh output
roots after investigating a failed stage. Prepared frame caps must match a rerun.

The report is `outputs/scannetpp_report/index.html`; compact metrics, configurations,
input hashes, and audit receipts are in `experiments/results/scannetpp_v2/`.
Each scene contains `baseline/`, `multiview/`, `multiview_depths/`, and `predictions/`.
Meshes are under each comparison's method folders. `progress.json` and per-stage
logs record active processing.

The comparison viewer currently keys scenes by the final directory name. Create
unique aliases before serving these nested result directories:

```bash
ln -s 0a76e06478/multiview outputs/scannetpp/bedroom_0a76e06478
ln -s fd361ab85f/multiview outputs/scannetpp/office_fd361ab85f
ln -s c0f5742640/multiview outputs/scannetpp/apartment_c0f5742640
.venv-nvblox/bin/python -m experiments.view_comparison \
  outputs/scannetpp/bedroom_0a76e06478 \
  outputs/scannetpp/office_fd361ab85f \
  outputs/scannetpp/apartment_c0f5742640 --host 127.0.0.1 --port 8092
.venv-nvblox/bin/python -m http.server 8093 --bind 127.0.0.1 \
  --directory outputs/scannetpp_report
```

See the [official data documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)
for the RGB/depth registration, millimeter encoding, aligned COLMAP poses,
anonymization masks, and released frame-filtering convention.

For remote access, run this on your own computer, then open
`http://localhost:8092` (Viser) and `http://localhost:8093` (report):

```bash
ssh -N -L 8092:127.0.0.1:8092 -L 8093:127.0.0.1:8093 \
  jingwen-orchtech@192.168.8.225
```

## Recorded run: 2026-11-16

All 2,172 released COLMAP frames were processed: 1,736 fusion frames and 436 held
out. DA3-LARGE revision `c54c26b16ec04d218e8d584ecf4bce082a9fcc20` was loaded from
the local cache. All 63 tests passed, including coordinate conversion, depth units,
privacy-mask propagation, and explicit global-fit abstention. Both localhost
servers passed HTTP checks; Viser delivered all six mesh variants over WebSocket.

F-score at 5 cm against the laser reference (%, higher is better):

| Scene | Fused / held out | Raw sensor | Regional | Sliding window |
| --- | ---: | ---: | ---: | ---: |
| Bedroom `0a76e06478` | 769 / 193 | 77.31 | 72.88 | 73.27 |
| Office `fd361ab85f` | 499 / 125 | 58.52 | 65.48 | 68.06 |
| Apartment `c0f5742640` | 468 / 118 | 40.90 | 66.35 | 68.21 |

Raw sensor / window held-out laser-depth MAE on shared rendered support is
5.31 / 9.52 cm (bedroom), 8.90 / 13.95 cm (office), and 8.53 / 8.89 cm (apartment).
Raw sensor also has lower visible-reference completeness distance in each scene.
The window reduces full-mesh mean distance and improves F-score versus regional
recovery in all three, but this is not a uniform improvement over raw sensor.
The sensor-anchor-only control is close to the window, and has lower held-out
error than the window in the office. Do not attribute all refitting gains to
cross-view constraints.

Eight office frames and one apartment frame had invalid global fits. Only that
optional baseline abstained; raw, regional, and window methods used every selected
fusion frame. Audits verify identical selections and trajectories, cache exclusion
of held-out frames, and exactly one window emission per selected frame.

[Full metrics](../experiments/results/scannetpp_v2/metrics.csv),
[audit](../experiments/results/scannetpp_v2/audit.json), and
[validation receipt](../experiments/results/scannetpp_v2/validation.json) are saved.

## Glass-scene follow-up batch

The visually selected follow-up scenes are `08bd80ce2a` (lounge), `1b379f1114`
(lounge), and `33f78338cd` (apartment). They were selected before inspecting their
reconstruction results. Their protocol is identical to the first batch, including
supplied poses, every-fifth-frame holdout, seed 0, DA3-LARGE, and nvblox settings.
The adapter accepts bare frame names and known `iphone/` or `video/` prefixes,
rejecting any collision after normalization. It does not rotate stored images.

```bash
.venv-nvblox/bin/python -m experiments.run_scannetpp \
  --scenes 08bd80ce2a 1b379f1114 33f78338cd
.venv-nvblox/bin/python -m experiments.report_scannetpp \
  --scenes 0a76e06478 fd361ab85f c0f5742640 08bd80ce2a 1b379f1114 33f78338cd
```

The [frozen batch protocol](../experiments/results/scannetpp_v2/extension_protocol.json)
records selection provenance and code hashes. All requested scenes are reported,
including unfavorable outcomes.

All three follow-up runs completed: **2,229 fused frames and 559 held-out frames**.
All 67 tests passed. The prediction cache, checkpoint revision, frame selection,
identical trajectories, window emission order, and actual integration counts were
audited. One global-baseline fit in `08bd80ce2a` abstained; all raw, regional, and
window frames were integrated. There was no scene-specific retuning or exclusion.

| Follow-up scene | Fused / held out | Raw F-score @ 5 cm | Regional F-score | Window F-score | Raw / window held-out MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `08bd80ce2a` (Lounge A) | 1,041 / 261 | 55.96% | 55.69% | 63.15% | 11.46 / 12.49 cm |
| `1b379f1114` (Lounge B) | 528 / 132 | 46.80% | 23.69% | 27.22% | 11.12 / 32.09 cm |
| `33f78338cd` (Apartment B) | 660 / 166 | 71.08% | 73.35% | 74.83% | 7.61 / 9.99 cm |

The window improves F-score over raw fusion in two follow-up scenes, but substantially
worsens it in `1b379f1114`. Raw depth retains lower held-out laser-depth MAE and
better visible-reference completeness in all three. Window recovery improves over
per-frame regional recovery in each, but visible glazing alone does not predict an
improvement over raw sensor fusion. The original apartment remains a selected
showcase, not a representative average or a general glass-reconstruction guarantee.

At completion of that batch, the combined report included six scenes. The saved
validation below records that six-scene state; the newer batch is documented next. Both servers still bind to
localhost on ports 8092 and 8093, so the existing SSH tunnel continues to work.
The [extension validation receipt](../experiments/results/scannetpp_v2/extension_validation.json)
records the six-scene viewer check and the follow-up audits.


## Second glass-scene follow-up batch

All five user-approved scenes from the second RGB search completed using the same
fixed settings: **2,229 fused frames and 559 held-out frames**. This coincidentally
matches the totals of the preceding three-scene batch. Across all eleven scenes,
the report now covers **6,194 fusion frames and 1,554 held-out frames**.

The [frozen protocol](../experiments/results/scannetpp_v2/extension_batch2_protocol.json)
records the pre-run shortlist hash. The lounge initially exhausted GPU memory
while exporting the sixth mesh. The export loop now releases each completed
mapper immediately, avoiding simultaneous retention of all GPU meshes and TSDF
volumes. Its multi-view stage was rerun with unchanged data and parameters;
the baseline/prediction cache was reused. The failed attempt is retained at
`outputs/scannetpp/7f68c514bd/failed_export_attempt`.
The [retry record](../experiments/results/scannetpp_v2/extension_batch2_retry.json)
records this implementation change. All 67 tests passed after the fix.

| Scene | Fused / held out | Raw F-score @ 5 cm | Regional | Window | Raw / window held-out MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `7f68c514bd` (Lounge C) | 775 / 194 | 56.32% | 54.40% | 56.56% | 11.39 / 14.76 cm |
| `511061232` (Winter garden) | 452 / 113 | 32.02% | 26.33% | 29.83% | 15.58 / 20.54 cm |
| `791a5c253d` (Conference) | 555 / 139 | 47.52% | 62.04% | 66.38% | 17.72 / 11.65 cm |
| `45ba9ea301` (Kitchen) | 152 / 39 | 69.30% | 65.90% | 68.24% | 3.43 / 4.41 cm |
| `8a20d62ac0` (Office B) | 295 / 74 | 77.63% | 89.28% | 89.25% | 5.21 / 6.24 cm |

Window F-score improves over raw fusion in three of five scenes, but the lounge
gain is only 0.25 percentage points. The conference room is the strongest new
result: +18.86 points, with lower held-out depth MAE and better completeness too.
The office gains 11.62 F-score points, but raw depth retains lower held-out MAE
and better completeness; its regional result is also marginally higher than the
window result. The winter garden and kitchen have lower window F-score than raw.

Six invalid global fits in the kitchen abstained only in the global baseline
(146 integrated versus 152 selected frames). All other methods integrated every
selected frame; there were no global abstentions in the other four scenes.
The final report audited identical poses/selections, prediction exclusion of held-out
frames, window emission order, and actual integration counts. These remain
full-scene laser-reference results, not glass-only accuracy measurements.
The original featured apartment remains the README headline.

The report and viewer include all eleven scenes, with the new conference room
initially selected in Viser. Existing SSH forwarding is unchanged:
[Viser](http://localhost:8092/) and [report](http://localhost:8093/).
The [new candidate gallery](http://localhost:8093/glass_candidates/batch2/)
preserves the original pre-run visual selection and links to completed results.

Reproduce the new runs and combined report:

```bash
.venv-nvblox/bin/python -m experiments.run_scannetpp \
  --scenes 7f68c514bd 511061232 791a5c253d 45ba9ea301 8a20d62ac0
.venv-nvblox/bin/python -m experiments.report_scannetpp \
  --scenes 0a76e06478 fd361ab85f c0f5742640 08bd80ce2a 1b379f1114 33f78338cd \
  7f68c514bd 511061232 791a5c253d 45ba9ea301 8a20d62ac0
```
