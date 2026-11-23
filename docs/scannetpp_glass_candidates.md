# More ScanNet++ scenes with substantial visible glazing

The featured result remains apartment **`c0f5742640`**. The candidates below were
selected by inspecting RGB, without running or inspecting new reconstruction scores.
The three primary scenes have now been evaluated; their selection did not imply
positive outcomes. See the [completed follow-up results](scannetpp.md#glass-scene-follow-up-batch).

## Selected follow-up scenes (completed)

| Priority | Scene | Released type | Visual reason | Representative frames | Released iPhone frames |
| --- | --- | --- | --- | --- | ---: |
| 1 | `08bd80ce2a` | Living room / lounge | Broad exterior window wall and glazed doors; closest visual match | `009260`, `014130` | 1,302 |
| 2 | `1b379f1114` | Living room / lounge | Large adjacent exterior panes; useful glass-facing views | `001750`, `003070` | 660 |
| 3 | `33f78338cd` | Apartment | Corner windows above the sofa; less glazing than the featured apartment | `003300`, `007180` | 826 |

Frame numbers denote `frame_NNNNNN.jpg`. These views contain prominent glazing;
we have **not measured glass pixel percentages**, and much of each full capture
looks at walls, floors, furniture, or ceilings. The second scene's stored RGB is
sideways; keep its image/calibration convention consistent during preparation.

![Two inspected views from each recommended scene](../experiments/results/scannetpp_v2/glass_candidates/shortlist.jpg)

For future controlled batches, use all released eligible frames with the existing
fixed parameters and holdout rule. The representative views are inspection aids,
not a license to restrict evaluation to favorable clips. Freeze the shortlist
before examining reconstruction outcomes and report every selected scene.

## Separate controls and reserve candidates

| Scene | Role | Why include it |
| --- | --- | --- |
| `1ae9e5d2a6` | Reflective/dark-exterior window control | Conference room with a window wall and strong reflections; 297 released frames |
| `0f25f24a4f` | Interior partition control | Glass/frosted partitions and doors, rather than broad exterior views; 2,775 frames |
| `07e6f56969` | Shower-glass control | Clear shower enclosure, a different material/background case; 454 frames |
| `0b0bb941b7` | Reserve | Tall glazed doors/windows at the end of a table-tennis room; only a small part of sampled views faces them; 766 frames |

All seven are in `nvs_sem_train`. Calibration, laser meshes, and five neighboring
released RGB/depth/mask frames around each representative view were checked.
`1b379f1114`, `07e6f56969`, and `0b0bb941b7` use `iphone/frame_*.jpg` names in
COLMAP while decoded files live under `iphone/rgb/frame_*.jpg`. The initial
screening note incorrectly called this a `video/` prefix. The preparation adapter
now normalizes both known prefixes with collision checks, preserving calibration
and poses. The other four use the single-OPENCV-camera/bare-name layout.
The requested follow-up run covers `08bd80ce2a`, `1b379f1114`, and `33f78338cd`;
its fixed settings are recorded in the
[extension protocol](../experiments/results/scannetpp_v2/extension_protocol.json).

## Screening scope and evidence

- There are 1,006 local scene directories. This first pass screened **200 eligible
  scenes in sorted scene-ID order**, encountering 20 excluded directories before
  reaching that limit. It is not an exhaustive dataset search.
- Eligibility required iPhone RGB/depth/mask directories, COLMAP images, semantic
  and reference meshes, and exclusion of NVS test scenes. Per-frame availability
  was checked for the shortlisted clips, not every file in all 200 scenes.
- Ranking used the fraction of mesh vertices labeled architectural glazing
  (windows, panes, glass walls/doors, shower glass). This is **not physical glass
  area or RGB coverage**; mesh tessellation and missing laser returns affect it.
- We reviewed six uniformly spaced RGB frames for the top 22 non-tested ranked
  scenes plus three earlier expansion suggestions, then 16 uniformly spaced
  views in seven promising scenes. Anonymization masks were applied to all figures.
- The highest mesh-proxy scene, `0c7962bd64`, was deprioritized after RGB inspection:
  it shows a small hallway/bathroom rather than a broad glass wall. `2162a7cf14`
  has substantial slatted-blind occlusion. The earlier expansion scenes
  `0a7cc12c0e`, `8a35ef3cfe`, and `4c5c60fa76` are lower priorities for this specific
  glass-coverage request based on their six sampled views.

The [shortlist manifest](../experiments/results/scannetpp_v2/glass_candidates/shortlist.json)
records observations, representative frames, neighboring clips, and adapter notes.
The [full screening](../experiments/results/scannetpp_v2/glass_scene_screening.json),
[initial review manifest](../experiments/results/scannetpp_v2/glass_candidates/review_manifest.json),
and [denser review manifest](../experiments/results/scannetpp_v2/glass_candidates/focus_manifest.json)
make the selection traceable. Contact sheets and focused views are saved alongside them.

With the existing report server and SSH forwarding, open the
[candidate gallery](http://localhost:8093/glass_candidates/). It includes both
recommended scenes and rejected/lower-priority candidates.

Reproduce the initial screening and contact sheets from the repository root:

```bash
.venv-nvblox/bin/python -m experiments.discover_glass_scenes --limit 200
.venv-nvblox/bin/python -m experiments.glass_candidate_gallery
```

Use `--limit 0` for an exhaustive semantic screening pass; this reads many large
meshes from the external disk. A broader search may reveal better matches, especially
where glass is missing or mislabeled in laser geometry.


## Second search: 300 additional scenes

Screened the next 300 eligible scenes, excluding the first screening's scene IDs
and exclusions: **500 distinct eligible scenes screened so far** out of 1,006 local
directories. This remains a partial search. Reviewed six uniformly spaced RGB
frames for the top 30 new candidates, then 16 frames each for six promising scenes.
Selection used RGB appearance only. All five recommended candidates have now been
reconstructed; see the [completed results](scannetpp.md#second-glass-scene-follow-up-batch).

| Priority | Scene | Visible glazing | Released frames |
| --- | --- | --- | ---: |
| 1 | `7f68c514bd` | Living/dining room with large balcony sliding glass doors; closest domestic match | 969 |
| 2 | `511061232` | Winter garden with broad glazed roof and side glazing; many vertical panes have blinds | 565 |
| 3 | `791a5c253d` | Conference room with a wide exterior window bank and partial blinds | 694 |
| Reserve | `45ba9ea301` | Compact kitchen with tall courtyard-facing glazing and close views | 191 |
| Reserve | `8a20d62ac0` | Office with a broad clear window bank behind a desk | 369 |

All five are in `nvs_sem_train`. The current adapter successfully parsed their
COLMAP camera/pose files with one camera each. RGB/depth/mask dimensions and
timestamps were checked for neighboring released frames around both representative
views; reference and anonymization mesh files are present. This is an input
preflight, not a full pipeline run. The nine-character ID `511061232` is the
actual local dataset directory name.

The high-ranked apartment `3db0a1c8f3` was deprioritized: dense inspection revealed
a translucent glass-block partition and shuttered windows. These recommendations
do not imply measured glass pixel coverage or superior reconstruction scores.

[Open the new gallery through the existing SSH tunnel](http://localhost:8093/glass_candidates/batch2/).
[Selection and preflight manifest](../experiments/results/scannetpp_v2/glass_candidates_batch2/shortlist.json).
[Screening evidence](../experiments/results/scannetpp_v2/glass_scene_screening_batch2.json).
Contact sheets and the six detailed sheets are saved beside the selection manifest.

Reproduce this screening and initial visual review:

```bash
.venv-nvblox/bin/python -m experiments.discover_glass_scenes --limit 300 \
  --exclude-screening experiments/results/scannetpp_v2/glass_scene_screening.json \
  --output experiments/results/scannetpp_v2/glass_scene_screening_batch2.json
.venv-nvblox/bin/python -m experiments.glass_candidate_gallery \
  --screening experiments/results/scannetpp_v2/glass_scene_screening_batch2.json \
  --output outputs/scannetpp_report/glass_candidates/batch2 --top 30 --no-earlier
```
