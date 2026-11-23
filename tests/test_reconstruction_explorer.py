import json

import numpy as np
import pytest
from PIL import Image

from experiments import render_reconstruction_explorer as explorer


def test_depth_views_share_crop_units_range_and_mask(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    scene = "8a20d62ac0"
    frame = explorer.VIEWS[scene][2]
    prepared = tmp_path / "data" / scene
    run = tmp_path / "outputs" / scene / "multiview"
    for folder in (prepared / "depth", prepared / "masks", run):
        folder.mkdir(parents=True)
    (prepared / "intrinsics.json").write_text(
        json.dumps(dict(width=3, height=4, fx=2, fy=2, cx=1, cy=1.5))
    )
    (run / "config.json").write_text(
        json.dumps(dict(depth_units_per_meter=1000, min_depth=0.1, max_depth=10))
    )
    raw = np.zeros((4, 3), dtype=np.uint16)
    raw[1:3] = [[1500, 0, 6000], [6500, 1000, 11000]]
    refined = np.zeros((4, 3), dtype=np.float32)
    refined[1:3] = [[1.5, np.nan, 3.0], [7.5, 1.0, 0.05]]
    mask = np.full((4, 3), 255, dtype=np.uint8)
    mask[2, 1] = 0
    Image.fromarray(raw).save(prepared / "depth" / f"{frame}.png")
    Image.fromarray(mask).save(prepared / "masks" / f"{frame}.png")
    np.save(tmp_path / "refined.npy", refined)
    monkeypatch.setattr(explorer, "WIDTH", 3)
    monkeypatch.setattr(explorer, "HEIGHT", 2)
    output = tmp_path / "web"
    explorer.render_input_depths(
        scene,
        tmp_path / "data",
        tmp_path / "outputs",
        tmp_path / "refined.npy",
        output,
        explorer.depth_palette(),
    )
    black, quarter, middle, far = (0, 0, 0), (59, 82, 139), (33, 145, 140), (253, 231, 37)
    for name, window_depth in (("raw", far), ("ours", middle)):
        with Image.open(output / scene / f"depth-{name}.webp") as image:
            expected = [[quarter, black, window_depth], [far, black, black]]
            np.testing.assert_array_equal(image, expected)


def test_depth_lookup_uses_exact_frame_and_prefers_fusion_cache(tmp_path):
    scene = "8a20d62ac0"
    cache, extra = tmp_path / "results", tmp_path / "supplemental"
    supplemental_frame = extra / scene / "frame_002450.npy"
    supplemental_frame.parent.mkdir(parents=True)
    (supplemental_frame.parent / "frame_002460.npy").touch()
    with pytest.raises(FileNotFoundError, match="8a20d62ac0/frame_002450"):
        explorer.refined_depth_path(scene, cache, extra)
    supplemental_frame.touch()
    assert explorer.refined_depth_path(scene, cache, extra) == supplemental_frame
    cached_frame = cache / scene / "multiview_depths" / "multiview" / "frame_002450.npy"
    cached_frame.parent.mkdir(parents=True)
    cached_frame.touch()
    assert explorer.refined_depth_path(scene, cache, extra) == cached_frame
