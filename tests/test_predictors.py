import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from glassrecon.predictors import CachedPredictor, DA3Predictor, resize_prediction


def test_cache_requires_explicit_depth_convention(tmp_path):
    with pytest.raises(ValueError, match="prediction-space"):
        CachedPredictor(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({"representation": "inverse_depth"}))
    np.save(tmp_path / "frame.npy", np.ones((4, 6)))
    predictor = CachedPredictor(tmp_path)
    assert predictor.representation == "inverse_depth"
    assert predictor.predict(np.zeros((8, 12, 3)), "frame").shape == (8, 12)
    with pytest.raises(ValueError, match="disagrees"):
        CachedPredictor(tmp_path, "depth")


def test_resizing_preserves_invalid_regions_and_valid_values():
    depth = np.full((4, 4), 2, dtype=np.float32)
    depth[:2, :2] = np.nan
    result = resize_prediction(depth, (8, 8))
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[:4, :4], 0)
    np.testing.assert_allclose(result[4:, 4:], 2)


def test_da3_adapter_passes_rgb_and_uses_resize_without_crop(monkeypatch):
    calls = {}

    class Model:
        @classmethod
        def from_pretrained(cls, model_id):
            calls["model_id"] = model_id
            return cls()

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            return self

        def inference(self, images, **kwargs):
            calls.update(kwargs)
            calls["image"] = images[0]
            return SimpleNamespace(depth=np.ones((1, 4, 6), dtype=np.float32))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", SimpleNamespace(DepthAnything3=Model))
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    predictor = DA3Predictor(device="cpu")
    assert predictor.predict(rgb, "frame").shape == (8, 12)
    assert calls["image"] is rgb
    assert calls["process_res_method"] == "upper_bound_resize"
    assert calls["model_id"] == "depth-anything/DA3-LARGE"
