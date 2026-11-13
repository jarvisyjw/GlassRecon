"""Lazy model adapters. Cached predictions do not require PyTorch or a GPU."""

import json
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


def resize_prediction(depth, shape):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D prediction, got {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    clean = np.where(valid, depth, 0).astype(np.float32)
    if depth.shape == tuple(shape):
        return clean
    size = (shape[1], shape[0])
    # Normalize interpolation by valid support to avoid bleeding zero holes.
    weights = np.asarray(
        Image.fromarray(valid.astype(np.float32)).resize(size, Image.Resampling.BILINEAR)
    )
    values = np.asarray(Image.fromarray(clean).resize(size, Image.Resampling.BILINEAR))
    nearest = np.asarray(Image.fromarray(valid).resize(size, Image.Resampling.NEAREST))
    output = np.zeros(shape, dtype=np.float32)
    np.divide(values, weights, out=output, where=nearest & (weights > 1e-6))
    return output


class DepthPredictor(Protocol):
    representation: str
    model_id: str

    def predict(self, rgb: np.ndarray, key: str) -> np.ndarray: ...


def _device(torch, requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class DA3Predictor:
    representation = "depth"

    def __init__(self, model_id=None, device="auto", process_res=504):
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "Install the official Depth-Anything-3 package; see README setup"
            ) from exc
        self.torch = torch
        self.model_id = model_id or "depth-anything/DA3-LARGE"
        self.model = DepthAnything3.from_pretrained(self.model_id).to(_device(torch, device)).eval()
        self.process_res = process_res

    def predict(self, rgb, key):
        with self.torch.inference_mode():
            result = self.model.inference(
                [rgb], process_res=self.process_res, process_res_method="upper_bound_resize"
            )
        return resize_prediction(result.depth[0], rgb.shape[:2])


class DA2Predictor:
    representation = "inverse_depth"

    def __init__(self, model_id=None, device="auto"):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise ImportError("Install the DA2 dependencies: pip install -e '.[da2]'") from exc
        self.torch = torch
        self.device = _device(torch, device)
        self.model_id = model_id or "depth-anything/Depth-Anything-V2-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device).eval()
        )

    def predict(self, rgb, key):
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs).predicted_depth[0].float().cpu().numpy()
        return resize_prediction(output, rgb.shape[:2])


class CachedPredictor:
    def __init__(self, folder, representation="auto"):
        self.folder = Path(folder)
        path = self.folder / "metadata.json"
        metadata = json.loads(path.read_text()) if path.exists() else {}
        recorded = metadata.get("representation")
        if representation == "auto":
            representation = recorded
        elif recorded is not None and representation != recorded:
            raise ValueError("Requested prediction space disagrees with cache metadata")
        if representation not in {"depth", "inverse_depth"}:
            raise ValueError("External predictions need --prediction-space depth or inverse_depth")
        self.representation = representation
        self.model_id = metadata.get("model_id", "external")

    def predict(self, rgb, key):
        depth = np.load(self.folder / f"{key}.npy", allow_pickle=False)
        return resize_prediction(depth, rgb.shape[:2])


def create_predictor(
    model,
    model_id=None,
    device="auto",
    prediction_dir=None,
    prediction_space="auto",
    process_res=504,
):
    if model == "cached":
        if prediction_dir is None:
            raise ValueError("--model cached requires --prediction-dir")
        return CachedPredictor(prediction_dir, prediction_space)
    if prediction_dir is not None or prediction_space != "auto":
        raise ValueError("--prediction-dir/--prediction-space apply only to --model cached")
    if model == "da3":
        return DA3Predictor(model_id, device, process_res)
    if model == "da2":
        return DA2Predictor(model_id, device)
    raise ValueError(f"Unknown depth model: {model}")


def write_prediction_metadata(folder, predictor, model):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": model,
                "model_id": predictor.model_id,
                "representation": predictor.representation,
                "resolution": "original RGB image size",
            },
            indent=2,
        )
        + "\n"
    )
