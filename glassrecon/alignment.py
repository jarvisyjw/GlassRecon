"""NumPy implementation of the regional sampling method in alignment_functions.py.

Candidates are least-squares affine fits to samples from individual partitions.
The winner minimizes absolute error over ALL valid sensor correspondences, just
as in the original code. There is no inlier threshold or consensus refitting.
"""

from dataclasses import dataclass

import numpy as np


class AlignmentError(ValueError):
    """The measurements do not constrain a usable metric transformation."""


@dataclass(frozen=True)
class AlignmentConfig:
    method: str = "local"
    rows: int = 4
    cols: int = 4
    samples: int = 10
    iterations: int = 100
    min_depth: float = 0.1
    max_depth: float = 10.0

    def __post_init__(self):
        if self.method not in {"local", "global"}:
            raise ValueError("Alignment method must be local or global")
        if min(self.rows, self.cols, self.iterations) < 1 or self.samples < 2:
            raise ValueError("Partitions/iterations must be positive and samples must be >= 2")
        if not 0 < self.min_depth < self.max_depth < float("inf"):
            raise ValueError("Require 0 < min_depth < max_depth < infinity")


@dataclass(frozen=True)
class AlignmentResult:
    scale: float
    shift: float
    valid_points: int
    mean_absolute_error: float
    candidates: int
    fallback: bool
    representation: str


def _fit(x, y):
    # Centering avoids the cancellation in det(X.T @ X) for large offsets.
    centered = x - x.mean()
    variance = np.dot(centered, centered)
    tolerance = np.finfo(np.float64).eps * max(np.dot(x, x), 1.0)
    if variance <= tolerance:
        raise AlignmentError("Prediction has insufficient variation to fit scale and shift")
    scale = float(np.dot(centered, y - y.mean()) / variance)
    shift = float(y.mean() - scale * x.mean())
    if not np.isfinite([scale, shift]).all() or scale <= 0:
        raise AlignmentError("Recovered scale must be finite and positive")
    return scale, shift


def recover_metric_depth(prediction, sensor_depth, representation="depth", config=None, rng=None):
    """Return metric depth (meters, zero = invalid) and alignment diagnostics.

    Sensor depth must already be registered to prediction pixels and in meters.
    Predictions represent relative depth or relative inverse depth. No metric
    ground truth or glass segmentation is used. Invalid predictions stay invalid;
    sensor holes can be filled wherever a valid prediction exists.
    """
    config = config or AlignmentConfig()
    rng = rng if rng is not None else np.random.default_rng(0)
    if representation not in {"depth", "inverse_depth"}:
        raise ValueError("representation must be depth or inverse_depth")
    prediction = np.asarray(prediction, dtype=np.float64)
    sensor = np.asarray(sensor_depth, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape != sensor.shape:
        raise ValueError("Prediction and sensor depth must be matching H x W arrays")
    pred_valid = np.isfinite(prediction) & (prediction > 0)
    valid = (
        pred_valid & np.isfinite(sensor) & (sensor > config.min_depth) & (sensor < config.max_depth)
    )
    count = int(valid.sum())
    minimum = config.samples if config.method == "local" else 2
    if count < minimum:
        raise AlignmentError(f"Only {count} valid correspondences; need at least {minimum}")
    x = prediction[valid]
    y = sensor[valid]
    if representation == "inverse_depth":
        y = 1.0 / y

    best = None
    best_error = float("inf")
    candidates = 0
    fallback = False
    if config.method == "local":
        # Index into valid vectors without allocating a full image mask per trial.
        pixel_rows, pixel_cols = np.nonzero(valid)
        h, w = prediction.shape
        # Include remainder rows/columns, unlike the original floor-step loop.
        row_edges = np.linspace(0, h, config.rows + 1, dtype=int)
        col_edges = np.linspace(0, w, config.cols + 1, dtype=int)
        for r0, r1 in zip(row_edges[:-1], row_edges[1:]):
            for c0, c1 in zip(col_edges[:-1], col_edges[1:]):
                indices = np.flatnonzero(
                    (pixel_rows >= r0) & (pixel_rows < r1) & (pixel_cols >= c0) & (pixel_cols < c1)
                )
                if indices.size < config.samples:
                    continue
                for _ in range(config.iterations):
                    sample = rng.choice(indices, config.samples, replace=False)
                    try:
                        scale, shift = _fit(x[sample], y[sample])
                    except AlignmentError:
                        continue
                    candidates += 1
                    error = float(np.abs(scale * x + shift - y).mean())
                    if error < best_error:
                        best, best_error = (scale, shift), error
        fallback = best is None

    if best is None:
        best = _fit(x, y)
        best_error = float(np.abs(best[0] * x + best[1] - y).mean())
        candidates += 1

    scale, shift = best
    transformed = np.zeros_like(prediction)
    transformed[pred_valid] = scale * prediction[pred_valid] + shift
    output_valid = pred_valid & np.isfinite(transformed) & (transformed > 1e-12)
    metric = np.zeros_like(prediction)
    if representation == "inverse_depth":
        metric[output_valid] = 1.0 / transformed[output_valid]
    else:
        metric[output_valid] = transformed[output_valid]
    metric[(metric < config.min_depth) | (metric > config.max_depth) | ~np.isfinite(metric)] = 0
    if not np.any(metric > 0):
        raise AlignmentError("Recovered depth contains no valid metric pixels")
    return metric.astype(np.float32), AlignmentResult(
        scale, shift, count, best_error, candidates, fallback, representation
    )
