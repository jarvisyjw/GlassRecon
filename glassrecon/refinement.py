"""Robust metric refinement and sensor-preserving, temporally checked depth completion.

Thresholds are fixed sensor-space heuristics, not learned confidence probabilities.
Only observed input depths enter refinement/history; clean evaluation depth never does.
"""

from dataclasses import dataclass

import numpy as np

from .alignment import AlignmentError, AlignmentResult


def shifted(array, dy, dx):
    """Zero-padded neighbor view; never wrap opposite image borders."""
    h, w = array.shape[:2]
    pads = [(2, 2), (2, 2)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, pads)[2 + dy : 2 + dy + h, 2 + dx : 2 + dx + w]


def noise_scale(depth):
    """Conservative metric tolerance: 1 cm + 0.5% of squared range."""
    return 0.01 + 0.005 * np.square(depth)


def interior_mask(depth):
    valid = np.isfinite(depth) & (depth > 0)
    interior = valid.copy()
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = shifted(depth, dy, dx)
        interior &= (neighbor > 0) & (np.abs(neighbor - depth) < 0.03 + 0.02 * depth)
    return interior


def filter_sensor(sensor, rgb):
    """Reject isolated spikes; average only similar RGB/depth neighbors, retaining edges."""
    sensor = np.where(np.isfinite(sensor) & (sensor > 0), sensor, 0).astype(np.float32)
    neighbors = np.stack([shifted(sensor, y, x) for y in [-1, 0, 1] for x in [-1, 0, 1]])
    # Zeros do not act as measured depth. Median is used only with >=6 valid neighbors.
    median = np.median(neighbors, axis=0)
    sigma = noise_scale(median)
    support = np.sum((neighbors > 0) & (np.abs(neighbors - median) < 2 * sigma), axis=0)
    rejected = (sensor > 0) & (support >= 6) & (np.abs(sensor - median) > 4 * sigma)
    valid = (sensor > 0) & ~rejected
    source = np.where(valid, sensor, 0)
    color = rgb.astype(np.float32) / 255
    total = np.zeros_like(sensor)
    weights = np.zeros_like(sensor)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            z = shifted(source, dy, dx)
            difference = z - sensor
            color_difference = np.mean((shifted(color, dy, dx) - color) ** 2, axis=-1)
            weight = np.exp(
                -(dy * dy + dx * dx) / 4.5
                - 0.5 * (difference / noise_scale(sensor)) ** 2
                - color_difference / (2 * 0.1**2)
            )
            weight *= valid & (z > 0) & (np.abs(difference) < 3 * noise_scale(sensor))
            total += weight * z
            weights += weight
    filtered = np.divide(total, weights, out=np.zeros_like(total), where=weights > 0)
    return filtered, rejected


def metric_transform(prediction, scale, shift, representation="depth", minimum=0.1, maximum=10):
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(prediction) & (prediction > 0)
    transformed = scale * np.where(valid, prediction, 0) + shift
    if representation == "inverse_depth":
        transformed = np.divide(
            1.0, transformed, out=np.zeros_like(transformed), where=transformed > 1e-12
        )
    elif representation != "depth":
        raise ValueError("Unknown prediction representation")
    return np.where(
        valid & (transformed >= minimum) & (transformed <= maximum), transformed, 0
    ).astype(np.float32)


def refine_alignment(prediction, sensor, initial: AlignmentResult, minimum=0.1, maximum=10):
    """IRLS consensus refit with a disjoint sensor-validation gate and edge exclusion.

    The original regional fit remains the fallback. Validation uses input sensor
    pixels only and is independent of the held-out reconstruction test frames.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    sensor = np.asarray(sensor, dtype=np.float64)
    yy, xx = np.indices(sensor.shape)
    anchors = interior_mask(sensor) & np.isfinite(pred) & (pred > 0)
    anchors &= (sensor > minimum) & (sensor < maximum)
    sampled = anchors & (yy % 4 == 0) & (xx % 4 == 0)
    validation = sampled & ((xx // 4 + yy // 4) % 3 == 0)
    training = sampled & ~validation
    scale, shift = initial.scale, initial.shift
    accepted = False
    before = after = None
    if training.sum() >= 100 and validation.sum() >= 50:
        x, z = pred[training], sensor[training]
        y = z if initial.representation == "depth" else 1 / z
        design = np.column_stack([x, np.ones_like(x)])
        coefficients = np.array([scale, shift])
        for _ in range(6):
            residual = design @ coefficients - y
            robust_sigma = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-5)
            weights = np.minimum(1, 1.5 * robust_sigma / np.maximum(np.abs(residual), 1e-12))
            weighted = design * np.sqrt(weights[:, None])
            if np.linalg.cond(weighted) > 1e6:
                break
            coefficients = np.linalg.lstsq(weighted, y * np.sqrt(weights), rcond=None)[0]
        original = metric_transform(pred, scale, shift, initial.representation, minimum, maximum)
        candidate = metric_transform(pred, *coefficients, initial.representation, minimum, maximum)
        before = float(np.abs(original[validation] - sensor[validation]).mean())
        after = float(np.abs(candidate[validation] - sensor[validation]).mean())
        if np.isfinite(coefficients).all() and coefficients[0] > 0 and after < before:
            scale, shift = map(float, coefficients)
            accepted = True
    result = metric_transform(pred, scale, shift, initial.representation, minimum, maximum)
    if not np.any(result > 0):
        raise AlignmentError("Refinement has no valid metric depth")
    return result, {
        "scale": scale,
        "shift": shift,
        "accepted": accepted,
        "anchors": int(anchors.sum()),
        "training_anchors": int(training.sum()),
        "validation_anchors": int(validation.sum()),
        "sensor_validation_mae_before_m": before,
        "sensor_validation_mae_after_m": after,
    }


def warp_depth(depth, source_pose, target_pose, camera):
    """Nearest-pixel, z-buffered reprojection of observed source depth into target camera."""
    yy, xx = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[yy, xx]
    xyz = np.column_stack([(xx - camera.cx) * z / camera.fx, (yy - camera.cy) * z / camera.fy, z])
    transform = np.linalg.inv(target_pose) @ source_pose
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    visible = xyz[:, 2] > 1e-6
    xyz = xyz[visible]
    u = np.rint(camera.fx * xyz[:, 0] / xyz[:, 2] + camera.cx).astype(int)
    v = np.rint(camera.fy * xyz[:, 1] / xyz[:, 2] + camera.cy).astype(int)
    inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    out = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    np.minimum.at(out, (v[inside], u[inside]), xyz[inside, 2])
    out[~np.isfinite(out)] = 0
    return out


@dataclass
class ConservativeFusion:
    camera: object
    minimum: float = 0.1
    maximum: float = 10.0
    previous_sensor: np.ndarray | None = None
    previous_pose: np.ndarray | None = None

    def fuse(self, sensor, recovered, rgb, pose):
        filtered, rejected = filter_sensor(sensor, rgb)
        residual = np.where(filtered > 0, filtered - recovered, 0)
        anchor = interior_mask(filtered) & (recovered > 0)
        total = np.zeros_like(filtered)
        square = np.zeros_like(filtered)
        count = np.zeros_like(filtered)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                adjacent = shifted(recovered, dy, dx)
                supported = shifted(anchor, dy, dx) & (recovered > 0)
                supported &= np.abs(adjacent - recovered) < 0.03 + 0.02 * recovered
                offset = shifted(residual, dy, dx)
                total += np.where(supported, offset, 0)
                square += np.where(supported, offset**2, 0)
                count += supported
        offset = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        variance = np.divide(square, count, out=np.zeros_like(square), where=count > 0) - offset**2
        candidate = recovered + offset
        eligible = (filtered == 0) & (recovered > 0) & (count >= 5)
        eligible &= variance < noise_scale(candidate) ** 2
        eligible &= (candidate > self.minimum) & (candidate < self.maximum)
        temporal = np.zeros_like(filtered, dtype=bool)
        if self.previous_sensor is not None:
            prior = warp_depth(self.previous_sensor, self.previous_pose, pose, self.camera)
            temporal = (prior > 0) & (np.abs(prior - candidate) < 2 * noise_scale(candidate))
        admitted = eligible & temporal
        fused = np.where(admitted, candidate, filtered).astype(np.float32)
        # Model completions NEVER enter the temporal evidence buffer.
        self.previous_sensor = filtered.copy()
        self.previous_pose = np.array(pose, copy=True)
        diagnostics = {
            "sensor_pixels": int((sensor > 0).sum()),
            "rejected_sensor_spikes": int(rejected.sum()),
            "spatially_supported_candidates": int(eligible.sum()),
            "temporally_supported_fills": int(admitted.sum()),
            "remaining_holes": int((fused == 0).sum()),
        }
        return fused, filtered, admitted, diagnostics
