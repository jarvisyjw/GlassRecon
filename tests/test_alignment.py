import numpy as np
import pytest

from glassrecon.alignment import AlignmentConfig, AlignmentError, recover_metric_depth


@pytest.mark.parametrize("representation", ["depth", "inverse_depth"])
@pytest.mark.parametrize("method", ["global", "local"])
def test_recovers_affine_and_fills_sensor_holes(representation, method):
    prediction = np.linspace(0.5, 2, 120).reshape(10, 12)
    target = 1.7 * prediction + 0.3
    truth = 1 / target if representation == "inverse_depth" else target
    sensor = truth.copy()
    sensor[:2, :2] = 0
    sensor[2, 0] = np.nan
    config = AlignmentConfig(method=method, rows=2, cols=2, iterations=10)
    result, fit = recover_metric_depth(prediction, sensor, representation, config)
    np.testing.assert_allclose(result, truth, rtol=1e-6)
    assert fit.scale == pytest.approx(1.7)
    assert fit.shift == pytest.approx(0.3)
    assert fit.valid_points == 115


def test_regional_sampling_rejects_local_sensor_corruption():
    prediction = np.random.default_rng(4).uniform(0.5, 2, (32, 32))
    truth = 2 * prediction + 0.2
    sensor = truth.copy()
    sensor[:8] += 3
    local, _ = recover_metric_depth(prediction, sensor, config=AlignmentConfig(iterations=15))
    global_result, _ = recover_metric_depth(
        prediction, sensor, config=AlignmentConfig(method="global")
    )
    assert np.abs(local - truth).mean() < 1e-5
    assert np.abs(global_result - truth).mean() > 0.5


@pytest.mark.parametrize("invalid", [0, -1, np.nan, np.inf])
def test_invalid_predictions_are_not_filled_by_shift(invalid):
    prediction = np.linspace(0.5, 2, 100).reshape(10, 10)
    sensor = 2 * prediction + 1
    prediction[0, 0] = invalid
    result, _ = recover_metric_depth(prediction, sensor, config=AlignmentConfig(method="global"))
    assert result[0, 0] == 0
    assert np.isfinite(result).all()


def test_inverse_depth_nonpositive_transform_stays_invalid():
    prediction = np.linspace(1, 2, 100).reshape(10, 10)
    sensor = 1 / (prediction - 0.5)
    prediction[0, 0], sensor[0, 0] = 0.1, 0
    result, _ = recover_metric_depth(
        prediction, sensor, "inverse_depth", AlignmentConfig(method="global")
    )
    assert result[0, 0] == 0


@pytest.mark.parametrize("sensor", [np.zeros((10, 10)), np.ones((10, 10))])
def test_degenerate_alignment_fails_instead_of_claiming_metric_scale(sensor):
    with pytest.raises(AlignmentError):
        recover_metric_depth(np.ones((10, 10)), sensor)


def test_remainder_pixels_participate_in_sampling():
    prediction = np.arange(49).reshape(7, 7) / 100 + 1
    sensor = 2 * prediction
    sensor[:4, :] = 0
    sensor[:, :4] = 0
    _, fit = recover_metric_depth(
        prediction, sensor, config=AlignmentConfig(rows=2, cols=2, samples=3, iterations=3)
    )
    assert not fit.fallback
    assert fit.scale == pytest.approx(2)


def test_sparse_partitions_use_global_fallback_and_rng_is_reproducible():
    p = np.linspace(1, 2, 100).reshape(10, 10)
    sensor = 2 * p + 0.4
    config = AlignmentConfig(rows=20, cols=20)
    a, fit_a = recover_metric_depth(p, sensor, config=config, rng=np.random.default_rng(42))
    b, fit_b = recover_metric_depth(p, sensor, config=config, rng=np.random.default_rng(42))
    assert fit_a.fallback and fit_a == fit_b
    np.testing.assert_array_equal(a, b)
