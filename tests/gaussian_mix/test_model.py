"""
Unit tests for gaussian_mix/src/model.py
"""

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

_src = Path(__file__).parent.parent.parent / "gaussian_mix" / "src" / "model.py"
_spec = importlib.util.spec_from_file_location("gaussian_mix.model", _src)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["gaussian_mix.model"] = _mod
_spec.loader.exec_module(_mod)

DEFAULT_MEANS = _mod.DEFAULT_MEANS
DEFAULT_SCALES = _mod.DEFAULT_SCALES
DEFAULT_WEIGHTS = _mod.DEFAULT_WEIGHTS
make_log_density = _mod.make_log_density


def test_default_weights_sum_to_one():
    assert np.isclose(DEFAULT_WEIGHTS.sum(), 1.0)


def test_default_means_shape():
    assert DEFAULT_MEANS.shape == (8, 2)


def test_default_scales_positive():
    assert np.all(DEFAULT_SCALES > 0)


def test_log_density_returns_scalar():
    log_density_fn = make_log_density()
    x = jnp.array([0.0, 0.0])
    result = log_density_fn(x)
    assert result.shape == ()


def test_log_density_higher_at_modes():
    """Log density at each mode center should exceed density at the origin."""
    log_density_fn = make_log_density()
    log_p_origin = log_density_fn(jnp.array([0.0, 0.0]))
    for mean in DEFAULT_MEANS:
        log_p_mode = log_density_fn(mean)
        assert log_p_mode > log_p_origin, (
            f"Expected higher density at mode {mean} than at origin"
        )


def test_log_density_decreases_far_from_modes():
    """Log density should be very low far from all modes."""
    log_density_fn = make_log_density()
    log_p_far = log_density_fn(jnp.array([100.0, 100.0]))
    log_p_near = log_density_fn(DEFAULT_MEANS[0])
    assert log_p_near > log_p_far


def test_custom_weights():
    """A single-component mixture should concentrate mass at that mean."""
    means = jnp.array([[2.0, 2.0], [-2.0, -2.0]])
    scales = jnp.array([0.5, 0.5])
    weights = jnp.array([1.0, 0.0])
    log_density_fn = make_log_density(means=means, scales=scales, weights=weights)
    log_p_mode0 = log_density_fn(means[0])
    log_p_mode1 = log_density_fn(means[1])
    assert log_p_mode0 > log_p_mode1


def test_density_at_modes_matches_weights():
    """Heavy modes (even indices) should have equal log density to each other,
    light modes (odd indices) likewise, and heavy > light."""
    log_density_fn = make_log_density()
    log_ps = [float(log_density_fn(mean)) for mean in DEFAULT_MEANS]
    heavy = [log_ps[k] for k in range(0, 8, 2)]
    light = [log_ps[k] for k in range(1, 8, 2)]
    assert np.allclose(heavy, heavy[0], atol=1e-5), f"Heavy modes not equal: {heavy}"
    assert np.allclose(light, light[0], atol=1e-5), f"Light modes not equal: {light}"
    assert heavy[0] > light[0], "Heavy modes should have higher density than light modes"
