"""
Unit tests for sajax/src/model.py
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

OUTPUT_DIR = Path(__file__).parent.parent / "output"

_src = Path(__file__).parent.parent.parent / "sajax" / "src" / "model.py"
_spec = importlib.util.spec_from_file_location("sajax.model", _src)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sajax.model"] = _mod
_spec.loader.exec_module(_mod)

GROUND_TRUTH = _mod.GROUND_TRUTH
OBS_LIGHT_CURVE = _mod.OBS_LIGHT_CURVE
PARAM_NAMES = _mod.PARAM_NAMES
PRIOR_DISTRIBUTIONS = _mod.PRIOR_DISTRIBUTIONS
SIGMA_NOISE = _mod.SIGMA_NOISE
TIMES = _mod.TIMES
TRUE_T14_TRANSIT = _mod.TRUE_T14_TRANSIT
STATIC_MODEL = _mod.STATIC_MODEL
TRUE_T0_TRANSIT = _mod.TRUE_T0_TRANSIT
_call_sajax = _mod._call_sajax
_compute_all_phases = _mod._compute_all_phases
compute_planet_sky_positions = _mod.compute_planet_sky_positions
rotate_active_region = _mod.rotate_active_region
generate_observations = _mod.generate_observations
make_log_density = _mod.make_log_density
plot_model = _mod.plot_model
sajax_model = _mod.sajax_model


# ---------------------------------------------------------------------------
# Ground-truth parameter vector matching the 17-element ordering in
# make_log_density. Inclination and arg_periapsis are in degrees, matching
# the convention in GROUND_TRUTH and the prior distributions.
# ---------------------------------------------------------------------------
GT_VECTOR = jnp.array([
    GROUND_TRUTH["spot_lat"],
    GROUND_TRUTH["spot_long"],
    GROUND_TRUTH["spot_size"],
    GROUND_TRUTH["spot_flux"],
    GROUND_TRUTH["fac_lat"],
    GROUND_TRUTH["fac_long"],
    GROUND_TRUTH["fac_size"],
    GROUND_TRUTH["fac_flux"],
    GROUND_TRUTH["p_rot"],
    GROUND_TRUTH["planet_radius"],
    GROUND_TRUTH["semimajor_axis"],
    GROUND_TRUTH["inclination"],
    GROUND_TRUTH["eccentricity"],
    GROUND_TRUTH["arg_periapsis"],
    GROUND_TRUTH["P_orb"],
    GROUND_TRUTH["LDC_u1"],
    GROUND_TRUTH["LDC_u2"],
])


def _sample_prior_vector(seed: int = 0) -> jnp.ndarray:
    """Draw a single in-prior parameter vector matching the make_log_density ordering."""
    rng = np.random.default_rng(seed)
    key_order = [
        "spot_lat", "spot_long", "spot_size", "spot_flux",
        "fac_lat", "fac_long", "fac_size", "fac_flux",
        "p_rot",
        "planet_radius", "semimajor_axis", "inclination",
        "eccentricity", "arg_periapsis", "P_orb",
        "ldc_u1", "ldc_u2",
    ]
    key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
    keys = jax.random.split(key, len(key_order))
    return jnp.array([
        float(PRIOR_DISTRIBUTIONS[name].sample(k))
        for name, k in zip(key_order, keys)
    ])


# ---------------------------------------------------------------------------
# Shape / contract tests
# ---------------------------------------------------------------------------

def test_obs_shape_matches_times():
    assert OBS_LIGHT_CURVE.shape == TIMES.shape


def test_gt_vector_length():
    assert GT_VECTOR.shape == (17,)


def test_param_names_match_ground_truth():
    assert PARAM_NAMES == list(GROUND_TRUTH.keys())


def test_prior_keys_consumed_by_model():
    """Every prior in PRIOR_DISTRIBUTIONS should appear as a sample site in
    sajax_model. Uses source-code string matching, so it catches dead/leftover
    priors but will not detect a prior that is referenced but never passed to
    numpyro.sample."""
    import inspect
    src = inspect.getsource(sajax_model)
    for name in PRIOR_DISTRIBUTIONS:
        assert f'"{name}"' in src, f"Prior '{name}' is never sampled in sajax_model"


# ---------------------------------------------------------------------------
# Transit geometry
# ---------------------------------------------------------------------------

def test_transit_duration_finite_and_positive():
    assert jnp.isfinite(TRUE_T14_TRANSIT)
    assert float(TRUE_T14_TRANSIT) > 0.0


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def test_generate_observations_deterministic():
    a = generate_observations(seed=123)
    b = generate_observations(seed=123)
    assert np.array_equal(a, b)


def test_generate_observations_different_seeds_differ():
    a = generate_observations(seed=1)
    b = generate_observations(seed=2)
    assert not np.array_equal(a, b)


def test_generate_observations_noise_scale():
    """Residuals between two seeds (both contain the same true light curve)
    should have std ~= sqrt(2) * SIGMA_NOISE."""
    a = generate_observations(seed=1)
    b = generate_observations(seed=2)
    diff_std = float(np.std(a - b))
    expected = float(np.sqrt(2.0) * SIGMA_NOISE)
    assert 0.5 * expected < diff_std < 1.5 * expected


# ---------------------------------------------------------------------------
# Log density: finiteness, JIT, gradient
# ---------------------------------------------------------------------------

def test_log_density_returns_scalar():
    log_density_fn = make_log_density()
    result = log_density_fn(GT_VECTOR)
    assert result.shape == ()


def test_log_density_finite_at_ground_truth():
    log_density_fn = make_log_density()
    result = float(log_density_fn(GT_VECTOR))
    assert np.isfinite(result)


def test_log_density_finite_at_prior_draws():
    log_density_fn = make_log_density()
    for seed in range(3):
        x = _sample_prior_vector(seed=seed)
        val = float(log_density_fn(x))
        assert np.isfinite(val), f"Non-finite log density at prior draw seed={seed}"


def test_log_density_jit_compatible():
    log_density_fn = make_log_density()
    jitted = jax.jit(log_density_fn)
    val = float(jitted(GT_VECTOR))
    assert np.isfinite(val)


def test_log_density_gradient_finite():
    log_density_fn = make_log_density()
    grad_fn = jax.grad(log_density_fn)
    g = grad_fn(GT_VECTOR)
    assert g.shape == GT_VECTOR.shape
    assert np.all(np.isfinite(np.asarray(g)))


def test_log_density_higher_at_ground_truth_than_prior_draw():
    """Smoke test: the posterior should prefer the truth over a random prior draw."""
    log_density_fn = make_log_density()
    ld_truth = float(log_density_fn(GT_VECTOR))
    ld_random = float(log_density_fn(_sample_prior_vector(seed=7)))
    assert ld_truth > ld_random


def test_ground_truth_residuals_at_noise_level():
    """Reconstructed light curve via the one-shot API (_call_sajax) at ground
    truth should match OBS_LIGHT_CURVE to within ~SIGMA_NOISE. A large residual
    std indicates the one-shot forward model does not round-trip correctly."""
    result = _call_sajax(
        TIMES,
        jnp.array([GROUND_TRUTH["spot_lat"], GROUND_TRUTH["fac_lat"]]),
        jnp.array([GROUND_TRUTH["spot_long"], GROUND_TRUTH["fac_long"]]),
        jnp.array([GROUND_TRUTH["spot_size"], GROUND_TRUTH["fac_size"]]),
        np.stack([np.array([GROUND_TRUTH["spot_flux"]]),
                  np.array([GROUND_TRUTH["fac_flux"]])]),
        GROUND_TRUTH["p_rot"],
        GROUND_TRUTH["planet_radius"],
        GROUND_TRUTH["semimajor_axis"],
        jnp.deg2rad(GROUND_TRUTH["inclination"]),
        GROUND_TRUTH["eccentricity"],
        jnp.deg2rad(GROUND_TRUTH["arg_periapsis"]),
        GROUND_TRUTH["P_orb"],
        GROUND_TRUTH["LDC_u1"],
        GROUND_TRUTH["LDC_u2"],
    )
    lc_reconstructed = np.array(result["lc"])
    residuals = OBS_LIGHT_CURVE - lc_reconstructed
    residual_std = float(np.std(residuals))
    assert residual_std < 5 * float(SIGMA_NOISE), (
        f"Residual std {residual_std:.2e} is >>SIGMA_NOISE={SIGMA_NOISE:.2e} — "
        "forward model does not round-trip at ground truth"
    )


def test_two_stage_residuals_at_noise_level():
    """Reconstructed light curve via the two-stage API (_compute_all_phases, the
    path used by sajax_model during sampling) at ground truth should match
    OBS_LIGHT_CURVE to within ~SIGMA_NOISE. Ensures the sampler's forward model
    is consistent with the one-shot API used to generate the observations."""
    gt = GROUND_TRUTH
    m = STATIC_MODEL

    spot_lat      = gt["spot_lat"]
    spot_long     = gt["spot_long"]
    spot_size     = gt["spot_size"]
    spot_flux     = gt["spot_flux"]
    fac_lat       = gt["fac_lat"]
    fac_long      = gt["fac_long"]
    fac_size      = gt["fac_size"]
    fac_flux      = gt["fac_flux"]
    P_rot         = gt["p_rot"]
    LDC_u1        = gt["LDC_u1"]
    LDC_u2        = gt["LDC_u2"]
    planet_radius = gt["planet_radius"]
    semimajor     = gt["semimajor_axis"]
    inclination   = jnp.deg2rad(gt["inclination"])
    eccentricity  = gt["eccentricity"]
    arg_periapsis = jnp.deg2rad(gt["arg_periapsis"])
    P_orb         = gt["P_orb"]

    dynamic_phases_rot = (m["times"] / P_rot * 360.0) % 360.0

    planet_xyz_all = compute_planet_sky_positions(
        times        = m["times"],
        t0           = TRUE_T0_TRANSIT,
        period       = P_orb,
        a_over_rstar = semimajor,
        inclination  = inclination,
        ecc          = eccentricity,
        omega_peri   = arg_periapsis,
    )

    ar_lat  = jnp.array([spot_lat, fac_lat])
    ar_long = jnp.array([spot_long, fac_long])
    ar_size = jnp.array([spot_size, fac_size])

    spr = m["star_pixel_rad"]
    ar_cart = jnp.stack([
        spr * jnp.sin(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
        spr * jnp.sin(jnp.deg2rad(ar_lat)),
        spr * jnp.cos(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
    ], axis=-1)

    all_ar_carts = jax.vmap(lambda p: jax.vmap(
        lambda c: rotate_active_region(c, p, m["inc_star"])
    )(ar_cart))(dynamic_phases_rot)

    flux_active = jnp.stack([
        jnp.broadcast_to(spot_flux, (1,)),
        jnp.broadcast_to(fac_flux, (1,)),
    ])

    lc, _, _ = _compute_all_phases(
        all_ar_carts,
        planet_xyz_all,
        wavelength        = m["wavelength"],
        flux_quiet_interp = m["flux_quiet"],
        flux_active_interp= flux_active,
        ldc_coeffs        = jnp.array([[LDC_u1, LDC_u2]]),
        I_profile         = m["I_profile"],
        mu_profile_pts    = m["mu_profile_pts"],
        x_disc            = m["x_disc"],
        y_disc            = m["y_disc"],
        mu_disc           = m["mu_disc"],
        vel_disc          = m["vel_disc"],
        star_pixel_rad    = spr,
        total_pixels      = m["total_pixels"],
        arsize_rads       = jnp.deg2rad(ar_size),
        k                 = planet_radius,
        ldc_mode          = m["ldc_mode"],
        ar_overlap_mode   = m["ar_overlap_mode"],
        plot_map_wavelength = m["plot_map_wavelength"],
        n                 = m["n"],
        flat_indices      = m["flat_indices"],
    )

    lc_reconstructed = np.array(lc)
    residuals = OBS_LIGHT_CURVE - lc_reconstructed
    residual_std = float(np.std(residuals))
    assert residual_std < 5 * float(SIGMA_NOISE), (
        f"Two-stage residual std {residual_std:.2e} is >>SIGMA_NOISE={SIGMA_NOISE:.2e} — "
        "bug is in the two-stage (_compute_all_phases) code path"
    )


# ---------------------------------------------------------------------------
# One-shot API edge cases
# ---------------------------------------------------------------------------

def test_call_sajax_activity_only_runs():
    """One-shot API with planet_radius=0 (stellar activity only, no transit)
    should produce a finite light curve — guards against division-by-zero or
    NaN propagation in the no-planet edge case."""
    result = _call_sajax(
        TIMES,
        jnp.array([GROUND_TRUTH["spot_lat"], GROUND_TRUTH["fac_lat"]]),
        jnp.array([GROUND_TRUTH["spot_long"], GROUND_TRUTH["fac_long"]]),
        jnp.array([GROUND_TRUTH["spot_size"], GROUND_TRUTH["fac_size"]]),
        np.stack([np.array([GROUND_TRUTH["spot_flux"]]),
                  np.array([GROUND_TRUTH["fac_flux"]])]),
        GROUND_TRUTH["p_rot"],
        0.0,
        GROUND_TRUTH["semimajor_axis"],
        jnp.deg2rad(GROUND_TRUTH["inclination"]),
        GROUND_TRUTH["eccentricity"],
        jnp.deg2rad(GROUND_TRUTH["arg_periapsis"]),
        GROUND_TRUTH["P_orb"],
        GROUND_TRUTH["LDC_u1"],
        GROUND_TRUTH["LDC_u2"],
    )
    lc = np.array(result["lc"])
    assert lc.shape == TIMES.shape
    assert np.all(np.isfinite(lc))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def test_plot_model_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(_mod, "OUTPUT_DIR", tmp_path)
    plot_model(filename="diagnostic.png")
    assert (tmp_path / "diagnostic.png").exists()


def test_plot_api_comparison():
    """Save a visual comparison of the one-shot and two-stage light curves at
    ground truth alongside OBS_LIGHT_CURVE. Makes no assertion — purely a
    visual diagnostic saved to tests/output/api_comparison.png."""
    gt = GROUND_TRUTH
    m = STATIC_MODEL

    # One-shot API
    lc_one_shot = np.array(_call_sajax(
        TIMES,
        jnp.array([gt["spot_lat"], gt["fac_lat"]]),
        jnp.array([gt["spot_long"], gt["fac_long"]]),
        jnp.array([gt["spot_size"], gt["fac_size"]]),
        np.stack([np.array([gt["spot_flux"]]), np.array([gt["fac_flux"]])]),
        gt["p_rot"],
        gt["planet_radius"],
        gt["semimajor_axis"],
        jnp.deg2rad(gt["inclination"]),
        gt["eccentricity"],
        jnp.deg2rad(gt["arg_periapsis"]),
        gt["P_orb"],
        gt["LDC_u1"],
        gt["LDC_u2"],
    )["lc"])

    # Two-stage API (replicating sajax_model compute path)
    P_rot         = gt["p_rot"]
    LDC_u1        = gt["LDC_u1"]
    LDC_u2        = gt["LDC_u2"]
    planet_radius = gt["planet_radius"]
    semimajor     = gt["semimajor_axis"]
    inclination   = jnp.deg2rad(gt["inclination"])
    eccentricity  = gt["eccentricity"]
    arg_periapsis = jnp.deg2rad(gt["arg_periapsis"])
    P_orb         = gt["P_orb"]
    ar_lat        = jnp.array([gt["spot_lat"], gt["fac_lat"]])
    ar_long       = jnp.array([gt["spot_long"], gt["fac_long"]])
    ar_size       = jnp.array([gt["spot_size"], gt["fac_size"]])

    dynamic_phases_rot = (m["times"] / P_rot * 360.0) % 360.0
    planet_xyz_all = compute_planet_sky_positions(
        times=m["times"], t0=TRUE_T0_TRANSIT, period=P_orb,
        a_over_rstar=semimajor, inclination=inclination,
        ecc=eccentricity, omega_peri=arg_periapsis,
    )
    spr = m["star_pixel_rad"]
    ar_cart = jnp.stack([
        spr * jnp.sin(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
        spr * jnp.sin(jnp.deg2rad(ar_lat)),
        spr * jnp.cos(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
    ], axis=-1)
    all_ar_carts = jax.vmap(lambda p: jax.vmap(
        lambda c: rotate_active_region(c, p, m["inc_star"])
    )(ar_cart))(dynamic_phases_rot)
    flux_active = jnp.stack([
        jnp.broadcast_to(gt["spot_flux"], (1,)),
        jnp.broadcast_to(gt["fac_flux"], (1,)),
    ])
    lc_two_stage_raw, _, _ = _compute_all_phases(
        all_ar_carts, planet_xyz_all,
        wavelength=m["wavelength"], flux_quiet_interp=m["flux_quiet"],
        flux_active_interp=flux_active, ldc_coeffs=jnp.array([[LDC_u1, LDC_u2]]),
        I_profile=m["I_profile"], mu_profile_pts=m["mu_profile_pts"],
        x_disc=m["x_disc"], y_disc=m["y_disc"], mu_disc=m["mu_disc"],
        vel_disc=m["vel_disc"], star_pixel_rad=spr, total_pixels=m["total_pixels"],
        arsize_rads=jnp.deg2rad(ar_size), k=planet_radius,
        ldc_mode=m["ldc_mode"], ar_overlap_mode=m["ar_overlap_mode"],
        plot_map_wavelength=m["plot_map_wavelength"],
        n=m["n"], flat_indices=m["flat_indices"],
    )
    lc_two_stage = np.array(lc_two_stage_raw)

    times = np.array(TIMES)
    residuals = lc_one_shot - lc_two_stage

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.scatter(times, OBS_LIGHT_CURVE, s=4, color="orange", alpha=0.6, label="Observations", zorder=1)
    ax.plot(times, lc_one_shot,  lw=2, color="steelblue", label="One-shot API", zorder=2)
    ax.plot(times, lc_two_stage, lw=2, color="crimson", linestyle="--", label="Two-stage API", zorder=3)
    ax.set_ylabel("Normalised flux")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(times, residuals * 1e6, lw=1.5, color="black")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline( float(SIGMA_NOISE) * 1e6, color="gray", linestyle="--", linewidth=0.8, label=r"±1σ noise")
    ax.axhline(-float(SIGMA_NOISE) * 1e6, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Residual [ppm]")
    ax.set_xlabel("Time [days]")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("One-shot vs two-stage API comparison at ground truth", y=1.01)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "api_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved API comparison plot to {out_path}")
