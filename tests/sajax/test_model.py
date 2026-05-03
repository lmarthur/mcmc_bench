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
import numpyro.distributions as dist
import pytest

OUTPUT_DIR = Path(__file__).parent.parent / "output"

_src = Path(__file__).parent.parent.parent / "sajax" / "src" / "model.py"
_spec = importlib.util.spec_from_file_location("sajax.model", _src)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sajax.model"] = _mod
_spec.loader.exec_module(_mod)

GROUND_TRUTH             = _mod.GROUND_TRUTH
OBS_LIGHT_CURVE          = _mod.OBS_LIGHT_CURVE
PARAM_NAMES              = _mod.PARAM_NAMES
PRIOR_DISTRIBUTIONS      = _mod.PRIOR_DISTRIBUTIONS
SIGMA_NOISE              = _mod.SIGMA_NOISE
TIMES                    = _mod.TIMES
TRUE_T14_TRANSIT         = _mod.TRUE_T14_TRANSIT
STATIC_MODEL             = _mod.STATIC_MODEL
TRUE_T0_TRANSIT          = _mod.TRUE_T0_TRANSIT
TRUE_LDC_U1              = _mod.TRUE_LDC_U1
TRUE_LDC_U2              = _mod.TRUE_LDC_U2
TRUE_P_ORB               = _mod.TRUE_P_ORB
_call_sajax              = _mod._call_sajax
_compute_all_phases      = _mod._compute_all_phases
compute_planet_sky_positions = _mod.compute_planet_sky_positions
rotate_active_region     = _mod.rotate_active_region
generate_observations    = _mod.generate_observations
make_inference_fns       = _mod.make_inference_fns
make_constrain_fn        = _mod.make_constrain_fn
make_log_ref             = _mod.make_log_ref
plot_model               = _mod.plot_model
sajax_model              = _mod.sajax_model
sample_initial_positions = _mod.sample_initial_positions


# ---------------------------------------------------------------------------
# Shared inference functions — created once to avoid re-tracing the model.
# ---------------------------------------------------------------------------

_RNG = jax.random.PRNGKey(0)
_LOG_DENSITY_FN, _POSTPROCESS_FN, _INIT_Z = make_inference_fns(_RNG)

# Gradient at the default prior draw — computed once and reused across gradient tests.

# ---------------------------------------------------------------------------
# Shape / contract tests
# ---------------------------------------------------------------------------

def test_obs_shape_matches_times():
    assert OBS_LIGHT_CURVE.shape == TIMES.shape


def test_param_names_match_ground_truth():
    assert PARAM_NAMES == list(GROUND_TRUTH.keys())
    assert len(PARAM_NAMES) == 17


def test_param_names_all_present_in_postprocess_output():
    """Every sampled parameter name must appear in postprocess_fn output.
    Catches key casing mismatches between GROUND_TRUTH and numpyro sample sites."""
    constrained = _POSTPROCESS_FN(_INIT_Z)
    for name in PARAM_NAMES:
        assert name in constrained, (
            f"PARAM_NAMES entry '{name}' is missing from postprocess_fn output. "
            f"Available keys: {list(constrained.keys())}"
        )


def test_prior_keys_consumed_by_model():
    """Every prior in PRIOR_DISTRIBUTIONS should appear as a sample site in
    sajax_model."""
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
    """Residuals between two seeds should have std ~= sqrt(2) * SIGMA_NOISE."""
    a = generate_observations(seed=1)
    b = generate_observations(seed=2)
    diff_std = float(np.std(a - b))
    expected = float(np.sqrt(2.0) * SIGMA_NOISE)
    assert 0.5 * expected < diff_std < 1.5 * expected


# ---------------------------------------------------------------------------
# Log density: finiteness, JIT, gradient (unconstrained space)
# ---------------------------------------------------------------------------

def test_log_density_returns_scalar():
    result = _LOG_DENSITY_FN(_INIT_Z)
    assert result.shape == ()


def test_log_density_finite_at_prior_init():
    """Log density must be finite at a prior draw in unconstrained space."""
    result = float(_LOG_DENSITY_FN(_INIT_Z))
    assert np.isfinite(result)


def test_log_density_finite_at_multiple_prior_draws():
    for seed in range(3):
        _, _, z = make_inference_fns(jax.random.PRNGKey(seed + 10))
        val = float(_LOG_DENSITY_FN(z))
        assert np.isfinite(val), f"Non-finite log density at prior draw seed={seed}"


def test_log_density_jit_compatible():
    jitted = jax.jit(_LOG_DENSITY_FN)
    val = float(jitted(_INIT_Z))
    assert np.isfinite(val)



def test_log_density_finite_at_two_draws():
    """Both a near-truth and a distant draw should yield finite log density."""
    ld_near = float(_LOG_DENSITY_FN(_INIT_Z))
    _, _, z_far = make_inference_fns(jax.random.PRNGKey(99))
    ld_far = float(_LOG_DENSITY_FN(z_far))
    assert np.isfinite(ld_near)
    assert np.isfinite(ld_far)


# ---------------------------------------------------------------------------
# postprocess_fn: constrained values within prior bounds
# ---------------------------------------------------------------------------

def test_postprocess_fn_returns_dict():
    constrained = _POSTPROCESS_FN(_INIT_Z)
    assert isinstance(constrained, dict)


def test_postprocess_fn_includes_deterministics():
    """postprocess_fn must include derived sites eccentricity and arg_periapsis."""
    constrained = _POSTPROCESS_FN(_INIT_Z)
    assert "eccentricity"  in constrained
    assert "arg_periapsis" in constrained


def test_postprocess_fn_uniform_params_within_bounds():
    """All Uniform-prior parameters must be within their bounds after postprocessing,
    across multiple prior draws.  This is the key test that was impossible to pass
    with the old constrained-space log_density approach."""
    bounds = {
        "spot_lat":    (-90.0,  90.0),
        "spot_long":   (  0.0, 360.0),
        "spot_size":   (  1.0,  90.0),
        "fac_lat":     (-90.0,  90.0),
        "fac_long":    (  0.0, 360.0),
        "fac_size":    (  1.0,  90.0),
        "inclination": ( 80.0, 100.0),
        "ldc_q1":      (  0.0,   1.0),
        "ldc_q2":      (  0.0,   1.0),
    }
    for seed in range(5):
        _, _, z = make_inference_fns(jax.random.PRNGKey(seed))
        c = _POSTPROCESS_FN(z)
        for name, (lo, hi) in bounds.items():
            val = float(c[name])
            assert lo <= val <= hi, (
                f"seed={seed}: {name}={val:.4f} is outside [{lo}, {hi}]"
            )


def test_postprocess_fn_eccentricity_non_negative():
    for seed in range(5):
        _, _, z = make_inference_fns(jax.random.PRNGKey(seed))
        c = _POSTPROCESS_FN(z)
        assert float(c["eccentricity"]) >= 0.0


# ---------------------------------------------------------------------------
# make_constrain_fn: must agree with postprocess_fn on sampled parameters
# ---------------------------------------------------------------------------

def test_constrain_fn_agrees_with_postprocess_fn():
    """make_constrain_fn must give the same constrained values as postprocess_fn
    for the sampled parameters.  Verifies that biject_to uses the same bijection
    as initialize_model internally."""
    constrain_fn = make_constrain_fn()
    c_fast = constrain_fn(_INIT_Z)
    c_full = _POSTPROCESS_FN(_INIT_Z)
    for name in PRIOR_DISTRIBUTIONS:
        assert jnp.allclose(
            jnp.array(c_fast[name]), jnp.array(c_full[name]), atol=1e-5
        ), (
            f"Mismatch for '{name}': "
            f"constrain_fn={float(c_fast[name]):.6f}, "
            f"postprocess_fn={float(c_full[name]):.6f}"
        )


def test_constrain_fn_vmappable():
    """make_constrain_fn result must be safe to jax.vmap over a batch of samples."""
    import jax.flatten_util
    constrain_fn = make_constrain_fn()
    _, unravel_fn = jax.flatten_util.ravel_pytree(_INIT_Z)
    flat = jax.flatten_util.ravel_pytree(_INIT_Z)[0]
    batch = jnp.stack([flat] * 4)
    result = jax.vmap(lambda x: constrain_fn(unravel_fn(x)))(batch)
    for name in PRIOR_DISTRIBUTIONS:
        assert result[name].shape == (4,), f"Unexpected shape for '{name}'"


# ---------------------------------------------------------------------------
# make_log_ref: prior log density in unconstrained space
# ---------------------------------------------------------------------------

def test_log_ref_finite_at_prior_draw():
    log_ref = make_log_ref(jax.random.PRNGKey(1))
    val = float(log_ref(_INIT_Z))
    assert np.isfinite(val)


def test_log_ref_finite_at_multiple_draws():
    log_ref = make_log_ref(jax.random.PRNGKey(2))
    for seed in range(3):
        _, _, z = make_inference_fns(jax.random.PRNGKey(seed + 20))
        assert np.isfinite(float(log_ref(z))), f"Non-finite log_ref at seed={seed}"


# ---------------------------------------------------------------------------
# Ground-truth residuals: one-shot and two-stage API
# ---------------------------------------------------------------------------

def test_ground_truth_residuals_at_noise_level():
    """Reconstructed light curve via the one-shot API at ground truth should
    match OBS_LIGHT_CURVE to within ~SIGMA_NOISE."""
    ecc_h = GROUND_TRUTH["ecc_h"]
    ecc_k = GROUND_TRUTH["ecc_k"]
    semimajor_axis = jnp.abs(
        GROUND_TRUTH["impact_param"] / jnp.cos(jnp.deg2rad(GROUND_TRUTH["inclination"]))
    )
    result = _call_sajax(
        TIMES,
        jnp.array([GROUND_TRUTH["spot_lat"], GROUND_TRUTH["fac_lat"]]),
        jnp.array([GROUND_TRUTH["spot_long"], GROUND_TRUTH["fac_long"]]),
        jnp.array([GROUND_TRUTH["spot_size"], GROUND_TRUTH["fac_size"]]),
        np.stack([np.array([GROUND_TRUTH["spot_flux"]]),
                  np.array([GROUND_TRUTH["fac_flux"]])]),
        GROUND_TRUTH["p_rot"],
        GROUND_TRUTH["planet_radius"],
        semimajor_axis,
        jnp.deg2rad(GROUND_TRUTH["inclination"]),
        ecc_h**2 + ecc_k**2,
        jnp.arctan2(ecc_k, ecc_h),
        TRUE_P_ORB,
        TRUE_LDC_U1,
        TRUE_LDC_U2,
    )
    lc_reconstructed = np.array(result["lc"])
    residuals = OBS_LIGHT_CURVE - lc_reconstructed
    residual_std = float(np.std(residuals))
    assert residual_std < 5 * float(SIGMA_NOISE), (
        f"Residual std {residual_std:.2e} is >>SIGMA_NOISE={SIGMA_NOISE:.2e} — "
        "forward model does not round-trip at ground truth"
    )


def test_two_stage_residuals_at_noise_level():
    """Reconstructed light curve via the two-stage API (_compute_all_phases) at
    ground truth should match OBS_LIGHT_CURVE to within ~SIGMA_NOISE."""
    gt = GROUND_TRUTH
    m  = STATIC_MODEL

    spot_lat      = gt["spot_lat"]
    spot_long     = gt["spot_long"]
    spot_size     = gt["spot_size"]
    spot_flux     = gt["spot_flux"]
    fac_lat       = gt["fac_lat"]
    fac_long      = gt["fac_long"]
    fac_size      = gt["fac_size"]
    fac_flux      = gt["fac_flux"]
    P_rot         = gt["p_rot"]
    LDC_u1        = TRUE_LDC_U1
    LDC_u2        = TRUE_LDC_U2
    planet_radius = gt["planet_radius"]
    inclination   = jnp.deg2rad(gt["inclination"])
    semimajor     = jnp.abs(gt["impact_param"] / jnp.cos(inclination))
    eccentricity  = gt["ecc_h"]**2 + gt["ecc_k"]**2
    arg_periapsis = jnp.arctan2(gt["ecc_k"], gt["ecc_h"])
    P_orb         = TRUE_P_ORB

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
        jnp.broadcast_to(fac_flux,  (1,)),
    ])

    lc, _, _ = _compute_all_phases(
        all_ar_carts,
        planet_xyz_all,
        wavelength         = m["wavelength"],
        flux_quiet_interp  = m["flux_quiet"],
        flux_active_interp = flux_active,
        ldc_coeffs         = jnp.array([[LDC_u1, LDC_u2]]),
        I_profile          = m["I_profile"],
        mu_profile_pts     = m["mu_profile_pts"],
        x_disc             = m["x_disc"],
        y_disc             = m["y_disc"],
        mu_disc            = m["mu_disc"],
        vel_disc           = m["vel_disc"],
        star_pixel_rad     = spr,
        total_pixels       = m["total_pixels"],
        arsize_rads        = jnp.deg2rad(ar_size),
        k                  = planet_radius,
        ldc_mode           = m["ldc_mode"],
        ar_overlap_mode    = m["ar_overlap_mode"],
        plot_map_wavelength= m["plot_map_wavelength"],
        n                  = m["n"],
        flat_indices       = m["flat_indices"],
    )

    lc_reconstructed = np.array(lc)
    residuals = OBS_LIGHT_CURVE - lc_reconstructed
    residual_std = float(np.std(residuals))
    assert residual_std < 5 * float(SIGMA_NOISE), (
        f"Two-stage residual std {residual_std:.2e} is >>SIGMA_NOISE={SIGMA_NOISE:.2e}"
    )


# ---------------------------------------------------------------------------
# One-shot API edge cases
# ---------------------------------------------------------------------------

def test_call_sajax_activity_only_runs():
    """One-shot API with planet_radius=0 should produce a finite light curve."""
    ecc_h = GROUND_TRUTH["ecc_h"]
    ecc_k = GROUND_TRUTH["ecc_k"]
    semimajor_axis = jnp.abs(
        GROUND_TRUTH["impact_param"] / jnp.cos(jnp.deg2rad(GROUND_TRUTH["inclination"]))
    )
    result = _call_sajax(
        TIMES,
        jnp.array([GROUND_TRUTH["spot_lat"], GROUND_TRUTH["fac_lat"]]),
        jnp.array([GROUND_TRUTH["spot_long"], GROUND_TRUTH["fac_long"]]),
        jnp.array([GROUND_TRUTH["spot_size"], GROUND_TRUTH["fac_size"]]),
        np.stack([np.array([GROUND_TRUTH["spot_flux"]]),
                  np.array([GROUND_TRUTH["fac_flux"]])]),
        GROUND_TRUTH["p_rot"],
        0.0,
        semimajor_axis,
        jnp.deg2rad(GROUND_TRUTH["inclination"]),
        ecc_h**2 + ecc_k**2,
        jnp.arctan2(ecc_k, ecc_h),
        TRUE_P_ORB,
        TRUE_LDC_U1,
        TRUE_LDC_U2,
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
    m  = STATIC_MODEL

    ecc_h = gt["ecc_h"]
    ecc_k = gt["ecc_k"]
    gt_semimajor = jnp.abs(gt["impact_param"] / jnp.cos(jnp.deg2rad(gt["inclination"])))
    lc_one_shot = np.array(_call_sajax(
        TIMES,
        jnp.array([gt["spot_lat"], gt["fac_lat"]]),
        jnp.array([gt["spot_long"], gt["fac_long"]]),
        jnp.array([gt["spot_size"], gt["fac_size"]]),
        np.stack([np.array([gt["spot_flux"]]), np.array([gt["fac_flux"]])]),
        gt["p_rot"],
        gt["planet_radius"],
        gt_semimajor,
        jnp.deg2rad(gt["inclination"]),
        ecc_h**2 + ecc_k**2,
        jnp.arctan2(ecc_k, ecc_h),
        TRUE_P_ORB,
        TRUE_LDC_U1,
        TRUE_LDC_U2,
    )["lc"])

    P_rot         = gt["p_rot"]
    LDC_u1        = TRUE_LDC_U1
    LDC_u2        = TRUE_LDC_U2
    planet_radius = gt["planet_radius"]
    inclination   = jnp.deg2rad(gt["inclination"])
    semimajor     = jnp.abs(gt["impact_param"] / jnp.cos(inclination))
    eccentricity  = ecc_h**2 + ecc_k**2
    arg_periapsis = jnp.arctan2(ecc_k, ecc_h)
    P_orb         = TRUE_P_ORB
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
        jnp.broadcast_to(gt["fac_flux"],  (1,)),
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
    ax.scatter(times, OBS_LIGHT_CURVE, s=4, color="orange", alpha=0.6,
               label="Observations", zorder=1)
    ax.plot(times, lc_one_shot,  lw=2, color="steelblue", label="One-shot API",  zorder=2)
    ax.plot(times, lc_two_stage, lw=2, color="crimson",   linestyle="--",
            label="Two-stage API", zorder=3)
    ax.set_ylabel("Normalised flux")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(times, residuals * 1e6, lw=1.5, color="black")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline( float(SIGMA_NOISE) * 1e6, color="gray", linestyle="--",
               linewidth=0.8, label=r"±1σ noise")
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


# ---------------------------------------------------------------------------
# sample_initial_positions
# ---------------------------------------------------------------------------

def test_sample_initial_positions_shape_dict():
    n = 5
    positions = sample_initial_positions(jax.random.PRNGKey(0), n)
    for name in PRIOR_DISTRIBUTIONS:
        assert positions[name].shape == (n,), (
            f"Expected shape ({n},) for '{name}', got {positions[name].shape}"
        )


def test_sample_initial_positions_shape_flat():
    n = 4
    ndim = len(PRIOR_DISTRIBUTIONS)
    coords = sample_initial_positions(jax.random.PRNGKey(0), n, return_flat=True)
    assert coords.shape == (n, ndim), (
        f"Expected shape ({n}, {ndim}), got {coords.shape}"
    )


def test_sample_initial_positions_within_prior_bounds():
    """Constrained values from the dict form must fall within each prior's support."""
    constrain_fn = make_constrain_fn()
    positions = sample_initial_positions(jax.random.PRNGKey(7), 10)
    constrained = jax.vmap(constrain_fn)(positions)
    bounds = {
        "spot_lat":    (-90.0,  90.0),
        "spot_long":   (  0.0, 360.0),
        "spot_size":   (  1.0,  90.0),
        "fac_lat":     (-90.0,  90.0),
        "fac_long":    (  0.0, 360.0),
        "fac_size":    (  1.0,  90.0),
        "inclination": ( 80.0, 100.0),
        "ldc_q1":      (  0.0,   1.0),
        "ldc_q2":      (  0.0,   1.0),
    }
    for name, (lo, hi) in bounds.items():
        vals = np.array(constrained[name])
        assert np.all(vals >= lo) and np.all(vals <= hi), (
            f"'{name}' values outside [{lo}, {hi}]: {vals}"
        )


def test_sample_initial_positions_flat_gives_finite_log_density():
    """Flat positions unravelled with init_z's unravel_fn must yield finite log density.

    This catches key-ordering mismatches between sample_initial_positions and
    initialize_model: a scrambled flat vector produces garbage but no error.
    """
    import jax.flatten_util
    _, unravel_fn = jax.flatten_util.ravel_pytree(_INIT_Z)
    coords = sample_initial_positions(jax.random.PRNGKey(0), 4, return_flat=True)
    for i in range(4):
        z = unravel_fn(coords[i])
        ld = float(_LOG_DENSITY_FN(z))
        assert np.isfinite(ld), (
            f"Non-finite log density ({ld}) for flat position {i} — "
            "possible key-ordering mismatch between sample_initial_positions and initialize_model"
        )


def test_sample_initial_positions_deterministic():
    a = sample_initial_positions(jax.random.PRNGKey(42), 3, return_flat=True)
    b = sample_initial_positions(jax.random.PRNGKey(42), 3, return_flat=True)
    assert jnp.allclose(a, b)


