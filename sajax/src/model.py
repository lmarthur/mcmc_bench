"""
Model of an exoplanetary transit with a spot crossing for sampler benchmarking.

Default configuration: a star of radius 1 with quadratic limb darkening,
a single circular spot on the stellar disk. A planet transits the star,
with the transit light curve computed via jaxoplanet and the stellar
activity modulation computed via sajax.

The combined light curve is:
    lc_total = lc_activity * (1 + lc_transit)

where lc_activity comes from sajax (rotational modulation from the spot)
and lc_transit comes from jaxoplanet (limb-darkened transit model).

A NumPyro model is used to define the joint distribution over the spot
parameters and the observed data, and numpyro.infer.util.initialize_model is
used to extract BlackJAX-compatible inference functions in unconstrained space.
"""

from pathlib import Path

import jax
from sajax.core import _compute_all_phases, rotate_active_region, build_combined_model, compute_combined_light_curve
from sajax.planet import compute_planet_sky_positions
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
import astropy.units as u
from astropy.constants import G

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ---------------------------------------------------------------------------
# Host star physical constants (M1V)
# ---------------------------------------------------------------------------
TRUE_M_STAR_MSUN = 0.50          # M_sun
TRUE_R_STAR_RSUN = 0.50          # R_sun

# ---------------------------------------------------------------------------
# Planet transit parameters (jaxoplanet)
# ---------------------------------------------------------------------------
TRUE_PLANET_RADIUS = 0.1                                                 #stellar-radii
TRUE_P_ORB         = 1.0
A_METERS           = ( (G.value * (TRUE_M_STAR_MSUN * u.M_sun).to(u.kg).value * (TRUE_P_ORB * 24 * 3600)**2)/(4 * jnp.pi**2) )**(1/3)
TRUE_SEMI_MAJOR    = A_METERS / (TRUE_R_STAR_RSUN * (1.0 * u.R_sun).to(u.m).value)
TRUE_INCLINATION   = jnp.deg2rad(90.0)       
TRUE_ECCENTRICITY  = 0.0        
TRUE_ARG_PERIAPSIS = 0.0        
TRUE_LDC_U1        = 0.4        
TRUE_LDC_U2        = 0.2
TRUE_T0_TRANSIT    = 0.0        

#%%%% Calculate transit duration
# Convert angles to radians
# Impact parameter (eccentricity-corrected)
TRUE_IMPACT_PARAM = (
    TRUE_SEMI_MAJOR * jnp.cos(TRUE_INCLINATION)
    * (1 - TRUE_ECCENTRICITY**2) / (1 + TRUE_ECCENTRICITY * jnp.sin(TRUE_ARG_PERIAPSIS))
)
# Argument inside arcsin
arg = (
    (1/TRUE_SEMI_MAJOR)
    * jnp.sqrt((1 + TRUE_PLANET_RADIUS)**2 - TRUE_IMPACT_PARAM**2)
    / jnp.sin(TRUE_INCLINATION)
)
# Numerical safety
arg = jnp.clip(arg, -1.0, 1.0)
# Transit duration
TRUE_T14_TRANSIT = (
    (TRUE_P_ORB / jnp.pi)
    * jnp.sqrt(1 - TRUE_ECCENTRICITY**2) / (1 + TRUE_ECCENTRICITY * jnp.sin(TRUE_ARG_PERIAPSIS))
    * jnp.arcsin(arg)
)

# ---------------------------------------------------------------------------
# Fixed observation setup
# ---------------------------------------------------------------------------

# Time / phase setup
low_t = -3.5*TRUE_T14_TRANSIT                                           #days
high_t = 3.5*TRUE_T14_TRANSIT                                           #days
exposure_time = 250                                                     #seconds
num_t = jnp.floor((((high_t - low_t) * 24 * 3600)/exposure_time))       #number of points
TIMES = jnp.linspace(low_t, high_t, int(num_t))

TRUE_P_ROT = 0.5                                                        #days

# Synthetic flat spectra — single wavelength bin for broadband benchmark
WAVELENGTH = np.array([550.0])       # nm
FLUX_QUIET = np.array([1.0])
FLUX_ACTIVE_SPOT = np.array([0.7])   # spot is 30% darker

STELLAR_INC = 90.0          
STELLAR_GRID_SIZE = 100
VE = 2.0                 

SIGMA_NOISE = 100e-6     # ~100 ppm

# ---------------------------------------------------------------------------
# Physical constants for M1V host star (used to set physically motivated priors)
# ---------------------------------------------------------------------------
# T_eff ~ 3700 K for M1V
T_STAR = 3700.0                  # K — photospheric temperature

# TRUE_M_STAR_MSUN and TRUE_R_STAR_RSUN are defined at the top of the file.
R_STAR_SIGMA_FRAC = 0.05         # fractional uncertainty from spectroscopy

# ---------------------------------------------------------------------------
# Simulated "prior measurements" — realistic measurement noise applied to true
# values. Priors are centered on these, not on the true values, to simulate
# the realistic case where the prior comes from an independent measurement.
# ---------------------------------------------------------------------------
_meas_rng = np.random.default_rng(seed=7830)  # fixed seed for reproducibility

# a P ~ 0.5 day star achieves ~0.2% precision → σ ~ 0.001 days (~90 s).
P_ROT_SIGMA = 0.001              # days
P_ROT_MEASURED = float(TRUE_P_ROT + _meas_rng.normal(0.0, P_ROT_SIGMA))

# Orbital period: ~5 transit observations, each with ~3 s timing uncertainty;
# σ_P ≈ σ_t_c / (N-1) ~ 3 s / 4 ~ 1e-5 days.
P_ORB_SIGMA = 1e-5               # days
P_ORB_MEASURED = float(TRUE_P_ORB + _meas_rng.normal(0.0, P_ORB_SIGMA))

# Stellar radius: 5% fractional uncertainty from spectroscopic classification.
R_STAR_MEASURED_RSUN = float(
    TRUE_R_STAR_RSUN + _meas_rng.normal(0.0, TRUE_R_STAR_RSUN * R_STAR_SIGMA_FRAC)
)

# True temperature deviation for ground-truth spot:
# FLUX_ACTIVE_SPOT[0] = (T_active/T_STAR)^4  →  T_active = T_STAR * flux^0.25
# delta_T = T_active - T_STAR
TRUE_DELTA_T = float(T_STAR * (FLUX_ACTIVE_SPOT[0] ** 0.25 - 1.0))

# ---------------------------------------------------------------------------
# Ground-truth spot
# ---------------------------------------------------------------------------
TRUE_SPOT_LAT = 5.0
TRUE_SPOT_LONG = 5.0
TRUE_SPOT_SIZE = 11.0

# ---------------------------------------------------------------------------
# Prior bounds: active regions
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = -90.0, 90.0
LONG_MIN, LONG_MAX = 0.0, 360.0
SIZE_MIN, SIZE_MAX = 1.0, 90.0
FLUX_MIN, FLUX_MAX = 0.1, 2.0

# ---------------------------------------------------------------------------
# Prior bounds: planet transit
# ---------------------------------------------------------------------------
PLANET_RADIUS_MIN, PLANET_RADIUS_MAX = 0.001, 0.3  # Rp/Rs
SEMI_MAJOR_MIN, SEMI_MAJOR_MAX = 0.0, 10.0         # a/R* (semi-major axis in stellar radii)
INCLINATION_MIN, INCLINATION_MAX = 80.0, 100.0     # inclination [degrees]

# ---------------------------------------------------------------------------
# Prior distributions — single source of truth for all samplers.
# ---------------------------------------------------------------------------

# Wide (physically motivated) priors
PRIOR_DISTRIBUTIONS = {
    # --- Spot geometry ---
    # sin_lat ~ Uniform(-1, 1): sampling sin(latitude) gives isotropic distribution
    # on the sphere — corrects for the cos(lat) Jacobian factor so that equal
    # prior probability is assigned to equal solid angles.
    "sin_lat":       dist.Uniform(-1.0, 1.0),
    "spot_long":     dist.Uniform(LONG_MIN, LONG_MAX), # TODO: Consider making this a circular/closed prior
    # log-uniform: spot size is a scale parameter (Jeffreys prior);
    # range spans detection floor (~1°) to giant polar spots (~45°).
    "spot_size":     dist.LogUniform(1.0, 45.0),

    # --- Spot flux via temperature deviation ---
    # delta_T = T_active - T_star; Normal(0, 300 K) spans spot (negative)
    # and facula (positive) regimes, with most probability near featureless
    # photosphere (delta_T = 0).
    "delta_T":       dist.Normal(0.0, 300.0),

    # --- Stellar rotation ---
    # Prior centered on independently measured rotation period.
    # σ ~ 0.2% for P~0.5 day star.
    "p_rot":         dist.Normal(P_ROT_MEASURED, P_ROT_SIGMA),

    # --- Planet geometry ---
    # planet_radius in R_p/R_star: log-uniform (scale parameter).
    # [0.01, 0.3]: lower bound ~ detection floor at 100 ppm; upper bound generous.
    "planet_radius": dist.LogUniform(0.01, 0.3),

    # semimajor_axis in a/R_star: log-uniform (scale parameter).
    # Lower bound: Roche limit for rocky planet.
    # Upper bound: detectability limit (~30-day orbit for M1V → a ~ 50).
    # LogUniform prior on a encodes geometric transit selection bias:
    # p(a) ∝ 1/a matches transit probability P_transit ∝ R_star/a.
    "semimajor_axis": dist.LogUniform(2.5, 50.0),

    # impact_param = (a/R_star) * cos(i): exact geometric prior on b given
    # that a transit was observed is Uniform (see derivation in model docs).
    "impact_param":  dist.Uniform(-1.0, 1.0),

    # --- Eccentricity (Ford parameterization) ---
    # ecc_h = √e · cos(ω),  ecc_k = √e · sin(ω).
    # Independent Normal(0, σ) components → e ~ Rayleigh(σ) prior.
    # P_orb = 1 day → tidal circularization timescale << stellar age for M
    # dwarfs; eccentricity is expected to be near zero.
    # σ = 0.05 → 90th percentile e ~ 0.11.
    "ecc_h":         dist.Normal(0.0, 0.05),
    "ecc_k":         dist.Normal(0.0, 0.05),

    # --- Orbital period ---
    # Prior centered on independently measured orbital period.
    # ~5 transit observations, each with ~3 s timing uncertainty;
    # σ_P ≈ σ_tc / (N-1) ~ 3s/4 ~ 1e-5 days.
    "P_orb":         dist.Normal(P_ORB_MEASURED, P_ORB_SIGMA),

    # --- Limb darkening (Kipping parameterization) ---
    # u1 = 2√q1·q2,  u2 = √q1·(1 - 2q2).
    "ldc_q1":        dist.Uniform(0.0, 1.0),
    "ldc_q2":        dist.Uniform(0.0, 1.0),
}

# ---------------------------------------------------------------------------
# Narrow (debugging) prior block — comment out the wide block above and
# uncomment this to test sampler near the ground truth.
# ---------------------------------------------------------------------------
# import jax.numpy as _jnp
# PRIOR_DISTRIBUTIONS = {
#     "sin_lat":       dist.Uniform(
#                          float(_jnp.sin(_jnp.deg2rad(TRUE_SPOT_LAT - 1.0))),
#                          float(_jnp.sin(_jnp.deg2rad(TRUE_SPOT_LAT + 1.0)))),
#     "spot_long":     dist.Uniform(TRUE_SPOT_LONG - 1.0, TRUE_SPOT_LONG + 1.0),
#     "spot_size":     dist.Uniform(TRUE_SPOT_SIZE - 1.0, TRUE_SPOT_SIZE + 1.0),
#     "delta_T":       dist.Normal(TRUE_DELTA_T, 20.0),
#     "p_rot":         dist.Normal(P_ROT_MEASURED, P_ROT_SIGMA),
#     "planet_radius": dist.LogUniform(0.095, 0.115),
#     "semimajor_axis":dist.LogUniform(float(TRUE_SEMI_MAJOR) * 0.95,
#                                      float(TRUE_SEMI_MAJOR) * 1.05),
#     "impact_param":  dist.Uniform(-0.1, 0.1),
#     "ecc_h":         dist.Uniform(-0.01, 0.01),
#     "ecc_k":         dist.Uniform(-0.01, 0.01),
#     "P_orb":         dist.Normal(P_ORB_MEASURED, P_ORB_SIGMA),
#     "ldc_q1":        dist.Uniform(0.34, 0.38),
#     "ldc_q2":        dist.Uniform(0.31, 0.35),
# }

# ---------------------------------------------------------------------------
# Pre-build the Static Model for MCMC (Two-Stage API)
# ---------------------------------------------------------------------------
STATIC_PARAMS_SAJAX = dict(
    ldc_coeffs=[TRUE_LDC_U1, TRUE_LDC_U2], 
    inc_star=STELLAR_INC,         
) 

STATIC_TRANSIT_PARAMS = dict(
    t0           = TRUE_T0_TRANSIT,
    period       = TRUE_P_ORB,
    a_over_rstar = TRUE_SEMI_MAJOR,
    inclination  = TRUE_INCLINATION,    
    k            = TRUE_PLANET_RADIUS,            
    ecc          = TRUE_ECCENTRICITY,
    omega_peri   = TRUE_ARG_PERIAPSIS,
)

STATIC_MODEL = build_combined_model(
    wavelength        = WAVELENGTH,
    flux_quiet        = FLUX_QUIET,
    params            = STATIC_PARAMS_SAJAX,
    times             = TIMES,
    P_rot             = TRUE_P_ROT,
    transit_params    = STATIC_TRANSIT_PARAMS,
    stellar_grid_size = STELLAR_GRID_SIZE,
    ve                = VE,
    ldc_mode          = "quadratic",
)


# ---------------------------------------------------------------------------
# Transit light curve (Plotting / One-Shot API)
# ---------------------------------------------------------------------------

def _call_sajax(
    times: jnp.ndarray,
    ar_lat: jnp.ndarray,
    ar_long: jnp.ndarray,
    ar_size: jnp.ndarray,
    flux_active: jnp.ndarray,
    P_rot: float,
    planet_radius: float,
    semimajor_axis: float,
    inclination: float,
    eccentricity: float,
    arg_periapsis: float,
    P_orb: float,
    LDC_u1: float,
    LDC_u2: float,
) -> dict:
    """
    Call sajax's compute_combined_light_curve and return the broadband light curve.
    Used primarily for plotting and ground-truth generation.
    """

    params_sajax = dict(
        ldc_coeffs=[LDC_u1, LDC_u2],  
        inc_star=STELLAR_INC,         
    ) 

    transit_params = dict(
        t0           = TRUE_T0_TRANSIT,
        period       = P_orb,
        a_over_rstar = semimajor_axis,
        inclination  = inclination,    
        k            = planet_radius,            
        ecc          = eccentricity,
        omega_peri   = arg_periapsis,
    )

    return compute_combined_light_curve(
        wavelength        = WAVELENGTH,
        flux_quiet        = FLUX_QUIET,
        flux_active       = flux_active,
        params            = params_sajax,
        ar_lat            = ar_lat.tolist(),
        ar_long           = ar_long.tolist(),
        ar_size           = ar_size.tolist(),
        times             = times,
        P_rot             = P_rot,
        transit_params    = transit_params,
        stellar_grid_size = STELLAR_GRID_SIZE,
        ve                = VE,
        ldc_mode          = "quadratic",
        plot_map_wavelength = WAVELENGTH[0] # Ensure maps are generated for this wavelength
    )

# ---------------------------------------------------------------------------
# Synthetic observations
# ---------------------------------------------------------------------------

def generate_observations(seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic noisy light curve from ground-truth parameters.
    Includes both stellar activity and planet transit.
    """

    lc_true = np.array(
        _call_sajax(
            TIMES,
            jnp.array([TRUE_SPOT_LAT]),
            jnp.array([TRUE_SPOT_LONG]),
            jnp.array([TRUE_SPOT_SIZE]),
            np.stack([FLUX_ACTIVE_SPOT]),
            TRUE_P_ROT,
            TRUE_PLANET_RADIUS,
            TRUE_SEMI_MAJOR,
            TRUE_INCLINATION,
            TRUE_ECCENTRICITY,
            TRUE_ARG_PERIAPSIS,
            TRUE_P_ORB,
            TRUE_LDC_U1,
            TRUE_LDC_U2,
        )["lc"]
    )

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, SIGMA_NOISE, size=lc_true.shape)
    return lc_true + noise


# Generate observations once at import time
OBS_LIGHT_CURVE = generate_observations(seed=42)

LC_TRUE = np.array(
    _call_sajax(
        TIMES,
        jnp.array([TRUE_SPOT_LAT]),
        jnp.array([TRUE_SPOT_LONG]),
        jnp.array([TRUE_SPOT_SIZE]),
        np.stack([FLUX_ACTIVE_SPOT]),
        TRUE_P_ROT,
        TRUE_PLANET_RADIUS,
        TRUE_SEMI_MAJOR,
        TRUE_INCLINATION,
        TRUE_ECCENTRICITY,
        TRUE_ARG_PERIAPSIS,
        TRUE_P_ORB,
        TRUE_LDC_U1,
        TRUE_LDC_U2,
    )["lc"]
)

# ---------------------------------------------------------------------------
# NumPyro model (MCMC API)
# ---------------------------------------------------------------------------
def sajax_model(y_obs: jnp.ndarray = jnp.array(OBS_LIGHT_CURVE), model_dict: dict = STATIC_MODEL):
    """
    NumPyro model for the spot + planet posterior.
    Uses pre-built STATIC_MODEL for JIT-compatibility.
    """
    sin_lat = numpyro.sample("sin_lat", PRIOR_DISTRIBUTIONS["sin_lat"])
    spot_lat = numpyro.deterministic("spot_lat", jnp.rad2deg(jnp.arcsin(sin_lat)))
    spot_long = numpyro.sample("spot_long", PRIOR_DISTRIBUTIONS["spot_long"])
    spot_size = numpyro.sample("spot_size", PRIOR_DISTRIBUTIONS["spot_size"])
    delta_T = numpyro.sample("delta_T", PRIOR_DISTRIBUTIONS["delta_T"])
    spot_flux = numpyro.deterministic("spot_flux", ((T_STAR + delta_T) / T_STAR) ** 4)

    P_rot = numpyro.sample("p_rot", PRIOR_DISTRIBUTIONS["p_rot"])
    ldc_q1 = numpyro.sample("ldc_q1", PRIOR_DISTRIBUTIONS["ldc_q1"])
    ldc_q2 = numpyro.sample("ldc_q2", PRIOR_DISTRIBUTIONS["ldc_q2"])
    LDC_u1 = numpyro.deterministic("ldc_u1", 2 * jnp.sqrt(ldc_q1) * ldc_q2)
    LDC_u2 = numpyro.deterministic("ldc_u2", jnp.sqrt(ldc_q1) * (1 - 2 * ldc_q2))

    # Planet parameters
    planet_radius = numpyro.sample("planet_radius", PRIOR_DISTRIBUTIONS["planet_radius"])
    semimajor_axis = numpyro.sample("semimajor_axis", PRIOR_DISTRIBUTIONS["semimajor_axis"])
    impact_param = numpyro.sample("impact_param", PRIOR_DISTRIBUTIONS["impact_param"])
    inclination = numpyro.deterministic(
        "inclination", jnp.rad2deg(jnp.arccos(impact_param / semimajor_axis))
    )
    ecc_h = numpyro.sample("ecc_h", PRIOR_DISTRIBUTIONS["ecc_h"])
    ecc_k = numpyro.sample("ecc_k", PRIOR_DISTRIBUTIONS["ecc_k"])
    eccentricity = numpyro.deterministic("eccentricity", ecc_h**2 + ecc_k**2)
    arg_periapsis = numpyro.deterministic("arg_periapsis", jnp.arctan2(ecc_k, ecc_h))
    P_orb = numpyro.sample("P_orb", PRIOR_DISTRIBUTIONS["P_orb"])

    # --- DYNAMIC CALCULATIONS (JAX) ---

    # Recompute Stellar Rotation Phases based on sampled p_rot
    # We use the static 'times' from the model_dict
    dynamic_phases_rot = (model_dict["times"] / P_rot * 360.0) % 360.0

    planet_xyz_all = compute_planet_sky_positions(
        times=model_dict["times"],
        t0=TRUE_T0_TRANSIT,
        period=P_orb,
        a_over_rstar=semimajor_axis,
        inclination=jnp.deg2rad(inclination),
        ecc=eccentricity,
        omega_peri=arg_periapsis
    )

    # Rotate Active Regions
    ar_lat = jnp.array([spot_lat])
    ar_long = jnp.array([spot_long])
    ar_size = jnp.array([spot_size])

    spr = model_dict["star_pixel_rad"]
    ar_cart = jnp.stack([
        spr * jnp.sin(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
        spr * jnp.sin(jnp.deg2rad(ar_lat)),
        spr * jnp.cos(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
    ], axis=-1)

    # Vmap rotation over the dynamic phases
    all_ar_carts = jax.vmap(lambda p: jax.vmap(
        lambda c: rotate_active_region(c, p, model_dict["inc_star"])
    )(ar_cart))(dynamic_phases_rot)

    # Integrate Light Curve
    flux_active = jnp.stack([
        jnp.broadcast_to(spot_flux, (1,)),
    ])

    # Compute Flux (JAX)
    lc, _, _ = _compute_all_phases(
        all_ar_carts,
        planet_xyz_all,
        wavelength=model_dict["wavelength"],
        flux_quiet_interp=model_dict["flux_quiet"],
        flux_active_interp=flux_active,
        ldc_coeffs=jnp.array([[LDC_u1, LDC_u2]]),
        I_profile=model_dict["I_profile"],
        mu_profile_pts=model_dict["mu_profile_pts"],
        x_disc=model_dict["x_disc"],
        y_disc=model_dict["y_disc"],
        mu_disc=model_dict["mu_disc"],
        vel_disc=model_dict["vel_disc"],
        star_pixel_rad=spr,
        total_pixels=model_dict["total_pixels"],
        arsize_rads=jnp.deg2rad(ar_size),
        k=planet_radius,
        ldc_mode=model_dict["ldc_mode"],
        ar_overlap_mode=model_dict["ar_overlap_mode"],
        plot_map_wavelength=model_dict["plot_map_wavelength"],
        n=model_dict["n"],
        flat_indices=model_dict["flat_indices"]
    )

    numpyro.sample("y_obs", dist.Normal(lc, SIGMA_NOISE), obs=y_obs)


def make_inference_fns(rng_key, y_obs: np.ndarray = OBS_LIGHT_CURVE, model_dict: dict = STATIC_MODEL):
    """
    Returns (log_density_fn, postprocess_fn, init_z) for use with BlackJAX/emcee_jax.

    All three operate in the unconstrained space that samplers work in:
      - log_density_fn : dict of unconstrained params → scalar log density
      - postprocess_fn : dict of unconstrained params → dict of constrained physical
                         values, including deterministic sites
      - init_z         : initial unconstrained position dict, sampled from the prior
    """
    param_info, potential_fn, postprocess_fn, _ = initialize_model(
        rng_key,
        sajax_model,
        model_args=(),
        model_kwargs={"y_obs": jnp.array(y_obs), "model_dict": model_dict},
    )
    return lambda x: -potential_fn(x), postprocess_fn, param_info.z


def make_constrain_fn():
    """
    Returns a lightweight unconstrained → constrained mapping.

    Unlike postprocess_fn from initialize_model, this never calls the forward
    model.  It only applies the analytical bijection for each prior distribution
    and computes the deterministic sites analytically.

    Safe to jax.vmap over arbitrarily large sample arrays without the memory
    blowup that occurs when postprocess_fn (which runs _compute_all_phases) is
    vmapped over thousands of MCMC samples.
    """
    from numpyro.distributions import biject_to
    transforms = {name: biject_to(d.support) for name, d in PRIOR_DISTRIBUTIONS.items()}

    def constrain_fn(z):
        c = {name: transforms[name](z[name]) for name in transforms}
        # Deterministic sites (all derived quantities computed analytically)
        c["spot_lat"] = jnp.rad2deg(jnp.arcsin(c["sin_lat"]))
        c["spot_flux"] = ((T_STAR + c["delta_T"]) / T_STAR) ** 4
        c["inclination"] = jnp.rad2deg(jnp.arccos(c["impact_param"] / c["semimajor_axis"]))
        c["eccentricity"]  = c["ecc_h"]**2 + c["ecc_k"]**2
        c["arg_periapsis"] = jnp.arctan2(c["ecc_k"], c["ecc_h"])
        c["ldc_u1"] = 2 * jnp.sqrt(c["ldc_q1"]) * c["ldc_q2"]
        c["ldc_u2"] = jnp.sqrt(c["ldc_q1"]) * (1 - 2 * c["ldc_q2"])
        return c

    return constrain_fn


def sample_initial_positions(key: jax.Array, n: int, return_flat: bool = False):
    """Sample n starting positions from the prior in unconstrained space.

    If return_flat=True, returns shape (n, ndim) suitable for ensemble samplers.
    If return_flat=False, returns a pytree dict with each value shape (n,).
    """
    import jax.flatten_util
    from numpyro.distributions import biject_to
    inv_transforms = {name: biject_to(d.support).inv for name, d in PRIOR_DISTRIBUTIONS.items()}

    chain_keys = jax.random.split(key, n)
    positions = []
    for ck in chain_keys:
        param_keys = jax.random.split(ck, len(PRIOR_DISTRIBUTIONS))
        z_dict = {
            name: inv_transforms[name](d.sample(pk))
            for pk, (name, d) in zip(param_keys, PRIOR_DISTRIBUTIONS.items())
        }
        positions.append(z_dict)

    if return_flat:
        flat_positions = [jax.flatten_util.ravel_pytree(z)[0] for z in positions]
        return jnp.stack(flat_positions)
    return jax.tree.map(lambda *arrays: jnp.stack(arrays), *positions)


def make_log_likelihood(y_obs: np.ndarray = OBS_LIGHT_CURVE, model_dict: dict = STATIC_MODEL):
    """
    Returns log p(y_obs | params) for a constrained (physical) parameter dict.
    For use with nested samplers (e.g. JAXNS) that handle the prior separately.

    New parameterization (matching PRIOR_DISTRIBUTIONS):
      - sin_lat    : sin of spot latitude → spot_lat via arcsin
      - delta_T    : temperature deviation (K) → spot_flux via Stefan-Boltzmann
      - semimajor_axis : a/R_star → inclination derived as arccos(impact_param/semimajor_axis)
      - ecc_h/ecc_k, ldc_q1/ldc_q2 are the sampled parameterization;
        eccentricity/arg_periapsis/ldc_u1/ldc_u2 are derived internally.
    """
    y_obs_arr = jnp.array(y_obs)

    def log_likelihood(params):
        # --- Spot geometry: sin_lat → spot_lat ---
        sin_lat   = params["sin_lat"]
        spot_lat  = jnp.rad2deg(jnp.arcsin(sin_lat))
        spot_long = params["spot_long"]
        spot_size = params["spot_size"]

        # --- Spot flux: delta_T → spot_flux (Stefan-Boltzmann) ---
        delta_T   = params["delta_T"]
        spot_flux = ((T_STAR + delta_T) / T_STAR) ** 4

        # --- Rotation and limb darkening ---
        P_rot  = params["p_rot"]
        ldc_q1 = params["ldc_q1"]
        ldc_q2 = params["ldc_q2"]
        LDC_u1 = 2 * jnp.sqrt(ldc_q1) * ldc_q2
        LDC_u2 = jnp.sqrt(ldc_q1) * (1 - 2 * ldc_q2)

        # --- Orbital geometry: semimajor_axis + impact_param → inclination ---
        planet_radius  = params["planet_radius"]
        semimajor_axis = params["semimajor_axis"]
        impact_param   = params["impact_param"]
        inclination    = jnp.rad2deg(jnp.arccos(impact_param / semimajor_axis))

        # --- Eccentricity ---
        ecc_h = params["ecc_h"]
        ecc_k = params["ecc_k"]
        eccentricity  = ecc_h ** 2 + ecc_k ** 2
        arg_periapsis = jnp.arctan2(ecc_k, ecc_h)
        P_orb = params["P_orb"]

        ar_lat  = jnp.array([spot_lat])
        ar_long = jnp.array([spot_long])
        ar_size = jnp.array([spot_size])

        dynamic_phases_rot = (model_dict["times"] / P_rot * 360.0) % 360.0

        planet_xyz_all = compute_planet_sky_positions(
            times=model_dict["times"],
            t0=TRUE_T0_TRANSIT,
            period=P_orb,
            a_over_rstar=semimajor_axis,
            inclination=jnp.deg2rad(inclination),
            ecc=eccentricity,
            omega_peri=arg_periapsis,
        )

        spr = model_dict["star_pixel_rad"]
        ar_cart = jnp.stack([
            spr * jnp.sin(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
            spr * jnp.sin(jnp.deg2rad(ar_lat)),
            spr * jnp.cos(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
        ], axis=-1)

        all_ar_carts = jax.vmap(lambda p: jax.vmap(
            lambda c: rotate_active_region(c, p, model_dict["inc_star"])
        )(ar_cart))(dynamic_phases_rot)

        flux_active = jnp.stack([
            jnp.broadcast_to(spot_flux, (1,)),
        ])

        lc, _, _ = _compute_all_phases(
            all_ar_carts,
            planet_xyz_all,
            wavelength=model_dict["wavelength"],
            flux_quiet_interp=model_dict["flux_quiet"],
            flux_active_interp=flux_active,
            ldc_coeffs=jnp.array([[LDC_u1, LDC_u2]]),
            I_profile=model_dict["I_profile"],
            mu_profile_pts=model_dict["mu_profile_pts"],
            x_disc=model_dict["x_disc"],
            y_disc=model_dict["y_disc"],
            mu_disc=model_dict["mu_disc"],
            vel_disc=model_dict["vel_disc"],
            star_pixel_rad=spr,
            total_pixels=model_dict["total_pixels"],
            arsize_rads=jnp.deg2rad(ar_size),
            k=planet_radius,
            ldc_mode=model_dict["ldc_mode"],
            ar_overlap_mode=model_dict["ar_overlap_mode"],
            plot_map_wavelength=model_dict["plot_map_wavelength"],
            n=model_dict["n"],
            flat_indices=model_dict["flat_indices"],
        )

        return dist.Normal(lc, SIGMA_NOISE).log_prob(y_obs_arr).sum()

    return log_likelihood


def make_log_ref(rng_key):
    """
    Returns the log density of the prior in unconstrained space.

    Used as the DEO/SEO reference distribution.  Both log_density_fn (from
    make_inference_fns) and this function include the same log-Jacobian
    correction, so likelihood-tempering interpolates consistently.
    """
    def _prior_only():
        for name, d in PRIOR_DISTRIBUTIONS.items():
            numpyro.sample(name, d)

    _, prior_potential_fn, _, _ = initialize_model(
        rng_key, _prior_only, model_args=(), model_kwargs={}
    )
    return lambda x: -prior_potential_fn(x)

def make_log_density(y_obs: np.ndarray = OBS_LIGHT_CURVE, model_dict: dict = STATIC_MODEL):
    """
    Returns log p(θ | y) ∝ log p(y | θ) + log p(θ) for a FLAT ARRAY of
    constrained parameters (ordered as PARAM_NAMES).

    For use with samplers that work directly in constrained space
    (e.g. DEO parallel tempering with RWMH).
    """
    log_likelihood_fn = make_log_likelihood(y_obs, model_dict)

    def log_density(x):
        # x is a flat jnp array of shape (ndim,), ordered as PARAM_NAMES
        params = {name: x[i] for i, name in enumerate(PARAM_NAMES)}
        log_lik = log_likelihood_fn(params)
        log_prior = jnp.array(0.0)
        for i, name in enumerate(PARAM_NAMES):
            log_prior = log_prior + PRIOR_DISTRIBUTIONS[name].log_prob(x[i])
        return log_lik + log_prior

    return log_density
    
# ---------------------------------------------------------------------------
# Diagnostic forward-model helpers
# ---------------------------------------------------------------------------

def compute_lc_from_constrained(constrained: dict, model_dict: dict = STATIC_MODEL) -> jnp.ndarray:
    """Compute the light curve from constrained parameters (mirrors sajax_model forward pass)."""
    dynamic_phases_rot = (model_dict["times"] / constrained["p_rot"] * 360.0) % 360.0

    planet_xyz_all = compute_planet_sky_positions(
        times=model_dict["times"],
        t0=TRUE_T0_TRANSIT,
        period=constrained["P_orb"],
        a_over_rstar=constrained["semimajor_axis"],
        inclination=jnp.deg2rad(constrained["inclination"]),
        ecc=constrained["eccentricity"],
        omega_peri=constrained["arg_periapsis"],
    )

    ar_lat  = jnp.array([constrained["spot_lat"]])
    ar_long = jnp.array([constrained["spot_long"]])
    ar_size = jnp.array([constrained["spot_size"]])

    spr = model_dict["star_pixel_rad"]
    ar_cart = jnp.stack([
        spr * jnp.sin(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
        spr * jnp.sin(jnp.deg2rad(ar_lat)),
        spr * jnp.cos(jnp.deg2rad(ar_long)) * jnp.cos(jnp.deg2rad(ar_lat)),
    ], axis=-1)

    all_ar_carts = jax.vmap(lambda p: jax.vmap(
        lambda c: rotate_active_region(c, p, model_dict["inc_star"])
    )(ar_cart))(dynamic_phases_rot)

    flux_active = jnp.stack([
        jnp.broadcast_to(jnp.asarray(constrained["spot_flux"]), (1,)),
    ])

    lc, _, _ = _compute_all_phases(
        all_ar_carts,
        planet_xyz_all,
        wavelength=model_dict["wavelength"],
        flux_quiet_interp=model_dict["flux_quiet"],
        flux_active_interp=flux_active,
        ldc_coeffs=jnp.array([[constrained["ldc_u1"], constrained["ldc_u2"]]]),
        I_profile=model_dict["I_profile"],
        mu_profile_pts=model_dict["mu_profile_pts"],
        x_disc=model_dict["x_disc"],
        y_disc=model_dict["y_disc"],
        mu_disc=model_dict["mu_disc"],
        vel_disc=model_dict["vel_disc"],
        star_pixel_rad=spr,
        total_pixels=model_dict["total_pixels"],
        arsize_rads=jnp.deg2rad(ar_size),
        k=constrained["planet_radius"],
        ldc_mode=model_dict["ldc_mode"],
        ar_overlap_mode=model_dict["ar_overlap_mode"],
        plot_map_wavelength=model_dict["plot_map_wavelength"],
        n=model_dict["n"],
        flat_indices=model_dict["flat_indices"],
    )
    return lc


def compute_chi2(constrained: dict, model_dict: dict = STATIC_MODEL) -> float:
    """Reduced chi-squared (obs vs model) for a set of constrained parameters."""
    lc = compute_lc_from_constrained(constrained, model_dict)
    n = len(TIMES)
    return float(jnp.sum(((jnp.array(OBS_LIGHT_CURVE) - lc) / SIGMA_NOISE) ** 2) / n)


# ---------------------------------------------------------------------------
# Ground truth dict — for sampler diagnostics.
# Contains TRUE values (not measured values) so posteriors can be compared
# to the actual data-generating parameters.
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    # sin(TRUE_SPOT_LAT in radians) — the sampled variable in the new parameterization
    "sin_lat":       float(jnp.sin(jnp.deg2rad(TRUE_SPOT_LAT))),
    "spot_long":     float(TRUE_SPOT_LONG),
    "spot_size":     float(TRUE_SPOT_SIZE),
    # Temperature deviation: TRUE_DELTA_T = T_STAR * (FLUX_ACTIVE_SPOT[0]^0.25 - 1)
    "delta_T":       float(TRUE_DELTA_T),
    # Use true rotation period (prior is centered on measured value, but truth is TRUE_P_ROT)
    "p_rot":         float(TRUE_P_ROT),
    "planet_radius": float(TRUE_PLANET_RADIUS),
    # semimajor axis: a/R_star computed from Kepler's 3rd law at model setup
    "semimajor_axis": float(TRUE_SEMI_MAJOR),
    # impact parameter: b = (a/R_star) * cos(i); i=90° → b=0 for edge-on
    "impact_param":  float(TRUE_SEMI_MAJOR * jnp.cos(TRUE_INCLINATION)),
    # ecc_h = √e·cos(ω); TRUE_ECCENTRICITY=0 → ecc_h=0
    "ecc_h":         float(jnp.sqrt(TRUE_ECCENTRICITY) * jnp.cos(TRUE_ARG_PERIAPSIS)),
    "ecc_k":         float(jnp.sqrt(TRUE_ECCENTRICITY) * jnp.sin(TRUE_ARG_PERIAPSIS)),
    # Use true orbital period (prior centered on measured, truth is TRUE_P_ORB)
    "P_orb":         float(TRUE_P_ORB),
    # Kipping q parameters: q1 = (u1+u2)², q2 = u1/(2(u1+u2))
    "ldc_q1":        float((TRUE_LDC_U1 + TRUE_LDC_U2) ** 2),
    "ldc_q2":        float(TRUE_LDC_U1 / (2 * (TRUE_LDC_U1 + TRUE_LDC_U2))),
}

PARAM_NAMES = list(GROUND_TRUTH.keys())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model(filename: str = "spot_transit_light_curve.png"):
    """
    Saves a diagnostic plot showing:
      - The ground-truth combined light curve and noisy observations
      - The transit component alone
      - The activity component alone
    """
    #Activity only
    lc_activity = np.array(
        _call_sajax(
            TIMES,
            jnp.array([TRUE_SPOT_LAT]),
            jnp.array([TRUE_SPOT_LONG]),
            jnp.array([TRUE_SPOT_SIZE]),
            np.stack([FLUX_ACTIVE_SPOT]),
            TRUE_P_ROT,
            0.,
            TRUE_SEMI_MAJOR,
            TRUE_INCLINATION,
            TRUE_ECCENTRICITY,
            TRUE_ARG_PERIAPSIS,
            TRUE_P_ORB,
            TRUE_LDC_U1,
            TRUE_LDC_U2,
        )["lc"]
    )
    #Transit only
    lc_transit = np.array(
        _call_sajax(
            TIMES,
            jnp.array([TRUE_SPOT_LAT]),
            jnp.array([TRUE_SPOT_LONG]),
            jnp.array([0.0001]),
            np.stack([FLUX_ACTIVE_SPOT]),
            TRUE_P_ROT,
            TRUE_PLANET_RADIUS,
            TRUE_SEMI_MAJOR,
            TRUE_INCLINATION,
            TRUE_ECCENTRICITY,
            TRUE_ARG_PERIAPSIS,
            TRUE_P_ORB,
            TRUE_LDC_U1,
            TRUE_LDC_U2,
        )["lc"]
    )
    #Combination
    full_result = _call_sajax(
            TIMES,
            jnp.array([TRUE_SPOT_LAT]),
            jnp.array([TRUE_SPOT_LONG]),
            jnp.array([TRUE_SPOT_SIZE]),
            np.stack([FLUX_ACTIVE_SPOT]),
            TRUE_P_ROT,
            TRUE_PLANET_RADIUS,
            TRUE_SEMI_MAJOR,
            TRUE_INCLINATION,
            TRUE_ECCENTRICITY,
            TRUE_ARG_PERIAPSIS,
            TRUE_P_ORB,
            TRUE_LDC_U1,
            TRUE_LDC_U2,
        )
    
    lc_combined = np.array(full_result['lc'])
    star_maps = np.array(full_result["star_maps"])

    # Define snapshot indices: Ingress, Mid-transit, Egress
    t_snap = [TRUE_T0_TRANSIT - 0.7*TRUE_T14_TRANSIT, 
              TRUE_T0_TRANSIT, 
              TRUE_T0_TRANSIT + 0.7*TRUE_T14_TRANSIT]
    idx_snap = [np.argmin(np.abs(TIMES - t)) for t in t_snap]

    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    # --- Top Row: System Snapshots ---
    for i, idx in enumerate(idx_snap):
        ax = fig.add_subplot(gs[0, i])
        # Display the map for the first (and only) wavelength bin
        im = ax.imshow(star_maps[idx, :, :], origin='lower', cmap='inferno', extent=[-1, 1, -1, 1])
        phase_snap = (TIMES[idx] / TRUE_P_ROT * 360.0) % 360.0    
        ax.set_title(rf"Phase: {phase_snap:.0f}$^\circ$", fontsize=12)
        ax.axis('off')
        if i == 2:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')

    # --- Bottom Row: Light Curve and Residuals ---
    ax_lc = fig.add_subplot(gs[1, :2])
    ax_res = fig.add_subplot(gs[1, 2])

    # Light Curve
    ax_lc.plot(TIMES, lc_activity - 0.002, '--', lw=2, alpha=0.8, color="green", label='Star', zorder=3)
    ax_lc.plot(TIMES, lc_transit, '--', lw=2, alpha=0.8, color="crimson", label='Planet', zorder=3)
    ax_lc.plot(TIMES, lc_combined, lw=3, label="Star + Planet", color="steelblue", zorder=1)
    ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=10, color="orange", label="Observations", zorder=2)
    
    # Mark snapshot locations on the light curve
    for t in t_snap:
        ax_lc.axvline(t, color='black', alpha=0.2, linestyle=':')

    ax_lc.legend(loc='lower left', frameon=False)
    ax_lc.set_xlabel("Time [days]")
    ax_lc.set_ylabel("Normalised flux")
    ax_lc.spines['top'].set_visible(False)
    ax_lc.spines['right'].set_visible(False)

    # Residuals
    res_ppm = (OBS_LIGHT_CURVE - lc_combined) * 1e6
    ax_res.hist(res_ppm, orientation='horizontal', histtype='stepfilled', bins=25, 
                color='orange', alpha=0.3, edgecolor='orange', lw=2)
    ax_res.axhline(np.std(res_ppm), color='black', linestyle='dotted', label=r'+/- 1$\sigma$')
    ax_res.axhline(-np.std(res_ppm), color='black', linestyle='dotted')
    ax_res.set_ylabel('Residuals [ppm]')
    ax_res.spines['right'].set_visible(False)
    ax_res.spines['top'].set_visible(False)
    ax_res.spines['bottom'].set_visible(False)
    ax_res.tick_params(bottom=False, labelbottom=False)
    ax_res.legend(fontsize=9)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostic plot with snapshots to {out_path}")


def plot_bestfit_lightcurve(constrained_samples: dict, output_dir: Path, map_params: dict = None):
    """Plot posterior mean and MAP light curve vs truth and observations."""
    mean_c = {name: float(np.array(v).mean()) for name, v in constrained_samples.items()}

    lc_mean = np.array(
        _call_sajax(
            TIMES,
            np.array([mean_c["spot_lat"]]),
            np.array([mean_c["spot_long"]]),
            np.array([mean_c["spot_size"]]),
            np.stack([np.array([mean_c["spot_flux"]])]),
            mean_c["p_rot"],
            mean_c["planet_radius"],
            mean_c["semimajor_axis"],
            np.deg2rad(mean_c["inclination"]),
            mean_c["eccentricity"],
            mean_c["arg_periapsis"],
            mean_c["P_orb"],
            mean_c["ldc_u1"],
            mean_c["ldc_u2"],
        )["lc"]
    )
    
    fig, (ax_lc, ax_res) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
    ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=4, color="orange", alpha=0.6,
                  label="Observations", zorder=1)
    ax_lc.plot(TIMES, LC_TRUE, lw=2, color="steelblue", label="True", zorder=2)
    ax_lc.plot(TIMES, lc_mean, lw=2, color="crimson", linestyle="--",
               label="Posterior mean fit", zorder=3)

    # Residual plot starts with mean residuals
    mean_residuals_ppm = (OBS_LIGHT_CURVE - lc_mean) * 1e6
    ax_res.scatter(TIMES, mean_residuals_ppm, s=4, color="crimson", alpha=0.6, label="Mean")

    # --- MAP light curve (only if map_params provided) ---
    if map_params is not None:
        mc = map_params
        lc_map = np.array(
            _call_sajax(
                TIMES,
                np.array([mc["spot_lat"]]),
                np.array([mc["spot_long"]]),
                np.array([mc["spot_size"]]),
                np.stack([np.array([mc["spot_flux"]])]),
                mc["p_rot"],
                mc["planet_radius"],
                mc["semimajor_axis"],
                np.deg2rad(mc["inclination"]),
                mc["eccentricity"],
                mc["arg_periapsis"],
                mc["P_orb"],
                mc["ldc_u1"],
                mc["ldc_u2"],
            )["lc"]
        )
        ax_lc.plot(TIMES, lc_map, lw=2, color="darkgreen", linestyle="--",
                   label="MAP fit", zorder=4)
        map_residuals_ppm = (OBS_LIGHT_CURVE - lc_map) * 1e6
        ax_res.scatter(TIMES, map_residuals_ppm, s=4, color="darkgreen", alpha=0.6, label="MAP")

    ax_lc.set_ylabel("Normalised flux")
    ax_lc.legend(frameon=False)
    ax_lc.spines["top"].set_visible(False)
    ax_lc.spines["right"].set_visible(False)

    ax_res.axhline(0, color="grey", lw=1, linestyle="--")
    ax_res.set_xlabel("Time [days]")
    ax_res.set_ylabel("Residuals [ppm]")
    ax_res.legend(frameon=False, fontsize=8)
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    lc_path = output_dir / "bestfit_lightcurve.png"
    fig.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved best-fit light curve to {lc_path}")


def plot_prior_posterior(constrained_samples: dict, output_dir: Path,
                         n_prior_samples: int = 3000, seed: int = 0):
    """
    One PNG per sampled parameter saved to output_dir/prior_posterior/.
    Each plot shows the prior PDF (blue line), posterior histogram (red fill),
    and a ground-truth vertical line (black dashed).
    """
    key = jax.random.PRNGKey(seed)
    pp_dir = output_dir / "prior_posterior"
    pp_dir.mkdir(parents=True, exist_ok=True)

    for name in PARAM_NAMES:
        d = PRIOR_DISTRIBUTIONS[name]

        key, sk = jax.random.split(key)
        prior_samps = np.array(d.sample(sk, (n_prior_samples,)))
        post_samps  = np.array(constrained_samples[name]).ravel()

        all_vals = np.concatenate([prior_samps, post_samps])
        x_lo = np.percentile(all_vals, 0.1)
        x_hi = np.percentile(all_vals, 99.9)
        pad  = 0.08 * (x_hi - x_lo) if x_hi > x_lo else 1e-6
        x_lo -= pad
        x_hi += pad

        x_grid    = np.linspace(x_lo, x_hi, 400)
        prior_pdf = np.exp(np.array(d.log_prob(jnp.array(x_grid))))

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(x_grid, prior_pdf, color="steelblue", lw=2, label="Prior")
        post_clip = post_samps[(post_samps >= x_lo) & (post_samps <= x_hi)]
        ax.hist(post_clip, bins=40, density=True,
                color="crimson", alpha=0.35, histtype="stepfilled",
                edgecolor="crimson", lw=1, label="Posterior")
        if name in GROUND_TRUTH:
            ax.axvline(GROUND_TRUTH[name], color="black", lw=1.5, ls="--", label="Truth")

        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_xlim(x_lo, x_hi)
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(pp_dir / f"{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved prior/posterior plots to {pp_dir}/")