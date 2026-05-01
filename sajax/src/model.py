"""
Model of an exoplanetary transit with a spot crossing for sampler benchmarking.

Default configuration: a star of radius 1 with quadratic limb darkening,
a single circular spot and a facula on the stellar disk. A planet transits
the star, with the transit light curve computed via jaxoplanet and the
stellar activity modulation computed via sajax.

The combined light curve is:
    lc_total = lc_activity * (1 + lc_transit)

where lc_activity comes from sajax (rotational modulation from spots/faculae)
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
# Planet transit parameters (jaxoplanet)
# ---------------------------------------------------------------------------
TRUE_PLANET_RADIUS = 0.1
TRUE_P_ORB         = 1.0
A_METERS           = ( (G.value * (1.0 * u.M_sun).to(u.kg).value * (TRUE_P_ORB * 24 * 3600)**2)/(4 * jnp.pi**2) )**(1/3)
TRUE_SEMI_MAJOR    = A_METERS / (1.0 * u.R_sun).to(u.m).value
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

TRUE_P_ROT = 0.5              

# Synthetic flat spectra — single wavelength bin for broadband benchmark
WAVELENGTH = np.array([550.0])       # nm
FLUX_QUIET = np.array([1.0])
FLUX_ACTIVE_SPOT = np.array([0.7])   # spot is 30% darker
FLUX_ACTIVE_FACULA = np.array([1.1]) # facula is 10% brighter

STELLAR_INC = 90.0          
STELLAR_GRID_SIZE = 100
VE = 2.0                 

SIGMA_NOISE = 100e-6     # ~100 ppm

# ---------------------------------------------------------------------------
# Ground-truth spot and facula
# ---------------------------------------------------------------------------
TRUE_SPOT_LAT = 5.0      
TRUE_SPOT_LONG = 5.0      
TRUE_SPOT_SIZE = 11.0     

TRUE_FACULA_LAT = -20.0
TRUE_FACULA_LONG = 165.0
TRUE_FACULA_SIZE = 16.0

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
ECC_H_SCALE = 0.3   # √e·cos(ω) prior scale
ECC_K_SCALE = 0.3   # √e·sin(ω) prior scale

# ---------------------------------------------------------------------------
# Prior distributions (numpyro) — single source of truth for all samplers
# ---------------------------------------------------------------------------
PRIOR_DISTRIBUTIONS = {
    "spot_lat":      dist.Uniform(4.0, 6.0),
    "spot_long":     dist.Uniform(4.0, 6.0),
    "spot_size":     dist.Uniform(10.0, 12.0),
    "spot_flux":     dist.Uniform(0.65, 0.75),
    "fac_lat":       dist.Uniform(-25.0, -15.0),
    "fac_long":      dist.Uniform(160.0, 170.0),
    "fac_size":      dist.Uniform(15.0, 17.0),
    "fac_flux":      dist.Uniform(1.05, 1.15),
    "p_rot":         dist.Normal(TRUE_P_ROT, 0.000001),
    "planet_radius": dist.Uniform(0.095, 0.15),
    "semimajor_axis":dist.Uniform(4.0, 4.5),
    "inclination":   dist.Uniform(89.0, 91.0),
    "ecc_h":         dist.Uniform(-0.01, 0.01),
    "ecc_k":         dist.Uniform(-0.01, 0.01),
    "P_orb":         dist.Normal(TRUE_P_ORB, 0.0005),
    "ldc_q1":        dist.Uniform(0.39, 0.41),
    "ldc_q2":        dist.Uniform(0.19, 0.21),
}


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
            jnp.array([TRUE_SPOT_LAT, TRUE_FACULA_LAT]),
            jnp.array([TRUE_SPOT_LONG, TRUE_FACULA_LONG]),
            jnp.array([TRUE_SPOT_SIZE, TRUE_FACULA_SIZE]),
            np.stack([FLUX_ACTIVE_SPOT, FLUX_ACTIVE_FACULA]),
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
        jnp.array([TRUE_SPOT_LAT, TRUE_FACULA_LAT]),
        jnp.array([TRUE_SPOT_LONG, TRUE_FACULA_LONG]),
        jnp.array([TRUE_SPOT_SIZE, TRUE_FACULA_SIZE]),
        np.stack([FLUX_ACTIVE_SPOT, FLUX_ACTIVE_FACULA]),
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
    NumPyro model for the spot + facula + planet posterior.
    Uses pre-built STATIC_MODEL for JIT-compatibility.
    """
    spot_lat = numpyro.sample("spot_lat", PRIOR_DISTRIBUTIONS["spot_lat"])
    spot_long = numpyro.sample("spot_long", PRIOR_DISTRIBUTIONS["spot_long"])
    spot_size = numpyro.sample("spot_size", PRIOR_DISTRIBUTIONS["spot_size"])
    spot_flux = numpyro.sample("spot_flux", PRIOR_DISTRIBUTIONS["spot_flux"])

    fac_lat = numpyro.sample("fac_lat", PRIOR_DISTRIBUTIONS["fac_lat"])
    fac_long = numpyro.sample("fac_long", PRIOR_DISTRIBUTIONS["fac_long"])
    fac_size = numpyro.sample("fac_size", PRIOR_DISTRIBUTIONS["fac_size"])
    fac_flux = numpyro.sample("fac_flux", PRIOR_DISTRIBUTIONS["fac_flux"])

    P_rot = numpyro.sample("p_rot", PRIOR_DISTRIBUTIONS["p_rot"])
    ldc_q1 = numpyro.sample("ldc_q1", PRIOR_DISTRIBUTIONS["ldc_q1"])
    ldc_q2 = numpyro.sample("ldc_q2", PRIOR_DISTRIBUTIONS["ldc_q2"])
    LDC_u1 = numpyro.deterministic("ldc_u1", 2 * jnp.sqrt(ldc_q1) * ldc_q2)
    LDC_u2 = numpyro.deterministic("ldc_u2", jnp.sqrt(ldc_q1) * (1 - 2 * ldc_q2))

    # Planet parameters
    planet_radius = numpyro.sample("planet_radius", PRIOR_DISTRIBUTIONS["planet_radius"])
    semimajor_axis = numpyro.sample("semimajor_axis", PRIOR_DISTRIBUTIONS["semimajor_axis"])
    inclination = numpyro.sample("inclination", PRIOR_DISTRIBUTIONS["inclination"])
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
    ar_lat = jnp.array([spot_lat, fac_lat])
    ar_long = jnp.array([spot_long, fac_long])
    ar_size = jnp.array([spot_size, fac_size])

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
        jnp.broadcast_to(fac_flux, (1,)),
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
    Expects dict keys matching PRIOR_DISTRIBUTIONS (ecc_h/ecc_k and ldc_q1/ldc_q2
    are the sampled parameterization; eccentricity/arg_periapsis/ldc_u1/ldc_u2 are derived).
    """
    y_obs_arr = jnp.array(y_obs)

    def log_likelihood(params):
        P_rot = params["p_rot"]
        ldc_q1 = params["ldc_q1"]
        ldc_q2 = params["ldc_q2"]
        LDC_u1 = 2 * jnp.sqrt(ldc_q1) * ldc_q2
        LDC_u2 = jnp.sqrt(ldc_q1) * (1 - 2 * ldc_q2)
        planet_radius = params["planet_radius"]
        semimajor_axis = params["semimajor_axis"]
        inclination = params["inclination"]
        ecc_h = params["ecc_h"]
        ecc_k = params["ecc_k"]
        eccentricity = ecc_h**2 + ecc_k**2
        arg_periapsis = jnp.arctan2(ecc_k, ecc_h)
        P_orb = params["P_orb"]

        ar_lat = jnp.array([params["spot_lat"], params["fac_lat"]])
        ar_long = jnp.array([params["spot_long"], params["fac_long"]])
        ar_size = jnp.array([params["spot_size"], params["fac_size"]])

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
            jnp.broadcast_to(params["spot_flux"], (1,)),
            jnp.broadcast_to(params["fac_flux"], (1,)),
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

    ar_lat  = jnp.array([constrained["spot_lat"],  constrained["fac_lat"]])
    ar_long = jnp.array([constrained["spot_long"], constrained["fac_long"]])
    ar_size = jnp.array([constrained["spot_size"],  constrained["fac_size"]])

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
        jnp.broadcast_to(jnp.asarray(constrained["fac_flux"]),  (1,)),
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
# Ground truth dict — for sampler diagnostics
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "spot_lat": TRUE_SPOT_LAT,
    "spot_long": TRUE_SPOT_LONG,
    "spot_size": TRUE_SPOT_SIZE,
    "spot_flux": FLUX_ACTIVE_SPOT[0],
    "fac_lat": TRUE_FACULA_LAT,
    "fac_long": TRUE_FACULA_LONG,
    "fac_size": TRUE_FACULA_SIZE,
    "fac_flux": FLUX_ACTIVE_FACULA[0],
    "p_rot": TRUE_P_ROT,
    "planet_radius": TRUE_PLANET_RADIUS,
    "semimajor_axis": TRUE_SEMI_MAJOR,
    "inclination": float(jnp.rad2deg(TRUE_INCLINATION)),
    "ecc_h": float(jnp.sqrt(TRUE_ECCENTRICITY) * jnp.cos(TRUE_ARG_PERIAPSIS)),
    "ecc_k": float(jnp.sqrt(TRUE_ECCENTRICITY) * jnp.sin(TRUE_ARG_PERIAPSIS)),
    "P_orb": TRUE_P_ORB,
    "ldc_q1": (TRUE_LDC_U1 + TRUE_LDC_U2) ** 2,
    "ldc_q2": TRUE_LDC_U1 / (2 * (TRUE_LDC_U1 + TRUE_LDC_U2)),
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
            jnp.array([TRUE_SPOT_LAT, TRUE_FACULA_LAT]),
            jnp.array([TRUE_SPOT_LONG, TRUE_FACULA_LONG]),
            jnp.array([TRUE_SPOT_SIZE, TRUE_FACULA_SIZE]),
            np.stack([FLUX_ACTIVE_SPOT, FLUX_ACTIVE_FACULA]),
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
            jnp.array([TRUE_SPOT_LAT, TRUE_FACULA_LAT]),
            jnp.array([TRUE_SPOT_LONG, TRUE_FACULA_LONG]),
            jnp.array([0.0001, 0.0001]),
            np.stack([FLUX_ACTIVE_SPOT, FLUX_ACTIVE_FACULA]),
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
            jnp.array([TRUE_SPOT_LAT, TRUE_FACULA_LAT]),
            jnp.array([TRUE_SPOT_LONG, TRUE_FACULA_LONG]),
            jnp.array([TRUE_SPOT_SIZE, TRUE_FACULA_SIZE]),
            np.stack([FLUX_ACTIVE_SPOT, FLUX_ACTIVE_FACULA]),
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


def plot_bestfit_lightcurve(constrained_samples: dict, output_dir: Path):
    """Plot posterior mean and posterior maximum light curve vs truth and observations, save to output_dir."""
    mean_c = {name: float(np.array(v).mean()) for name, v in constrained_samples.items()}
    max_c = {name: float(np.array(v).max()) for name, v in constrained_samples.items()}

    lc_mean = np.array(
        _call_sajax(
            TIMES,
            np.array([mean_c["spot_lat"], mean_c["fac_lat"]]),
            np.array([mean_c["spot_long"], mean_c["fac_long"]]),
            np.array([mean_c["spot_size"], mean_c["fac_size"]]),
            np.stack([np.array([mean_c["spot_flux"]]), np.array([mean_c["fac_flux"]])]),
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

    lc_max = np.array(
        _call_sajax(
            TIMES,
            np.array([max_c["spot_lat"], max_c["fac_lat"]]),
            np.array([max_c["spot_long"], max_c["fac_long"]]),
            np.array([max_c["spot_size"], max_c["fac_size"]]),
            np.stack([np.array([max_c["spot_flux"]]), np.array([max_c["fac_flux"]])]),
            max_c["p_rot"],
            max_c["planet_radius"],
            max_c["semimajor_axis"],
            np.deg2rad(max_c["inclination"]),
            max_c["eccentricity"],
            max_c["arg_periapsis"],
            max_c["P_orb"],
            max_c["ldc_u1"],
            max_c["ldc_u2"],
        )["lc"]
    )

    fig, (ax_lc, ax_res) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
    ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=4, color="orange", alpha=0.6,
                  label="Observations", zorder=1)
    ax_lc.plot(TIMES, LC_TRUE, lw=2, color="steelblue", label="True", zorder=2)
    ax_lc.plot(TIMES, lc_mean, lw=2, color="crimson", linestyle="--",
               label="Posterior mean fit", zorder=3)
    ax_lc.plot(TIMES, lc_max, lw=2, color="darkgreen", linestyle="--",
               label="Posterior maximum fit", zorder=4)
    ax_lc.set_ylabel("Normalised flux")
    ax_lc.legend(frameon=False)
    ax_lc.spines["top"].set_visible(False)
    ax_lc.spines["right"].set_visible(False)

    mean_residuals_ppm = (OBS_LIGHT_CURVE - lc_mean) * 1e6
    max_residuals_ppm = (OBS_LIGHT_CURVE - lc_max) * 1e6
    ax_res.scatter(TIMES, mean_residuals_ppm, s=4, color="orange", alpha=0.6)
    ax_res.scatter(TIMES, max_residuals_ppm, s=4, color="darkgreen", alpha=0.6)
    ax_res.axhline(0, color="crimson", lw=1, linestyle="--")
    ax_res.set_xlabel("Time [days]")
    ax_res.set_ylabel("Residuals [ppm]")
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    lc_path = output_dir / "bestfit_lightcurve.png"
    fig.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved best-fit light curve to {lc_path}")