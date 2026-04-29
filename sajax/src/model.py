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
parameters and the observed data, and numpyro.infer.util.log_density is
used to extract a BlackJAX-compatible log-density function for sampling.
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
from numpyro.infer.util import log_density
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
P_ROT_MIN, P_ROT_MAX = 0.1, 5.0
LDC_U1_MIN, LDC_U1_MAX = 0.0, 1.0
LDC_U2_MIN, LDC_U2_MAX = 0.0, 1.0

# ---------------------------------------------------------------------------
# Prior bounds: planet transit
# ---------------------------------------------------------------------------
PLANET_RADIUS_MIN, PLANET_RADIUS_MAX = 0.001, 0.3  # Rp/Rs
SEMI_MAJOR_MIN, SEMI_MAJOR_MAX = 0.0, 10.0         # a/R* (semi-major axis in stellar radii)
INCLINATION_MIN, INCLINATION_MAX = 80.0, 100.0     # inclination [degrees]
P_ORB_MIN, P_ORB_MAX = 1.0, 10.0                   # orbital period [days]
ECC_H_SCALE = 0.3   # √e·cos(ω) prior scale
ECC_K_SCALE = 0.3   # √e·sin(ω) prior scale

# ---------------------------------------------------------------------------
# Prior distributions (numpyro) — single source of truth for all samplers
# ---------------------------------------------------------------------------
PRIOR_DISTRIBUTIONS = {
    "spot_lat":      dist.Uniform(LAT_MIN, LAT_MAX),
    "spot_long":     dist.Uniform(LONG_MIN, LONG_MAX),
    "spot_size":     dist.Uniform(SIZE_MIN, SIZE_MAX),
    "spot_flux":     dist.Uniform(FLUX_MIN, FLUX_MAX),
    "fac_lat":       dist.Uniform(LAT_MIN, LAT_MAX),
    "fac_long":      dist.Uniform(LONG_MIN, LONG_MAX),
    "fac_size":      dist.Uniform(SIZE_MIN, SIZE_MAX),
    "fac_flux":      dist.Uniform(FLUX_MIN, FLUX_MAX),
    "p_rot":         dist.LogNormal(jnp.log(TRUE_P_ROT), 1.0),
    "planet_radius": dist.LogNormal(jnp.log(TRUE_PLANET_RADIUS), 0.5),
    "semimajor_axis":dist.LogNormal(jnp.log(5.0), 0.5),
    "inclination":   dist.Uniform(INCLINATION_MIN, INCLINATION_MAX),
    "ecc_h":         dist.Normal(0.0, ECC_H_SCALE),
    "ecc_k":         dist.Normal(0.0, ECC_K_SCALE),
    "P_orb":         dist.Normal(TRUE_P_ORB, 0.0005),
    "ldc_u1":        dist.Normal(0.0, 5.0),
    "ldc_u2":        dist.Normal(0.0, 5.0),
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
    LDC_u1 = numpyro.sample("ldc_u1", PRIOR_DISTRIBUTIONS["ldc_u1"])
    LDC_u2 = numpyro.sample("ldc_u2", PRIOR_DISTRIBUTIONS["ldc_u2"])

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


def make_log_density(y_obs: np.ndarray = OBS_LIGHT_CURVE, model_dict: dict = STATIC_MODEL):
    """
    Returns a BlackJAX-compatible log-density function.

    The returned function accepts a 17-element parameter vector:
        x = [spot_lat, spot_long, spot_size, spot_flux,
             fac_lat, fac_long, fac_size, fac_flux,
             p_rot,
             planet_radius, semimajor_axis, inclination (degrees),
             ecc_h, ecc_k, P_orb,
             ldc_u1, ldc_u2]
    """
    y_obs_jnp = jnp.array(y_obs)

    def log_density_fn(x):
        params = {
            "spot_lat": x[0],
            "spot_long": x[1],
            "spot_size": x[2],
            "spot_flux": x[3],
            "fac_lat": x[4],
            "fac_long": x[5],
            "fac_size": x[6],
            "fac_flux": x[7],
            "p_rot": x[8],
            "planet_radius": x[9],
            "semimajor_axis": x[10],
            "inclination": x[11],
            "ecc_h": x[12],
            "ecc_k": x[13],
            "P_orb": x[14],
            "ldc_u1": x[15],
            "ldc_u2": x[16],
        }
        ld, _ = log_density(
            sajax_model,
            model_args=(),
            model_kwargs={"y_obs": y_obs_jnp, "model_dict": model_dict},
            params=params,
        )
        return ld

    return log_density_fn


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
    "LDC_u1": TRUE_LDC_U1,
    "LDC_u2": TRUE_LDC_U2,
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
    log_density_fn = make_log_density()

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