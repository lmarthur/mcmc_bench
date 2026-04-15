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
import sajax
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
#%%%% Define G in units needed now to avoid JAX tracing issues
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
R_star = (1.0 * u.R_sun).value

TRUE_PLANET_RADIUS = 0.1        # planet radius in stellar radii (Rp/Rs)
TRUE_P_ORB         = 1.0        # planet orbital period [days]
A_METERS           = ( (G.value * (1.0 * u.M_sun).to(u.kg).value * (TRUE_P_ORB * 24 * 3600)**2)/(4 * jnp.pi**2) )**(1/3)           
TRUE_SEMI_MAJOR    = A_METERS / (1.0 * u.R_sun).to(u.m).value # semi-major axis
TRUE_INCLINATION   = jnp.deg2rad(90.0)       # orbital inclination [degrees]
TRUE_ECCENTRICITY  = 0.0        # orbital eccentricity
TRUE_ARG_PERIAPSIS = 0.0        # argument of periastron [degrees]
TRUE_LDC_U1        = 0.4        # limb-darkening for transit (matches sajax)
TRUE_LDC_U2        = 0.2
TRUE_T0_TRANSIT    = 0.0        # mid-transit time [days]

#%%%% Calculate transit duration
# Convert angles to radians
# Impact parameter (eccentricity-corrected)
TRUE_IMPACT_PARAM = (
    (TRUE_SEMI_MAJOR * jnp.cos(TRUE_INCLINATION)) / R_star
    * (1 - TRUE_ECCENTRICITY**2) / (1 + TRUE_ECCENTRICITY * jnp.sin(TRUE_ARG_PERIAPSIS))
)
# Argument inside arcsin
arg = (
    (1/TRUE_SEMI_MAJOR)
    * jnp.sqrt((1 + TRUE_PLANET_RADIUS)**2 - TRUE_IMPACT_PARAM**2)
    / jnp.sin(TRUE_INCLINATION)
)
# Numerical safety
arg = np.clip(arg, -1.0, 1.0)
# Transit duration
TRUE_T14_TRANSIT = (
    (TRUE_P_ORB / jnp.pi)
    * jnp.sqrt(1 - TRUE_ECCENTRICITY**2) / (1 + TRUE_ECCENTRICITY * jnp.sin(TRUE_ARG_PERIAPSIS * jnp.pi / 180.0))
    * jnp.arcsin(arg)
)

# ---------------------------------------------------------------------------
# Fixed observation setup — shared across all sampler scripts
# ---------------------------------------------------------------------------

# Time / phase setup
low_t = -3.5*TRUE_T14_TRANSIT                                           #days
high_t = 3.5*TRUE_T14_TRANSIT                                           #days
exposure_time = 250                                                     #seconds
num_t = jnp.floor((((high_t - low_t) * 24 * 3600)/exposure_time))       #number of points
TIMES = jnp.linspace(low_t, high_t, int(num_t))

# Rotational phases for sajax [degrees]
TRUE_P_ROT = 0.5              # stellar rotation period [days]
PHASES_ROT = (TIMES / TRUE_P_ROT * 360.0) % 360.0

# Synthetic flat spectra — single wavelength bin for broadband benchmark
WAVELENGTH = np.array([550.0])       # nm
FLUX_QUIET = np.array([1.0])
FLUX_ACTIVE_SPOT = np.array([0.7])   # spot is 30% darker
FLUX_ACTIVE_FACULA = np.array([1.1]) # facula is 10% brighter

# Fixed stellar / instrument parameters
STELLAR_INC = 90.0          # stellar inclination (equator-on)

STELLAR_GRID_SIZE = 200  # stellar radius in pixels
VE = 2.0                 # equatorial velocity [km/s]

# Noise level
SIGMA_NOISE = 100e-6     # ~100 ppm

# ---------------------------------------------------------------------------
# Ground-truth spot and facula
# ---------------------------------------------------------------------------
TRUE_SPOT_LAT = 0.0      # degrees
TRUE_SPOT_LONG = 5.0      # degrees
TRUE_SPOT_SIZE = 11.0     # degrees radius

TRUE_FACULA_LAT = -20.0
TRUE_FACULA_LONG = 165.0
TRUE_FACULA_SIZE = 16.0

# ---------------------------------------------------------------------------
# Prior bounds: active regions
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = -90.0, 90.0
LONG_MIN, LONG_MAX = 0.0, 360.0
SIZE_MIN, SIZE_MAX = 1.0, 90.0
FLUX_MIN, FLUX_MAX = 0.5, 1.5
P_ROT_MIN, P_ROT_MAX = 5.0, 20.0
LDC_U1_MIN, LDC_U1_MAX = 0.0, 1.0
LDC_U2_MIN, LDC_U2_MAX = 0.0, 1.0

# ---------------------------------------------------------------------------
# Prior bounds: planet transit
# ---------------------------------------------------------------------------
PLANET_RADIUS_MIN, PLANET_RADIUS_MAX = 0.01, 0.3   # Rp/Rs
SEMI_MAJOR_MIN, SEMI_MAJOR_MAX = 0.0, 1.0          # impact parameter
INCLINATION_MIN, INCLINATION_MAX = 80.0, 100.0     # inclination [degrees]
P_ORB_MIN, P_ORB_MAX = 1.0, 10.0                   # orbital period [days]
ECCENTRICITY_MIN, ECCENTRICITY_MAX = 0.0, 0.5      # eccentricity
ARG_PERIAPSIS_MIN, ARG_PERIAPSIS_MAX = 0.0, 360.0  # argument of periastron [degrees]

# ---------------------------------------------------------------------------
# Transit light curve (sajax)
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
) -> jnp.ndarray:
    """
    Call sajax's compute_light_curve and return the broadband light curve
    as a 1-D JAX array of shape (N_PHASES,).
    """

    params_sajax = dict(
    ldc_coeffs=[LDC_u1, LDC_u2],  # quadratic limb-darkening [u1, u2]
    inc_star=STELLAR_INC,         # equator-on view
    ) 

    transit_params = dict(
        t0           = TRUE_SPOT_SIZE,
        period       = P_orb,
        a_over_rstar = semimajor_axis,
        inclination  = inclination,    
        k            = planet_radius,            
        ecc          = eccentricity,
        omega_peri   = arg_periapsis,
    )

    result = sajax.compute_combined_light_curve(
        wavelength        =WAVELENGTH,
        flux_quiet        =FLUX_QUIET,
        flux_active       =flux_active,
        params            =params_sajax,
        ar_lat            =ar_lat.tolist(),
        ar_long           =ar_long.tolist(),
        ar_size           =ar_size.tolist(),
        times             = times,
        P_rot             = P_rot,
        transit_params    = transit_params,
        stellar_grid_size = STELLAR_GRID_SIZE,
        ve                = VE,
        ldc_mode          = "quadratic",
    )

    return jnp.array(result["lc"])


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
        )
    )

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, SIGMA_NOISE, size=lc_true.shape)
    return lc_true + noise


# Generate observations once at import time
OBS_LIGHT_CURVE = generate_observations(seed=42)


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

def spot_model(y_obs: jnp.ndarray = jnp.array(OBS_LIGHT_CURVE)):
    """
    NumPyro model for the stellar spot + facula posterior.

    Latent variables (spot):
        spot_lat   ~ Uniform(LAT_MIN, LAT_MAX)
        spot_long  ~ Uniform(LONG_MIN, LONG_MAX)
        spot_size  ~ Uniform(SIZE_MIN, SIZE_MAX)
        spot_flux  ~ Uniform(FLUX_MIN, FLUX_MAX)

    Latent variables (facula):
        fac_lat    ~ Uniform(LAT_MIN, LAT_MAX)
        fac_long   ~ Uniform(LONG_MIN, LONG_MAX)
        fac_size   ~ Uniform(SIZE_MIN, SIZE_MAX)
        fac_flux   ~ Uniform(FLUX_MIN, FLUX_MAX)
    """
    # Star parameters
    spot_lat = numpyro.sample("spot_lat", dist.Uniform(LAT_MIN, LAT_MAX))
    spot_long = numpyro.sample("spot_long", dist.Uniform(LONG_MIN, LONG_MAX))
    spot_size = numpyro.sample("spot_size", dist.Uniform(SIZE_MIN, SIZE_MAX))
    spot_flux = numpyro.sample("spot_flux", dist.Uniform(FLUX_MIN, FLUX_MAX))
    fac_lat = numpyro.sample("fac_lat", dist.Uniform(LAT_MIN, LAT_MAX))
    fac_long = numpyro.sample("fac_long", dist.Uniform(LONG_MIN, LONG_MAX))
    fac_size = numpyro.sample("fac_size", dist.Uniform(SIZE_MIN, SIZE_MAX))
    fac_flux = numpyro.sample("fac_flux", dist.Uniform(FLUX_MIN, FLUX_MAX))
    P_rot = numpyro.sample("p_rot", dist.Uniform(P_ROT_MIN, P_ROT_MAX))
    LDC_u1 = numpyro.sample("ldc_u1", dist.Uniform(LDC_U1_MIN, LDC_U1_MAX))
    LDC_u2 = numpyro.sample("ldc_u2", dist.Uniform(LDC_U2_MIN, LDC_U2_MAX))

    # Planet parameters
    planet_radius = numpyro.sample("planet_radius", dist.Uniform(PLANET_RADIUS_MIN, PLANET_RADIUS_MAX))
    semimajor_axis = numpyro.sample("semimajor_axis", dist.Uniform(SEMI_MAJOR_MIN, SEMI_MAJOR_MAX))
    inclination = numpyro.sample("inclination", dist.Uniform(INCLINATION_MIN, INCLINATION_MAX))
    eccentricity = numpyro.sample("eccentricity", dist.Uniform(ECCENTRICITY_MIN, ECCENTRICITY_MAX))
    arg_periapsis = numpyro.sample("arg_periapsis", dist.Uniform(ARG_PERIAPSIS_MIN, ARG_PERIAPSIS_MAX))
    P_orb = numpyro.sample("P_orb", dist.Uniform(P_ORB_MIN, P_ORB_MAX))

    # Build arrays
    ar_lat = jnp.array([spot_lat, fac_lat])
    ar_long = jnp.array([spot_long, fac_long])
    ar_size = jnp.array([spot_size, fac_size])
    flux_active = jnp.stack([
        jnp.broadcast_to(spot_flux, (1,)),
        jnp.broadcast_to(fac_flux, (1,)),
    ])

    lc_model = _call_sajax(
        ar_lat, 
        ar_long, 
        ar_size, 
        flux_active,
        P_rot,
        planet_radius,
        semimajor_axis,
        inclination,
        eccentricity,
        arg_periapsis,
        P_orb,
        LDC_u1,
        LDC_u2,
        )

    numpyro.sample(
        "y_obs",
        dist.Normal(lc_model, SIGMA_NOISE),
        obs=y_obs,
    )


def make_log_density(y_obs: np.ndarray = OBS_LIGHT_CURVE):
    """
    Returns a BlackJAX-compatible log-density function.

    The returned function accepts a 17-element parameter vector:
        x = [spot_lat, spot_long, spot_size, spot_flux,
             fac_lat, fac_long, fac_size, fac_flux, P_rot, 
             planet_radius, semimajor_axis, inclination, 
             eccentricity, arg_periapsis, P_orb, LDC_u1, LDC_u2]
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
            "P_rot": x[8],
            "planet_radius": x[9],
            "semimajor_axis": x[10],
            "inclination": x[11],
            "eccentricity": x[12],
            "arg_periapsis": x[13], 
            "P_orb": x[14], 
            "LDC_u1": x[15],
            "LDC_u2": x[16],
        }
        ld, _ = log_density(
            spot_model,
            model_args=(),
            model_kwargs={"y_obs": y_obs_jnp},
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
    "planet_radius": TRUE_PLANET_RADIUS,
    "semimajor_axis": TRUE_SEMI_MAJOR,
    "inclination": TRUE_INCLINATION,
    "eccentricity": TRUE_ECCENTRICITY,
    "arg_periapsis": TRUE_ARG_PERIAPSIS,
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
      - A 2D marginal heatmap of log p(spot_lat, spot_long | y_obs)
    """
    log_density_fn = make_log_density()

    # Individual components
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
        )
    )
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
        )
    )

    #Combination
    lc_combined = np.array(
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
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Combined light curve ---
    ax = axes[0, 0]
    ax.plot(TIMES, lc_combined, lw=1.5, label="True combined", color="steelblue")
    ax.scatter(
        TIMES, OBS_LIGHT_CURVE,
        s=1, alpha=0.3, color="orange", label="Observed (+ noise)",
    )
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Combined: activity × transit")
    ax.legend()

    # --- Activity only ---
    ax = axes[0, 1]
    ax.plot(TIMES, lc_activity, lw=1.5, color="green")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Stellar activity only (sajax)")

    # --- Transit only ---
    ax = axes[1, 0]
    # Zoom into transit window
    t_mid = TRUE_T0_TRANSIT
    t_window = 3 * TRUE_T14_TRANSIT
    mask = np.abs(TIMES - t_mid) < t_window
    ax.plot(TIMES[mask], lc_transit[mask], lw=2, color="crimson")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Planet transit only (jaxoplanet)")

    # --- Log-posterior slice: spot_lat vs spot_long ---
    resolution = 40
    lats = np.linspace(LAT_MIN, LAT_MAX, resolution)
    longs = np.linspace(LONG_MIN, LONG_MAX, resolution)
    LL, LO = np.meshgrid(lats, longs)

    # Fix all other params to ground truth
    log_p = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            x = jnp.array([
                LL[i, j], LO[i, j], TRUE_SPOT_SIZE, FLUX_ACTIVE_SPOT[0],
                TRUE_FACULA_LAT, TRUE_FACULA_LONG, TRUE_FACULA_SIZE,
                FLUX_ACTIVE_FACULA[0],
            ])
            log_p[i, j] = log_density_fn(x)

    ax = axes[1, 1]
    im = ax.pcolormesh(lats, longs, log_p, shading="auto", cmap="viridis")
    ax.scatter(
        [TRUE_SPOT_LAT], [TRUE_SPOT_LONG],
        color="red", s=60, zorder=5, label="True spot",
    )
    ax.set_xlabel("spot_lat [deg]")
    ax.set_ylabel("spot_long [deg]")
    ax.set_title("log p(lat, long | data) [other params fixed]")
    ax.legend()
    plt.colorbar(im, ax=ax, label="log p")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():

    # Individual components
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
        )
    )
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
        )
    )
    #Combination
    lc_combined = np.array(
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
        )
    )

    obs = generate_observations(seed=12)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Combined
    ax = axes[0]
    ax.plot(TIMES, lc_combined, ".-", label="True combined", color="steelblue")
    ax.plot(TIMES, obs, ".", alpha=0.7, color="orange", label="Observed")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Combined light curve: stellar activity x planet transit")
    ax.legend()

    ax = axes[1]
    ax.plot(TIMES, lc_activity, ".-", color="green", label="Stellar activity (sajax)")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Stellar activity modulation")
    ax.legend()

    # Transit only
    ax = axes[2]
    ax.plot(TIMES, lc_transit, ".-", color="crimson", label="Planet transit (jaxoplanet)")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Planet transit")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # plot_model()


if __name__ == "__main__":
    main()