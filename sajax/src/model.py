"""
Model of an exoplanetary transit with a spot crossing for sampler benchmarking.

Default configuration: a star of radius 1 with quadratic limb darkening and a single circular spot of radius 0.2,
centered at (0.5, 0) on the stellar disk. A planet of radius 0.1 transits across the star along the x-axis,
with mid-transit at (0, 0) and total duration of 1 time unit. The spot crossing occurs at t=0.2, causing a temporary brightening in the light curve.
The target distribution is the posterior over the spot parameters (position and radius) given a noisy light curve observation.
This model is designed to be multimodal due to the symmetry of the stellar disk and the degeneracy between spot size and contrast, making it a challenging test case for samplers.
The likelihood is defined by the difference between the observed light curve and the model-predicted light curve, which depends on the spot parameters.

A NumPyro model is used to define the joint distribution over the spot parameters and the observed data,
and numpyro.infer.util.log_density is used to extract a BlackJAX-compatible log-density function for sampling.
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

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ---------------------------------------------------------------------------
# Fixed observation setup — shared across all sampler scripts
# ---------------------------------------------------------------------------

# Rotational phases: 60 evenly spaced points over one full rotation
N_PHASES = 60
PHASES_ROT = np.linspace(0, 360, N_PHASES, endpoint=False)

# Synthetic flat spectra — a single wavelength bin is enough for a broadband
# benchmark. sajax needs arrays, so we pass 1-element arrays.
# flux_quiet = 1.0 (normalised stellar continuum)
# flux_active = 0.7 (spot is 30% darker than the quiet photosphere)
WAVELENGTH = np.array([550.0])   # nm, single broadband bin
FLUX_QUIET = np.array([1.0])
FLUX_ACTIVE = np.array([0.7])

# Fixed stellar / instrument parameters
PARAMS_STELLAR = dict(
    u1=0.4,          # quadratic limb-darkening coefficient 1
    u2=0.2,          # quadratic limb-darkening coefficient 2
    inc_star=90.0,   # equator-on view — maximises latitude degeneracy
)
STELLAR_GRID_SIZE = 100   # stellar radius in pixels; 100 is fast and accurate
VE = 2.0                  # equatorial velocity [km/s] — sets line broadening

# Noise level on the observed light curve (normalised flux units)
SIGMA_NOISE = 5e-4   # ~500 ppm, realistic for a bright star with good photometry

# ---------------------------------------------------------------------------
# Ground-truth spot (used to generate synthetic observations)
# ---------------------------------------------------------------------------
TRUE_LAT  = 20.0    # degrees
TRUE_LONG = 45.0    # degrees
TRUE_SIZE = 5.0     # degrees radius

# ---------------------------------------------------------------------------
# Prior bounds on the three inferred spot parameters
# ---------------------------------------------------------------------------
LAT_MIN,  LAT_MAX  =  -90.0, 90.0   # degrees latitude
LONG_MIN, LONG_MAX =   0.0, 360.0   # degrees longitude
SIZE_MIN, SIZE_MAX =   1.0,  90.0   # degrees radius


def _call_sajax(ar_lat: float, ar_long: float, ar_size: float) -> jnp.ndarray:
    """
    Call sajax's compute_light_curve for a single spot and return the
    broadband light curve as a 1-D JAX array of shape (N_PHASES,).

    sajax expects Python lists for ar_lat / ar_long / ar_size, which is fine
    because this function is called inside numpyro.infer.util.log_density
    (no JIT through the sajax call itself).
    """
    result = sajax.compute_light_curve(
        wavelength=WAVELENGTH,
        flux_quiet=FLUX_QUIET,
        flux_active=FLUX_ACTIVE,
        params=PARAMS_STELLAR,
        ar_lat=[float(ar_lat)],
        ar_long=[float(ar_long)],
        ar_size=[float(ar_size)],
        phases_rot=PHASES_ROT,
        stellar_grid_size=STELLAR_GRID_SIZE,
        ve=VE,
        ldc_mode="quadratic",
    )

    return jnp.array(result["lc"])


def generate_observations(seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic noisy light curve from the ground-truth spot parameters.

    Returns:
        y_obs: (N_PHASES,) array of observed normalised flux values
    """
    lc_true = np.array(_call_sajax(TRUE_LAT, TRUE_LONG, TRUE_SIZE))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, SIGMA_NOISE, size=lc_true.shape)
    return lc_true + noise


# Generate observations once at import time so every sampler uses the same data.
OBS_LIGHT_CURVE = generate_observations(seed=42)


def spot_model(y_obs: jnp.ndarray = jnp.array(OBS_LIGHT_CURVE)):
    """
    NumPyro model for the stellar spot posterior.

    Latent variables:
        ar_lat  ~ Uniform(LAT_MIN,  LAT_MAX)   — spot latitude  [degrees]
        ar_long ~ Uniform(LONG_MIN, LONG_MAX)  — spot longitude [degrees]
        ar_size ~ Uniform(SIZE_MIN, SIZE_MAX)  — spot radius    [degrees]

    Likelihood:
        y_obs ~ Normal(lc(ar_lat, ar_long, ar_size), SIGMA_NOISE)
    """
    ar_lat  = numpyro.sample("ar_lat",  dist.Uniform(LAT_MIN,  LAT_MAX))
    ar_long = numpyro.sample("ar_long", dist.Uniform(LONG_MIN, LONG_MAX))
    ar_size = numpyro.sample("ar_size", dist.Uniform(SIZE_MIN, SIZE_MAX))

    lc_model = _call_sajax(ar_lat, ar_long, ar_size)

    numpyro.sample(
        "y_obs",
        dist.Normal(lc_model, SIGMA_NOISE),
        obs=y_obs,
    )


def make_log_density(y_obs: np.ndarray = OBS_LIGHT_CURVE):
    """
    Returns a BlackJAX-compatible log-density function for the spot posterior.

    The returned function accepts a 3-element parameter vector:
        x = [ar_lat, ar_long, ar_size]
    and returns scalar log p(x | y_obs).

    Args:
        y_obs: (N_PHASES,) observed light curve (default: module-level synthetic data)

    Returns:
        log_density_fn(x): scalar log p(x | y_obs)
    """
    y_obs_jnp = jnp.array(y_obs)

    def log_density_fn(x):
        ar_lat, ar_long, ar_size = x[0], x[1], x[2]
        ld, _ = log_density(
            spot_model,
            model_args=(),
            model_kwargs={"y_obs": y_obs_jnp},
            params={"ar_lat": ar_lat, "ar_long": ar_long, "ar_size": ar_size},
        )
        return ld

    return log_density_fn


# ---------------------------------------------------------------------------
# Default parameter names and ground truth — used by sampler scripts for
# diagnostics (replaces DEFAULT_MEANS / DEFAULT_WEIGHTS from the Gaussian model)
# ---------------------------------------------------------------------------
PARAM_NAMES   = ["ar_lat", "ar_long", "ar_size"]
PARAM_BOUNDS  = jnp.array([[LAT_MIN, LAT_MAX], [LONG_MIN, LONG_MAX], [SIZE_MIN, SIZE_MAX]])
TRUE_PARAMS   = jnp.array([TRUE_LAT, TRUE_LONG, TRUE_SIZE])
NDIM          = 3


def plot_model(filename: str = "spot_light_curve.png"):
    """
    Saves a diagnostic plot showing:
      - The ground-truth and observed light curves
      - A 2D marginal heatmap of log p(ar_lat, ar_long | y_obs) with ar_size fixed

    Saved to OUTPUT_DIR / filename.
    """
    log_density_fn = make_log_density()

    # --- Ground-truth vs observed light curve ---
    lc_true = np.array(_call_sajax(TRUE_LAT, TRUE_LONG, TRUE_SIZE))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(PHASES_ROT, lc_true, lw=2, label="True light curve", color="steelblue")
    ax.scatter(PHASES_ROT, OBS_LIGHT_CURVE, s=10, alpha=0.7, color="orange", label="Observed (+ noise)")
    ax.set_xlabel("Rotational phase [deg]")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Spot light curve")
    ax.legend()

    # --- Log-posterior slice: lat vs long at true ar_size ---
    resolution = 60
    lats  = np.linspace(LAT_MIN,  LAT_MAX,  resolution)
    longs = np.linspace(LONG_MIN, LONG_MAX, resolution)
    LL, LO = np.meshgrid(lats, longs)
    grid = np.stack([LL.ravel(), LO.ravel(), np.full(LL.size, TRUE_SIZE)], axis=-1)

    log_p = np.array([log_density_fn(jnp.array(row)) for row in grid])
    log_p = log_p.reshape(resolution, resolution)

    ax = axes[1]
    im = ax.pcolormesh(lats, longs, log_p, shading="auto", cmap="viridis")
    ax.scatter([TRUE_LAT], [TRUE_LONG], color="red", s=60, zorder=5, label="True params")
    ax.set_xlabel("ar_lat [deg]")
    ax.set_ylabel("ar_long [deg]")
    ax.set_title(f"log p(lat, long | data)  [ar_size = {TRUE_SIZE}°]")
    ax.legend()
    plt.colorbar(im, ax=ax, label="log p")

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")