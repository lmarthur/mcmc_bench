"""
Run Affine Invariant MCMC on the sajax planet+activity model.
"""

import json
import sys
import time
import warnings
from pathlib import Path

# Add src to path if necessary
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az
import emcee_jax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import our specific model components
from model import (
    make_log_density, 
    plot_model, 
    OUTPUT_DIR, 
    PARAM_NAMES, 
    GROUND_TRUTH,
    # Import bounds for initialization
    LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX, SIZE_MIN, SIZE_MAX, 
    FLUX_MIN, FLUX_MAX, P_ROT_MIN, P_ROT_MAX, LDC_U1_MIN, LDC_U1_MAX,
    LDC_U2_MIN, LDC_U2_MAX, PLANET_RADIUS_MIN, PLANET_RADIUS_MAX,
    SEMI_MAJOR_MIN, SEMI_MAJOR_MAX, INCLINATION_MIN, INCLINATION_MAX,
    ECCENTRICITY_MIN, ECCENTRICITY_MAX, ARG_PERIAPSIS_MIN, ARG_PERIAPSIS_MAX,
    P_ORB_MIN, P_ORB_MAX
)

AFFINV_OUTPUT_DIR = OUTPUT_DIR / "affinv"

NUM_BURNIN = 500   # Increased for a high-dimensional model
NUM_SAMPLES = 2000
NUM_WALKERS = 64    # Generally want walkers > 2 * NDIM (17 * 2 = 34)
NDIM = len(PARAM_NAMES)

def get_initial_coords(key, num_walkers):
    """
    Initialize walkers uniformly within a tight-ish region of the prior 
    to avoid starting in zero-probability regions.
    """
    # Create an array of mins and maxes based on your model's priors
    mins = np.array([
        LAT_MIN, LONG_MIN, SIZE_MIN, FLUX_MIN,          # Spot
        LAT_MIN, LONG_MIN, SIZE_MIN, FLUX_MIN,          # Facula
        P_ROT_MIN,                                      # P_rot
        PLANET_RADIUS_MIN, SEMI_MAJOR_MIN, INCLINATION_MIN, 
        ECCENTRICITY_MIN, ARG_PERIAPSIS_MIN, P_ORB_MIN, # Planet
        LDC_U1_MIN, LDC_U2_MIN                           # LDC
    ])
    maxes = np.array([
        LAT_MAX, LONG_MAX, SIZE_MAX, FLUX_MAX,
        LAT_MAX, LONG_MAX, SIZE_MAX, FLUX_MAX,
        P_ROT_MAX,
        PLANET_RADIUS_MAX, SEMI_MAJOR_MAX, INCLINATION_MAX,
        ECCENTRICITY_MAX, ARG_PERIAPSIS_MAX, P_ORB_MAX,
        LDC_U1_MAX, LDC_U2_MAX
    ])

    # Narrow the range slightly so we don't start exactly on a hard boundary
    center = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    width = (maxes - mins) * 0.1 
    
    # Randomize around ground truth or within bounds
    low = np.maximum(mins, center - width)
    high = np.minimum(maxes, center + width)
    
    return jax.random.uniform(key, shape=(num_walkers, NDIM), minval=low, maxval=high)

def main(seed=0, save_outputs=True):
    init_key, state_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # --- Initialize walkers ---
    coords = get_initial_coords(init_key, NUM_WALKERS)

    # --- Initialize sampler ---
    sampler = emcee_jax.EnsembleSampler(log_density_fn)
    state = sampler.init(state_key, coords)

    # --- Run chains ---
    _print(f"Sampling sajax model ({NUM_SAMPLES} steps, {NUM_WALKERS} walkers, {NDIM} params)...")
    trace = sampler.sample_parallel(sample_key, state, NUM_SAMPLES, progress=save_outputs)

    # Reshape: (NUM_STEPS, NUM_WALKERS, NDIM) -> (NUM_WALKERS, NUM_SAMPLES, NDIM)
    raw = np.asarray(trace.samples.coordinates)
    samples = raw.transpose(1, 0, 2)
    samples = samples[:, NUM_BURNIN:, :] 

    # --- Diagnostics ---
    accepted = np.asarray(trace.sample_stats['accept_prob'])
    acceptance = float(accepted.mean())
    total_log_density_evals = NUM_WALKERS * NUM_SAMPLES

    # ArviZ setup for all 17 parameters
    posterior_dict = {PARAM_NAMES[i]: samples[:, :, i] for i in range(NDIM)}
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={"acceptance_rate": accepted[NUM_BURNIN:, :].T},
    )
    
    summary = az.summary(idata)
    total_bulk_ess = summary["ess_bulk"].mean()
    ess_per_logp_eval = total_bulk_ess / total_log_density_evals

    _print("\n=== Diagnostics ===")
    _print(f"  Mean acceptance rate:          {acceptance:.3f}")
    _print(f"  Total log-density evaluations: {int(total_log_density_evals)}")
    _print("\n  ArviZ summary (first 5 params):")
    _print(summary.head().to_string())
    _print(f"\n  Average Bulk ESS per log-density eval: {ess_per_logp_eval:.4f}")

    wall_time_s = time.perf_counter() - t0
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    # --- Save Results ---
    if save_outputs:
        AFFINV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(AFFINV_OUTPUT_DIR / "sajax_idata.nc"))
        
        # Summary plots
        # 1. Trace Plots (subset for readability)
        axes = az.plot_trace(idata, var_names=PARAM_NAMES[:6])
        plt.tight_layout()
        plt.savefig(AFFINV_OUTPUT_DIR / "traces_subset.png")
        plt.close()

        # 2. Corner Plot for planet parameters
        planet_vars = ["planet_radius", "semimajor_axis", "inclination", "P_orb"]
        az.plot_pair(idata, var_names=planet_vars, kind="kde", marginals=True)
        plt.savefig(AFFINV_OUTPUT_DIR / "planet_corner.png")
        plt.close()

    return summary

if __name__ == "__main__":
    main()