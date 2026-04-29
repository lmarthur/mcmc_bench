"""
Run Affine Invariant MCMC on the sajax planet+activity model.
"""

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
    _call_sajax,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    TIMES,
    OBS_LIGHT_CURVE,
    PRIOR_DISTRIBUTIONS,
)

AFFINV_OUTPUT_DIR = OUTPUT_DIR / "affinv"

NUM_BURNIN = 1000
NUM_SAMPLES = 2000
NUM_WALKERS = 64
NDIM = len(PARAM_NAMES)

def get_initial_coords(key, num_walkers):
    """Sample each walker's starting position independently from the prior."""
    coords = []
    for name in PARAM_NAMES:
        key, subkey = jax.random.split(key)
        samples = PRIOR_DISTRIBUTIONS[name].sample(subkey, sample_shape=(num_walkers,))
        coords.append(samples)
    return jnp.stack(coords, axis=-1)

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
        # 1. Trace Plots (subset for readability) — skip any param with zero variance
        plot_vars = [
            p for p in PARAM_NAMES[:6]
            if np.ptp(samples[:, :, PARAM_NAMES.index(p)]) > 1e-8
        ]
        if plot_vars:
            axes = az.plot_trace(idata, var_names=plot_vars)
            plt.tight_layout()
            plt.savefig(AFFINV_OUTPUT_DIR / "traces_subset.png")
            plt.close()

        # 2. Corner plot — all parameters
        az.rcParams["plot.max_subplots"] = len(PARAM_NAMES) ** 2
        az.plot_pair(
            idata,
            var_names=PARAM_NAMES,
            kind="kde",
            marginals=True,
            figsize=(24, 24),
        )
        plt.savefig(AFFINV_OUTPUT_DIR / "corner_all.png", dpi=120, bbox_inches="tight")
        plt.close()

        # 3. Best-fit light curve using posterior mean
        mean_params = samples.mean(axis=(0, 1))
        mean_dict = {name: float(mean_params[i]) for i, name in enumerate(PARAM_NAMES)}
        mean_ecc = mean_dict["ecc_h"]**2 + mean_dict["ecc_k"]**2
        mean_omega = float(np.arctan2(mean_dict["ecc_k"], mean_dict["ecc_h"]))
        mean_u1 = 2 * np.sqrt(mean_dict["ldc_q1"]) * mean_dict["ldc_q2"]
        mean_u2 = np.sqrt(mean_dict["ldc_q1"]) * (1 - 2 * mean_dict["ldc_q2"])

        lc_bestfit = np.array(
            _call_sajax(
                TIMES,
                np.array([mean_dict["spot_lat"], mean_dict["fac_lat"]]),
                np.array([mean_dict["spot_long"], mean_dict["fac_long"]]),
                np.array([mean_dict["spot_size"], mean_dict["fac_size"]]),
                np.stack([np.array([mean_dict["spot_flux"]]), np.array([mean_dict["fac_flux"]])]),
                mean_dict["p_rot"],
                mean_dict["planet_radius"],
                mean_dict["semimajor_axis"],
                np.deg2rad(mean_dict["inclination"]),
                mean_ecc,
                mean_omega,
                mean_dict["P_orb"],
                mean_u1,
                mean_u2,
            )["lc"]
        )

        gt_ecc = GROUND_TRUTH["ecc_h"]**2 + GROUND_TRUTH["ecc_k"]**2
        gt_omega = float(np.arctan2(GROUND_TRUTH["ecc_k"], GROUND_TRUTH["ecc_h"]))
        gt_u1 = 2 * np.sqrt(GROUND_TRUTH["ldc_q1"]) * GROUND_TRUTH["ldc_q2"]
        gt_u2 = np.sqrt(GROUND_TRUTH["ldc_q1"]) * (1 - 2 * GROUND_TRUTH["ldc_q2"])
        lc_true = np.array(
            _call_sajax(
                TIMES,
                np.array([GROUND_TRUTH["spot_lat"], GROUND_TRUTH["fac_lat"]]),
                np.array([GROUND_TRUTH["spot_long"], GROUND_TRUTH["fac_long"]]),
                np.array([GROUND_TRUTH["spot_size"], GROUND_TRUTH["fac_size"]]),
                np.stack([np.array([GROUND_TRUTH["spot_flux"]]), np.array([GROUND_TRUTH["fac_flux"]])]),
                GROUND_TRUTH["p_rot"],
                GROUND_TRUTH["planet_radius"],
                GROUND_TRUTH["semimajor_axis"],
                np.deg2rad(GROUND_TRUTH["inclination"]),
                gt_ecc,
                gt_omega,
                GROUND_TRUTH["P_orb"],
                gt_u1,
                gt_u2,
            )["lc"]
        )

        fig, (ax_lc, ax_res) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                             gridspec_kw={"height_ratios": [3, 1]})

        ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=4, color="orange", alpha=0.6,
                      label="Observations", zorder=1)
        ax_lc.plot(TIMES, lc_true, lw=2, color="steelblue", label="True", zorder=2)
        ax_lc.plot(TIMES, lc_bestfit, lw=2, color="crimson", linestyle="--",
                   label="Posterior mean fit", zorder=3)
        ax_lc.set_ylabel("Normalised flux")
        ax_lc.legend(frameon=False)
        ax_lc.spines["top"].set_visible(False)
        ax_lc.spines["right"].set_visible(False)

        residuals_ppm = (OBS_LIGHT_CURVE - lc_bestfit) * 1e6
        ax_res.scatter(TIMES, residuals_ppm, s=4, color="orange", alpha=0.6)
        ax_res.axhline(0, color="crimson", lw=1, linestyle="--")
        ax_res.set_xlabel("Time [days]")
        ax_res.set_ylabel("Residuals [ppm]")
        ax_res.spines["top"].set_visible(False)
        ax_res.spines["right"].set_visible(False)

        fig.tight_layout()
        lc_path = AFFINV_OUTPUT_DIR / "bestfit_lightcurve.png"
        fig.savefig(lc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return summary

if __name__ == "__main__":
    main()