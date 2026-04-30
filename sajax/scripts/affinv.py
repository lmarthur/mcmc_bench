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
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import our specific model components
from model import (
    make_inference_fns,
    make_constrain_fn,
    plot_model,
    _call_sajax,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    TIMES,
    OBS_LIGHT_CURVE,
    PRIOR_DISTRIBUTIONS,
    TRUE_LDC_U1,
    TRUE_LDC_U2,
    TRUE_P_ORB,
)

AFFINV_OUTPUT_DIR = OUTPUT_DIR / "affinv"

NUM_BURNIN = 500
NUM_SAMPLES = 1000
NUM_WALKERS = 64
NDIM = len(PARAM_NAMES)


def get_initial_coords(key, num_walkers, unravel_fn):
    """Sample each walker's starting position from the prior in unconstrained space."""
    from numpyro.distributions import biject_to
    inv_transforms = {name: biject_to(d.support).inv for name, d in PRIOR_DISTRIBUTIONS.items()}

    walker_keys = jax.random.split(key, num_walkers)
    flat_positions = []
    for wk in walker_keys:
        param_keys = jax.random.split(wk, len(PRIOR_DISTRIBUTIONS))
        z_dict = {
            name: inv_transforms[name](d.sample(pk))
            for pk, (name, d) in zip(param_keys, PRIOR_DISTRIBUTIONS.items())
        }
        flat_z, _ = jax.flatten_util.ravel_pytree(z_dict)
        flat_positions.append(flat_z)
    return jnp.stack(flat_positions)


def main(seed=0, save_outputs=True):
    init_key, state_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn, _, init_z = make_inference_fns(init_key)
    constrain_fn = make_constrain_fn()
    _, unravel_fn = jax.flatten_util.ravel_pytree(init_z)
    log_density_flat = lambda x: log_density_fn(unravel_fn(x))

    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # --- Initialize walkers ---
    coords = get_initial_coords(init_key, NUM_WALKERS, unravel_fn)

    n_show = min(8, NUM_WALKERS)
    coords_constrained = jax.vmap(lambda x: constrain_fn(unravel_fn(x)))(coords[:n_show])
    _print(f"\nInitial walker positions (constrained space, first {n_show} of {NUM_WALKERS}):")
    _print(f"  {'param':20s}  " + "  ".join(f"walker{i:02d}" for i in range(n_show)))
    for name in PARAM_NAMES:
        vals = np.array(coords_constrained[name])
        _print(f"  {name:20s}  " + "  ".join(f"{v:8.4f}" for v in vals))

    # --- Initialize sampler ---
    sampler = emcee_jax.EnsembleSampler(log_density_flat)
    state = sampler.init(state_key, coords)

    # --- Run chains ---
    _print(f"Sampling sajax model ({NUM_SAMPLES} steps, {NUM_WALKERS} walkers, {NDIM} params)...")
    trace = sampler.sample_parallel(sample_key, state, NUM_SAMPLES, progress=save_outputs)

    # Reshape: (NUM_STEPS, NUM_WALKERS, NDIM) -> (NUM_WALKERS, NUM_SAMPLES, NDIM)
    raw = np.asarray(trace.samples.coordinates)
    samples_unc = raw.transpose(1, 0, 2)
    samples_unc = samples_unc[:, NUM_BURNIN:, :]

    # Convert unconstrained samples to constrained space
    flat_unc = samples_unc.reshape(-1, samples_unc.shape[-1])
    constrained = jax.vmap(lambda x: constrain_fn(unravel_fn(x)))(flat_unc)
    # Split walkers into 2 equal groups to satisfy ArviZ's minimum-2-chains
    # requirement for R-hat. Walkers within each half are combined into draws.
    n_post = samples_unc.shape[1]
    half = NUM_WALKERS // 2
    cold_samples = {name: np.array(constrained[name]).reshape(2, half * n_post)
                    for name in PARAM_NAMES}

    # --- Diagnostics ---
    accepted = np.asarray(trace.sample_stats['accept_prob'])
    acceptance = float(accepted.mean())
    total_log_density_evals = NUM_WALKERS * NUM_SAMPLES

    posterior_dict = {name: cold_samples[name] for name in PARAM_NAMES}
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={"acceptance_rate": accepted[NUM_BURNIN:, :].T.reshape(2, half * n_post)},
    )

    summary = az.summary(idata)
    total_bulk_ess = summary["ess_bulk"].sum()
    ess_per_logp_eval = total_bulk_ess / total_log_density_evals

    _print("\n=== Diagnostics ===")
    _print(f"  Mean acceptance rate:          {acceptance:.3f}")
    _print(f"  Total log-density evaluations: {int(total_log_density_evals)}")
    _print("\n  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print(f"\n  Total Bulk ESS per log-density eval: {ess_per_logp_eval:.4f}")

    wall_time_s = time.perf_counter() - t0

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = np.array([np.array(cold_samples[p]).mean() for p in PARAM_NAMES])
    param_bias = posterior_means - gt_array

    _print("\n  Parameter recovery (posterior mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    diagnostics = {
        "sampler": "AffineInvariantEnsemble",
        "num_walkers": NUM_WALKERS,
        "num_burnin": NUM_BURNIN,
        "num_samples": NUM_SAMPLES,
        "ndim": NDIM,
        "mean_acceptance_rate": float(acceptance),
        "total_log_density_evals": int(total_log_density_evals),
        "wall_time_s": float(wall_time_s),
        "total_bulk_ess": float(total_bulk_ess),
        "bulk_ess_per_logp_eval": float(ess_per_logp_eval),
        "posterior_means": {name: float(pm) for name, pm in zip(PARAM_NAMES, posterior_means)},
        "ground_truth": {k: float(v) for k, v in GROUND_TRUTH.items()},
        "param_bias": {name: float(b) for name, b in zip(PARAM_NAMES, param_bias)},
        "arviz_summary": json.loads(summary.to_json()),
    }

    # --- Save Results ---
    if save_outputs:
        AFFINV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(AFFINV_OUTPUT_DIR / "sajax_idata.nc"))
        diag_path = AFFINV_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {AFFINV_OUTPUT_DIR / 'sajax_idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

        # 1. Trace Plots (subset for readability)
        plot_vars = PARAM_NAMES[:6]
        az.plot_trace(idata, var_names=plot_vars)
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
        mean_c = {name: float(np.array(v).mean()) for name, v in constrained.items()}

        lc_bestfit = np.array(
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
                TRUE_P_ORB,
                mean_c["ldc_u1"],
                mean_c["ldc_u2"],
            )["lc"]
        )

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
                GROUND_TRUTH["ecc_h"]**2 + GROUND_TRUTH["ecc_k"]**2,
                float(np.arctan2(GROUND_TRUTH["ecc_k"], GROUND_TRUTH["ecc_h"])),
                TRUE_P_ORB,
                TRUE_LDC_U1,
                TRUE_LDC_U2,
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

    return diagnostics


if __name__ == "__main__":
    main()
