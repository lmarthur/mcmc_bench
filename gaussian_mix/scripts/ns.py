"""
Run Nested Sampling (JAXNS) on the 2D Gaussian mixture model and save trace plots.
"""

import json
import sys
import time
import logging
import warnings
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az
import jaxns
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability.substrates.jax as tfp

# from jaxns import DefaultNestedSampler, Model, Prior, resample

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS, DEFAULT_WEIGHTS, PRIOR_LOW, PRIOR_HIGH

tfpd = tfp.distributions

NS_OUTPUT_DIR = OUTPUT_DIR / "ns"


MAX_SAMPLES = 1e5          # Max samples budget — nested sampling runs until convergence, but this caps cost.
NUM_POSTERIOR_DRAWS = 5000 # Number of uniformly-weighted posterior draws to resample for diagnostics/plots.
NUM_LIVE_POINTS = 1000      # Number of live points for nested sampling — more points gives better accuracy but higher cost.


def main(seed=0, save_outputs=True):
    rng_key = jax.random.PRNGKey(seed)
    resample_key, run_key = jax.random.split(rng_key)
    _print = print if save_outputs else lambda *a, **kw: None

    log_density_fn = make_log_density()
    if save_outputs:
        plot_model()

    # --- Define JAXNS prior and likelihood ---
    # JAXNS Prior uses a generator (yield) pattern.
    def prior_model():
        x1 = yield jaxns.Prior(
            tfpd.Uniform(low=PRIOR_LOW, high=PRIOR_HIGH), name="x1"
        )
        x2 = yield jaxns.Prior(
            tfpd.Uniform(low=PRIOR_LOW, high=PRIOR_HIGH), name="x2"
        )
        return jnp.stack([x1, x2])

    model = jaxns.Model(prior_model=prior_model, log_likelihood=log_density_fn)
    if save_outputs:
        model.sanity_check(jax.random.PRNGKey(1), S=100)

    # --- Run nested sampler ---
    ns = jaxns.NestedSampler(model=model,max_samples=MAX_SAMPLES,num_live_points=NUM_LIVE_POINTS)#,difficult_model=True)

    _print("Running nested sampling...")
    t0 = time.perf_counter()
    termination_reason, state = jax.jit(ns)(run_key)
    results = ns.to_results(termination_reason=termination_reason, state=state)
    wall_time_s = time.perf_counter() - t0

    _print(f"\nTermination reason: {termination_reason}")

    # --- Resample to uniform posterior draws for diagnostics ---
    # results.samples is a dict {"x1": ..., "x2": ...} of weighted dead points.
    # resample() draws S uniformly-weighted samples using importance resampling.
    uniform_samples = jaxns.resample(
        key=resample_key,
        samples=results.samples,
        log_weights=results.log_dp_mean,
        S=NUM_POSTERIOR_DRAWS,
        replace=True,
    )
    x1_samples = np.array(uniform_samples["x1"])  # (NUM_POSTERIOR_DRAWS,)
    x2_samples = np.array(uniform_samples["x2"])  # (NUM_POSTERIOR_DRAWS,)
    # Stack into (1, NUM_POSTERIOR_DRAWS, 2) — a single "chain" for ArviZ
    samples_2d = np.stack([x1_samples, x2_samples], axis=-1)[None, :, :]

    # --- Diagnostics ---
    means_np = np.array(DEFAULT_MEANS)
    num_modes = len(means_np)

    # Mode assignment for the resampled posterior
    dists = np.linalg.norm(
        samples_2d[:, :, None, :] - means_np[None, None, :, :], axis=-1
    )
    chain_assignments = np.argmin(dists, axis=-1)  # (1, NUM_POSTERIOR_DRAWS)
    flat_assignments = chain_assignments.ravel()

    mode_weights = np.bincount(flat_assignments, minlength=num_modes) / flat_assignments.size

    # Nested sampling has no chains, so inter-mode transitions aren't meaningful.
    # Instead report ESS directly from JAXNS (Kish's estimate on the weighted samples).
    jaxns_ess = float(results.ESS)
    total_likelihood_evals = int(results.total_num_likelihood_evaluations)
    ess_per_likelihood_eval = jaxns_ess / total_likelihood_evals

    # Log evidence
    log_z = float(results.log_Z_mean)
    log_z_uncert = float(results.log_Z_uncert)

    # ArviZ summary on the resampled draws (single pseudo-chain)
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    if not save_outputs:
        _az_log.setLevel(logging.ERROR)
    idata = az.from_dict(
        posterior={"x1": samples_2d[:, :, 0], "x2": samples_2d[:, :, 1]}
    )
    summary = az.summary(idata, var_names=["x1", "x2"])
    _az_log.setLevel(_az_prev)

    _print("\n=== Diagnostics ===")
    _print(f"  log Z (evidence):              {log_z:.3f} ± {log_z_uncert:.3f}")
    _print(f"  JAXNS ESS (Kish estimate):     {jaxns_ess:.1f}")
    _print(f"  Total likelihood evaluations:  {total_likelihood_evals}")
    _print(f"  Likelihood evals / sample:     {results.total_num_likelihood_evaluations / max(1, int(results.total_num_samples)):.1f}")
    _print(f"  ESS per likelihood eval:       {ess_per_likelihood_eval:.4f}")
    _print(f"  Wall-clock time:               {wall_time_s:.2f}s")
    _print()
    true_weights = np.array(DEFAULT_WEIGHTS)
    _print(f"  Mode weight recovery (empirical vs true):")
    for k, (w, tw) in enumerate(zip(mode_weights, true_weights)):
        _print(f"    Mode {k}: {w:.3f}  (true: {tw:.3f})")
    _print()
    # Note: R-hat is 1.0 by construction (single chain) — only ESS/MCSE are meaningful here
    _print("  ArviZ summary (ESS, MCSE — R-hat is trivially 1.0 for a single chain):")
    _print(summary.to_string())

    # --- Results ---
    diagnostics = {
        "sampler": "NestedSampling_JAXNS",
        "wall_time_s": wall_time_s,
        "num_posterior_draws": NUM_POSTERIOR_DRAWS,
        "prior_low": PRIOR_LOW,
        "prior_high": PRIOR_HIGH,
        "log_Z_mean": log_z,
        "log_Z_uncert": log_z_uncert,
        "total_likelihood_evals": total_likelihood_evals,
        "total_ns_samples": int(results.total_num_samples),
        "likelihood_evals_per_ns_sample": float(
            results.total_num_likelihood_evaluations / max(1, int(results.total_num_samples))
        ),
        "jaxns_ess_kish": jaxns_ess,
        "ess_per_likelihood_eval": ess_per_likelihood_eval,
        "mode_weights": mode_weights.tolist(),
        "true_mode_weights": np.array(DEFAULT_WEIGHTS).tolist(),
        "arviz_summary": json.loads(summary.to_json()),
    }
    if save_outputs:
        NS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(NS_OUTPUT_DIR / "idata.nc"))
        diag_path = NS_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)

    if not save_outputs:
        return diagnostics

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(x1_samples, x2_samples, alpha=0.15, s=2, color="steelblue")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title(f"Nested Sampling posterior ({NUM_POSTERIOR_DRAWS} resampled draws)")
    axes[0].set_aspect("equal")

    # Nested sampling dead-point log-likelihood trace (shrinkage curve)
    # This replaces the MCMC trace plot — shows the NS compression of prior volume
    log_L_dead = np.array(results.log_L_samples[: int(results.total_num_samples)])
    axes[1].plot(log_L_dead, lw=0.6, color="steelblue", alpha=0.8)
    axes[1].set_xlabel("dead point index")
    axes[1].set_ylabel("log L")
    axes[1].set_title("NS shrinkage: log-likelihood of dead points")

    fig.tight_layout()
    out_path = NS_OUTPUT_DIR / "samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved samples plot to {out_path}")

    # --- Corner plot ---
    corner_axes = az.plot_pair(
        idata,
        var_names=["x1", "x2"],
        kind=["scatter", "kde"],
        scatter_kwargs={"alpha": 0.05, "s": 2},
        kde_kwargs={"contourf_kwargs": {"alpha": 0.3}},
        marginals=True,
        figsize=(6, 6),
    )
    corner_fig = corner_axes.ravel()[0].get_figure()
    corner_path = NS_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    _print(f"Saved corner plot to {corner_path}")
    return diagnostics

if __name__ == "__main__":
    main()