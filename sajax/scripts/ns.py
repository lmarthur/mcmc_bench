"""
Run Nested Sampling (JAXNS) on the SAJAX planet+activity model and save outputs.
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
import numpyro.distributions as dist
import tensorflow_probability.substrates.jax as tfp

from model import (
    make_log_likelihood,
    make_constrain_fn,
    plot_model,
    plot_bestfit_lightcurve,
    compute_chi2,
    compute_lc_from_constrained,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    OBS_LIGHT_CURVE,
    LC_TRUE,
    TIMES,
    PRIOR_DISTRIBUTIONS,
)

tfpd = tfp.distributions

NS_OUTPUT_DIR = OUTPUT_DIR / "ns"

MAX_SAMPLES = 1e5
NUM_POSTERIOR_DRAWS = 5000
NUM_LIVE_POINTS = 100
NUM_SLICES = 25
DLOGZ_THRESHOLD = 100.0

# Diagnostic stride — print intermediate results every DIAG_STRIDE dead points
DIAG_STRIDE = 50
PLOT_STRIDE = 200

_DIAG_PARAMS = [
    "spot_lat", "spot_long", "spot_size", "spot_flux",
    "p_rot", "planet_radius", "semimajor_axis", "P_orb",
]


def _numpyro_to_tfp(d):
    if isinstance(d, dist.Uniform):
        return tfpd.Uniform(low=d.low, high=d.high)
    elif isinstance(d, dist.LogNormal):
        return tfpd.LogNormal(loc=d.loc, scale=d.scale)
    elif isinstance(d, dist.Normal):
        return tfpd.Normal(loc=d.loc, scale=d.scale)
    elif isinstance(d, dist.Beta):
        return tfpd.Beta(
            concentration1=jnp.float64(d.concentration1),
            concentration0=jnp.float64(d.concentration0),
        )
    elif isinstance(d, dist.LogUniform):
        return tfpd.TransformedDistribution(
            distribution=tfpd.Uniform(low=jnp.log(d.low), high=jnp.log(d.high)),
            bijector=tfp.bijectors.Exp(),
        )
    raise TypeError(f"No TFP equivalent known for {type(d)}")


def run_nested_sampling_diagnostics(results, output_dir=None):
    """
    Analyze dead points from nested sampling and print diagnostics at regular intervals.
    
    Saves an animated GIF of LC snapshots to output_dir/lc_evolution.gif
    every PLOT_STRIDE dead points.
    """
    from io import BytesIO
    from PIL import Image

    constrain_fn = make_constrain_fn()
    
    samples_dict = results.samples
    log_L = np.array(results.log_L_samples[: int(results.total_num_samples)])
    log_weights = np.array(results.log_dp_mean[: int(results.total_num_samples)])
    
    n_dead = len(log_L)
    
    print(f"\n=== Nested Sampling Diagnostics ===")
    print(f"(Analyzing {n_dead} dead points, stride={DIAG_STRIDE})\n")
    
    col_w = 13
    header = f"{'dead_idx':>6}  {'log_L':>10}  {'chi2_red':>9}  " + "  ".join(f"{p:>{col_w}}" for p in _DIAG_PARAMS)
    sep    = "=" * len(header)
    print(header)
    print(sep)
    
    frames = []
    
    for idx in range(0, n_dead, DIAG_STRIDE):
        constrained = {name: np.array(samples_dict[name])[idx] for name in PARAM_NAMES}

        constrained["spot_lat"] = np.rad2deg(np.arcsin(constrained["sin_lat"]))
        constrained["inclination"] = np.rad2deg(np.arccos(constrained["impact_param"] / constrained["semimajor_axis"]))
        constrained["eccentricity"] = constrained["ecc_h"]**2 + constrained["ecc_k"]**2
        constrained["arg_periapsis"] = np.arctan2(constrained["ecc_k"], constrained["ecc_h"])
        constrained["ldc_u1"] = 2 * np.sqrt(constrained["ldc_q1"]) * constrained["ldc_q2"]
        constrained["ldc_u2"] = np.sqrt(constrained["ldc_q1"]) * (1 - 2 * constrained["ldc_q2"])

        chi2 = compute_chi2(constrained)
        
        param_str = "  ".join(f"{float(constrained[p]):>{col_w}.5f}" for p in _DIAG_PARAMS)
        print(f"{idx:>6}  {log_L[idx]:>10.3f}  {chi2:>9.4f}  {param_str}")
        
        if output_dir is not None and idx % PLOT_STRIDE == 0:
            lc_model = np.array(compute_lc_from_constrained(constrained))
            fig, (ax_lc, ax_res) = plt.subplots(
                2, 1, figsize=(10, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=3, color="orange", alpha=0.5, label="Obs")
            ax_lc.plot(TIMES, LC_TRUE, lw=1.5, color="steelblue", label="True")
            ax_lc.plot(TIMES, lc_model, lw=1.5, color="crimson", ls="--",
                       label=f"Dead point {idx}  log L={log_L[idx]:.3f}  χ²_r={chi2:.3f}")
            ax_lc.legend(frameon=False, fontsize=9)
            ax_lc.set_ylabel("Flux")
            ax_lc.spines["top"].set_visible(False)
            ax_lc.spines["right"].set_visible(False)
            
            res_ppm = (OBS_LIGHT_CURVE - lc_model) * 1e6
            ax_res.scatter(TIMES, res_ppm, s=3, color="orange", alpha=0.5)
            ax_res.axhline(0, color="crimson", lw=1, ls="--")
            ax_res.set_xlabel("Time [days]")
            ax_res.set_ylabel("Res. [ppm]")
            ax_res.spines["top"].set_visible(False)
            ax_res.spines["right"].set_visible(False)
            
            fig.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGBA"))
    
    if frames and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gif_path = output_dir / "lc_evolution.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0,
        )
        print(f"\nSaved LC evolution GIF ({len(frames)} frames) to {gif_path}")


def main(seed=0, save_outputs=True):
    rng_key = jax.random.PRNGKey(seed)
    resample_key, run_key = jax.random.split(rng_key)
    _print = print if save_outputs else lambda *a, **kw: None

    log_likelihood_fn = make_log_likelihood(OBS_LIGHT_CURVE)
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    # --- Define JAXNS prior and likelihood ---
    def prior_model():
        samples = {}
        for name, d in PRIOR_DISTRIBUTIONS.items():
            samples[name] = yield jaxns.Prior(_numpyro_to_tfp(d), name=name)
        return samples

    model = jaxns.Model(prior_model=prior_model, log_likelihood=log_likelihood_fn)
    if save_outputs:
        model.sanity_check(jax.random.PRNGKey(1), S=100)

    # --- Run nested sampler ---
    ns = jaxns.NestedSampler(model=model, 
                            max_samples=MAX_SAMPLES, 
                            num_live_points=NUM_LIVE_POINTS,
                            num_slices=NUM_SLICES,
                            verbose=True)

    _print("Running nested sampling...")
    t0 = time.perf_counter()
    term_cond = jaxns.TerminationCondition(dlogZ=DLOGZ_THRESHOLD)
    termination_reason, state = jax.jit(ns)(run_key, term_cond=term_cond)
    results = ns.to_results(termination_reason=termination_reason, state=state)
    wall_time_s = time.perf_counter() - t0

    _print(f"\nTermination reason: {termination_reason}")

    # --- Run step-by-step diagnostics on dead points ---
    if save_outputs:
        run_nested_sampling_diagnostics(results, output_dir=NS_OUTPUT_DIR)

    # --- Resample to uniform posterior draws ---
    uniform_samples = jaxns.resample(
        key=resample_key,
        samples=results.samples,
        log_weights=results.log_dp_mean,
        S=NUM_POSTERIOR_DRAWS,
        replace=True,
    )

    # --- Build constrained samples with derived quantities ---
    ecc_h = np.array(uniform_samples["ecc_h"])
    ecc_k = np.array(uniform_samples["ecc_k"])
    ldc_q1 = np.array(uniform_samples["ldc_q1"])
    ldc_q2 = np.array(uniform_samples["ldc_q2"])
    impact_param_arr   = np.array(uniform_samples["impact_param"])
    semimajor_axis_arr = np.array(uniform_samples["semimajor_axis"])
    constrained_with_derived = {
        **{name: np.array(uniform_samples[name]) for name in PARAM_NAMES},
        "spot_lat": np.rad2deg(np.arcsin(np.array(uniform_samples["sin_lat"]))),
        "inclination": np.rad2deg(np.arccos(impact_param_arr / semimajor_axis_arr)),
        "eccentricity": ecc_h**2 + ecc_k**2,
        "arg_periapsis": np.arctan2(ecc_k, ecc_h),
        "ldc_u1": 2 * np.sqrt(ldc_q1) * ldc_q2,
        "ldc_u2": np.sqrt(ldc_q1) * (1 - 2 * ldc_q2),
    }

    # --- Diagnostics ---
    jaxns_ess = float(results.ESS)
    total_likelihood_evals = int(results.total_num_likelihood_evaluations)
    ess_per_likelihood_eval = jaxns_ess / total_likelihood_evals
    log_z = float(results.log_Z_mean)
    log_z_uncert = float(results.log_Z_uncert)

    # uniform_samples keys match PRIOR_DISTRIBUTIONS (and thus PARAM_NAMES)
    posterior_dict = {name: np.array(uniform_samples[name])[None, :] for name in PARAM_NAMES}
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    if not save_outputs:
        _az_log.setLevel(logging.ERROR)
    idata = az.from_dict(posterior=posterior_dict)
    summary = az.summary(idata)
    _az_log.setLevel(_az_prev)

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = np.array([np.array(uniform_samples[name]).mean() for name in PARAM_NAMES])
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  log Z (evidence):              {log_z:.3f} ± {log_z_uncert:.3f}")
    _print(f"  JAXNS ESS (Kish estimate):     {jaxns_ess:.1f}")
    _print(f"  Total likelihood evaluations:  {total_likelihood_evals}")
    _print(f"  Likelihood evals / NS sample:  {results.total_num_likelihood_evaluations / max(1, int(results.total_num_samples)):.1f}")
    _print(f"  ESS per likelihood eval:       {ess_per_likelihood_eval:.4f}")
    _print(f"  Wall-clock time:               {wall_time_s:.2f}s")
    _print()
    _print("  Parameter recovery (posterior mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print()
    _print("  ArviZ summary (ESS, MCSE — R-hat is trivially 1.0 for a single chain):")
    _print(summary.to_string())

    # --- Results ---
    diagnostics = {
        "sampler": "NestedSampling_JAXNS",
        "wall_time_s": float(wall_time_s),
        "num_posterior_draws": NUM_POSTERIOR_DRAWS,
        "num_live_points": NUM_LIVE_POINTS,
        "log_Z_mean": log_z,
        "log_Z_uncert": log_z_uncert,
        "total_likelihood_evals": total_likelihood_evals,
        "total_ns_samples": int(results.total_num_samples),
        "likelihood_evals_per_ns_sample": float(
            results.total_num_likelihood_evaluations / max(1, int(results.total_num_samples))
        ),
        "jaxns_ess_kish": jaxns_ess,
        "ess_per_likelihood_eval": ess_per_likelihood_eval,
        "posterior_means": {name: float(pm) for name, pm in zip(PARAM_NAMES, posterior_means)},
        "ground_truth": {k: float(v) for k, v in GROUND_TRUTH.items()},
        "param_bias": {name: float(b) for name, b in zip(PARAM_NAMES, param_bias)},
        "arviz_summary": json.loads(summary.to_json()),
    }

    if save_outputs:
        NS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(NS_OUTPUT_DIR / "sajax_idata.nc"))
        diag_path = NS_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {NS_OUTPUT_DIR / 'sajax_idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # 1. NS shrinkage curve
    fig, ax = plt.subplots(figsize=(10, 4))
    log_L_dead = np.array(results.log_L_samples[: int(results.total_num_samples)])
    ax.plot(log_L_dead, lw=0.6, color="steelblue", alpha=0.8)
    ax.set_xlabel("dead point index")
    ax.set_ylabel("log L")
    ax.set_title("NS shrinkage: log-likelihood of dead points")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    shrinkage_path = NS_OUTPUT_DIR / "shrinkage.png"
    fig.savefig(shrinkage_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved shrinkage plot to {shrinkage_path}")

    # 2. Corner plot — all parameters with truth / MAP / mean reference lines
    n_params = len(PARAM_NAMES)
    az.rcParams["plot.max_subplots"] = n_params ** 2

    axes = az.plot_pair(
        idata,
        var_names=PARAM_NAMES,
        kind="kde",
        marginals=True,
        figsize=(24, 24),
    )

    # Reference values for each parameter
    truth_vals = [float(GROUND_TRUTH[p]) for p in PARAM_NAMES]
    mean_vals  = [float(posterior_means[i]) for i in range(n_params)]

    # Consistent colors
    color_truth = "steelblue"
    color_mean  = "crimson"
    lw = 1.5
    marker_size = 30

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if ax is None:
                continue

            if i == j:
                ax.axvline(truth_vals[i], color=color_truth, ls="-",  lw=lw, alpha=0.8)
                ax.axvline(mean_vals[i],  color=color_mean,  ls=":",  lw=lw, alpha=0.8)

            elif i > j:
                ax.axvline(truth_vals[j], color=color_truth, ls="-",  lw=lw, alpha=0.5)
                ax.axvline(mean_vals[j],  color=color_mean,  ls=":",  lw=lw, alpha=0.5)

                ax.axhline(truth_vals[i], color=color_truth, ls="-",  lw=lw, alpha=0.5)
                ax.axhline(mean_vals[i],  color=color_mean,  ls=":",  lw=lw, alpha=0.5)

                ax.scatter(truth_vals[j], truth_vals[i], color=color_truth,
                           marker="s", s=marker_size, zorder=10, edgecolors="white", linewidths=0.5)
                ax.scatter(mean_vals[j],  mean_vals[i],  color=color_mean,
                           marker="s", s=marker_size, zorder=10, edgecolors="white", linewidths=0.5)

    from matplotlib.lines import Line2D
    fig = axes[0, 0].get_figure()
    legend_handles = [
        Line2D([0], [0], color=color_truth, ls="-",  lw=lw, label="Truth"),
        Line2D([0], [0], color=color_mean,  ls=":",  lw=lw, label="Posterior mean"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
        frameon=True,
        framealpha=0.9,
        fontsize=12,
        title="Reference",
        title_fontsize=13,
    )

    corner_path = NS_OUTPUT_DIR / "corner_all.png"
    fig.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved full corner plot to {corner_path}")

    # 3. Best-fit light curve — delegate to model.py
    plot_bestfit_lightcurve(constrained_with_derived, NS_OUTPUT_DIR, map_params=None)

    return diagnostics


if __name__ == "__main__":
    main()