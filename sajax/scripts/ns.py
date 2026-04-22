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
import tensorflow_probability.substrates.jax as tfp

from model import (
    make_log_density,
    plot_model,
    _call_sajax,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    TIMES,
    OBS_LIGHT_CURVE,
    TRUE_P_ROT, TRUE_PLANET_RADIUS,
    LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX, SIZE_MIN, SIZE_MAX,
    FLUX_MIN, FLUX_MAX, P_ROT_MIN, P_ROT_MAX, LDC_U1_MIN, LDC_U1_MAX,
    LDC_U2_MIN, LDC_U2_MAX, PLANET_RADIUS_MIN, PLANET_RADIUS_MAX,
    SEMI_MAJOR_MIN, SEMI_MAJOR_MAX, INCLINATION_MIN, INCLINATION_MAX,
    ECCENTRICITY_MIN, ECCENTRICITY_MAX, ARG_PERIAPSIS_MIN, ARG_PERIAPSIS_MAX,
    P_ORB_MIN, P_ORB_MAX,
)

tfpd = tfp.distributions

NS_OUTPUT_DIR = OUTPUT_DIR / "ns"

MAX_SAMPLES = 1e5
NUM_POSTERIOR_DRAWS = 5000
NUM_LIVE_POINTS = 500


def main(seed=0, save_outputs=True):
    rng_key = jax.random.PRNGKey(seed)
    resample_key, run_key = jax.random.split(rng_key)
    _print = print if save_outputs else lambda *a, **kw: None

    log_density_fn = make_log_density()
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    # --- Define JAXNS prior and likelihood ---
    # Parameter order must match PARAM_NAMES / the flat array expected by log_density_fn:
    # [spot_lat, spot_long, spot_size, spot_flux,
    #  fac_lat,  fac_long,  fac_size,  fac_flux,
    #  p_rot, planet_radius, semimajor_axis, inclination,
    #  eccentricity, arg_periapsis, P_orb, ldc_u1, ldc_u2]
    def prior_model():
        spot_lat      = yield jaxns.Prior(tfpd.Uniform(low=LAT_MIN,            high=LAT_MAX),            name="spot_lat")
        spot_long     = yield jaxns.Prior(tfpd.Uniform(low=LONG_MIN,           high=LONG_MAX),           name="spot_long")
        spot_size     = yield jaxns.Prior(tfpd.Uniform(low=SIZE_MIN,           high=SIZE_MAX),           name="spot_size")
        spot_flux     = yield jaxns.Prior(tfpd.Uniform(low=FLUX_MIN,           high=FLUX_MAX),           name="spot_flux")
        fac_lat       = yield jaxns.Prior(tfpd.Uniform(low=LAT_MIN,            high=LAT_MAX),            name="fac_lat")
        fac_long      = yield jaxns.Prior(tfpd.Uniform(low=LONG_MIN,           high=LONG_MAX),           name="fac_long")
        fac_size      = yield jaxns.Prior(tfpd.Uniform(low=SIZE_MIN,           high=SIZE_MAX),           name="fac_size")
        fac_flux      = yield jaxns.Prior(tfpd.Uniform(low=FLUX_MIN,           high=FLUX_MAX),           name="fac_flux")
        p_rot         = yield jaxns.Prior(tfpd.LogNormal(loc=jnp.log(TRUE_P_ROT),      scale=1.0),      name="p_rot")
        planet_radius = yield jaxns.Prior(tfpd.LogNormal(loc=jnp.log(TRUE_PLANET_RADIUS), scale=0.5),   name="planet_radius")
        semimajor     = yield jaxns.Prior(tfpd.LogNormal(loc=jnp.log(5.0),     scale=0.5),              name="semimajor_axis")
        inclination   = yield jaxns.Prior(tfpd.Uniform(low=INCLINATION_MIN,    high=INCLINATION_MAX),    name="inclination")
        eccentricity  = yield jaxns.Prior(tfpd.Beta(concentration0=jnp.float64(10.0), concentration1=jnp.float64(2.0)), name="eccentricity")
        arg_periapsis = yield jaxns.Prior(tfpd.Uniform(low=ARG_PERIAPSIS_MIN,  high=ARG_PERIAPSIS_MAX),  name="arg_periapsis")
        P_orb         = yield jaxns.Prior(tfpd.Normal(loc=1.0,                 scale=0.01),              name="P_orb")
        ldc_u1        = yield jaxns.Prior(tfpd.Uniform(low=LDC_U1_MIN,         high=LDC_U1_MAX),         name="ldc_u1")
        ldc_u2        = yield jaxns.Prior(tfpd.Uniform(low=LDC_U2_MIN,         high=LDC_U2_MAX),         name="ldc_u2")
        return jnp.stack([
            spot_lat, spot_long, spot_size, spot_flux,
            fac_lat,  fac_long,  fac_size,  fac_flux,
            p_rot, planet_radius, semimajor, inclination,
            eccentricity, arg_periapsis, P_orb, ldc_u1, ldc_u2,
        ])

    def log_likelihood(x):
        return log_density_fn(x)

    model = jaxns.Model(prior_model=prior_model, log_likelihood=log_likelihood)
    if save_outputs:
        model.sanity_check(jax.random.PRNGKey(1), S=100)

    # --- Run nested sampler ---
    ns = jaxns.NestedSampler(model=model, max_samples=MAX_SAMPLES, num_live_points=NUM_LIVE_POINTS, verbose=True)

    _print("Running nested sampling...")
    t0 = time.perf_counter()
    termination_reason, state = jax.jit(ns)(run_key)
    results = ns.to_results(termination_reason=termination_reason, state=state)
    wall_time_s = time.perf_counter() - t0

    _print(f"\nTermination reason: {termination_reason}")

    # --- Resample to uniform posterior draws ---
    uniform_samples = jaxns.resample(
        key=resample_key,
        samples=results.samples,
        log_weights=results.log_dp_mean,
        S=NUM_POSTERIOR_DRAWS,
        replace=True,
    )

    # Stack into (1, NUM_POSTERIOR_DRAWS, NDIM) — single pseudo-chain for ArviZ
    param_arrays = [np.array(uniform_samples[name]) for name in [
        "spot_lat", "spot_long", "spot_size", "spot_flux",
        "fac_lat",  "fac_long",  "fac_size",  "fac_flux",
        "p_rot", "planet_radius", "semimajor_axis", "inclination",
        "eccentricity", "arg_periapsis", "P_orb", "ldc_u1", "ldc_u2",
    ]]
    samples_nd = np.stack(param_arrays, axis=-1)[None, :, :]  # (1, NUM_POSTERIOR_DRAWS, 17)

    # --- Diagnostics ---
    jaxns_ess = float(results.ESS)
    total_likelihood_evals = int(results.total_num_likelihood_evaluations)
    ess_per_likelihood_eval = jaxns_ess / total_likelihood_evals
    log_z = float(results.log_Z_mean)
    log_z_uncert = float(results.log_Z_uncert)

    posterior_dict = {PARAM_NAMES[i]: samples_nd[:, :, i] for i in range(len(PARAM_NAMES))}
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    if not save_outputs:
        _az_log.setLevel(logging.ERROR)
    idata = az.from_dict(posterior=posterior_dict)
    summary = az.summary(idata)
    _az_log.setLevel(_az_prev)

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = samples_nd[0].mean(axis=0)
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

    # NS shrinkage curve
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

    # Corner plot — all parameters
    az.rcParams["plot.max_subplots"] = len(PARAM_NAMES) ** 2
    az.plot_pair(
        idata,
        var_names=PARAM_NAMES,
        kind="kde",
        marginals=True,
        figsize=(24, 24),
    )
    corner_path = NS_OUTPUT_DIR / "corner_all.png"
    plt.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close()
    _print(f"Saved full corner plot to {corner_path}")

    # Best-fit light curve using posterior mean
    mean_dict = {name: float(posterior_means[i]) for i, name in enumerate(PARAM_NAMES)}

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
            mean_dict["eccentricity"],
            mean_dict["arg_periapsis"],
            mean_dict["P_orb"],
            mean_dict["LDC_u1"],
            mean_dict["LDC_u2"],
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
            GROUND_TRUTH["eccentricity"],
            GROUND_TRUTH["arg_periapsis"],
            GROUND_TRUTH["P_orb"],
            GROUND_TRUTH["LDC_u1"],
            GROUND_TRUTH["LDC_u2"],
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
    lc_path = NS_OUTPUT_DIR / "bestfit_lightcurve.png"
    fig.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved best-fit light curve to {lc_path}")

    return diagnostics


if __name__ == "__main__":
    main()
