"""
Run Random Walk Metropolis-Hastings on the SAJAX planet+activity model and save outputs.
"""

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import (
    make_log_density,
    plot_model,
    _call_sajax,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    TIMES,
    OBS_LIGHT_CURVE,
    LAT_MIN, LAT_MAX, LONG_MIN, LONG_MAX, SIZE_MIN, SIZE_MAX,
    FLUX_MIN, FLUX_MAX, P_ROT_MIN, P_ROT_MAX, LDC_U1_MIN, LDC_U1_MAX,
    LDC_U2_MIN, LDC_U2_MAX, PLANET_RADIUS_MIN, PLANET_RADIUS_MAX,
    SEMI_MAJOR_MIN, SEMI_MAJOR_MAX, INCLINATION_MIN, INCLINATION_MAX,
    ECCENTRICITY_MIN, ECCENTRICITY_MAX, ARG_PERIAPSIS_MIN, ARG_PERIAPSIS_MAX,
    P_ORB_MIN, P_ORB_MAX,
)

RWMH_OUTPUT_DIR = OUTPUT_DIR / "rwmh"

NDIM = len(PARAM_NAMES)
NUM_BURNIN = 500
NUM_SAMPLES = 1000
NUM_CHAINS = 8
# Roberts, Gelman & Gilks (1997): 2.38/sqrt(d) targets ~23.4% acceptance
STEP_SIZE = 2.38 / np.sqrt(NDIM)  # ≈ 0.577 for d=17


# Prior bounds array — used for initialisation
PRIOR_MINS = np.array([
    LAT_MIN, LONG_MIN, SIZE_MIN, FLUX_MIN,           # spot
    LAT_MIN, LONG_MIN, SIZE_MIN, FLUX_MIN,           # facula
    P_ROT_MIN,                                       # p_rot
    PLANET_RADIUS_MIN, SEMI_MAJOR_MIN, INCLINATION_MIN,
    ECCENTRICITY_MIN, ARG_PERIAPSIS_MIN, P_ORB_MIN,  # planet
    LDC_U1_MIN, LDC_U2_MIN,                          # LDC
])
PRIOR_MAXS = np.array([
    LAT_MAX, LONG_MAX, SIZE_MAX, FLUX_MAX,
    LAT_MAX, LONG_MAX, SIZE_MAX, FLUX_MAX,
    P_ROT_MAX,
    PLANET_RADIUS_MAX, SEMI_MAJOR_MAX, INCLINATION_MAX,
    ECCENTRICITY_MAX, ARG_PERIAPSIS_MAX, P_ORB_MAX,
    LDC_U1_MAX, LDC_U2_MAX,
])


def get_initial_positions(key: jax.Array, num_chains: int) -> jnp.ndarray:
    """
    Initialise chains near the ground truth.
    Uniform-prior parameters: sample within ±10% of prior range around ground truth.
    LogNormal/Beta parameters: sample from a tight version of their prior (σ=0.1 in
    log-space for LogNormal; Uniform(0, 0.1) for eccentricity).
    """
    key, base_key, k_prot, k_rp, k_sma, k_ecc = jax.random.split(key, 6)

    center = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    width = (PRIOR_MAXS - PRIOR_MINS) * 0.10
    low = np.maximum(PRIOR_MINS, center - width)
    high = np.minimum(PRIOR_MAXS, center + width)
    positions = jax.random.uniform(base_key, shape=(num_chains, NDIM), minval=low, maxval=high)

    # Override the four parameters whose priors changed from Uniform
    # idx 8  — p_rot:          LogNormal(ln(true), 1.0)
    # idx 9  — planet_radius:  LogNormal(ln(true), 0.5)
    # idx 10 — semimajor_axis: LogNormal(ln(5.0),  0.5)
    # idx 12 — eccentricity:   Beta(2, 10)
    positions = positions.at[:, 8].set(
        jnp.exp(jax.random.normal(k_prot, (num_chains,)) * 0.1 + jnp.log(center[8])))
    positions = positions.at[:, 9].set(
        jnp.exp(jax.random.normal(k_rp,   (num_chains,)) * 0.1 + jnp.log(center[9])))
    positions = positions.at[:, 10].set(
        jnp.exp(jax.random.normal(k_sma,  (num_chains,)) * 0.1 + jnp.log(5.0)))
    positions = positions.at[:, 12].set(
        jax.random.uniform(k_ecc, (num_chains,), minval=0.0, maxval=0.1))

    return positions


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


def main(seed: int = 0, save_outputs: bool = True):
    init_key, burnin_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # --- Initialise chains near ground truth ---
    initial_positions = get_initial_positions(init_key, NUM_CHAINS)

    kernel = blackjax.rmh(
        log_density_fn,
        proposal_generator=blackjax.mcmc.random_walk.normal(sigma=STEP_SIZE),
    )
    init_fn = jax.vmap(kernel.init)
    initial_states = init_fn(initial_positions)

    @jax.vmap
    def run_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_BURNIN)

    # --- Burn-in ---
    if NUM_BURNIN > 0:
        _print(f"Running burn-in ({NUM_BURNIN} steps, {NUM_CHAINS} chains)...")
        burnin_keys = jax.random.split(burnin_key, NUM_CHAINS)
        burnin_states, _ = run_chain(burnin_keys, initial_states)
        post_burnin_states = jax.tree.map(lambda x: x[:, -1], burnin_states)
    else:
        post_burnin_states = initial_states

    # --- Sample ---
    _print(f"Sampling ({NUM_SAMPLES} steps, {NUM_CHAINS} chains, {NDIM} params)...")

    @jax.vmap
    def run_sample_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_SAMPLES)

    chain_sample_keys = jax.random.split(sample_key, NUM_CHAINS)
    all_states, all_infos = run_sample_chain(chain_sample_keys, post_burnin_states)

    # all_states.position: (NUM_CHAINS, NUM_SAMPLES, NDIM)
    samples = np.array(all_states.position)

    # --- Diagnostics ---
    acceptance = float(np.mean(np.array(all_infos.acceptance_rate)))
    total_log_density_evals = NUM_CHAINS * (NUM_BURNIN + NUM_SAMPLES)

    # ArviZ summary
    posterior_dict = {PARAM_NAMES[i]: samples[:, :, i] for i in range(NDIM)}
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={"acceptance_rate": np.array(all_infos.acceptance_rate)},
    )
    summary = az.summary(idata)
    total_bulk_ess = summary["ess_bulk"].sum()
    ess_per_logp_eval = total_bulk_ess / total_log_density_evals

    # Parameter recovery vs ground truth
    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = samples.mean(axis=(0, 1))  # mean over chains and samples
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  Mean acceptance rate:          {acceptance:.3f}")
    _print(f"  Total log-density evaluations: {int(total_log_density_evals)}")
    _print()
    _print("  Parameter recovery (posterior mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print()
    _print(f"  Total bulk ESS: {total_bulk_ess:.1f}")
    _print(f"  Bulk ESS per log-density eval: {ess_per_logp_eval:.4f}")

    wall_time_s = time.perf_counter() - t0
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "RandomWalkMetropolisHastings",
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_BURNIN,
        "num_samples": NUM_SAMPLES,
        "ndim": NDIM,
        "step_size": float(STEP_SIZE),
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

    if save_outputs:
        RWMH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(RWMH_OUTPUT_DIR / "idata.nc"))
        diag_path = RWMH_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {RWMH_OUTPUT_DIR / 'idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # Trace plots for first 6 parameters
    axes = az.plot_trace(idata, var_names=PARAM_NAMES[:6], figsize=(14, 10))
    plt.tight_layout()
    trace_path = RWMH_OUTPUT_DIR / "traces_subset.png"
    plt.savefig(trace_path, dpi=150, bbox_inches="tight")
    plt.close()
    _print(f"Saved trace plot to {trace_path}")

    # Corner plot — all parameters
    az.rcParams["plot.max_subplots"] = len(PARAM_NAMES) ** 2
    az.plot_pair(
        idata,
        var_names=PARAM_NAMES,
        kind="kde",
        marginals=True,
        figsize=(24, 24),
    )
    corner_path = RWMH_OUTPUT_DIR / "corner_all.png"
    plt.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close()
    _print(f"Saved full corner plot to {corner_path}")

    # Best-fit light curve using posterior mean
    mean_params = samples.mean(axis=(0, 1))
    mean_dict = {name: float(mean_params[i]) for i, name in enumerate(PARAM_NAMES)}

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
    lc_path = RWMH_OUTPUT_DIR / "bestfit_lightcurve.png"
    fig.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved best-fit light curve to {lc_path}")

    return diagnostics


if __name__ == "__main__":
    main()
