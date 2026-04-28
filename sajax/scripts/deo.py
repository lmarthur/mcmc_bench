"""
Run DEO (non-reversible) parallel tempering on the SAJAX planet+activity model and save outputs.

DEO uses a deterministic even-odd parity schedule for swap moves, which is
non-reversible and achieves a round-trip rate independent of the number of chains.
"""

import json
import sys
import time
import logging
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az
import blackjax
import blackjax.mcmc.random_walk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pt_jax

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

DEO_OUTPUT_DIR = OUTPUT_DIR / "deo"

# ===========================================================================
# Tunable parameters
# ===========================================================================

# Number of parallel chains (including the reference/hottest chain).
# More chains → smoother temperature ladder, better mode mixing, higher cost.
NUM_CHAINS = 17

# Number of warm-up (burn-in) steps discarded before collecting samples.
NUM_WARMUP = 5000

# Number of posterior samples collected from the cold chain.
NUM_SAMPLES = 1000

# Step sizes for the local RWMH kernel.
# STEP_SIZE_HOT: used for the hottest chain (β≈0); needs broad proposals.
# STEP_SIZE_COLD: used for the coldest chain (β=1, target); tune for ~23% acceptance.
# Intermediate chains receive linearly interpolated step sizes.
STEP_SIZE_HOT = 2.0
STEP_SIZE_COLD = 0.3

# Scale of the reference distribution (independent Normal per dimension).
# Each parameter's σ is set to REF_SCALE_FRACTION × (prior std dev).
# Increase if the reference fails to cover the prior; decrease to concentrate it.
REF_SCALE_FRACTION = 0.5

# Base for the geometric temperature ladder: β_k = base^{-(N-1-k)}, β_{N-1}=1.
# Larger base → wider spacing between adjacent temperatures (higher swap rejection).
# Smaller base → more chains needed to span the same range.
ANNEALING_BASE = 1.4142135623730951  # sqrt(2), the pt_jax default

# ===========================================================================


def get_initial_positions(key: jax.Array, num_chains: int) -> jnp.ndarray:
    """Sample each chain's starting position independently from the prior."""
    positions = []
    for name in PARAM_NAMES:
        key, subkey = jax.random.split(key)
        prior_key = name.lower() if name.startswith("LDC") else name
        samples = PRIOR_DISTRIBUTIONS[prior_key].sample(subkey, sample_shape=(num_chains,))
        positions.append(samples)
    return jnp.stack(positions, axis=-1)


def rwmh_kernel_generator(log_p, step_size):
    rmh = blackjax.rmh(
        log_p,
        proposal_generator=blackjax.mcmc.random_walk.normal(sigma=step_size),
    )

    def kernel(key, position):
        state = rmh.init(position)
        new_state, _ = rmh.step(key, state)
        return new_state.position

    return kernel


def main(seed: int = 0, save_outputs: bool = True):
    init_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))
    _print = print if save_outputs else lambda *a, **kw: None

    ndim = len(PARAM_NAMES)

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # --- Reference distribution: diagonal Normal, σ = REF_SCALE_FRACTION × prior std ---
    ref_scales = jnp.array([
        float(jnp.sqrt(PRIOR_DISTRIBUTIONS[
            name.lower() if name.startswith("LDC") else name
        ].variance))
        for name in PARAM_NAMES
    ]) * REF_SCALE_FRACTION

    def log_ref(x):
        return dist.Normal(jnp.zeros(ndim), ref_scales).to_event().log_prob(x)

    # --- Temperature schedule ---
    betas = pt_jax.annealing.annealing_exponential(NUM_CHAINS, base=ANNEALING_BASE)

    # --- Build kernels ---
    step_sizes = jnp.linspace(STEP_SIZE_HOT, STEP_SIZE_COLD, NUM_CHAINS)

    K_ind = pt_jax.kernels.generate_independent_annealed_kernel(
        log_prob=log_density_fn,
        log_ref=log_ref,
        annealing_schedule=betas,
        kernel_generator=rwmh_kernel_generator,
        params=step_sizes,
    )
    K_deo = pt_jax.swap.generate_deo_extended_kernel(
        log_prob=log_density_fn,
        log_ref=log_ref,
        annealing_schedule=betas,
    )

    # --- Initialise chains from prior ---
    x0 = get_initial_positions(init_key, NUM_CHAINS)

    # --- Run DEO sampling loop ---
    _print(f"Running DEO (RWMH local kernel, {NUM_CHAINS} chains, "
           f"{NUM_SAMPLES} samples, {NUM_WARMUP} warmup, {ndim} params)...")

    samples, rejection_rates = pt_jax.swap.deo_sampling_loop(
        key=sample_key,
        x0=x0,
        kernel_local=K_ind,
        kernel_deo=K_deo,
        n_samples=NUM_SAMPLES,
        warmup=NUM_WARMUP,
    )
    # samples:         (NUM_SAMPLES, NUM_CHAINS, NDIM)
    # rejection_rates: (NUM_SAMPLES, NUM_CHAINS - 1)

    wall_time_s = time.perf_counter() - t0

    # Cold chain (β=1, target distribution) is the last chain.
    cold_samples = np.array(samples[:, -1, :])  # (NUM_SAMPLES, NDIM)
    mean_swap_rejection = np.array(rejection_rates.mean(axis=0))  # (NUM_CHAINS - 1,)

    # --- Diagnostics ---
    total_log_density_evals = NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)

    posterior_dict = {PARAM_NAMES[i]: cold_samples[None, :, i] for i in range(ndim)}
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    if not save_outputs:
        _az_log.setLevel(logging.ERROR)
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={"swap_rejection_rate": np.mean(mean_swap_rejection)},
    )
    summary = az.summary(idata)
    _az_log.setLevel(_az_prev)

    total_bulk_ess = float(summary["ess_bulk"].sum())
    ess_per_logp_eval = total_bulk_ess / total_log_density_evals

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = cold_samples.mean(axis=0)
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  Total log-density evaluations: {int(total_log_density_evals)}")
    _print()
    _print("  Mean per-pair swap rejection rates:")
    for i, r in enumerate(mean_swap_rejection):
        _print(f"    chain {i} <-> {i+1}  (beta {betas[i]:.4f} <-> {betas[i+1]:.4f}): {r:.3f}")
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
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "DEO_ParallelTempering",
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "ndim": ndim,
        "step_size_hot": float(STEP_SIZE_HOT),
        "step_size_cold": float(STEP_SIZE_COLD),
        "annealing_base": float(ANNEALING_BASE),
        "ref_scale_fraction": float(REF_SCALE_FRACTION),
        "beta_schedule": np.array(betas).tolist(),
        "step_sizes": np.array(step_sizes).tolist(),
        "mean_swap_rejection_rates": mean_swap_rejection.tolist(),
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
        DEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(DEO_OUTPUT_DIR / "idata.nc"))
        diag_path = DEO_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {DEO_OUTPUT_DIR / 'idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # Trace plots for first 6 parameters
    axes = az.plot_trace(idata, var_names=PARAM_NAMES[:6], figsize=(14, 10))
    plt.tight_layout()
    trace_path = DEO_OUTPUT_DIR / "traces_subset.png"
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
    corner_path = DEO_OUTPUT_DIR / "corner_all.png"
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
    lc_path = DEO_OUTPUT_DIR / "bestfit_lightcurve.png"
    fig.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved best-fit light curve to {lc_path}")

    # Per-pair swap rejection rates
    pair_labels = [f"{betas[i]:.3f}↔{betas[i+1]:.3f}" for i in range(NUM_CHAINS - 1)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(NUM_CHAINS - 1), mean_swap_rejection, color="steelblue", alpha=0.8)
    ax.set_xticks(range(NUM_CHAINS - 1))
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean rejection rate")
    ax.set_xlabel("Adjacent chain pair (β values)")
    ax.set_title("DEO swap rejection rates per adjacent pair")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    swap_path = DEO_OUTPUT_DIR / "swap_rates.png"
    fig.savefig(swap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved swap rates plot to {swap_path}")

    return diagnostics


if __name__ == "__main__":
    main()
