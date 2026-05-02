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
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pt_jax

from model import (
    make_inference_fns,
    make_log_ref,
    make_constrain_fn,
    sample_initial_positions,
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

DEO_OUTPUT_DIR = OUTPUT_DIR / "deo"

# ===========================================================================
# Tunable parameters
# ===========================================================================

# Number of parallel chains (including the reference/hottest chain).
# More chains → smoother temperature ladder, better mode mixing, higher cost.
NUM_CHAINS = 15

# Number of warm-up (burn-in) steps discarded before collecting samples.
NUM_WARMUP = 500

# Number of posterior samples collected from the cold chain.
NUM_SAMPLES = 2000

# Local exploration kernel: "mala" (gradient-based) or "rwmh" (gradient-free).
KERNEL = "rwmh"

# Step sizes for the local kernel. Per-chain values are linearly interpolated
# from STEP_SIZE_HOT (β≈0) down to STEP_SIZE_COLD (β=1).
# RWMH: tune cold for ~23% acceptance. MALA: tune cold for ~57% acceptance.
# RWMH and MALA have very different optimal scales, so they're set separately.
#
# RWMH proposal is per-dimension preconditioned: sigma_i = α · (2.38/√d) · prior_std_i,
# following the Roberts-Gelman-Gilks (1997) optimal scaling rule. The values below
# are the global multiplier α (dimensionless). α=1 starts at the rule's prediction;
# tune cold to ~23% acceptance, hot larger to broaden hot-chain proposals.
STEP_SIZE_HOT_RWMH = 1.0
STEP_SIZE_COLD_RWMH = 0.1
# TODO: MALA step sizes likely need further tuning. With the current values the
# cold chain freezes near the mode (proposal scale too large for local geometry,
# and chains can land outside bounded prior support and get pinned at log_p=-inf).
# Likely need much smaller cold step size and/or per-dimension scaling and/or
# reparameterisation of bounded parameters before MALA is usable here.
STEP_SIZE_HOT_MALA = 0.1
STEP_SIZE_COLD_MALA = 0.01

# Base for the geometric temperature ladder: β_k = base^{-(N-1-k)}, β_{N-1}=1.
# Larger base → wider spacing between adjacent temperatures (higher swap rejection).
# Smaller base → more chains needed to span the same range.
ANNEALING_BASE = 1.4142135623730951  # sqrt(2)

# Diagnostic stride — print intermediate results every DIAG_STRIDE dead points
DIAG_STRIDE = 50
PLOT_STRIDE = 200

_DIAG_PARAMS = [
    "spot_lat", "spot_long", "spot_size", "spot_flux",
    "fac_lat", "fac_long", "fac_size", "fac_flux",
    "p_rot", "planet_radius", "inclination", "P_orb",
]

# ===========================================================================


def make_rwmh_kernel_generator(proposal_scale):
    """Return a pt_jax kernel_generator that uses sigma = alpha · proposal_scale.

    proposal_scale is a length-ndim vector setting the per-dimension RWMH proposal
    sigma at alpha=1; the per-chain alpha is supplied by pt_jax via `params=`.
    """
    def rwmh_kernel_generator(log_p, alpha):
        sigma = alpha * proposal_scale
        rmh = blackjax.rmh(
            log_p,
            proposal_generator=blackjax.mcmc.random_walk.normal(sigma=sigma),
        )

        def kernel(key, position):
            state = rmh.init(position)
            new_state, _ = rmh.step(key, state)
            return new_state.position

        return kernel

    return rwmh_kernel_generator


def mala_kernel_generator(log_p, step_size):
    mala = blackjax.mala(log_p, step_size)

    def kernel(key, position):
        state = mala.init(position)
        new_state, _ = mala.step(key, state)
        return new_state.position

    return kernel


def unc_to_constrained(sample_unc, constrain_fn, unravel_fn):
    """
    Convert a single unconstrained flat array to a constrained dict
    with all derived quantities.
    """
    z_dict = unravel_fn(jnp.array(sample_unc))
    c_raw = constrain_fn(z_dict)
    c = {k: float(v) for k, v in c_raw.items()}
    c["eccentricity"] = c["ecc_h"] ** 2 + c["ecc_k"] ** 2
    c["arg_periapsis"] = np.arctan2(c["ecc_k"], c["ecc_h"])
    c["ldc_u1"] = 2 * np.sqrt(c["ldc_q1"]) * c["ldc_q2"]
    c["ldc_u2"] = np.sqrt(c["ldc_q1"]) * (1 - 2 * c["ldc_q2"])
    return c


def unc_array_to_constrained(samples_unc, constrain_fn, unravel_fn):
    """
    Convert an array of unconstrained samples (n_samples, ndim)
    to a constrained dict with arrays, including derived quantities.
    """
    c_all = jax.vmap(lambda x: constrain_fn(unravel_fn(x)))(jnp.array(samples_unc))
    c = {k: np.array(v) for k, v in c_all.items()}
    c["eccentricity"] = np.array(c["ecc_h"]) ** 2 + np.array(c["ecc_k"]) ** 2
    c["arg_periapsis"] = np.arctan2(np.array(c["ecc_k"]), np.array(c["ecc_h"]))
    c["ldc_u1"] = 2 * np.sqrt(np.array(c["ldc_q1"])) * np.array(c["ldc_q2"])
    c["ldc_u2"] = np.sqrt(np.array(c["ldc_q1"])) * (1 - 2 * np.array(c["ldc_q2"]))
    return c


def run_deo_diagnostics(cold_samples_unc, constrain_fn, unravel_fn,
                        save_lcs=False, output_dir=None):
    """
    Iterate through the cold chain samples (unconstrained) and print a
    per-step table of constrained parameters and reduced chi-squared.
    """
    n_steps, _ = cold_samples_unc.shape

    print(f"\n=== DEO Step-by-Step Diagnostics ===")
    print(f"(Analyzing {n_steps} cold-chain samples, stride={DIAG_STRIDE})\n")

    col_w = 13
    header = f"{'step':>5}  {'chi2_red':>9}  " + "  ".join(f"{p:>{col_w}}" for p in _DIAG_PARAMS)
    sep = "=" * len(header)
    print(header)
    print(sep)

    if save_lcs and output_dir is not None:
        lc_dir = output_dir / "step_lcs"
        lc_dir.mkdir(parents=True, exist_ok=True)
    else:
        lc_dir = None

    for step_idx in range(0, n_steps, DIAG_STRIDE):
        c = unc_to_constrained(cold_samples_unc[step_idx], constrain_fn, unravel_fn)

        chi2 = compute_chi2(c)
        param_str = "  ".join(f"{float(c[p]):>{col_w}.5f}" for p in _DIAG_PARAMS)
        print(f"{step_idx:>5}  {chi2:>9.4f}  {param_str}")

        if lc_dir is not None and step_idx % PLOT_STRIDE == 0:
            lc_model = np.array(compute_lc_from_constrained(c))
            fig, (ax_lc, ax_res) = plt.subplots(
                2, 1, figsize=(10, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=3, color="orange", alpha=0.5, label="Obs")
            ax_lc.plot(TIMES, LC_TRUE, lw=1.5, color="steelblue", label="True")
            ax_lc.plot(TIMES, lc_model, lw=1.5, color="crimson", ls="--",
                       label=f"Step {step_idx}  χ²_r={chi2:.3f}")
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
            fig.savefig(lc_dir / f"lc_step_{step_idx:05d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    if lc_dir is not None:
        print(f"\nLC snapshots saved to {lc_dir}/")


def main(seed: int = 0, save_outputs: bool = True):
    init_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))
    _print = print if save_outputs else lambda *a, **kw: None

    ndim = len(PARAM_NAMES)

    # --- Model in unconstrained space ---
    log_density_fn_dict, _, init_z = make_inference_fns(init_key)
    flat_init, unravel_fn = jax.flatten_util.ravel_pytree(init_z)

    log_density_fn = lambda x: log_density_fn_dict(unravel_fn(x))

    log_ref_dict = make_log_ref(init_key)
    log_ref = lambda x: log_ref_dict(unravel_fn(x))

    constrain_fn = make_constrain_fn()

    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # --- Temperature schedule ---
    betas = pt_jax.annealing.annealing_exponential(NUM_CHAINS, base=ANNEALING_BASE)

    # --- Build kernels ---
    if KERNEL == "mala":
        step_size_hot, step_size_cold = STEP_SIZE_HOT_MALA, STEP_SIZE_COLD_MALA
        kernel_generator = mala_kernel_generator
        proposal_scale = None
    elif KERNEL == "rwmh":
        step_size_hot, step_size_cold = STEP_SIZE_HOT_RWMH, STEP_SIZE_COLD_RWMH
        # Unit scale in unconstrained space
        proposal_scale = jnp.ones(ndim)
        kernel_generator = make_rwmh_kernel_generator(proposal_scale)
    else:
        raise ValueError(f"Unknown KERNEL {KERNEL!r}; expected 'mala' or 'rwmh'.")
    step_sizes = jnp.linspace(step_size_hot, step_size_cold, NUM_CHAINS)

    K_ind = pt_jax.kernels.generate_independent_annealed_kernel(
        log_prob=log_density_fn,
        log_ref=log_ref,
        annealing_schedule=betas,
        kernel_generator=kernel_generator,
        params=step_sizes,
    )
    K_deo = pt_jax.swap.generate_deo_extended_kernel(
        log_prob=log_density_fn,
        log_ref=log_ref,
        annealing_schedule=betas,
    )

    # --- Initialise chains from prior (unconstrained space) ---
    x0 = sample_initial_positions(init_key, NUM_CHAINS, return_flat=True)

    # --- Diagnostic: initial log-density per chain (cold chain is index -1) ---
    init_log_targets = np.array([float(log_density_fn(x0[i])) for i in range(NUM_CHAINS)])
    _print("Initial log target density per chain (β increases left→right):")
    for i, lp in enumerate(init_log_targets):
        _print(f"    chain {i:2d}  beta={float(betas[i]):.4f}  log_target(x0)={lp:+.4e}")
    _print(f"  Cold-chain initial log_target: {init_log_targets[-1]:+.4e}")

    # --- Diagnostic: acceptance test at truth ---
    from numpyro.distributions import biject_to
    inv_transforms = {name: biject_to(d.support).inv
                      for name, d in PRIOR_DISTRIBUTIONS.items()}
    gt_unc_dict = {name: inv_transforms[name](jnp.array(GROUND_TRUTH[name]))
                   for name in PARAM_NAMES}
    gt_unc_flat = jax.flatten_util.ravel_pytree(gt_unc_dict)[0]
    # Reorder to match unravel_fn ordering
    gt_unc_flat = jax.flatten_util.ravel_pytree(
        {k: inv_transforms[k](jnp.array(GROUND_TRUTH[k]))
         for k in unravel_fn(flat_init).keys()}
    )[0]
    log_p_truth = float(log_density_fn(gt_unc_flat))
    _print(f"\n  log_density at ground truth (unconstrained): {log_p_truth:.2f}")

    test_key = jax.random.PRNGKey(99)
    n_test = 200
    n_in_bounds = 0
    n_accept = 0
    for i in range(n_test):
        test_key, prop_key = jax.random.split(test_key)
        noise = step_size_cold * jax.random.normal(prop_key, shape=(ndim,))
        proposal = gt_unc_flat + noise
        log_p_prop = float(log_density_fn(proposal))
        if np.isfinite(log_p_prop):
            n_in_bounds += 1
            if np.log(np.random.random() + 1e-300) < (log_p_prop - log_p_truth):
                n_accept += 1
    _print(f"  Test proposals finite: {n_in_bounds}/{n_test}")
    _print(f"  Test proposals accepted: {n_accept}/{n_test}")
    _print(f"  → Estimated acceptance rate: {n_accept / n_test:.1%}")

    # --- Run DEO sampling loop ---
    _print(f"\nRunning DEO ({KERNEL.upper()} local kernel, {NUM_CHAINS} chains, "
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
    cold_samples_unc = np.array(samples[:, -1, :])  # (NUM_SAMPLES, NDIM)
    mean_swap_rejection = np.array(rejection_rates.mean(axis=0))

    # --- Step-by-step diagnostics ---
    if save_outputs:
        run_deo_diagnostics(cold_samples_unc, constrain_fn, unravel_fn,
                            save_lcs=True, output_dir=DEO_OUTPUT_DIR)

    # --- Cold chain movement check ---
    _print("\nCold-chain movement check (first vs last sample, constrained):")
    c_first = unc_to_constrained(cold_samples_unc[0], constrain_fn, unravel_fn)
    c_last = unc_to_constrained(cold_samples_unc[-1], constrain_fn, unravel_fn)
    _print(f"  {'param':20s}  {'first':>12s}  {'last':>12s}  {'|Δ|':>12s}")
    for name in PARAM_NAMES:
        first = c_first[name]
        last = c_last[name]
        _print(f"  {name:20s}  {first:12.6f}  {last:12.6f}  {abs(last - first):12.6f}")
    if np.allclose(cold_samples_unc[0], cold_samples_unc[-1]):
        _print("  WARNING: cold chain first and last samples are identical — chain never moved.")

    # --- Convert all cold samples to constrained space ---
    constrained_all = unc_array_to_constrained(cold_samples_unc, constrain_fn, unravel_fn)

    # --- Diagnostics ---
    total_local_evals = NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)
    cost_label = "gradient_evals" if KERNEL == "mala" else "log_density_evals"

    posterior_dict = {name: constrained_all[name][None, :] for name in PARAM_NAMES}
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
    ess_per_local_eval = total_bulk_ess / total_local_evals

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = np.array([constrained_all[p].mean() for p in PARAM_NAMES])
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  Kernel:        {KERNEL.upper()}")
    _print(f"  Total {cost_label}: {int(total_local_evals)}")
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
    _print(f"  Bulk ESS per {cost_label.replace('_', '-')}: {ess_per_local_eval:.4f}")
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "DEO_ParallelTempering",
        "kernel": KERNEL,
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "ndim": ndim,
        "step_size_hot": float(step_size_hot),
        "step_size_cold": float(step_size_cold),
        "annealing_base": float(ANNEALING_BASE),
        "reference_distribution": "joint_prior_unconstrained",
        "beta_schedule": np.array(betas).tolist(),
        "step_sizes": np.array(step_sizes).tolist(),
        "mean_swap_rejection_rates": mean_swap_rejection.tolist(),
        cost_label: int(total_local_evals),
        "wall_time_s": float(wall_time_s),
        "total_bulk_ess": float(total_bulk_ess),
        f"bulk_ess_per_{cost_label}": float(ess_per_local_eval),
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
    az.plot_trace(idata, var_names=PARAM_NAMES[:6], figsize=(14, 10))
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

    # Best-fit light curve
    plot_bestfit_lightcurve(constrained_all, DEO_OUTPUT_DIR)

    return diagnostics


if __name__ == "__main__":
    main()
