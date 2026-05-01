"""
Run SEO (reversible) parallel tempering on the SAJAX planet+activity model and save outputs.

SEO uses a stochastic even-odd parity schedule for swap moves, which is
reversible (unlike DEO). At each step the parity (even or odd adjacent pairs)
is chosen randomly with equal probability.
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
    make_constrain_fn,
    make_log_ref,
    plot_model,
    sample_initial_positions,
    plot_bestfit_lightcurve,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    PRIOR_DISTRIBUTIONS,
)

SEO_OUTPUT_DIR = OUTPUT_DIR / "seo"

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
STEP_SIZE_HOT_RWMH = 5.0
STEP_SIZE_COLD_RWMH = 1.0
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

# ===========================================================================


def _unconstrained_prior_std(d) -> float:
    """Return the std of prior d in unconstrained (bijected) space analytically.

    Uniform(a, b): inverse bijection is logit → logistic distribution, std = π/√3.
    Normal(μ, σ) / LogNormal(μ, σ): bijection is identity / log, unconstrained std = σ.
    """
    from numpyro.distributions import Uniform, Normal, LogNormal
    if isinstance(d, Uniform):
        return float(np.pi / np.sqrt(3.0))
    elif isinstance(d, (Normal, LogNormal)):
        return float(d.scale)
    raise TypeError(f"No analytical unconstrained std for {type(d).__name__}")


def make_rwmh_kernel_generator(proposal_scale):
    """Return a pt_jax kernel_generator that uses sigma = alpha · proposal_scale.

    proposal_scale is a length-ndim vector setting the per-dimension RWMH proposal
    σ at α=1; the per-chain α is supplied by pt_jax via `params=`.
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


def main(seed: int = 0, save_outputs: bool = True):
    init_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))
    _print = print if save_outputs else lambda *a, **kw: None

    ndim = len(PARAM_NAMES)

    # --- Model ---
    log_density_fn, _, init_z = make_inference_fns(init_key)
    _, unravel_fn = jax.flatten_util.ravel_pytree(init_z)
    log_density_flat = lambda x: log_density_fn(unravel_fn(x))
    constrain_fn = make_constrain_fn()

    log_ref_dict = make_log_ref(init_key)
    log_ref_flat = lambda x: log_ref_dict(unravel_fn(x))

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
        # Per-dimension RWMH proposal scale in unconstrained space: (2.38/√d) · prior_std_i.
        # The per-chain α multiplies this vector to form the proposal sigma.
        prior_stds = jnp.array([
            _unconstrained_prior_std(PRIOR_DISTRIBUTIONS[name])
            for name in PARAM_NAMES
        ])
        proposal_scale = (2.38 / jnp.sqrt(ndim)) * prior_stds
        kernel_generator = make_rwmh_kernel_generator(proposal_scale)
    else:
        raise ValueError(f"Unknown KERNEL {KERNEL!r}; expected 'mala' or 'rwmh'.")
    step_sizes = jnp.linspace(step_size_hot, step_size_cold, NUM_CHAINS)

    K_ind = pt_jax.kernels.generate_independent_annealed_kernel(
        log_prob=log_density_flat,
        log_ref=log_ref_flat,
        annealing_schedule=betas,
        kernel_generator=kernel_generator,
        params=step_sizes,
    )
    K_seo = pt_jax.swap.generate_seo_extended_kernel(
        log_prob=log_density_flat,
        log_ref=log_ref_flat,
        annealing_schedule=betas,
    )

    # --- Initialise chains from prior in unconstrained space ---
    x0 = sample_initial_positions(init_key, NUM_CHAINS, return_flat=True)

    # --- Diagnostic: initial chain positions in constrained space ---
    x0_constrained = jax.vmap(lambda x: constrain_fn(unravel_fn(x)))(x0)
    _print("Initial chain positions (constrained space):")
    _print(f"  {'param':20s}  " + "  ".join(f"chain{i:02d}" for i in range(NUM_CHAINS)))
    for name in PARAM_NAMES:
        vals = np.array(x0_constrained[name])
        _print(f"  {name:20s}  " + "  ".join(f"{v:8.4f}" for v in vals))

    # --- Diagnostic: initial log-density per chain (cold chain is index -1) ---
    init_log_targets = np.array([float(log_density_flat(x0[i])) for i in range(NUM_CHAINS)])
    _print("Initial log target density per chain (β increases left→right):")
    for i, lp in enumerate(init_log_targets):
        _print(f"    chain {i:2d}  beta={float(betas[i]):.4f}  log_target(x0)={lp:+.4e}")
    _print(f"  Cold-chain initial log_target: {init_log_targets[-1]:+.4e}")

    # --- Run SEO sampling loop ---
    _print(f"Running SEO ({KERNEL.upper()} local kernel, {NUM_CHAINS} chains, "
           f"{NUM_SAMPLES} samples, {NUM_WARMUP} warmup, {ndim} params)...")

    samples, rejection_rates = pt_jax.swap.seo_sampling_loop(
        key=sample_key,
        x0=x0,
        kernel_local=K_ind,
        kernel_seo=K_seo,
        n_samples=NUM_SAMPLES,
        warmup=NUM_WARMUP,
    )
    # samples:         (NUM_SAMPLES, NUM_CHAINS, NDIM)
    # rejection_rates: (NUM_SAMPLES, NUM_CHAINS - 1)

    wall_time_s = time.perf_counter() - t0

    # Cold chain (β=1, target distribution) is the last chain.
    cold_samples_flat = np.array(samples[:, -1, :])  # (NUM_SAMPLES, NDIM)
    mean_swap_rejection = np.array(rejection_rates.mean(axis=0))  # (NUM_CHAINS - 1,)

    # Convert unconstrained cold-chain samples to constrained space
    constrained_samples = jax.vmap(lambda x: constrain_fn(unravel_fn(x)))(
        jnp.array(cold_samples_flat)
    )

    # --- Diagnostic: did the cold chain move? Compare first vs last sample ---
    _print("\nCold-chain movement check (first vs last sample):")
    _print(f"  {'param':20s}  {'first':>12s}  {'last':>12s}  {'|Δ|':>12s}")
    for name in PARAM_NAMES:
        first = float(np.array(constrained_samples[name])[0])
        last = float(np.array(constrained_samples[name])[-1])
        _print(f"  {name:20s}  {first:12.6f}  {last:12.6f}  {abs(last - first):12.6f}")
    if np.allclose(cold_samples_flat[0], cold_samples_flat[-1]):
        _print("  WARNING: cold chain first and last samples are identical — chain never moved.")

    # --- Diagnostics ---
    # MALA: one gradient eval per chain per local step. RWMH: one log-density eval.
    # Swap moves cost only log-density evals in either case.
    total_local_evals = NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)
    cost_label = "gradient_evals" if KERNEL == "mala" else "log_density_evals"

    posterior_dict = {name: np.array(constrained_samples[name])[None, :] for name in PARAM_NAMES}
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
    posterior_means = np.array([np.array(constrained_samples[p]).mean() for p in PARAM_NAMES])
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
        "sampler": "SEO_ParallelTempering",
        "kernel": KERNEL,
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "ndim": ndim,
        "step_size_hot": float(step_size_hot),
        "step_size_cold": float(step_size_cold),
        "annealing_base": float(ANNEALING_BASE),
        "reference_distribution": "joint_prior",
        "beta_schedule": np.array(betas).tolist(),
        "step_sizes": np.array(step_sizes).tolist(),
        "rwmh_proposal_scale": (
            np.array(proposal_scale).tolist() if proposal_scale is not None else None
        ),
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
        SEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(SEO_OUTPUT_DIR / "idata.nc"))
        diag_path = SEO_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {SEO_OUTPUT_DIR / 'idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # Trace plots for first 6 parameters
    axes = az.plot_trace(idata, var_names=PARAM_NAMES[:6], figsize=(14, 10))
    plt.tight_layout()
    trace_path = SEO_OUTPUT_DIR / "traces_subset.png"
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
    corner_path = SEO_OUTPUT_DIR / "corner_all.png"
    plt.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close()
    _print(f"Saved full corner plot to {corner_path}")

    # Best-fit light curve using posterior mean
    plot_bestfit_lightcurve(constrained_samples, SEO_OUTPUT_DIR)

    # Per-pair swap rejection rates
    pair_labels = [f"{betas[i]:.3f}↔{betas[i+1]:.3f}" for i in range(NUM_CHAINS - 1)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(NUM_CHAINS - 1), mean_swap_rejection, color="steelblue", alpha=0.8)
    ax.set_xticks(range(NUM_CHAINS - 1))
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean rejection rate")
    ax.set_xlabel("Adjacent chain pair (β values)")
    ax.set_title("SEO swap rejection rates per adjacent pair")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    swap_path = SEO_OUTPUT_DIR / "swap_rates.png"
    fig.savefig(swap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved swap rates plot to {swap_path}")

    return diagnostics


if __name__ == "__main__":
    main()
