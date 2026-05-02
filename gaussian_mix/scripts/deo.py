"""
Run DEO (non-reversible) parallel tempering on the 2D Gaussian mixture model.

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
import pt_jax

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS, DEFAULT_WEIGHTS, PRIOR_LOW, PRIOR_HIGH

DEO_OUTPUT_DIR = OUTPUT_DIR / "deo"

# Local exploration kernel: "mala" (gradient-based) or "rwmh" (gradient-free).
KERNEL = "rwmh"

NUM_CHAINS = 8
NUM_SAMPLES = 2500
NUM_WARMUP = 250


# ---------------------------------------------------------------------------
# Reference distribution: uniform prior over [PRIOR_LOW, PRIOR_HIGH]^2.
# ---------------------------------------------------------------------------

def log_ref(x):
    return jax.scipy.stats.uniform.logpdf(x, loc=PRIOR_LOW, scale=PRIOR_HIGH - PRIOR_LOW).sum()


# ---------------------------------------------------------------------------
# Local kernel generators (pt-jax interface: (log_p, param) -> (key, x) -> x)
# ---------------------------------------------------------------------------

def mala_kernel_generator(log_p, step_size):
    mala = blackjax.mala(log_p, step_size)

    def kernel(key, position):
        state = mala.init(position)
        new_state, _ = mala.step(key, state)
        return new_state.position

    return kernel


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=0, save_outputs=True):
    kernel_type = KERNEL

    init_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model()

    # --- Temperature schedule ---
    betas = pt_jax.annealing.annealing_exponential(NUM_CHAINS)

    # --- Build kernels ---
    sigma_prior = (PRIOR_HIGH - PRIOR_LOW) / jnp.sqrt(12)
    step_sizes = (2.38 / jnp.sqrt(2)) * sigma_prior / jnp.sqrt(betas)
    kernel_generator = mala_kernel_generator if kernel_type == "mala" else rwmh_kernel_generator

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

    x0 = jax.random.uniform(init_key, shape=(NUM_CHAINS, 2), minval=PRIOR_LOW, maxval=PRIOR_HIGH)

    # --- Run DEO sampling loop ---
    _print(f"Running DEO ({kernel_type.upper()} local kernel, {NUM_CHAINS} chains, "
          f"{NUM_SAMPLES} samples, {NUM_WARMUP} warmup)...")
    t0 = time.perf_counter()
    samples, rejection_rates = pt_jax.swap.deo_sampling_loop(
        key=sample_key,
        x0=x0,
        kernel_local=K_ind,
        kernel_deo=K_deo,
        n_samples=NUM_SAMPLES,
        warmup=NUM_WARMUP,
    )
    # samples:         (NUM_SAMPLES, NUM_CHAINS, 2)
    # rejection_rates: (NUM_SAMPLES, NUM_CHAINS - 1)
    wall_time_s = time.perf_counter() - t0

    mean_swap_rejection = np.array(rejection_rates.mean(axis=0))  # (NUM_CHAINS - 1,)

    # Cold chain (beta=1, target distribution) is the last chain.
    cold_samples = np.array(samples[:, -1, :])  # (NUM_SAMPLES, 2)

    # --- Diagnostics ---
    means_np = np.array(DEFAULT_MEANS)
    num_modes = len(means_np)

    # Mode assignment for the cold chain
    cold_expanded = cold_samples[:, None, :]         # (NUM_SAMPLES, 1, 2)
    dists_to_modes = np.linalg.norm(cold_expanded - means_np[None, :, :], axis=-1)
    assignments = np.argmin(dists_to_modes, axis=-1)  # (NUM_SAMPLES,)

    mode_weights = np.bincount(assignments, minlength=num_modes) / len(assignments)

    # Inter-mode transitions on the cold chain
    transitions = int(np.sum(np.diff(assignments) != 0))

    # Stuck detection (treat cold chain as a single chain)
    stuck = np.unique(assignments).size == 1

    # Cost metric
    if kernel_type == "mala":
        # MALA: one gradient evaluation per chain per local step.
        # Swap moves require only log-density evaluations, not gradients.
        total_cost = NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)
        cost_label = "gradient_evals"
    else:
        # RWMH: one log-density evaluation per chain per local step.
        total_cost = NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)
        cost_label = "log_density_evals"

    # ArviZ summary on cold chain (treated as a single chain)
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    if not save_outputs:
        _az_log.setLevel(logging.ERROR)
    idata = az.from_dict(
        posterior={"x1": cold_samples[None, :, 0], "x2": cold_samples[None, :, 1]},
    )
    summary = az.summary(idata, var_names=["x1", "x2"])
    _az_log.setLevel(_az_prev)
    total_bulk_ess = float(summary["ess_bulk"].sum())
    ess_per_cost = total_bulk_ess / total_cost

    _print("\n=== Diagnostics ===")
    _print(f"  Kernel:       {kernel_type.upper()}")
    _print(f"  Total {cost_label}: {int(total_cost)}")
    _print()
    true_weights = np.array(DEFAULT_WEIGHTS)
    _print("  Mode weight recovery (empirical vs true):")
    for k, (w, tw) in enumerate(zip(mode_weights, true_weights)):
        _print(f"    Mode {k}: {w:.3f}  (true: {tw:.3f})")
    _print()
    _print(f"  Inter-mode transitions (cold chain): {transitions}")
    if stuck:
        _print("  WARNING: cold chain never left one mode.")
    else:
        _print("  Cold chain not stuck.")
    _print()
    _print("  Mean per-pair swap rejection rates:")
    for i, r in enumerate(mean_swap_rejection):
        _print(f"    chain {i} <-> {i+1}  (beta {betas[i]:.4f} <-> {betas[i+1]:.4f}): {r:.3f}")
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print()
    _print(f"  Bulk ESS per {cost_label.replace('_', '-')}: {ess_per_cost:.4f}")
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "DEO_ParallelTempering",
        "wall_time_s": wall_time_s,
        "kernel_type": kernel_type,
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "beta_schedule": np.array(betas).tolist(),
        "step_sizes": np.array(step_sizes).tolist(),
        "mean_swap_rejection_rates": mean_swap_rejection.tolist(),
        cost_label: int(total_cost),
        f"bulk_ess_per_{cost_label}": float(ess_per_cost),
        "mode_weights": mode_weights.tolist(),
        "true_mode_weights": np.array(DEFAULT_WEIGHTS).tolist(),
        "inter_mode_transitions_cold_chain": transitions,
        "cold_chain_stuck": bool(stuck),
        "arviz_summary": json.loads(summary.to_json()),
    }
    if save_outputs:
        DEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(DEO_OUTPUT_DIR / "idata.nc"))
        diag_path = DEO_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # 1. Scatter (all chains, colored by temperature) + cold chain traces
    all_samples_np = np.array(samples)  # (NUM_SAMPLES, NUM_CHAINS, 2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    cmap = plt.cm.coolwarm
    chain_colors = cmap(np.linspace(0, 1, NUM_CHAINS))
    for c in range(NUM_CHAINS):
        label = f"β={betas[c]:.2f}" if c in (0, NUM_CHAINS - 1) else None
        axes[0].scatter(
            all_samples_np[::5, c, 0], all_samples_np[::5, c, 1],
            alpha=0.1, s=2, color=chain_colors[c], label=label,
        )
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title(f"DEO ({kernel_type.upper()}) — all {NUM_CHAINS} chains")
    axes[0].set_aspect("equal")
    axes[0].legend(markerscale=4, fontsize=7)

    for i, label in enumerate(["x1", "x2"]):
        axes[i + 1].plot(cold_samples[:, i], lw=0.4, alpha=0.8, color=chain_colors[-1])
        axes[i + 1].set_xlabel("iteration")
        axes[i + 1].set_ylabel(label)
        axes[i + 1].set_title(f"Cold chain trace: {label} (β=1)")

    fig.tight_layout()
    samples_path = DEO_OUTPUT_DIR / "samples.png"
    fig.savefig(samples_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved samples plot to {samples_path}")

    # 2. Corner plot (cold chain)
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
    corner_path = DEO_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    _print(f"Saved corner plot to {corner_path}")

    # 3. Per-pair swap rejection rates
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
