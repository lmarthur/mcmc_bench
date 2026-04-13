"""
Run SEO (reversible) parallel tempering on the 2D Gaussian mixture model.

SEO uses a stochastic even-odd parity schedule for swap moves, which is
reversible (unlike DEO). At each step the parity (even or odd adjacent pairs)
is chosen randomly with equal probability.
"""

import json
import sys
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

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS, DEFAULT_WEIGHTS

SEO_OUTPUT_DIR = OUTPUT_DIR / "seo"

# Local exploration kernel: "mala" (gradient-based) or "rwmh" (gradient-free).
KERNEL = "mala"

NUM_CHAINS = 10
NUM_SAMPLES = 5000
NUM_WARMUP = 1000
# Per-chain step sizes: larger for hot chains (broad exploration), smaller for cold.
STEP_SIZE_HOT = 0.5
STEP_SIZE_COLD = 0.05


# ---------------------------------------------------------------------------
# Reference distribution: isotropic Gaussian broad enough to cover the mixture
# (components sit on a circle of radius 7).
# ---------------------------------------------------------------------------
_REF_SCALE = 10.0


def log_ref(x):
    return dist.Normal(jnp.zeros(2), _REF_SCALE).to_event().log_prob(x)


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

def main():
    kernel_type = KERNEL

    sample_key = jax.random.PRNGKey(0)

    # --- Model ---
    log_density_fn = make_log_density()
    print("Plotting model...")
    plot_model()

    # --- Temperature schedule ---
    betas = pt_jax.annealing.annealing_exponential(NUM_CHAINS)
    print(f"Temperature schedule (betas): {np.array(betas).round(4).tolist()}")

    # --- Build kernels ---
    step_sizes = jnp.linspace(STEP_SIZE_HOT, STEP_SIZE_COLD, NUM_CHAINS)
    kernel_generator = mala_kernel_generator if kernel_type == "mala" else rwmh_kernel_generator

    K_ind = pt_jax.kernels.generate_independent_annealed_kernel(
        log_prob=log_density_fn,
        log_ref=log_ref,
        annealing_schedule=betas,
        kernel_generator=kernel_generator,
        params=step_sizes,
    )
    K_seo = pt_jax.swap.generate_seo_extended_kernel(
        log_prob=log_density_fn,
        log_ref=log_ref,
        annealing_schedule=betas,
    )

    # --- Initialize: all chains start at origin ---
    x0 = jnp.zeros((NUM_CHAINS, 2))

    # --- Run SEO sampling loop ---
    print(f"Running SEO ({kernel_type.upper()} local kernel, {NUM_CHAINS} chains, "
          f"{NUM_SAMPLES} samples, {NUM_WARMUP} warmup)...")
    samples, rejection_rates = pt_jax.swap.seo_sampling_loop(
        key=sample_key,
        x0=x0,
        kernel_local=K_ind,
        kernel_seo=K_seo,
        n_samples=NUM_SAMPLES,
        warmup=NUM_WARMUP,
    )
    # samples:         (NUM_SAMPLES, NUM_CHAINS, 2)
    # rejection_rates: (NUM_SAMPLES, NUM_CHAINS - 1)

    mean_swap_rejection = np.array(rejection_rates.mean(axis=0))  # (NUM_CHAINS - 1,)

    # Cold chain (beta=1, target distribution) is the last chain.
    cold_samples = np.array(samples[:, -1, :])  # (NUM_SAMPLES, 2)

    # --- Diagnostics ---
    means_np = np.array(DEFAULT_MEANS)
    num_modes = len(means_np)

    # Mode assignment for the cold chain
    cold_expanded = cold_samples[:, None, :]
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
        total_cost = NUM_CHAINS * NUM_SAMPLES
        cost_label = "gradient_evals"
    else:
        # RWMH: one log-density evaluation per chain per local step.
        total_cost = NUM_CHAINS * NUM_SAMPLES
        cost_label = "log_density_evals"

    # ArviZ summary on cold chain (treated as a single chain)
    idata = az.from_dict(
        posterior={"x1": cold_samples[None, :, 0], "x2": cold_samples[None, :, 1]},
    )
    summary = az.summary(idata, var_names=["x1", "x2"])
    total_bulk_ess = float(summary["ess_bulk"].sum())
    ess_per_cost = total_bulk_ess / total_cost

    print("\n=== Diagnostics ===")
    print(f"  Kernel:       {kernel_type.upper()}")
    print(f"  Total {cost_label}: {int(total_cost)}")
    print()
    true_weights = np.array(DEFAULT_WEIGHTS)
    print("  Mode weight recovery (empirical vs true):")
    for k, (w, tw) in enumerate(zip(mode_weights, true_weights)):
        print(f"    Mode {k}: {w:.3f}  (true: {tw:.3f})")
    print()
    print(f"  Inter-mode transitions (cold chain): {transitions}")
    if stuck:
        print("  WARNING: cold chain never left one mode.")
    else:
        print("  Cold chain not stuck.")
    print()
    print("  Mean per-pair swap rejection rates:")
    for i, r in enumerate(mean_swap_rejection):
        print(f"    chain {i} <-> {i+1}  (beta {betas[i]:.4f} <-> {betas[i+1]:.4f}): {r:.3f}")
    print()
    print("  ArviZ summary (R-hat, ESS, MCSE):")
    print(summary.to_string())
    print()
    print(f"  Bulk ESS per {cost_label.replace('_', '-')}: {ess_per_cost:.4f}")

    # --- Save results ---
    SEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idata.to_netcdf(str(SEO_OUTPUT_DIR / "idata.nc"))
    print(f"\nSaved InferenceData to {SEO_OUTPUT_DIR / 'idata.nc'}")

    diagnostics = {
        "sampler": "SEO_ParallelTempering",
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
    diag_path = SEO_OUTPUT_DIR / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics to {diag_path}")

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
    axes[0].set_title(f"SEO ({kernel_type.upper()}) — all {NUM_CHAINS} chains")
    axes[0].set_aspect("equal")
    axes[0].legend(markerscale=4, fontsize=7)

    for i, label in enumerate(["x1", "x2"]):
        axes[i + 1].plot(cold_samples[:, i], lw=0.4, alpha=0.8, color=chain_colors[-1])
        axes[i + 1].set_xlabel("iteration")
        axes[i + 1].set_ylabel(label)
        axes[i + 1].set_title(f"Cold chain trace: {label} (β=1)")

    fig.tight_layout()
    samples_path = SEO_OUTPUT_DIR / "samples.png"
    fig.savefig(samples_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved samples plot to {samples_path}")

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
    corner_path = SEO_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    print(f"Saved corner plot to {corner_path}")

    # 3. Per-pair swap rejection rates
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
    print(f"Saved swap rates plot to {swap_path}")


if __name__ == "__main__":
    main()
