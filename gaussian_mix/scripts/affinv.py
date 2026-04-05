"""
Run Affine Invariant MCMC on the 2D Gaussian mixture model and save trace plots.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az
import emcee_jax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS, DEFAULT_WEIGHTS

AFFINV_OUTPUT_DIR = OUTPUT_DIR / "affinv"

NUM_BURNIN = 1000
NUM_SAMPLES = 5000
NUM_WALKERS = 5
NDIM = 2


def main():
    init_key, state_key, sample_key = jax.random.split(jax.random.PRNGKey(0), 3)

    # --- Model ---
    log_density_fn = make_log_density()
    print("Plotting model...")
    plot_model()

    # --- Initialize chains from random starting positions ---
    coords = jax.random.normal(init_key, shape=(NUM_WALKERS, NDIM))

    # --- Initialize sampler ---
    sampler = emcee_jax.EnsembleSampler(log_density_fn)
    state = sampler.init(state_key, coords)

    # --- Run chains in parallel ---
    print(f"Sampling ({NUM_SAMPLES} steps, {NUM_WALKERS} walkers)...")
    trace = sampler.sample_parallel(sample_key, state, NUM_SAMPLES)

    # --- Retrieve samples ---
    samples = np.asarray(trace.samples.coordinates).reshape(NUM_WALKERS, NUM_SAMPLES, NDIM)  # shape (walkers, nsteps, nparams)

    # --- Diagnostics ---
    means_np = np.array(DEFAULT_MEANS)
    num_modes = len(means_np)

    # Assign each sample to nearest mode: (NUM_WALKERS, NUM_SAMPLES)
    dists = np.linalg.norm(
        samples[:, :, None, :] - means_np[None, None, :, :], axis=-1
    )
    chain_assignments = np.argmin(dists, axis=-1)
    flat_assignments = chain_assignments.ravel()

    # Mode weights
    mode_weights = np.bincount(flat_assignments, minlength=num_modes) / flat_assignments.size

    # Inter-mode transitions per chain
    transitions = np.sum(np.diff(chain_assignments, axis=1) != 0, axis=1)

    # Stuck chains (never leave one mode)
    stuck_chains = [c for c in range(NUM_WALKERS) if np.unique(chain_assignments[c]).size == 1]

    # ESS per gradient evaluation (NUTS: num_integration_steps ~ grad evals per step)
    num_integration_steps = np.array(all_infos.num_integration_steps)

    # ArviZ summary: R-hat, bulk/tail ESS, MCSE
    idata = az.from_dict(
        posterior={"x1": samples[:, :, 0], "x2": samples[:, :, 1]},
        sample_stats={
            "acceptance_rate": np.array(trace.sample_stats.accept_prob),
            "n_steps": num_integration_steps,
        },
    )
    summary = az.summary(idata, var_names=["x1", "x2"])
    total_grad_evals = num_integration_steps.sum()
    total_bulk_ess = summary["ess_bulk"].sum()
    ess_per_grad = total_bulk_ess / total_grad_evals

    # Tree depth saturation (fraction of steps hitting the max observed tree depth)
    max_steps = num_integration_steps.max()
    saturation_frac = np.mean(num_integration_steps == max_steps)

    # Acceptance rate
    acceptance = np.mean(np.array(trace.sample_stats.accept_prob))

    print("\n=== Diagnostics ===")
    print(f"  Mean acceptance rate:       {acceptance:.3f}")
    print(f"  Total gradient evaluations: {int(total_grad_evals)}")
    print(f"  Mean tree depth (steps):    {num_integration_steps.mean():.1f}  (max observed: {int(max_steps)})")
    print(f"  Tree depth saturation:      {saturation_frac:.1%}")
    print()
    true_weights = np.array(DEFAULT_WEIGHTS)
    print(f"  Mode weight recovery (empirical vs true):")
    for k, (w, tw) in enumerate(zip(mode_weights, true_weights)):
        print(f"    Mode {k}: {w:.3f}  (true: {tw:.3f})")
    print()
    print(f"  Inter-mode transitions per chain: {transitions.tolist()}")
    if stuck_chains:
        print(f"  WARNING: stuck chains (never left one mode): {stuck_chains}")
    else:
        print(f"  No stuck chains detected.")
    print()
    print("  ArviZ summary (R-hat, ESS, MCSE):")
    print(summary.to_string())
    print()
    print(f"  Bulk ESS per gradient eval: {ess_per_grad:.4f}")

    # --- Save results ---
    AFFINV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idata.to_netcdf(str(AFFINV_OUTPUT_DIR / "idata.nc"))
    print(f"Saved InferenceData to {AFFINV_OUTPUT_DIR / 'idata.nc'}")

    diagnostics = {
        "sampler": "NUTS",
        "num_chains": NUM_WALKERS,
        "num_warmup": NUM_BURNIN,
        "num_samples": NUM_SAMPLES,
        "mean_acceptance_rate": float(acceptance),
        "total_grad_evals": int(total_grad_evals),
        "mean_integration_steps": float(num_integration_steps.mean()),
        "max_integration_steps": int(max_steps),
        "tree_depth_saturation": float(saturation_frac),
        "mode_weights": mode_weights.tolist(),
        "true_mode_weights": np.array(DEFAULT_WEIGHTS).tolist(),
        "inter_mode_transitions": transitions.tolist(),
        "stuck_chains": stuck_chains,
        "bulk_ess_per_grad_eval": float(ess_per_grad),
        "arviz_summary": json.loads(summary.to_json()),
    }
    diag_path = AFFINV_OUTPUT_DIR / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics to {diag_path}")

    # --- Plots ---

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_WALKERS))
    for c in range(NUM_WALKERS):
        axes[0].scatter(samples[c, :, 0], samples[c, :, 1], alpha=0.15, s=2, color=colors[c])
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title(f"NUTS samples ({NUM_WALKERS} chains)")
    axes[0].set_aspect("equal")

    for i, label in enumerate(["x1", "x2"]):
        for c in range(NUM_WALKERS):
            axes[i + 1].plot(samples[c, :, i], lw=0.4, alpha=0.6, color=colors[c])
        axes[i + 1].set_xlabel("iteration")
        axes[i + 1].set_ylabel(label)
        axes[i + 1].set_title(f"Trace: {label}")

    fig.tight_layout()
    out_path = AFFINV_OUTPUT_DIR / "samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved samples plot to {out_path}")

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
    corner_path = AFFINV_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    print(f"Saved corner plot to {corner_path}")


if __name__ == "__main__":
    main()
