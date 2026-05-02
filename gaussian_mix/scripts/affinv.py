"""
Run Affine Invariant MCMC on the 2D Gaussian mixture model and save trace plots.
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
import emcee_jax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS, DEFAULT_WEIGHTS, PRIOR_LOW, PRIOR_HIGH

AFFINV_OUTPUT_DIR = OUTPUT_DIR / "affinv"

NUM_BURNIN = 500
NUM_SAMPLES = 2500
NUM_WALKERS = 64
NDIM = 2


def main(seed=0, save_outputs=True):
    init_key, state_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model()

    # total timer: covers initialization, JIT compilation, sampling, and host transfer
    t_total = time.perf_counter()

    # --- Initialize walkers from random starting positions ---
    coords = jax.random.uniform(init_key, shape=(NUM_WALKERS, NDIM), minval=PRIOR_LOW, maxval=PRIOR_HIGH)

    # --- Initialize sampler ---
    sampler = emcee_jax.EnsembleSampler(log_density_fn)
    state = sampler.init(state_key, coords)

    # AOT compile to exclude JIT from core timer; progress=False required inside jit
    _c_sample = jax.jit(
        lambda k, s: sampler.sample_parallel(k, s, NUM_SAMPLES, progress=False)
    ).lower(sample_key, state).compile()

    # core timer: sampling only, no JIT overhead
    t_core = time.perf_counter()

    _print(f"Sampling ({NUM_SAMPLES} steps, {NUM_WALKERS} walkers)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = _c_sample(sample_key, state)
    jax.block_until_ready(trace.samples.coordinates)
    wall_time_core_s = time.perf_counter() - t_core

    # samples shape: (NUM_STEPS, NUM_WALKERS, NDIM) -> reshape to (NUM_WALKERS, NUM_SAMPLES, NDIM)
    raw = np.asarray(trace.samples.coordinates)  # (NUM_SAMPLES, NUM_WALKERS, NDIM)
    samples = raw.transpose(1, 0, 2)             # (NUM_WALKERS, NUM_SAMPLES, NDIM)
    samples = samples[:, NUM_BURNIN:, :]         # discard burn-in
    wall_time_total_s = time.perf_counter() - t_total

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

    # Inter-mode transitions per walker
    transitions = np.sum(np.diff(chain_assignments, axis=1) != 0, axis=1)

    # Stuck walkers (never leave one mode)
    stuck_chains = [c for c in range(NUM_WALKERS) if np.unique(chain_assignments[c]).size == 1]

    # Acceptance rate: emcee stores per-step acceptance as a boolean array
    # trace.samples has an `accept_prob` field: (NUM_SAMPLES, NUM_WALKERS)
    accepted = np.asarray(trace.sample_stats['accept_prob'])   # (NUM_SAMPLES, NUM_WALKERS)
    acceptance = float(accepted.mean())

    # Cost metric: each step proposes one move per walker -> NUM_WALKERS log-density evals per step
    total_log_density_evals = NUM_WALKERS * NUM_SAMPLES

    # ArviZ summary: R-hat, bulk/tail ESS, MCSE
    # emcee has no per-step tree-depth info, so sample_stats only carries acceptance
    idata = az.from_dict(
        posterior={"x1": samples[:, :, 0], "x2": samples[:, :, 1]},
        sample_stats={"acceptance_rate": accepted.T},  # (NUM_WALKERS, NUM_SAMPLES)
    )
    summary = az.summary(idata, var_names=["x1", "x2"])
    total_bulk_ess = summary["ess_bulk"].sum()

    # ESS per log-density evaluation (analogous to NUTS's ESS per grad eval)
    ess_per_logp_eval = total_bulk_ess / total_log_density_evals

    _print("\n=== Diagnostics ===")
    _print(f"  Mean acceptance rate:          {acceptance:.3f}")
    _print(f"  Total log-density evaluations: {int(total_log_density_evals)}")
    _print()
    true_weights = np.array(DEFAULT_WEIGHTS)
    _print(f"  Mode weight recovery (empirical vs true):")
    for k, (w, tw) in enumerate(zip(mode_weights, true_weights)):
        _print(f"    Mode {k}: {w:.3f}  (true: {tw:.3f})")
    _print()
    _print(f"  Inter-mode transitions per walker: {transitions.tolist()}")
    if stuck_chains:
        _print(f"  WARNING: stuck walkers (never left one mode): {stuck_chains}")
    else:
        _print(f"  No stuck walkers detected.")
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print()
    _print(f"  Bulk ESS per log-density eval: {ess_per_logp_eval:.4f}")
    _print(f"\n  Wall-clock time (core):  {wall_time_core_s:.2f}s")
    _print(f"  Wall-clock time (total): {wall_time_total_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "AffineInvariantMCMC",
        "num_walkers": NUM_WALKERS,
        "num_burnin": NUM_BURNIN,
        "num_samples": NUM_SAMPLES,
        "wall_time_s": wall_time_core_s,
        "wall_time_core_s": wall_time_core_s,
        "wall_time_total_s": wall_time_total_s,
        "mean_acceptance_rate": float(acceptance),
        "total_log_density_evals": int(total_log_density_evals),
        "bulk_ess_per_logp_eval": float(ess_per_logp_eval),
        "mode_weights": mode_weights.tolist(),
        "true_mode_weights": np.array(DEFAULT_WEIGHTS).tolist(),
        "inter_mode_transitions": transitions.tolist(),
        "stuck_chains": stuck_chains,
        "arviz_summary": json.loads(summary.to_json()),
    }
    if save_outputs:
        AFFINV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(AFFINV_OUTPUT_DIR / "idata.nc"))
        diag_path = AFFINV_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)

    if not save_outputs:
        return diagnostics

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_WALKERS))
    for c in range(NUM_WALKERS):
        axes[0].scatter(samples[c, :, 0], samples[c, :, 1], alpha=0.15, s=2, color=colors[c])
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title(f"Affine Invariant MCMC samples ({NUM_WALKERS} walkers)")
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
    corner_path = AFFINV_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    _print(f"Saved corner plot to {corner_path}")
    return diagnostics


if __name__ == "__main__":
    main()
