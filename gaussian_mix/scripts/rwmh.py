"""
Run Random Walk Metropolis-Hastings on the 2D Gaussian mixture model and save trace plots.
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

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS, DEFAULT_WEIGHTS, PRIOR_LOW, PRIOR_HIGH

RWMH_OUTPUT_DIR = OUTPUT_DIR / "rwmh"

NUM_BURNIN = 500
NUM_SAMPLES = 10000
NUM_CHAINS = 4
STEP_SIZE = (2.38 / np.sqrt(2)) * (PRIOR_HIGH - PRIOR_LOW) / np.sqrt(12)  # 2.38/sqrt(d) * sigma_prior, d=2


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


def main(seed=0, save_outputs=True):
    init_key, burnin_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model()

    # total timer: covers initialization, JIT compilation, sampling, and host transfer
    t_total = time.perf_counter()

    # --- Initialize chains from random starting positions ---
    initial_positions = jax.random.uniform(init_key, shape=(NUM_CHAINS, 2), minval=PRIOR_LOW, maxval=PRIOR_HIGH)

    kernel = blackjax.rmh(log_density_fn, proposal_generator=blackjax.mcmc.random_walk.normal(sigma=STEP_SIZE))
    init_fn = jax.vmap(kernel.init)
    initial_states = init_fn(initial_positions)

    burnin_keys = jax.random.split(burnin_key, NUM_CHAINS)
    chain_sample_keys = jax.random.split(sample_key, NUM_CHAINS)

    @jax.vmap
    def run_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_BURNIN)

    @jax.vmap
    def run_sample_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_SAMPLES)

    # AOT compile to exclude JIT from core timer; use initial_states as shape proxy for post-burnin states
    if NUM_BURNIN > 0:
        _c_burnin = jax.jit(run_chain).lower(burnin_keys, initial_states).compile()
    _c_sample = jax.jit(run_sample_chain).lower(chain_sample_keys, initial_states).compile()

    # core timer: burn-in + sampling only, no JIT overhead
    t_core = time.perf_counter()

    # --- Burn-in (discard, just to move chains away from starting positions) ---
    if NUM_BURNIN > 0:
        _print(f"Running burn-in ({NUM_BURNIN} steps)...")
        burnin_states, _ = _c_burnin(burnin_keys, initial_states)
        post_burnin_states = jax.tree.map(lambda x: x[:, -1], burnin_states)
    else:
        post_burnin_states = initial_states

    # --- Sample ---
    _print(f"Sampling ({NUM_SAMPLES} steps, {NUM_CHAINS} chains)...")
    all_states, all_infos = _c_sample(chain_sample_keys, post_burnin_states)
    jax.block_until_ready(all_states)
    wall_time_core_s = time.perf_counter() - t_core

    # all_states.position: (NUM_CHAINS, NUM_SAMPLES, 2)
    samples = np.array(all_states.position)
    wall_time_total_s = time.perf_counter() - t_total

    # --- Diagnostics ---
    means_np = np.array(DEFAULT_MEANS)
    num_modes = len(means_np)

    # Assign each sample to nearest mode: (NUM_CHAINS, NUM_SAMPLES)
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
    stuck_chains = [c for c in range(NUM_CHAINS) if np.unique(chain_assignments[c]).size == 1]

    # Acceptance rate: blackjax rmh populates all_infos.acceptance_rate per step
    # shape: (NUM_CHAINS, NUM_SAMPLES)
    acceptance = float(np.mean(np.array(all_infos.acceptance_rate)))

    # Cost metric: RWMH is gradient-free; each step requires exactly 1 log-density
    # evaluation per chain (one proposal evaluated, no leapfrog integration).
    total_log_density_evals = NUM_CHAINS * (NUM_BURNIN + NUM_SAMPLES)

    # ArviZ summary: R-hat, bulk/tail ESS, MCSE
    idata = az.from_dict(
        posterior={"x1": samples[:, :, 0], "x2": samples[:, :, 1]},
        sample_stats={"acceptance_rate": np.array(all_infos.acceptance_rate)},
    )
    summary = az.summary(idata, var_names=["x1", "x2"])
    total_bulk_ess = summary["ess_bulk"].sum()

    # ESS per log-density evaluation
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
    _print(f"  Inter-mode transitions per chain: {transitions.tolist()}")
    if stuck_chains:
        _print(f"  WARNING: stuck chains (never left one mode): {stuck_chains}")
    else:
        _print(f"  No stuck chains detected.")
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print()
    _print(f"  Bulk ESS per log-density eval: {ess_per_logp_eval:.4f}")
    _print(f"\n  Wall-clock time (core):  {wall_time_core_s:.2f}s")
    _print(f"  Wall-clock time (total): {wall_time_total_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "RandomWalkMetropolisHastings",
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_BURNIN,
        "num_samples": NUM_SAMPLES,
        "adapted_step_size": float(STEP_SIZE),
        "mean_acceptance_rate": float(acceptance),
        "total_log_density_evals": int(total_log_density_evals),
        "wall_time_s": wall_time_core_s,
        "wall_time_core_s": wall_time_core_s,
        "wall_time_total_s": wall_time_total_s,
        "bulk_ess_per_logp_eval": float(ess_per_logp_eval),
        "mode_weights": mode_weights.tolist(),
        "true_mode_weights": np.array(DEFAULT_WEIGHTS).tolist(),
        "inter_mode_transitions": transitions.tolist(),
        "stuck_chains": stuck_chains,
        "arviz_summary": json.loads(summary.to_json()),
    }
    if save_outputs:
        RWMH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(RWMH_OUTPUT_DIR / "idata.nc"))
        diag_path = RWMH_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)

    if not save_outputs:
        return diagnostics

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CHAINS))
    for c in range(NUM_CHAINS):
        axes[0].scatter(samples[c, :, 0], samples[c, :, 1], alpha=0.15, s=2, color=colors[c])
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title(f"RWMH samples ({NUM_CHAINS} chains)")
    axes[0].set_aspect("equal")

    for i, label in enumerate(["x1", "x2"]):
        for c in range(NUM_CHAINS):
            axes[i + 1].plot(samples[c, :, i], lw=0.4, alpha=0.6, color=colors[c])
        axes[i + 1].set_xlabel("iteration")
        axes[i + 1].set_ylabel(label)
        axes[i + 1].set_title(f"Trace: {label}")

    fig.tight_layout()
    out_path = RWMH_OUTPUT_DIR / "samples.png"
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
    corner_path = RWMH_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    _print(f"Saved corner plot to {corner_path}")
    return diagnostics


if __name__ == "__main__":
    main()
