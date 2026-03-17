"""
Run NUTS on the 2D Gaussian mixture model and save trace plots.
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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS

NUTS_OUTPUT_DIR = OUTPUT_DIR / "nuts"

NUM_WARMUP = 1000
NUM_SAMPLES = 5000
NUM_CHAINS = 5


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


def main():
    rng_key = jax.random.PRNGKey(0)

    # --- Model ---
    log_density_fn = make_log_density()
    print("Plotting model...")
    plot_model()

    # --- Warmup: adapt one chain, share parameters across all chains ---
    print(f"Running warmup ({NUM_WARMUP} steps)...")
    warmup_key, sample_key = jax.random.split(rng_key)
    warmup = blackjax.window_adaptation(blackjax.nuts, log_density_fn)
    initial_position = jnp.array([0.0, 0.0])
    (_, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=NUM_WARMUP)
    print(f"  Adapted step size: {parameters['step_size']:.4f}")

    # --- Initialize chains from random starting positions ---
    init_key, sample_key = jax.random.split(sample_key)
    initial_positions = jax.random.normal(init_key, shape=(NUM_CHAINS, 2)) * 3.0

    kernel = blackjax.nuts(log_density_fn, **parameters)
    init_fn = jax.vmap(kernel.init)
    initial_states = init_fn(initial_positions)

    # --- Run chains in parallel with vmap ---
    print(f"Sampling ({NUM_SAMPLES} steps, {NUM_CHAINS} chains)...")
    chain_sample_keys = jax.random.split(sample_key, NUM_CHAINS)

    @jax.vmap
    def run_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_SAMPLES)

    all_states, all_infos = run_chain(chain_sample_keys, initial_states)

    # all_states.position: (NUM_CHAINS, NUM_SAMPLES, 2)
    samples = np.array(all_states.position)

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

    # ESS per gradient evaluation (NUTS: num_integration_steps ~ grad evals per step)
    num_integration_steps = np.array(all_infos.num_integration_steps)

    # ArviZ summary: R-hat, bulk/tail ESS, MCSE
    idata = az.from_dict(
        posterior={"x1": samples[:, :, 0], "x2": samples[:, :, 1]},
        sample_stats={
            "acceptance_rate": np.array(all_infos.acceptance_rate),
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
    acceptance = np.mean(np.array(all_infos.acceptance_rate))

    print("\n=== Diagnostics ===")
    print(f"  Mean acceptance rate:       {acceptance:.3f}")
    print(f"  Total gradient evaluations: {int(total_grad_evals)}")
    print(f"  Mean tree depth (steps):    {num_integration_steps.mean():.1f}  (max observed: {int(max_steps)})")
    print(f"  Tree depth saturation:      {saturation_frac:.1%}")
    print()
    print(f"  Mode weight recovery (true = {1/num_modes:.3f} each):")
    for k, w in enumerate(mode_weights):
        print(f"    Mode {k}: {w:.3f}")
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
    NUTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idata.to_netcdf(str(NUTS_OUTPUT_DIR / "idata.nc"))
    print(f"Saved InferenceData to {NUTS_OUTPUT_DIR / 'idata.nc'}")

    diagnostics = {
        "sampler": "NUTS",
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "adapted_step_size": float(parameters["step_size"]),
        "mean_acceptance_rate": float(acceptance),
        "total_grad_evals": int(total_grad_evals),
        "mean_integration_steps": float(num_integration_steps.mean()),
        "max_integration_steps": int(max_steps),
        "tree_depth_saturation": float(saturation_frac),
        "mode_weights": mode_weights.tolist(),
        "true_mode_weight": 1 / num_modes,
        "inter_mode_transitions": transitions.tolist(),
        "stuck_chains": stuck_chains,
        "bulk_ess_per_grad_eval": float(ess_per_grad),
        "arviz_summary": json.loads(summary.to_json()),
    }
    diag_path = NUTS_OUTPUT_DIR / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics to {diag_path}")

    # --- Plots ---

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CHAINS))
    for c in range(NUM_CHAINS):
        axes[0].scatter(samples[c, :, 0], samples[c, :, 1], alpha=0.15, s=2, color=colors[c])
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title(f"NUTS samples ({NUM_CHAINS} chains)")
    axes[0].set_aspect("equal")

    for i, label in enumerate(["x1", "x2"]):
        for c in range(NUM_CHAINS):
            axes[i + 1].plot(samples[c, :, i], lw=0.4, alpha=0.6, color=colors[c])
        axes[i + 1].set_xlabel("iteration")
        axes[i + 1].set_ylabel(label)
        axes[i + 1].set_title(f"Trace: {label}")

    fig.tight_layout()
    out_path = NUTS_OUTPUT_DIR / "samples.png"
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
    corner_path = NUTS_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    print(f"Saved corner plot to {corner_path}")


if __name__ == "__main__":
    main()
