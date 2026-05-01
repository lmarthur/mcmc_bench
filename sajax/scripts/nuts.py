"""
Run NUTS on the SAJAX planet+activity model and save outputs.
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
    make_inference_fns,
    make_constrain_fn,
    plot_model,
    sample_initial_positions,
    plot_bestfit_lightcurve,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    OBS_LIGHT_CURVE,
)

NUTS_OUTPUT_DIR = OUTPUT_DIR / "nuts"

NDIM = len(PARAM_NAMES)
NUM_WARMUP = 250
NUM_SAMPLES = 1000
NUM_CHAINS = 4


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


def main(seed: int = 0, save_outputs: bool = True):
    init_key, warmup_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn, _, _ = make_inference_fns(init_key, OBS_LIGHT_CURVE)
    constrain_fn = make_constrain_fn()

    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    # --- Diagnostic 1: JAX autodiff vs finite-difference per parameter ---
    _print("=== Diagnostic 1: JAX grad vs finite-difference grad (h=1e-4) ===")
    warmup_start = jax.tree.map(lambda x: x[0], sample_initial_positions(init_key, 1))
    grad = jax.grad(log_density_fn)(warmup_start)
    h = 1e-4
    _print(f"  {'Parameter':<22} {'JAX grad':>14} {'FD grad':>14} {'ratio':>10} {'match?':>8}")
    _print("  " + "-" * 70)
    for name in PARAM_NAMES:
        def perturb(z, delta, _name=name):
            return {k: v + delta if k == _name else v for k, v in z.items()}
        fd_grad = float(
            (log_density_fn(perturb(warmup_start, h)) - log_density_fn(perturb(warmup_start, -h))) / (2 * h)
        )
        jg = float(grad[name])
        if abs(fd_grad) > 1e-30:
            ratio = jg / fd_grad
        elif abs(jg) < 1e-30:
            ratio = float("nan")
        else:
            ratio = float("inf")
        match = "OK" if abs(jg - fd_grad) < max(1e-2 * max(abs(jg), abs(fd_grad)), 1e-6) else "MISMATCH"
        _print(f"  {name:<22} {jg:>14.4g} {fd_grad:>14.4g} {ratio:>10.4f} {match:>8}")
    _print()

    t0 = time.perf_counter()

    # --- Warmup: adapt one chain, share parameters across all chains ---
    _print(f"Running warmup ({NUM_WARMUP} steps)...")
    warmup = blackjax.window_adaptation(blackjax.nuts, log_density_fn)
    (_, parameters), _ = warmup.run(warmup_key, warmup_start, num_steps=NUM_WARMUP)
    _print(f"  Adapted step size: {parameters['step_size']:.4f}")

    # --- Diagnostic 3: adapted inverse mass matrix ---
    _print("\n=== Diagnostic 3: Adapted inverse mass matrix ===")
    inv_mass = parameters["inverse_mass_matrix"]
    if isinstance(inv_mass, dict):
        _print(f"  {'Parameter':<22} {'inv_M diag':>14} {'eff. std':>12}")
        _print("  " + "-" * 50)
        for name in PARAM_NAMES:
            v = float(inv_mass[name])
            eff_std = float(np.sqrt(v)) if v >= 0 else float("nan")
            _print(f"  {name:<22} {v:>14.4g} {eff_std:>12.4g}")
    else:
        inv_mass_arr = np.array(inv_mass)
        if inv_mass_arr.ndim == 1:
            _print(f"  {'Parameter':<22} {'inv_M diag':>14} {'eff. std':>12}")
            _print("  " + "-" * 50)
            for name, v in zip(PARAM_NAMES, inv_mass_arr):
                eff_std = float(np.sqrt(v)) if v >= 0 else float("nan")
                _print(f"  {name:<22} {float(v):>14.4g} {eff_std:>12.4g}")
        else:
            _print(f"  {'Parameter':<22} {'inv_M diag':>14}")
            _print("  " + "-" * 38)
            for name, v in zip(PARAM_NAMES, np.diag(inv_mass_arr)):
                _print(f"  {name:<22} {float(v):>14.4g}")
    _print()

    # --- Initialize chains ---
    init_key2, sample_key = jax.random.split(sample_key)
    initial_positions = sample_initial_positions(init_key2, NUM_CHAINS)

    kernel = blackjax.nuts(log_density_fn, **parameters)
    init_fn = jax.vmap(kernel.init)
    initial_states = init_fn(initial_positions)

    # --- Run chains in parallel ---
    _print(f"Sampling ({NUM_SAMPLES} steps, {NUM_CHAINS} chains, {NDIM} params)...")
    chain_sample_keys = jax.random.split(sample_key, NUM_CHAINS)

    @jax.vmap
    def run_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_SAMPLES)

    all_states, all_infos = run_chain(chain_sample_keys, initial_states)

    # all_states.position: dict, each leaf shape (NUM_CHAINS, NUM_SAMPLES)
    unc_positions = all_states.position
    constrained_positions = jax.vmap(jax.vmap(constrain_fn))(unc_positions)

    # --- Diagnostics ---
    num_integration_steps = np.array(all_infos.num_integration_steps)
    total_grad_evals = int(num_integration_steps.sum())
    acceptance = float(np.mean(np.array(all_infos.acceptance_rate)))

    posterior_dict = {name: np.array(constrained_positions[name]) for name in PARAM_NAMES}
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={
            "acceptance_rate": np.array(all_infos.acceptance_rate),
            "n_steps": num_integration_steps,
        },
    )
    summary = az.summary(idata)
    total_bulk_ess = summary["ess_bulk"].sum()
    ess_per_grad = total_bulk_ess / total_grad_evals

    max_steps = num_integration_steps.max()
    saturation_frac = float(np.mean(num_integration_steps == max_steps))

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = np.array([np.array(constrained_positions[p]).mean() for p in PARAM_NAMES])
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  Mean acceptance rate:       {acceptance:.3f}")
    _print(f"  Total gradient evaluations: {total_grad_evals}")
    _print(f"  Mean tree depth (steps):    {num_integration_steps.mean():.1f}  (max observed: {int(max_steps)})")
    _print(f"  Tree depth saturation:      {saturation_frac:.1%}")
    _print()
    _print("  Parameter recovery (posterior mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print()
    _print(f"  Total bulk ESS: {total_bulk_ess:.1f}")
    _print(f"  Bulk ESS per gradient eval: {ess_per_grad:.4f}")

    wall_time_s = time.perf_counter() - t0
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "NUTS",
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "ndim": NDIM,
        "adapted_step_size": float(parameters["step_size"]),
        "mean_acceptance_rate": float(acceptance),
        "total_grad_evals": total_grad_evals,
        "mean_integration_steps": float(num_integration_steps.mean()),
        "max_integration_steps": int(max_steps),
        "tree_depth_saturation": saturation_frac,
        "wall_time_s": float(wall_time_s),
        "total_bulk_ess": float(total_bulk_ess),
        "bulk_ess_per_grad_eval": float(ess_per_grad),
        "posterior_means": {name: float(pm) for name, pm in zip(PARAM_NAMES, posterior_means)},
        "ground_truth": {k: float(v) for k, v in GROUND_TRUTH.items()},
        "param_bias": {name: float(b) for name, b in zip(PARAM_NAMES, param_bias)},
        "arviz_summary": json.loads(summary.to_json()),
    }

    if save_outputs:
        NUTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(NUTS_OUTPUT_DIR / "sajax_idata.nc"))
        diag_path = NUTS_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {NUTS_OUTPUT_DIR / 'sajax_idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    def _is_degenerate(arr):
        # relative std < 1e-6 catches near-constant chains that pass a raw absolute check
        s = arr.std(axis=1).min()
        m = abs(arr.mean())
        return s / (m + 1.0) < 1e-6

    stuck_params = [
        p for p in PARAM_NAMES
        if _is_degenerate(np.array(constrained_positions[p]))
    ]
    if stuck_params:
        print(f"WARNING: {len(stuck_params)} parameter(s) have near-zero within-chain variance "
              f"(chains stuck): {stuck_params}")

    # Trace plots for first 6 non-stuck parameters
    plot_vars = [p for p in PARAM_NAMES[:6] if p not in stuck_params]
    if plot_vars:
        az.plot_trace(idata, var_names=plot_vars, figsize=(14, 10))
        plt.tight_layout()
        trace_path = NUTS_OUTPUT_DIR / "traces_subset.png"
        plt.savefig(trace_path, dpi=150, bbox_inches="tight")
        plt.close()
        _print(f"Saved trace plot to {trace_path}")
    else:
        print("WARNING: All parameters are stuck — skipping trace plot.")

    # Corner plot — exclude degenerate parameters to avoid KDE failure
    corner_vars = [p for p in PARAM_NAMES if p not in stuck_params]
    az.rcParams["plot.max_subplots"] = len(corner_vars) ** 2
    az.plot_pair(
        idata,
        var_names=corner_vars,
        kind="kde",
        marginals=True,
        figsize=(24, 24),
    )
    corner_path = NUTS_OUTPUT_DIR / "corner_all.png"
    plt.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close()
    _print(f"Saved full corner plot to {corner_path}")

    # Best-fit light curve using posterior mean
    plot_bestfit_lightcurve(constrained_positions, NUTS_OUTPUT_DIR)

    return diagnostics


if __name__ == "__main__":
    main()
