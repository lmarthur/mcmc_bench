"""
Run Random Walk Metropolis-Hastings on the SAJAX planet+activity model and save outputs.
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
    PRIOR_DISTRIBUTIONS,
)

RWMH_OUTPUT_DIR = OUTPUT_DIR / "rwmh"

NDIM = len(PARAM_NAMES)
NUM_BURNIN  = 2500
NUM_SAMPLES = 2500
NUM_CHAINS  = 8


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


# Roberts, Gelman & Gilks (1997): σ_proposal = 2.38/√d × σ_min, where σ_min is the
# smallest prior std in unconstrained space across all parameters.
_UNC_PRIOR_STDS = {name: _unconstrained_prior_std(d) for name, d in PRIOR_DISTRIBUTIONS.items()}
_SIGMA_MIN = min(_UNC_PRIOR_STDS.values())
STEP_SIZE = 2.38 / np.sqrt(NDIM) * _SIGMA_MIN



def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)
    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


def main(seed: int = 0, save_outputs: bool = True):
    init_key, burnin_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    y_obs = OBS_LIGHT_CURVE

    log_density_fn, _, _ = make_inference_fns(init_key, y_obs)
    constrain_fn = make_constrain_fn()

    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # Initialise chains in unconstrained space
    x0 = sample_initial_positions(init_key, NUM_CHAINS)

    x0_constrained = jax.vmap(constrain_fn)(x0)
    _print("Initial chain positions (constrained space):")
    _print(f"  {'param':20s}  " + "  ".join(f"chain{i:02d}" for i in range(NUM_CHAINS)))
    for name in PARAM_NAMES:
        vals = np.array(x0_constrained[name])
        _print(f"  {name:20s}  " + "  ".join(f"{v:8.4f}" for v in vals))

    _print("\nDiagnostic: log density at initial positions")
    _print(f"  {'chain':>6}  {'log_density':>14}  {'finite?':>8}  {'nan?':>6}")
    _print("  " + "-" * 40)
    init_log_densities = []
    for i in range(NUM_CHAINS):
        x_i = jax.tree.map(lambda leaf: leaf[i], x0)
        ld = float(log_density_fn(x_i))
        init_log_densities.append(ld)
        _print(f"  {i:>6}  {ld:>14.4f}  {str(np.isfinite(ld)):>8}  {str(np.isnan(ld)):>6}")
    n_nan = sum(np.isnan(d) for d in init_log_densities)
    n_neginf = sum(np.isneginf(d) for d in init_log_densities)
    if n_nan or n_neginf:
        _print(f"\n  WARNING: {n_nan} chain(s) have NaN log density, "
               f"{n_neginf} have -inf. Chains will be stuck — "
               f"check the light curve computation for numerical issues.")
    else:
        _print(f"\n  All {NUM_CHAINS} initial log densities are finite.")

    kernel = blackjax.rmh(
        log_density_fn,
        proposal_generator=blackjax.mcmc.random_walk.normal(sigma=STEP_SIZE),
    )
    init_fn = jax.vmap(kernel.init)
    initial_states = init_fn(x0)

    @jax.vmap
    def run_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_BURNIN)

    if NUM_BURNIN > 0:
        _print(f"Running burn-in ({NUM_BURNIN} steps, {NUM_CHAINS} chains)...")
        burnin_keys = jax.random.split(burnin_key, NUM_CHAINS)
        burnin_states, _ = run_chain(burnin_keys, initial_states)
        post_burnin_states = jax.tree.map(lambda x: x[:, -1], burnin_states)
    else:
        post_burnin_states = initial_states

    _print(f"Sampling ({NUM_SAMPLES} steps, {NUM_CHAINS} chains, {NDIM} params)...")

    @jax.vmap
    def run_sample_chain(rng_key, initial_state):
        return inference_loop(rng_key, kernel.step, initial_state, NUM_SAMPLES)

    chain_sample_keys = jax.random.split(sample_key, NUM_CHAINS)
    all_states, all_infos = run_sample_chain(chain_sample_keys, post_burnin_states)

    # all_states.position: dict, each leaf shape (NUM_CHAINS, NUM_SAMPLES)
    # Map unconstrained → constrained for diagnostics and plotting.
    unc_positions = all_states.position
    constrained_positions = jax.vmap(jax.vmap(constrain_fn))(unc_positions)

    acceptance = float(np.mean(np.array(all_infos.acceptance_rate)))
    total_log_density_evals = NUM_CHAINS * (NUM_BURNIN + NUM_SAMPLES)

    posterior_dict = {k: np.array(v) for k, v in constrained_positions.items()}
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={"acceptance_rate": np.array(all_infos.acceptance_rate)},
    )
    summary = az.summary(idata)
    total_bulk_ess = summary["ess_bulk"].sum()
    ess_per_logp_eval = total_bulk_ess / total_log_density_evals

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = np.array([
        np.array(constrained_positions[p]).mean() for p in PARAM_NAMES
    ])
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  Mean acceptance rate:          {acceptance:.3f}")
    _print(f"  Total log-density evaluations: {int(total_log_density_evals)}")
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

    wall_time_s = time.perf_counter() - t0
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    diagnostics = {
        "sampler": "RandomWalkMetropolisHastings",
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_BURNIN,
        "num_samples": NUM_SAMPLES,
        "ndim": NDIM,
        "step_size": float(STEP_SIZE),
        "mean_acceptance_rate": float(acceptance),
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
        RWMH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(RWMH_OUTPUT_DIR / "idata.nc"))
        diag_path = RWMH_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {RWMH_OUTPUT_DIR / 'idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    stuck_params = [
        p for p in PARAM_NAMES
        if np.array(constrained_positions[p]).std(axis=1).min() <= 1e-10
    ]
    if stuck_params:
        print(f"WARNING: {len(stuck_params)} parameter(s) have zero within-chain variance "
              f"(chains stuck): {stuck_params}")
    plot_vars = [p for p in PARAM_NAMES[:6] if p not in stuck_params]
    if plot_vars:
        az.plot_trace(idata, var_names=plot_vars, figsize=(14, 10))
        plt.tight_layout()
        plt.savefig(RWMH_OUTPUT_DIR / "traces_subset.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        print("WARNING: All parameters are stuck — skipping trace plot.")

    az.rcParams["plot.max_subplots"] = len(PARAM_NAMES) ** 2
    az.plot_pair(idata, var_names=PARAM_NAMES, kind="kde", marginals=True, figsize=(24, 24))
    plt.savefig(RWMH_OUTPUT_DIR / "corner_all.png", dpi=120, bbox_inches="tight")
    plt.close()

    plot_bestfit_lightcurve(constrained_positions, RWMH_OUTPUT_DIR)

    return diagnostics


if __name__ == "__main__":
    main()
