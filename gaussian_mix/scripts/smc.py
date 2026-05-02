"""
Run Adaptive Tempered SMC on the 2D Gaussian mixture model and save results.

The target is split into a broad Gaussian prior and a "likelihood" equal to
the ratio log p_target / log p_prior, so that at temperature lambda=0 particles
are drawn from the prior and at lambda=1 they approximate the target.
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
import blackjax.smc.resampling as resampling
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS, DEFAULT_WEIGHTS, PRIOR_LOW, PRIOR_HIGH

SMC_OUTPUT_DIR = OUTPUT_DIR / "smc"

NUM_PARTICLES = 2000
TARGET_ESS = 0.75    # target ESS fraction for adaptive temperature step
NUM_MCMC_STEPS = 10  # RMH refreshment steps per SMC iteration
MAX_STEPS = 500      # safety cap on SMC iterations


def main(seed=0, save_outputs=True):
    rng_key = jax.random.PRNGKey(seed)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model()

    # total timer: covers initialization, JIT compilation, sampling, and host transfer
    t_total = time.perf_counter()

    # --- Prior / likelihood split for tempering ---
    # At lambda=0: sample from logprior (broad Gaussian)
    # At lambda=1: sample from logprior + loglikelihood = log_target
    def logprior_fn(x):
        return jax.scipy.stats.uniform.logpdf(x, loc=PRIOR_LOW, scale=PRIOR_HIGH - PRIOR_LOW).sum()

    def loglikelihood_fn(x):
        return log_density_fn(x) - logprior_fn(x)

    # --- Build adaptive tempered SMC with RMH inner kernel ---
    # sigma is computed from the current tempering parameter (lambda) at each
    # step using the 2.38/sqrt(d) * sigma_prior / sqrt(lambda) rule, matching
    # the per-chain step size used in SEO/DEO.  We bake this into a jitted
    # wrapper that reads state.tempering_param as a traced JAX value so the
    # computation graph adapts dynamically without re-compilation.
    _sigma_prior = (PRIOR_HIGH - PRIOR_LOW) / jnp.sqrt(12)
    _rmh_kernel = blackjax.rmh.build_kernel()

    @jax.jit
    def step_fn(key, state):
        lam = jnp.maximum(state.tempering_param, 1e-4)
        sigma = (2.38 / jnp.sqrt(2)) * _sigma_prior / jnp.sqrt(lam)

        def _normal_proposal(rng_key, pos):
            return pos + jax.random.normal(rng_key, shape=pos.shape) * sigma

        def rmh_step_fn(rng_key, st, logdensity_fn):
            return _rmh_kernel(rng_key, st, logdensity_fn,
                               transition_generator=_normal_proposal)

        return blackjax.adaptive_tempered_smc(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=rmh_step_fn,
            mcmc_init_fn=blackjax.rmh.init,
            mcmc_parameters={},
            resampling_fn=resampling.systematic,
            target_ess=TARGET_ESS,
            num_mcmc_steps=NUM_MCMC_STEPS,
        ).step(key, state)

    # --- Initialize particles from prior ---
    # Build a one-off SMC object just for .init(); the proposal sigma doesn't
    # matter here since it is only used during .step().
    init_key, loop_key = jax.random.split(rng_key)
    initial_positions = jax.random.uniform(init_key, shape=(NUM_PARTICLES, 2), minval=PRIOR_LOW, maxval=PRIOR_HIGH)
    _smc_init = blackjax.adaptive_tempered_smc(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=lambda rng_key, st, logdensity_fn: _rmh_kernel(
            rng_key, st, logdensity_fn,
            transition_generator=lambda k, p: p + jax.random.normal(k, p.shape),
        ),
        mcmc_init_fn=blackjax.rmh.init,
        mcmc_parameters={},
        resampling_fn=resampling.systematic,
        target_ess=TARGET_ESS,
        num_mcmc_steps=NUM_MCMC_STEPS,
    )
    state = _smc_init.init(initial_positions)

    # AOT compile step_fn to exclude JIT from core timer
    _warmup_key, _compile_key = jax.random.split(loop_key)
    _c_step = step_fn.lower(_compile_key, state).compile()

    # core timer: SMC loop only, no JIT overhead
    # Each iteration forces synchronization via float(state.tempering_param).
    t_core = time.perf_counter()
    all_lambdas = [float(state.tempering_param)]
    all_log_nc = []
    num_steps = 0

    _print(f"Running adaptive tempered SMC ({NUM_PARTICLES} particles)...")
    while state.tempering_param < 1.0 and num_steps < MAX_STEPS:
        loop_key, subkey = jax.random.split(loop_key)
        state, info = _c_step(subkey, state)
        lam = float(state.tempering_param)
        lnc = float(info.log_likelihood_increment)
        all_lambdas.append(lam)
        all_log_nc.append(lnc)
        num_steps += 1
        if num_steps % 10 == 0 or lam >= 1.0:
            _print(f"  Step {num_steps:3d}: lambda={lam:.4f}")

    if num_steps >= MAX_STEPS:
        _print(f"  WARNING: hit MAX_STEPS={MAX_STEPS} before lambda reached 1.")
    else:
        _print(f"  Completed in {num_steps} SMC steps.")

    wall_time_core_s = time.perf_counter() - t_core

    # --- Final particles and weights ---
    particles = np.array(state.particles)   # (NUM_PARTICLES, 2)
    weights = np.array(state.weights)       # already normalised
    wall_time_total_s = time.perf_counter() - t_total

    final_ess = float(1.0 / np.sum(weights ** 2))
    log_nc_total = float(np.sum(all_log_nc))

    # Total log-density evaluations:
    # Each SMC step performs NUM_MCMC_STEPS RMH moves per particle (1 logp eval each)
    # plus 2 evaluations per particle for the incremental weight update (loglikelihood
    # at old and new temperature).
    total_log_density_evals = num_steps * NUM_PARTICLES * (NUM_MCMC_STEPS + 2)

    # --- Mode weight recovery (weighted) ---
    means_np = np.array(DEFAULT_MEANS)
    num_modes = len(means_np)
    dists_to_modes = np.linalg.norm(
        particles[:, None, :] - means_np[None, :, :], axis=-1
    )
    assignments = np.argmin(dists_to_modes, axis=-1)
    mode_weights = np.bincount(assignments, weights=weights, minlength=num_modes)

    _print("\n=== Diagnostics ===")
    _print(f"  SMC steps:                {num_steps}")
    _print(f"  Final lambda:             {float(state.tempering_param):.6f}")
    _print(f"  Log normalizing constant: {log_nc_total:.3f}")
    _print(f"  Final ESS:                {final_ess:.1f} / {NUM_PARTICLES}")
    _print(f"  Total log-density evals:  {int(total_log_density_evals)}")
    _print(f"  Wall-clock time (core):   {wall_time_core_s:.2f}s")
    _print(f"  Wall-clock time (total):  {wall_time_total_s:.2f}s")
    _print()
    true_weights = np.array(DEFAULT_WEIGHTS)
    _print("  Mode weight recovery (empirical vs true):")
    for k, (w, tw) in enumerate(zip(mode_weights, true_weights)):
        _print(f"    Mode {k}: {w:.3f}  (true: {tw:.3f})")

    # ArviZ summary — treat the (weighted, post-resampling) particles as one chain
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    if not save_outputs:
        _az_log.setLevel(logging.ERROR)
    idata = az.from_dict(
        posterior={"x1": particles[:, 0][None, :], "x2": particles[:, 1][None, :]}
    )
    summary = az.summary(idata, var_names=["x1", "x2"])
    _az_log.setLevel(_az_prev)
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())

    # --- Results ---
    diagnostics = {
        "sampler": "AdaptiveTemperedSMC",
        "num_particles": NUM_PARTICLES,
        "target_ess": TARGET_ESS,
        "num_mcmc_steps_per_iter": NUM_MCMC_STEPS,
        "prior_low": PRIOR_LOW,
        "prior_high": PRIOR_HIGH,
        "sigma_proposal_formula": "2.38/sqrt(2) * sigma_prior / sqrt(lambda)",
        "wall_time_s": wall_time_core_s,
        "wall_time_core_s": wall_time_core_s,
        "wall_time_total_s": wall_time_total_s,
        "num_smc_steps": num_steps,
        "final_tempering_param": float(state.tempering_param),
        "log_normalizing_constant": log_nc_total,
        "final_ess": final_ess,
        "total_log_density_evals": int(total_log_density_evals),
        "tempering_schedule": [float(x) for x in all_lambdas],
        "log_nc_increments": [float(x) for x in all_log_nc],
        "mode_weights": mode_weights.tolist(),
        "true_mode_weights": np.array(DEFAULT_WEIGHTS).tolist(),
        "arviz_summary": json.loads(summary.to_json()),
    }
    if save_outputs:
        SMC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(SMC_OUTPUT_DIR / "idata.nc"))
        diag_path = SMC_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)

    if not save_outputs:
        return diagnostics

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(
        particles[:, 0], particles[:, 1],
        c=weights, s=5, alpha=0.6, cmap="viridis",
    )
    plt.colorbar(sc, ax=axes[0], label="weight")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title(f"SMC particles (N={NUM_PARTICLES})")
    axes[0].set_aspect("equal")

    axes[1].plot(all_lambdas, marker="o", markersize=2, lw=0.8)
    axes[1].set_xlabel("SMC step")
    axes[1].set_ylabel("lambda (tempering parameter)")
    axes[1].set_title("Adaptive tempering schedule")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = SMC_OUTPUT_DIR / "samples.png"
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
    corner_path = SMC_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    _print(f"Saved corner plot to {corner_path}")
    return diagnostics


if __name__ == "__main__":
    main()
