"""
Run Adaptive Tempered SMC on the 2D Gaussian mixture model and save results.

The target is split into a broad Gaussian prior and a "likelihood" equal to
the ratio log p_target / log p_prior, so that at temperature lambda=0 particles
are drawn from the prior and at lambda=1 they approximate the target.
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
import blackjax.smc.resampling as resampling
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import make_log_density, plot_model, OUTPUT_DIR, DEFAULT_MEANS, DEFAULT_WEIGHTS

SMC_OUTPUT_DIR = OUTPUT_DIR / "smc"

NUM_PARTICLES = 2000
TARGET_ESS = 0.5    # target ESS fraction for adaptive temperature step
NUM_MCMC_STEPS = 10  # RMH refreshment steps per SMC iteration
SIGMA_PRIOR = 15.0   # prior std dev — wide enough to cover modes at radius ~7
SIGMA_PROPOSAL = 1.0  # RMH proposal std dev — roughly one mode width
MAX_STEPS = 500      # safety cap on SMC iterations


def main():
    rng_key = jax.random.PRNGKey(0)

    # --- Model ---
    log_density_fn = make_log_density()
    print("Plotting model...")
    plot_model()

    # --- Prior / likelihood split for tempering ---
    # At lambda=0: sample from logprior (broad Gaussian)
    # At lambda=1: sample from logprior + loglikelihood = log_target
    def logprior_fn(x):
        return jax.scipy.stats.norm.logpdf(x, 0.0, SIGMA_PRIOR).sum()

    def loglikelihood_fn(x):
        return log_density_fn(x) - logprior_fn(x)

    # --- Build adaptive tempered SMC with RMH inner kernel ---
    # transition_generator is a callable and cannot be passed through
    # mcmc_parameters (which gets vmapped over particles).  Bake sigma into a
    # closure so the step function is self-contained and mcmc_parameters stays
    # empty, sidestepping the vmap shape issue entirely.
    _rmh_kernel = blackjax.rmh.build_kernel()
    _normal_proposal = lambda rng_key, pos: (
        pos + jax.random.normal(rng_key, shape=pos.shape) * SIGMA_PROPOSAL
    )

    def rmh_step_fn(rng_key, state, logdensity_fn):
        return _rmh_kernel(rng_key, state, logdensity_fn,
                           transition_generator=_normal_proposal)

    smc = blackjax.adaptive_tempered_smc(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=rmh_step_fn,
        mcmc_init_fn=blackjax.rmh.init,
        mcmc_parameters={},
        resampling_fn=resampling.systematic,
        target_ess=TARGET_ESS,
        num_mcmc_steps=NUM_MCMC_STEPS,
    )

    # --- Initialize particles from prior ---
    init_key, loop_key = jax.random.split(rng_key)
    initial_positions = jax.random.normal(init_key, shape=(NUM_PARTICLES, 2)) * SIGMA_PRIOR
    state = smc.init(initial_positions)

    # --- Run SMC loop (adaptive schedule; stop when lambda reaches 1) ---
    step_fn = jax.jit(smc.step)
    all_lambdas = [float(state.tempering_param)]
    all_log_nc = []
    num_steps = 0

    print(f"Running adaptive tempered SMC ({NUM_PARTICLES} particles)...")
    while state.tempering_param < 1.0 and num_steps < MAX_STEPS:
        loop_key, subkey = jax.random.split(loop_key)
        state, info = step_fn(subkey, state)
        lam = float(state.tempering_param)
        lnc = float(info.log_likelihood_increment)
        all_lambdas.append(lam)
        all_log_nc.append(lnc)
        num_steps += 1
        if num_steps % 10 == 0 or lam >= 1.0:
            print(f"  Step {num_steps:3d}: lambda={lam:.4f}")

    if num_steps >= MAX_STEPS:
        print(f"  WARNING: hit MAX_STEPS={MAX_STEPS} before lambda reached 1.")
    else:
        print(f"  Completed in {num_steps} SMC steps.")

    # --- Final particles and weights ---
    particles = np.array(state.particles)   # (NUM_PARTICLES, 2)
    weights = np.array(state.weights)       # already normalised

    final_ess = float(1.0 / np.sum(weights ** 2))
    log_nc_total = float(np.sum(all_log_nc))

    # --- Mode weight recovery (weighted) ---
    means_np = np.array(DEFAULT_MEANS)
    num_modes = len(means_np)
    dists_to_modes = np.linalg.norm(
        particles[:, None, :] - means_np[None, :, :], axis=-1
    )
    assignments = np.argmin(dists_to_modes, axis=-1)
    mode_weights = np.bincount(assignments, weights=weights, minlength=num_modes)

    print("\n=== Diagnostics ===")
    print(f"  SMC steps:                {num_steps}")
    print(f"  Final lambda:             {float(state.tempering_param):.6f}")
    print(f"  Log normalizing constant: {log_nc_total:.3f}")
    print(f"  Final ESS:                {final_ess:.1f} / {NUM_PARTICLES}")
    print()
    true_weights = np.array(DEFAULT_WEIGHTS)
    print("  Mode weight recovery (empirical vs true):")
    for k, (w, tw) in enumerate(zip(mode_weights, true_weights)):
        print(f"    Mode {k}: {w:.3f}  (true: {tw:.3f})")

    # ArviZ summary — treat the (weighted, post-resampling) particles as one chain
    idata = az.from_dict(
        posterior={"x1": particles[:, 0][None, :], "x2": particles[:, 1][None, :]}
    )
    summary = az.summary(idata, var_names=["x1", "x2"])
    print()
    print("  ArviZ summary (R-hat, ESS, MCSE):")
    print(summary.to_string())

    # --- Save results ---
    SMC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idata.to_netcdf(str(SMC_OUTPUT_DIR / "idata.nc"))
    print(f"Saved InferenceData to {SMC_OUTPUT_DIR / 'idata.nc'}")

    diagnostics = {
        "sampler": "AdaptiveTemperedSMC",
        "num_particles": NUM_PARTICLES,
        "target_ess": TARGET_ESS,
        "num_mcmc_steps_per_iter": NUM_MCMC_STEPS,
        "sigma_prior": SIGMA_PRIOR,
        "sigma_proposal": SIGMA_PROPOSAL,
        "num_smc_steps": num_steps,
        "final_tempering_param": float(state.tempering_param),
        "log_normalizing_constant": log_nc_total,
        "final_ess": final_ess,
        "tempering_schedule": [float(x) for x in all_lambdas],
        "log_nc_increments": [float(x) for x in all_log_nc],
        "mode_weights": mode_weights.tolist(),
        "true_mode_weights": np.array(DEFAULT_WEIGHTS).tolist(),
        "arviz_summary": json.loads(summary.to_json()),
    }
    diag_path = SMC_OUTPUT_DIR / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics to {diag_path}")

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
    corner_path = SMC_OUTPUT_DIR / "corner.png"
    corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
    plt.close(corner_fig)
    print(f"Saved corner plot to {corner_path}")


if __name__ == "__main__":
    main()
