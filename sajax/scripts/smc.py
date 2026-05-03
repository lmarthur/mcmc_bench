"""
Run Adaptive Tempered SMC on the SAJAX planet+activity model and save outputs.

Works in constrained (physical) parameter space with a flat array representation.
The joint posterior is split into prior and likelihood for likelihood tempering:
  - At lambda=0 particles are drawn from the prior.
  - At lambda=1 they approximate the target posterior.
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

from model import (
    make_log_density,
    plot_model,
    compute_chi2,
    compute_lc_from_constrained,
    plot_bestfit_lightcurve,
    plot_prior_posterior,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    OBS_LIGHT_CURVE,
    LC_TRUE,
    TIMES,
    PRIOR_DISTRIBUTIONS,
)

SMC_OUTPUT_DIR = OUTPUT_DIR / "smc"

NDIM = len(PARAM_NAMES)

# The sajax forward model vmaps over time steps and stellar pixels internally.
# When the SMC vmaps this over all particles simultaneously, intermediate arrays
# reach shape [N_particles, n_times, n_pixels] which exhausts GPU VRAM at
# N_particles=2000 (~47 GB per intermediate tensor on this model).
# Keep N_particles <= 500 for the L40S (48 GB); 200 is a safe default.
NUM_PARTICLES = 250
TARGET_ESS = 0.75    # target ESS fraction for adaptive temperature step
NUM_MCMC_STEPS = 25   # RMH refreshment steps per SMC iteration
MAX_STEPS = 500       # safety cap on SMC iterations

# Multiplier on (2.38/sqrt(d)) * current_particle_std — the RGG-optimal scale
# for the current tempered posterior.  1.0 = theoretically optimal; tune down
# if acceptance is low (proposal overshoots the posterior).
SIGMA_FACTOR = 1.0

DIAG_STRIDE = 2       # print step diagnostics every N SMC iterations
PLOT_STRIDE = 10      # save LC snapshots every N SMC iterations

_DIAG_PARAMS = [
    "spot_lat", "spot_long", "spot_size", "spot_flux",
    "fac_lat", "fac_long", "fac_size", "fac_flux",
    "p_rot", "planet_radius", "inclination", "P_orb",
]


def sample_prior_particles(key, num_particles):
    """Return shape (num_particles, NDIM) array sampled from the joint prior."""
    positions = []
    for name in PARAM_NAMES:
        key, subkey = jax.random.split(key)
        positions.append(PRIOR_DISTRIBUTIONS[name].sample(subkey, sample_shape=(num_particles,)))
    return jnp.stack(positions, axis=-1)


def particle_to_constrained_dict(x):
    """Convert a single flat particle to a named dict including derived quantities."""
    c = {name: float(x[i]) for i, name in enumerate(PARAM_NAMES)}
    c["semimajor_axis"] = float(np.abs(c["impact_param"] / np.cos(np.deg2rad(c["inclination"]))))
    c["eccentricity"]  = float(c["ecc_h"] ** 2 + c["ecc_k"] ** 2)
    c["arg_periapsis"] = float(np.arctan2(c["ecc_k"], c["ecc_h"]))
    c["ldc_u1"] = float(2 * np.sqrt(c["ldc_q1"]) * c["ldc_q2"])
    c["ldc_u2"] = float(np.sqrt(c["ldc_q1"]) * (1 - 2 * c["ldc_q2"]))
    return c


def main(seed: int = 0, save_outputs: bool = True):
    rng_key = jax.random.PRNGKey(seed)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    # --- Prior / likelihood split for tempering ---
    def logprior_fn(x):
        total = jnp.array(0.0)
        for i, name in enumerate(PARAM_NAMES):
            total = total + PRIOR_DISTRIBUTIONS[name].log_prob(x[i])
        return total

    def loglikelihood_fn(x):
        return log_density_fn(x) - logprior_fn(x)

    # --- Adaptive RMH kernel: proposal_scale is a traced JAX argument so the
    #     JIT compiles once and reuses for every per-step scale value. ---
    _rmh_kernel = blackjax.rmh.build_kernel()

    @jax.jit
    def step_fn(rng_key, state, proposal_scale):
        """Single adaptive-tempered SMC step with dynamic diagonal proposal."""
        def _proposal(k, pos):
            return pos + jax.random.normal(k, shape=pos.shape) * proposal_scale

        def _rmh_step(k, s, logdensity_fn):
            return _rmh_kernel(k, s, logdensity_fn, transition_generator=_proposal)

        kernel = blackjax.adaptive_tempered_smc(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            mcmc_step_fn=_rmh_step,
            mcmc_init_fn=blackjax.rmh.init,
            mcmc_parameters={},
            resampling_fn=resampling.systematic,
            target_ess=TARGET_ESS,
            num_mcmc_steps=NUM_MCMC_STEPS,
        )
        return kernel.step(rng_key, state)

    prior_stds = jnp.array([
        float(jnp.sqrt(PRIOR_DISTRIBUTIONS[name].variance))
        for name in PARAM_NAMES
    ])

    # --- Initialize particles from prior ---
    init_key, loop_key = jax.random.split(rng_key)
    initial_positions = sample_prior_particles(init_key, NUM_PARTICLES)
    # Use a throwaway kernel just to get the initial SMC state object.
    _init_kernel = blackjax.adaptive_tempered_smc(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=lambda k, s, ld: s,  # placeholder — never called
        mcmc_init_fn=blackjax.rmh.init,
        mcmc_parameters={},
        resampling_fn=resampling.systematic,
        target_ess=TARGET_ESS,
        num_mcmc_steps=NUM_MCMC_STEPS,
    )
    state = _init_kernel.init(initial_positions)

    t0 = time.perf_counter()

    # --- SMC loop ---
    all_lambdas = [float(state.tempering_param)]
    all_log_nc = []
    all_ess = []
    all_max_w = []
    all_dlam = []
    all_jump = []
    num_steps = 0
    lc_plots_saved = 0

    if save_outputs:
        lc_dir = SMC_OUTPUT_DIR / "step_lcs"
        lc_dir.mkdir(parents=True, exist_ok=True)
    else:
        lc_dir = None

    col_w = 13
    diag_header = (
        f"{'step':>5}  {'lambda':>8}  {'d_lam':>8}  {'ESS':>6}  {'max_w':>6}  "
        f"{'jump':>8}  {'chi2_red':>9}  "
        + "  ".join(f"{p:>{col_w}}" for p in _DIAG_PARAMS)
    )
    diag_sep = "=" * len(diag_header)
    _print(f"\nRunning adaptive tempered SMC ({NUM_PARTICLES} particles, {NDIM} params)...")
    _print(f"\n=== Step-by-Step Diagnostics (stride={DIAG_STRIDE}) ===")
    _print("ESS=effective sample size, max_w=max particle weight, jump=mean L2 step\n")
    _print(diag_header)
    _print(diag_sep)

    while state.tempering_param < 1.0 and num_steps < MAX_STEPS:
        prev_particles = np.array(state.particles)
        prev_lam = float(state.tempering_param)

        # Adaptive proposal: RGG scale applied to current particle spread,
        # floored at 1% of per-parameter prior std to prevent collapse when
        # particles concentrate, capped at prior std so it only shrinks as λ→1.
        p_arr = jnp.array(state.particles)
        w_arr = jnp.array(state.weights)
        w_mean = (p_arr * w_arr[:, None]).sum(axis=0)
        w_var  = (w_arr[:, None] * (p_arr - w_mean) ** 2).sum(axis=0)
        current_std = jnp.sqrt(jnp.maximum(w_var, 1e-8))
        proposal_scale = SIGMA_FACTOR * (2.38 / jnp.sqrt(NDIM)) * jnp.minimum(
            jnp.maximum(current_std, 0.01 * prior_stds),
            prior_stds,
        )

        loop_key, subkey = jax.random.split(loop_key)
        state, info = step_fn(subkey, state, proposal_scale)
        lam = float(state.tempering_param)
        lnc = float(info.log_likelihood_increment)
        dlam = lam - prev_lam

        particles_np = np.array(state.particles)
        weights_np   = np.array(state.weights)
        ess   = float(1.0 / np.sum(weights_np ** 2))
        max_w = float(np.max(weights_np))
        jump  = float(np.mean(np.linalg.norm(particles_np - prev_particles, axis=1)))

        all_lambdas.append(lam)
        all_log_nc.append(lnc)
        all_ess.append(ess)
        all_max_w.append(max_w)
        all_dlam.append(dlam)
        all_jump.append(jump)
        num_steps += 1

        if num_steps % DIAG_STRIDE == 0 or lam >= 1.0:
            mean_x = (particles_np * weights_np[:, None]).sum(axis=0)
            c = particle_to_constrained_dict(mean_x)
            chi2 = compute_chi2(c)
            param_str = "  ".join(f"{float(c[p]):>{col_w}.5f}" for p in _DIAG_PARAMS)
            _print(
                f"{num_steps:>5}  {lam:>8.4f}  {dlam:>8.4f}  {ess:>6.1f}  {max_w:>6.4f}  "
                f"{jump:>8.4f}  {chi2:>9.4f}  {param_str}"
            )

        if lc_dir is not None and (num_steps % PLOT_STRIDE == 0 or lam >= 1.0):
            if num_steps % DIAG_STRIDE != 0:
                # c / chi2 not computed in the diag block above; compute now
                mean_x = (particles_np * weights_np[:, None]).sum(axis=0)
                c = particle_to_constrained_dict(mean_x)
                chi2 = compute_chi2(c)
            lc_model = np.array(compute_lc_from_constrained(c))
            fig, (ax_lc, ax_res) = plt.subplots(
                2, 1, figsize=(10, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=3, color="orange", alpha=0.5, label="Obs")
            ax_lc.plot(TIMES, LC_TRUE, lw=1.5, color="steelblue", label="True")
            ax_lc.plot(TIMES, lc_model, lw=1.5, color="crimson", ls="--",
                       label=f"Step {num_steps}  λ={lam:.4f}  χ²_r={chi2:.3f}")
            ax_lc.legend(frameon=False, fontsize=9)
            ax_lc.set_ylabel("Flux")
            ax_lc.spines["top"].set_visible(False)
            ax_lc.spines["right"].set_visible(False)

            res_ppm = (OBS_LIGHT_CURVE - lc_model) * 1e6
            ax_res.scatter(TIMES, res_ppm, s=3, color="orange", alpha=0.5)
            ax_res.axhline(0, color="crimson", lw=1, ls="--")
            ax_res.set_xlabel("Time [days]")
            ax_res.set_ylabel("Res. [ppm]")
            ax_res.spines["top"].set_visible(False)
            ax_res.spines["right"].set_visible(False)

            fig.tight_layout()
            fig.savefig(lc_dir / f"lc_step_{num_steps:05d}.png", dpi=100,
                        bbox_inches="tight")
            plt.close(fig)
            lc_plots_saved += 1

    if num_steps >= MAX_STEPS:
        _print(f"\n  WARNING: hit MAX_STEPS={MAX_STEPS} before lambda reached 1.")
    else:
        _print(f"\n  Completed in {num_steps} SMC steps.")

    if lc_dir is not None:
        _print(f"\n{lc_plots_saved} LC snapshot(s) saved to {lc_dir}/")

    # --- Collapse / mixing summary ---
    _print(f"\n=== Particle Health Summary ===")
    ess_arr    = np.array(all_ess)
    dlam_arr   = np.array(all_dlam)
    jump_arr   = np.array(all_jump)
    max_w_arr  = np.array(all_max_w)

    collapse_step = next((i for i, e in enumerate(all_ess) if e < 0.1 * NUM_PARTICLES), None)
    _print(f"  ESS:        min={ess_arr.min():.1f}  max={ess_arr.max():.1f}  final={ess_arr[-1]:.1f}")
    if collapse_step is not None:
        _print(f"  ESS first fell below 10% at step {collapse_step + 1}  (λ={all_lambdas[collapse_step + 1]:.4f})")
    _print(f"  Max weight: min={max_w_arr.min():.4f}  max={max_w_arr.max():.4f}  final={max_w_arr[-1]:.4f}")
    _print(f"  Jump dist:  min={jump_arr.min():.4f}  mean={jump_arr.mean():.4f}  max={jump_arr.max():.4f}")
    _print(f"  λ delta:    min={dlam_arr.min():.5f}  mean={dlam_arr.mean():.5f}  max={dlam_arr.max():.5f}")

    wall_time_s = time.perf_counter() - t0

    # --- Final particles and weights ---
    particles = np.array(state.particles)   # (NUM_PARTICLES, NDIM)
    weights   = np.array(state.weights)     # normalized

    final_ess   = float(1.0 / np.sum(weights ** 2))
    log_nc_total = float(np.sum(all_log_nc))

    # Each SMC step: NUM_MCMC_STEPS RMH evals + 2 weight-update evals per particle.
    total_log_density_evals = num_steps * NUM_PARTICLES * (NUM_MCMC_STEPS + 2)

    posterior_means = (particles * weights[:, None]).sum(axis=0)

    # --- Build constrained samples (resampled to equal weights) ---
    rng = np.random.default_rng(seed + 1)
    indices   = rng.choice(NUM_PARTICLES, size=NUM_PARTICLES, replace=True, p=weights)
    resampled = particles[indices]
    impact_param_r = resampled[:, PARAM_NAMES.index("impact_param")]
    inclination_r  = resampled[:, PARAM_NAMES.index("inclination")]
    ecc_h_r  = resampled[:, PARAM_NAMES.index("ecc_h")]
    ecc_k_r  = resampled[:, PARAM_NAMES.index("ecc_k")]
    ldc_q1_r = resampled[:, PARAM_NAMES.index("ldc_q1")]
    ldc_q2_r = resampled[:, PARAM_NAMES.index("ldc_q2")]
    constrained_samples = {PARAM_NAMES[i]: resampled[:, i] for i in range(NDIM)}
    constrained_samples["semimajor_axis"] = np.abs(impact_param_r / np.cos(np.deg2rad(inclination_r)))
    constrained_samples["eccentricity"]  = ecc_h_r ** 2 + ecc_k_r ** 2
    constrained_samples["arg_periapsis"] = np.arctan2(ecc_k_r, ecc_h_r)
    constrained_samples["ldc_u1"] = 2 * np.sqrt(ldc_q1_r) * ldc_q2_r
    constrained_samples["ldc_u2"] = np.sqrt(ldc_q1_r) * (1 - 2 * ldc_q2_r)

    # --- ArviZ summary (treat weighted particles as a single chain) ---
    posterior_dict = {PARAM_NAMES[i]: particles[None, :, i] for i in range(NDIM)}
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    if not save_outputs:
        _az_log.setLevel(logging.ERROR)
    idata = az.from_dict(posterior=posterior_dict)
    summary = az.summary(idata)
    _az_log.setLevel(_az_prev)

    total_bulk_ess = float(summary["ess_bulk"].sum())
    ess_per_logp_eval = total_bulk_ess / total_log_density_evals

    gt_array   = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  SMC steps:                {num_steps}")
    _print(f"  Final lambda:             {float(state.tempering_param):.6f}")
    _print(f"  Log normalizing constant: {log_nc_total:.3f}")
    _print(f"  Final ESS:                {final_ess:.1f} / {NUM_PARTICLES}")
    _print(f"  Total log-density evals:  {int(total_log_density_evals)}")
    _print(f"  Total bulk ESS:           {total_bulk_ess:.1f}")
    _print(f"  Bulk ESS per logp eval:   {ess_per_logp_eval:.4f}")
    _print()
    _print("  Parameter recovery (weighted mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    diagnostics = {
        "sampler": "AdaptiveTemperedSMC",
        "num_particles": NUM_PARTICLES,
        "target_ess": TARGET_ESS,
        "num_mcmc_steps_per_iter": NUM_MCMC_STEPS,
        "sigma_factor": float(SIGMA_FACTOR),
        "ndim": NDIM,
        "wall_time_s": float(wall_time_s),
        "num_smc_steps": num_steps,
        "final_tempering_param": float(state.tempering_param),
        "log_normalizing_constant": log_nc_total,
        "final_ess": final_ess,
        "total_log_density_evals": int(total_log_density_evals),
        "total_bulk_ess": total_bulk_ess,
        "bulk_ess_per_logp_eval": float(ess_per_logp_eval),
        "tempering_schedule": [float(x) for x in all_lambdas],
        "log_nc_increments": [float(x) for x in all_log_nc],
        "posterior_means": {name: float(pm) for name, pm in zip(PARAM_NAMES, posterior_means)},
        "ground_truth": {k: float(v) for k, v in GROUND_TRUTH.items()},
        "param_bias": {name: float(b) for name, b in zip(PARAM_NAMES, param_bias)},
        "arviz_summary": json.loads(summary.to_json()),
    }

    if save_outputs:
        SMC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(SMC_OUTPUT_DIR / "idata.nc"))
        diag_path = SMC_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {SMC_OUTPUT_DIR / 'idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    diagnostics.update({
        "ess_per_step": [float(x) for x in all_ess],
        "dlam_per_step": [float(x) for x in all_dlam],
        "jump_per_step": [float(x) for x in all_jump],
        "max_weight_per_step": [float(x) for x in all_max_w],
    })

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # Particle health panel: ESS, max weight, jump distance
    steps = np.arange(1, num_steps + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    axes[0].plot(steps, all_ess, lw=1)
    axes[0].axhline(0.1 * NUM_PARTICLES, color="crimson", lw=0.8, ls="--", label="10% ESS")
    axes[0].set_ylabel("ESS")
    axes[0].set_title("Effective Sample Size")
    axes[0].set_xlabel("SMC step")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, all_max_w, lw=1, color="darkorange")
    axes[1].axhline(1.0 / NUM_PARTICLES, color="steelblue", lw=0.8, ls="--", label="uniform weight")
    axes[1].set_ylabel("Max weight")
    axes[1].set_title("Max particle weight (collapse indicator)")
    axes[1].set_xlabel("SMC step")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, all_jump, lw=1, color="purple")
    axes[2].set_ylabel("Mean L2 jump")
    axes[2].set_title("Mean particle displacement (MCMC mixing)")
    axes[2].set_xlabel("SMC step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    health_path = SMC_OUTPUT_DIR / "particle_health.png"
    fig.savefig(health_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved particle health plot to {health_path}")

    # Weighted particle scatter (first two params) + tempering schedule
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(
        particles[:, 0], particles[:, 1],
        c=weights, s=5, alpha=0.6, cmap="viridis",
    )
    plt.colorbar(sc, ax=axes[0], label="weight")
    axes[0].set_xlabel(PARAM_NAMES[0])
    axes[0].set_ylabel(PARAM_NAMES[1])
    axes[0].set_title(f"SMC particles (N={NUM_PARTICLES})")

    axes[1].plot(all_lambdas, marker="o", markersize=2, lw=0.8)
    axes[1].set_xlabel("SMC step")
    axes[1].set_ylabel("lambda (tempering parameter)")
    axes[1].set_title("Adaptive tempering schedule")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    scatter_path = SMC_OUTPUT_DIR / "samples.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved samples plot to {scatter_path}")

    # Trace plot (particle index acts as pseudo-chain for visualization)
    az.plot_trace(idata, var_names=PARAM_NAMES[:6], figsize=(14, 10))
    plt.tight_layout()
    trace_path = SMC_OUTPUT_DIR / "traces_subset.png"
    plt.savefig(trace_path, dpi=150, bbox_inches="tight")
    plt.close()
    _print(f"Saved trace plot to {trace_path}")

    # Corner plot — all parameters with truth / MAP / mean reference lines
    n_params = len(PARAM_NAMES)
    az.rcParams["plot.max_subplots"] = n_params ** 2

    axes = az.plot_pair(
        idata,
        var_names=PARAM_NAMES,
        kind="kde",
        marginals=True,
        figsize=(24, 24),
    )

    # Reference values for each parameter
    truth_vals = [float(GROUND_TRUTH[p]) for p in PARAM_NAMES]
    mean_vals  = [float(posterior_means[i]) for i in range(n_params)]

    # Consistent colors
    color_truth = "steelblue"
    color_mean  = "crimson"
    lw = 1.5
    marker_size = 30

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if ax is None:
                continue

            if i == j:
                ax.axvline(truth_vals[i], color=color_truth, ls="-",  lw=lw, alpha=0.8)
                ax.axvline(mean_vals[i],  color=color_mean,  ls=":",  lw=lw, alpha=0.8)

            elif i > j:
                ax.axvline(truth_vals[j], color=color_truth, ls="-",  lw=lw, alpha=0.5)
                ax.axvline(mean_vals[j],  color=color_mean,  ls=":",  lw=lw, alpha=0.5)

                ax.axhline(truth_vals[i], color=color_truth, ls="-",  lw=lw, alpha=0.5)
                ax.axhline(mean_vals[i],  color=color_mean,  ls=":",  lw=lw, alpha=0.5)

                ax.scatter(truth_vals[j], truth_vals[i], color=color_truth,
                           marker="s", s=marker_size, zorder=10, edgecolors="white", linewidths=0.5)
                ax.scatter(mean_vals[j],  mean_vals[i],  color=color_mean,
                           marker="s", s=marker_size, zorder=10, edgecolors="white", linewidths=0.5)

    from matplotlib.lines import Line2D
    fig = axes[0, 0].get_figure()
    legend_handles = [
        Line2D([0], [0], color=color_truth, ls="-",  lw=lw, label="Truth"),
        Line2D([0], [0], color=color_mean,  ls=":",  lw=lw, label="Posterior mean"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
        frameon=True,
        framealpha=0.9,
        fontsize=12,
        title="Reference",
        title_fontsize=13,
    )

    corner_path = SMC_OUTPUT_DIR / "corner_all.png"
    fig.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved full corner plot to {corner_path}")

    # Best-fit light curve — delegate to model.py
    plot_bestfit_lightcurve(constrained_samples, SMC_OUTPUT_DIR, map_params=None)

    # Per-parameter prior vs posterior plots
    plot_prior_posterior(constrained_samples, SMC_OUTPUT_DIR)

    return diagnostics


if __name__ == "__main__":
    main()
