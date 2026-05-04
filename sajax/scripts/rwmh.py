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
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from model import (
    make_inference_fns,
    make_constrain_fn,
    plot_model,
    sample_initial_positions,
    plot_bestfit_lightcurve,
    plot_prior_posterior,
    compute_chi2,
    compute_lc_from_constrained,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    OBS_LIGHT_CURVE,
    LC_TRUE,
    TIMES,
    PRIOR_DISTRIBUTIONS,
)

RWMH_OUTPUT_DIR = OUTPUT_DIR / "rwmh"

NDIM = len(PARAM_NAMES)
NUM_BURNIN  = 10000
NUM_SAMPLES = 5000
NUM_CHAINS  = 10


def _unconstrained_prior_std(d) -> float:
    """Return the std of prior d in unconstrained (bijected) space analytically.

    Uniform(a, b): inverse bijection is logit → logistic distribution, std = π/√3.
    Normal(μ, σ) / LogNormal(μ, σ): bijection is identity / log, unconstrained std = σ.
    """
    from numpyro.distributions import Uniform, Normal, LogNormal, LogUniform
    if isinstance(d, (Uniform, LogUniform)):
        # Both have bounded support; biject_to applies a logistic transform,
        # so the unconstrained variable is logistic(0,1) with std = π/√3.
        return float(np.pi / np.sqrt(3.0))
    elif isinstance(d, (Normal, LogNormal)):
        return float(d.scale)
    raise TypeError(f"No analytical unconstrained std for {type(d).__name__}")


# Roberts, Gelman & Gilks (1997): σ_proposal = 2.38/√d × σ_min, where σ_min is the
# smallest prior std in unconstrained space across all parameters.
_UNC_PRIOR_STDS = {name: _unconstrained_prior_std(d) for name, d in PRIOR_DISTRIBUTIONS.items()}
_SIGMA_MIN = min(_UNC_PRIOR_STDS.values())
# STEP_SIZE = 2.38 / np.sqrt(NDIM) *_SIGMA_MIN
STEP_SIZE = _SIGMA_MIN

DIAG_STRIDE = 100
PLOT_STRIDE = 1000

_DIAG_PARAMS = [
    "spot_lat", "spot_long", "spot_size", "spot_flux",
    "p_rot", "planet_radius", "semimajor_axis", "P_orb",
]


def run_step_diagnostics(raw, constrain_fn, save_lcs=False, output_dir=None):
    """
    Iterate through the full sample trace (including burn-in) and print a
    per-step table of walker-mean parameters and reduced chi-squared.

    Saves an animated GIF of LC snapshots to output_dir/lc_evolution.gif
    every PLOT_STRIDE steps when save_lcs=True.

    Parameters
    ----------
    raw : ndarray, shape (NUM_STEPS, NUM_WALKERS, NDIM)
        Raw unconstrained samples straight from trace.samples.coordinates.
    """
    from io import BytesIO
    from PIL import Image

    n_steps, n_walkers, _ = raw.shape

    print(f"\n=== Step-by-Step Diagnostics  "
          f"(steps 0–{n_steps-1}, stride={DIAG_STRIDE}, {n_walkers} walkers) ===")
    print(f"Values are the walker ensemble mean in constrained space.\n")

    col_w = 13
    header = f"{'step':>5}  {'chi2_red':>9}  " + "  ".join(f"{p:>{col_w}}" for p in _DIAG_PARAMS)
    sep    = "=" * len(header)
    print(header)
    print(sep)

    frames = []

    for step_idx in range(0, n_steps, DIAG_STRIDE):
        mean_unc = jnp.array(raw[step_idx].mean(axis=0))
        c = constrain_fn({name: mean_unc[i] for i, name in enumerate(PARAM_NAMES)})

        # Diagnostic 1: check arccos argument validity
        b = float(c["impact_param"])
        a = float(c["semimajor_axis"])
        ratio = b / a
        if abs(ratio) > 1.0:
            print(f"  [DIAG] step {step_idx}: |b/a| = {abs(ratio):.6f} > 1 "
                  f"(b={b:.4f}, a={a:.4f}) — arccos will produce NaN inclination")

        chi2 = compute_chi2(c)

        # Diagnostic 2: NaN chi2 — dump key constrained values
        if np.isnan(chi2):
            nan_keys = ["impact_param", "semimajor_axis", "inclination",
                        "eccentricity", "planet_radius", "p_rot"]
            vals = ", ".join(f"{k}={float(c[k]):.6f}" for k in nan_keys if k in c)
            print(f"  [DIAG] step {step_idx}: NaN chi2 — {vals}")
            if step_idx == 0:
                print("  [DIAG] NaN at step 0: problem is in initialization, "
                      "not burn-in drift")

        param_str = "  ".join(f"{float(c[p]):>{col_w}.5f}" for p in _DIAG_PARAMS)
        print(f"{step_idx:>5}  {chi2:>9.4f}  {param_str}")

        if save_lcs and output_dir is not None and step_idx % PLOT_STRIDE == 0:
            lc_model = np.array(compute_lc_from_constrained(c))
            fig, (ax_lc, ax_res) = plt.subplots(
                2, 1, figsize=(10, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax_lc.scatter(TIMES, OBS_LIGHT_CURVE, s=3, color="orange", alpha=0.5, label="Obs")
            ax_lc.plot(TIMES, LC_TRUE, lw=1.5, color="steelblue", label="True")
            ax_lc.plot(TIMES, lc_model, lw=1.5, color="crimson", ls="--",
                       label=f"Step {step_idx} mean  χ²_r={chi2:.3f}")
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

            # Render figure to in-memory PIL Image
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGBA"))

    if frames and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gif_path = output_dir / "lc_evolution.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,   # ms per frame
            loop=0,         # loop forever
        )
        print(f"\nSaved LC evolution GIF ({len(frames)} frames) to {gif_path}")


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

    init_log_densities = [
        float(log_density_fn(jax.tree.map(lambda leaf: leaf[i], x0)))
        for i in range(NUM_CHAINS)
    ]
    n_nan = sum(np.isnan(d) for d in init_log_densities)
    n_neginf = sum(np.isneginf(d) for d in init_log_densities)
    if n_nan or n_neginf:
        _print(f"WARNING: {n_nan} chain(s) have NaN log density, "
               f"{n_neginf} have -inf — check the light curve computation for numerical issues.")

    # Build a custom proposal generator: proposed = current + sigma * N(0, I).
    # blackjax.mcmc.random_walk.normal does not add to the current position when
    # the position is a pytree of scalars (one dict entry per parameter), so we
    # flatten/unflatten manually.
    _x0_single = jax.tree.map(lambda leaf: leaf[0], x0)
    _, _unravel_fn = jax.flatten_util.ravel_pytree(_x0_single)

    def _proposal_generator(rng_key, position):
        flat, _ = jax.flatten_util.ravel_pytree(position)
        noise = STEP_SIZE * jax.random.normal(rng_key, shape=flat.shape)
        return _unravel_fn(flat + noise)

    kernel = blackjax.rmh(
        log_density_fn,
        proposal_generator=_proposal_generator,
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
        burnin_unc_positions = burnin_states.position
        post_burnin_states = jax.tree.map(lambda x: x[:, -1], burnin_states)
    else:
        burnin_unc_positions = None
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

    if burnin_unc_positions is not None:
        all_unc_steps = {
            name: np.concatenate([
                np.array(burnin_unc_positions[name]), # (NUM_CHAINS, NUM_BURNIN)
                np.array(unc_positions[name]),         # (NUM_CHAINS, NUM_SAMPLES)
            ], axis=1)
            for name in unc_positions
        }
    else:
        all_unc_steps = {name: np.array(unc_positions[name]) for name in unc_positions}
    # Convert {name: (NUM_CHAINS, total_steps)} -> (total_steps, NUM_CHAINS, NDIM)
    all_unc_arr = np.stack(
        [np.array(all_unc_steps[name]) for name in PARAM_NAMES], axis=-1
    ).transpose(1, 0, 2)
    run_step_diagnostics(all_unc_arr, constrain_fn, save_lcs=save_outputs,
                         output_dir=RWMH_OUTPUT_DIR)
    
    # --- Extract MAP sample ---
    # BlackJAX RMH state already carries logdensity at every step
    log_probs = np.array(all_states.logdensity)  # (NUM_CHAINS, NUM_SAMPLES)
    map_idx = np.unravel_index(np.argmax(log_probs), log_probs.shape)
    i_chain, i_step = map_idx
    _print(f"  MAP sample at chain={i_chain}, step={i_step}, "
           f"log_prob={log_probs[i_chain, i_step]:.4f}")

    # Extract the constrained parameter dict at the MAP index
    map_params = {name: float(np.array(constrained_positions[name])[i_chain, i_step])
                  for name in constrained_positions}

    # --- Diagnostics ---
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
    _print("  MAP parameter values:")
    for name in PARAM_NAMES:
        _print(f"    {name:20s}  MAP={map_params[name]:8.4f}  truth={GROUND_TRUTH[name]:8.4f}")
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
        "map_params": {name: float(map_params[name]) for name in PARAM_NAMES},
        "map_log_prob": float(log_probs[i_chain, i_step]),
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

    # --- Plots ---
    # 1. Trace plots
    plot_vars = PARAM_NAMES[:6]
    az.plot_trace(idata, var_names=plot_vars, combined=True)
    plt.tight_layout()
    plt.savefig(RWMH_OUTPUT_DIR / "traces_subset.png")
    plt.close()

    # 2. Corner plot — all parameters with truth / MAP / mean reference lines
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
    map_vals   = [float(map_params[p]) for p in PARAM_NAMES]
    mean_vals  = [float(posterior_means[i]) for i in range(n_params)]

    # Consistent colors
    color_truth = "steelblue"
    color_map   = "darkgreen"
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
                ax.axvline(map_vals[i],   color=color_map,   ls="--", lw=lw, alpha=0.8)
                ax.axvline(mean_vals[i],  color=color_mean,  ls=":",  lw=lw, alpha=0.8)

            elif i > j:
                ax.axvline(truth_vals[j], color=color_truth, ls="-",  lw=lw, alpha=0.5)
                ax.axvline(map_vals[j],   color=color_map,   ls="--", lw=lw, alpha=0.5)
                ax.axvline(mean_vals[j],  color=color_mean,  ls=":",  lw=lw, alpha=0.5)

                ax.axhline(truth_vals[i], color=color_truth, ls="-",  lw=lw, alpha=0.5)
                ax.axhline(map_vals[i],   color=color_map,   ls="--", lw=lw, alpha=0.5)
                ax.axhline(mean_vals[i],  color=color_mean,  ls=":",  lw=lw, alpha=0.5)

                ax.scatter(truth_vals[j], truth_vals[i], color=color_truth,
                           marker="s", s=marker_size, zorder=10, edgecolors="white", linewidths=0.5)
                ax.scatter(map_vals[j],   map_vals[i],   color=color_map,
                           marker="s", s=marker_size, zorder=10, edgecolors="white", linewidths=0.5)
                ax.scatter(mean_vals[j],  mean_vals[i],  color=color_mean,
                           marker="s", s=marker_size, zorder=10, edgecolors="white", linewidths=0.5)

    from matplotlib.lines import Line2D
    fig = axes[0, 0].get_figure()
    legend_handles = [
        Line2D([0], [0], color=color_truth, ls="-",  lw=lw, label="Truth"),
        Line2D([0], [0], color=color_map,   ls="--", lw=lw, label="MAP"),
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

    corner_path = RWMH_OUTPUT_DIR / "corner_all.png"
    fig.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved full corner plot to {corner_path}")

    # 3. Best-fit light curve — delegate to model.py
    plot_bestfit_lightcurve(constrained_positions, RWMH_OUTPUT_DIR, map_params=map_params)

    # 4. Per-parameter prior vs posterior plots
    plot_prior_posterior(constrained_positions, RWMH_OUTPUT_DIR)

    return diagnostics


if __name__ == "__main__":
    main()
