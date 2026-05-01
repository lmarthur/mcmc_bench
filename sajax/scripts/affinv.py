"""
Run Affine Invariant MCMC on the sajax planet+activity model.
"""

import json
import sys
import time
import warnings
from pathlib import Path

# Add src to path if necessary
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import arviz as az
import emcee_jax
import jax
import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Import our specific model components
from model import (
    make_inference_fns,
    make_constrain_fn,
    plot_model,
    sample_initial_positions,
    plot_bestfit_lightcurve,
    compute_chi2,
    compute_lc_from_constrained,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    OBS_LIGHT_CURVE,
    LC_TRUE,
    TIMES,
    SIGMA_NOISE,
    STATIC_MODEL,
)

AFFINV_OUTPUT_DIR = OUTPUT_DIR / "affinv"

NUM_BURNIN = 500
NUM_SAMPLES = 1000
NUM_WALKERS = 64
NDIM = len(PARAM_NAMES)

# Diagnostic stride controls — print a table row every DIAG_STRIDE steps,
# save an LC snapshot every PLOT_STRIDE steps.
DIAG_STRIDE = 10
PLOT_STRIDE = 100



_DIAG_PARAMS = [
    "spot_lat", "spot_long", "spot_size", "spot_flux",
    "fac_lat", "fac_long", "fac_size", "fac_flux",
    "p_rot", "planet_radius", "inclination", "P_orb",
]


def run_step_diagnostics(raw, constrain_fn, unravel_fn, save_lcs=False, output_dir=None):
    """
    Iterate through the full sample trace (including burn-in) and print a
    per-step table of walker-mean parameters and reduced chi-squared.

    Saves an LC snapshot to output_dir/step_lcs/ every PLOT_STRIDE steps when
    save_lcs=True.

    Parameters
    ----------
    raw : ndarray, shape (NUM_STEPS, NUM_WALKERS, NDIM)
        Raw unconstrained samples straight from trace.samples.coordinates.
    """
    n_steps, n_walkers, _ = raw.shape

    print(f"\n=== Step-by-Step Diagnostics  "
          f"(steps 0–{n_steps-1}, stride={DIAG_STRIDE}, {n_walkers} walkers) ===")
    print(f"Values are the walker ensemble mean in constrained space.\n")

    col_w = 13
    header = f"{'step':>5}  {'chi2_red':>9}  " + "  ".join(f"{p:>{col_w}}" for p in _DIAG_PARAMS)
    sep    = "=" * len(header)
    print(header)
    print(sep)

    if save_lcs and output_dir is not None:
        lc_dir = output_dir / "step_lcs"
        lc_dir.mkdir(parents=True, exist_ok=True)
    else:
        lc_dir = None

    for step_idx in range(0, n_steps, DIAG_STRIDE):
        mean_unc = jnp.array(raw[step_idx].mean(axis=0))
        c = constrain_fn(unravel_fn(mean_unc))

        chi2 = compute_chi2(c)
        param_str = "  ".join(f"{float(c[p]):>{col_w}.5f}" for p in _DIAG_PARAMS)
        print(f"{step_idx:>5}  {chi2:>9.4f}  {param_str}")

        if lc_dir is not None and step_idx % PLOT_STRIDE == 0:
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
            fig.savefig(lc_dir / f"lc_step_{step_idx:05d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    if lc_dir is not None:
        print(f"\nLC snapshots saved to {lc_dir}/")


def main(seed=0, save_outputs=True):
    init_key, state_key, sample_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    _print = print if save_outputs else lambda *a, **kw: None

    # --- Model ---
    log_density_fn, _, init_z = make_inference_fns(init_key)
    constrain_fn = make_constrain_fn()
    _, unravel_fn = jax.flatten_util.ravel_pytree(init_z)
    log_density_flat = lambda x: log_density_fn(unravel_fn(x))

    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # --- Initialize walkers ---
    coords = sample_initial_positions(init_key, NUM_WALKERS, return_flat=True)

    n_show = min(8, NUM_WALKERS)
    coords_constrained = jax.vmap(lambda x: constrain_fn(unravel_fn(x)))(coords[:n_show])
    _print(f"\nInitial walker positions (constrained space, first {n_show} of {NUM_WALKERS}):")
    _print(f"  {'param':20s}  " + "  ".join(f"walker{i:02d}" for i in range(n_show)))
    for name in PARAM_NAMES:
        vals = np.array(coords_constrained[name])
        _print(f"  {name:20s}  " + "  ".join(f"{v:8.4f}" for v in vals))

    def compute_chi2(constrained: dict, model_dict: dict = STATIC_MODEL) -> float:
        """Reduced chi-squared (obs vs model) for a set of constrained parameters."""
        # Ensure all derived quantities are present
        c = dict(constrained)
        if "eccentricity" not in c:
            c["eccentricity"] = c["ecc_h"]**2 + c["ecc_k"]**2
        if "arg_periapsis" not in c:
            c["arg_periapsis"] = jnp.arctan2(c["ecc_k"], c["ecc_h"])
        if "ldc_u1" not in c:
            c["ldc_u1"] = 2 * jnp.sqrt(c["ldc_q1"]) * c["ldc_q2"]
        if "ldc_u2" not in c:
            c["ldc_u2"] = jnp.sqrt(c["ldc_q1"]) * (1 - 2 * c["ldc_q2"])
        
        lc = compute_lc_from_constrained(c, model_dict)
        n = len(TIMES)
        return float(jnp.sum(((jnp.array(OBS_LIGHT_CURVE) - lc) / SIGMA_NOISE) ** 2) / n)
    chi2_truth = compute_chi2(GROUND_TRUTH)
    print(f"χ² at ground truth: {chi2_truth}")

    # --- Initialize sampler ---
    sampler = emcee_jax.EnsembleSampler(log_density_flat)
    state = sampler.init(state_key, coords)

    # --- Run chains ---
    _print(f"Sampling sajax model ({NUM_SAMPLES} steps, {NUM_WALKERS} walkers, {NDIM} params)...")
    trace = sampler.sample_parallel(sample_key, state, NUM_SAMPLES, progress=save_outputs)

    # Reshape: (NUM_STEPS, NUM_WALKERS, NDIM) -> (NUM_WALKERS, NUM_SAMPLES, NDIM)
    raw = np.asarray(trace.samples.coordinates)

    # --- Step-by-step diagnostics (full trace, including burn-in) ---
    run_step_diagnostics(raw, constrain_fn, unravel_fn,
                         save_lcs=save_outputs, output_dir=AFFINV_OUTPUT_DIR)

    samples_unc = raw.transpose(1, 0, 2)
    samples_unc = samples_unc[:, NUM_BURNIN:, :]

    # Convert unconstrained samples to constrained space
    flat_unc = samples_unc.reshape(-1, samples_unc.shape[-1])
    constrained = jax.vmap(lambda x: constrain_fn(unravel_fn(x)))(flat_unc)
    # Split walkers into 2 equal groups to satisfy ArviZ's minimum-2-chains
    # requirement for R-hat. Walkers within each half are combined into draws.
    n_post = samples_unc.shape[1]
    half = NUM_WALKERS // 2
    cold_samples = {name: np.array(constrained[name]).reshape(2, half * n_post)
                    for name in PARAM_NAMES}

    # --- Diagnostics ---
    accepted = np.asarray(trace.sample_stats['accept_prob'])
    acceptance = float(accepted.mean())
    total_log_density_evals = NUM_WALKERS * NUM_SAMPLES

    posterior_dict = {name: cold_samples[name] for name in PARAM_NAMES}
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={"acceptance_rate": accepted[NUM_BURNIN:, :].T.reshape(2, half * n_post)},
    )

    summary = az.summary(idata)
    total_bulk_ess = summary["ess_bulk"].sum()
    ess_per_logp_eval = total_bulk_ess / total_log_density_evals

    _print("\n=== Diagnostics ===")
    _print(f"  Mean acceptance rate:          {acceptance:.3f}")
    _print(f"  Total log-density evaluations: {int(total_log_density_evals)}")
    _print("\n  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print(f"\n  Total Bulk ESS per log-density eval: {ess_per_logp_eval:.4f}")

    wall_time_s = time.perf_counter() - t0

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = np.array([np.array(cold_samples[p]).mean() for p in PARAM_NAMES])
    param_bias = posterior_means - gt_array

    _print("\n  Parameter recovery (posterior mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    diagnostics = {
        "sampler": "AffineInvariantEnsemble",
        "num_walkers": NUM_WALKERS,
        "num_burnin": NUM_BURNIN,
        "num_samples": NUM_SAMPLES,
        "ndim": NDIM,
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

    # --- Save Results ---
    if save_outputs:
        AFFINV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(AFFINV_OUTPUT_DIR / "sajax_idata.nc"))
        diag_path = AFFINV_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {AFFINV_OUTPUT_DIR / 'sajax_idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

        # 1. Trace Plots (subset for readability)
        plot_vars = PARAM_NAMES[:6]
        az.plot_trace(idata, var_names=plot_vars)
        plt.tight_layout()
        plt.savefig(AFFINV_OUTPUT_DIR / "traces_subset.png")
        plt.close()

        # 2. Corner plot — all parameters
        az.rcParams["plot.max_subplots"] = len(PARAM_NAMES) ** 2
        az.plot_pair(
            idata,
            var_names=PARAM_NAMES,
            kind="kde",
            marginals=True,
            figsize=(24, 24),
        )
        plt.savefig(AFFINV_OUTPUT_DIR / "corner_all.png", dpi=120, bbox_inches="tight")
        plt.close()

        # 3. Best-fit light curve using posterior mean
        plot_bestfit_lightcurve(constrained, AFFINV_OUTPUT_DIR)

    return diagnostics


if __name__ == "__main__":
    main()
