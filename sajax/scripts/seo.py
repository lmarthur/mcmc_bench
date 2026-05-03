"""
Run SEO (reversible) parallel tempering on the SAJAX planet+activity model and save outputs.

SEO uses a stochastic even-odd parity schedule for swap moves, which is
reversible (unlike DEO). At each step the parity (even or odd adjacent pairs)
is chosen randomly with equal probability.
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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pt_jax

from model import (
    make_log_density,
    plot_model,
    plot_bestfit_lightcurve,
    plot_prior_posterior,
    _call_sajax,
    compute_chi2,
    compute_lc_from_constrained,
    OUTPUT_DIR,
    PARAM_NAMES,
    GROUND_TRUTH,
    TIMES,
    OBS_LIGHT_CURVE,
    LC_TRUE,
    PRIOR_DISTRIBUTIONS,
)

SEO_OUTPUT_DIR = OUTPUT_DIR / "seo"

DIAG_STRIDE = 100
PLOT_STRIDE = 1000

_DIAG_PARAMS = [
    "spot_lat", "spot_long", "spot_size", "spot_flux",
    "fac_lat", "fac_long", "fac_size", "fac_flux",
    "p_rot", "planet_radius", "inclination", "P_orb",
]

# ===========================================================================
# Tunable parameters
# ===========================================================================

# Number of parallel chains (including the reference/hottest chain).
# More chains → smoother temperature ladder, better mode mixing, higher cost.
NUM_CHAINS = 15

# Number of warm-up (burn-in) steps discarded before collecting samples.
NUM_WARMUP = 500

# Number of posterior samples collected from the cold chain.
NUM_SAMPLES = 2000

# Local exploration kernel: "mala" (gradient-based) or "rwmh" (gradient-free).
KERNEL = "rwmh"

# Step sizes for the local kernel. Per-chain values are linearly interpolated
# from STEP_SIZE_HOT (β≈0) down to STEP_SIZE_COLD (β=1).
# RWMH: tune cold for ~23% acceptance. MALA: tune cold for ~57% acceptance.
# RWMH and MALA have very different optimal scales, so they're set separately.
#
# RWMH proposal is per-dimension preconditioned: sigma_i = α · (2.38/√d) · prior_std_i,
# following the Roberts-Gelman-Gilks (1997) optimal scaling rule. The values below
# are the global multiplier α (dimensionless). α=1 starts at the rule's prediction;
# tune cold to ~23% acceptance, hot larger to broaden hot-chain proposals.
STEP_SIZE_HOT_RWMH = 0.5
STEP_SIZE_COLD_RWMH = 0.002
# TODO: MALA step sizes likely need further tuning. With the current values the
# cold chain freezes near the mode (proposal scale too large for local geometry,
# and chains can land outside bounded prior support and get pinned at log_p=-inf).
# Likely need much smaller cold step size and/or per-dimension scaling and/or
# reparameterisation of bounded parameters before MALA is usable here.
STEP_SIZE_HOT_MALA = 0.1
STEP_SIZE_COLD_MALA = 0.01

# Base for the geometric temperature ladder: β_k = base^{-(N-1-k)}, β_{N-1}=1.
# Larger base → wider spacing between adjacent temperatures (higher swap rejection).
# Smaller base → more chains needed to span the same range.
ANNEALING_BASE = 1.4142135623730951  # sqrt(2)

# ===========================================================================


def get_initial_positions(key: jax.Array, num_chains: int) -> jnp.ndarray:
    """Sample each chain's starting position independently from the prior."""
    positions = []
    for name in PARAM_NAMES:
        key, subkey = jax.random.split(key)
        prior_key = name.lower() if name.startswith("LDC") else name
        samples = PRIOR_DISTRIBUTIONS[prior_key].sample(subkey, sample_shape=(num_chains,))
        positions.append(samples)
    return jnp.stack(positions, axis=-1)


def make_rwmh_kernel_generator(proposal_scale):
    """Return a pt_jax kernel_generator that uses sigma = alpha · proposal_scale.

    proposal_scale is a length-ndim vector setting the per-dimension RWMH proposal
    σ at α=1; the per-chain α is supplied by pt_jax via `params=`.
    """
    def rwmh_kernel_generator(log_p, alpha):
        # Build a custom proposal generator: proposed = current + alpha * proposal_scale * N(0, I).
        # Position is a flat array so no flatten/unflatten is needed (unlike rwmh.py where
        # positions are dicts and blackjax.mcmc.random_walk.normal fails on pytree scalars).
        def _proposal_generator(rng_key, position):
            noise = alpha * proposal_scale * jax.random.normal(rng_key, shape=position.shape)
            return position + noise

        rmh = blackjax.rmh(log_p, proposal_generator=_proposal_generator)

        def kernel(key, position):
            state = rmh.init(position)
            new_state, _ = rmh.step(key, state)
            return new_state.position

        return kernel

    return rwmh_kernel_generator


def mala_kernel_generator(log_p, step_size):
    mala = blackjax.mala(log_p, step_size)

    def kernel(key, position):
        state = mala.init(position)
        new_state, _ = mala.step(key, state)
        return new_state.position

    return kernel

def _to_physical_dict(param_vector):
    """Convert a flat parameter vector (indexed by PARAM_NAMES) to a dict
    including derived physical quantities needed by _call_sajax / compute_lc_from_constrained."""
    c = {name: float(param_vector[i]) for i, name in enumerate(PARAM_NAMES)}
    c["semimajor_axis"] = float(np.abs(c["impact_param"] / np.cos(np.deg2rad(c["inclination"]))))
    c["eccentricity"]  = float(c["ecc_h"] ** 2 + c["ecc_k"] ** 2)
    c["arg_periapsis"] = float(np.arctan2(c["ecc_k"], c["ecc_h"]))
    c["ldc_u1"] = float(2 * np.sqrt(c["ldc_q1"]) * c["ldc_q2"])
    c["ldc_u2"] = float(np.sqrt(c["ldc_q1"]) * (1 - 2 * c["ldc_q2"]))
    return c

def run_step_diagnostics(raw, save_lcs=False, output_dir=None):
    """
    Iterate through the cold-chain sample trace and print a per-step table of
    parameters and reduced chi-squared.

    Saves an animated GIF of LC snapshots to output_dir/lc_evolution.gif
    every PLOT_STRIDE steps when save_lcs=True.

    Parameters
    ----------
    raw : ndarray, shape (NUM_SAMPLES, NDIM)
        Constrained flat samples from the cold chain.
    """
    from io import BytesIO
    from PIL import Image

    n_samples, _ = raw.shape

    print(f"\n=== Step-by-Step Diagnostics  "
          f"(steps 0–{n_samples-1}, stride={DIAG_STRIDE}) ===")
    print(f"Values are the constrained parameter values.\n")

    col_w = 13
    header = f"{'step':>5}  {'chi2_red':>9}  " + "  ".join(f"{p:>{col_w}}" for p in _DIAG_PARAMS)
    sep    = "=" * len(header)
    print(header)
    print(sep)

    frames = []

    for step_idx in range(0, n_samples, DIAG_STRIDE):
        c = _to_physical_dict(raw[step_idx])

        chi2 = compute_chi2(c)
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


def main(seed: int = 0, save_outputs: bool = True):
    init_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))
    _print = print if save_outputs else lambda *a, **kw: None

    ndim = len(PARAM_NAMES)

    # --- Model ---
    log_density_fn = make_log_density()
    if save_outputs:
        plot_model(filename="sajax_ground_truth.png")

    t0 = time.perf_counter()

    # --- Reference distribution: the joint prior (likelihood-tempering path) ---
    prior_keys = [name.lower() if name.startswith("LDC") else name for name in PARAM_NAMES]

    def log_ref(x):
        total = jnp.array(0.0)
        for i, key in enumerate(prior_keys):
            total = total + PRIOR_DISTRIBUTIONS[key].log_prob(x[i])
        return total

    # --- Temperature schedule ---
    betas = pt_jax.annealing.annealing_exponential(NUM_CHAINS, base=ANNEALING_BASE)

    # --- Build kernels ---
    if KERNEL == "mala":
        step_size_hot, step_size_cold = STEP_SIZE_HOT_MALA, STEP_SIZE_COLD_MALA
        kernel_generator = mala_kernel_generator
        proposal_scale = None
    elif KERNEL == "rwmh":
        step_size_hot, step_size_cold = STEP_SIZE_HOT_RWMH, STEP_SIZE_COLD_RWMH
        # Per-dimension RWMH proposal scale: (2.38/√d) · prior_std_i.
        # The per-chain α multiplies this vector to form the proposal sigma.
        prior_stds = jnp.array([
            float(jnp.sqrt(PRIOR_DISTRIBUTIONS[
                name.lower() if name.startswith("LDC") else name
            ].variance))
            for name in PARAM_NAMES
        ])
        proposal_scale = (2.38 / jnp.sqrt(ndim)) * prior_stds
        kernel_generator = make_rwmh_kernel_generator(proposal_scale)
    else:
        raise ValueError(f"Unknown KERNEL {KERNEL!r}; expected 'mala' or 'rwmh'.")
    step_sizes = jnp.linspace(step_size_hot, step_size_cold, NUM_CHAINS)

    K_ind = pt_jax.kernels.generate_independent_annealed_kernel(
        log_prob=log_density_fn,
        log_ref=log_ref,
        annealing_schedule=betas,
        kernel_generator=kernel_generator,
        params=step_sizes,
    )
    K_seo = pt_jax.swap.generate_seo_extended_kernel(
        log_prob=log_density_fn,
        log_ref=log_ref,
        annealing_schedule=betas,
    )

    # --- Initialise chains from prior ---
    x0 = get_initial_positions(init_key, NUM_CHAINS)

    # --- Diagnostic: initial log-density per chain (cold chain is index -1) ---
    init_log_targets = np.array([float(log_density_fn(x0[i])) for i in range(NUM_CHAINS)])
    init_log_refs    = np.array([float(log_ref(x0[i]))         for i in range(NUM_CHAINS)])
    _print("Initial log-density per chain (β increases left→right):")
    _print(f"  {'chain':>5}  {'beta':>8}  {'log_target':>14}  {'log_ref':>14}  {'log_likelihood':>14}")
    for i in range(NUM_CHAINS):
        ll = init_log_targets[i] - init_log_refs[i]
        _print(f"  {i:>5}  {float(betas[i]):>8.4f}  {init_log_targets[i]:>14.4e}"
               f"  {init_log_refs[i]:>14.4e}  {ll:>14.4e}")

    # --- Diagnostic: RWMH proposal check on cold chain (5 test proposals) ---
    if KERNEL == "rwmh":
        _print("\nRWMH proposal diagnostic (cold chain, 5 test proposals):")
        diag_key, sample_key = jax.random.split(sample_key)
        cold_x0 = x0[-1]
        _print(f"  Cold chain x0 log_density : {float(log_density_fn(cold_x0)):+.4e}")
        _print(f"  Cold chain x0 log_ref     : {float(log_ref(cold_x0)):+.4e}")
        _print(f"  {'trial':>5}  {'log_target':>14}  {'log_ref':>14}  {'log_MH_ratio':>14}  values[:4]")
        for trial in range(5):
            diag_key, subkey = jax.random.split(diag_key)
            noise = jax.random.normal(subkey, shape=cold_x0.shape)
            proposal = cold_x0 + STEP_SIZE_COLD_RWMH * proposal_scale * noise
            prop_lp  = float(log_density_fn(proposal))
            prop_ref = float(log_ref(proposal))
            log_mh   = prop_lp - float(log_density_fn(cold_x0))
            vals_str = "  ".join(f"{float(v):.4f}" for v in np.array(proposal)[:4])
            _print(f"  {trial:>5}  {prop_lp:>14.4e}  {prop_ref:>14.4e}"
                   f"  {log_mh:>14.4e}  [{vals_str} ...]")

    # --- Diagnostic: vmap consistency check for log_density_fn ---
    _print("\nVmap consistency check for log_density_fn:")
    ld_vmap = jax.vmap(log_density_fn)(x0)
    _print(f"  {'chain':>5}  {'sequential':>14}  {'vmap':>14}  {'match':>6}")
    for i in range(NUM_CHAINS):
        seq_val  = init_log_targets[i]
        vmap_val = float(ld_vmap[i])
        match    = "OK" if abs(seq_val - vmap_val) < 1.0 else "MISMATCH"
        _print(f"  {i:>5}  {seq_val:>14.4e}  {vmap_val:>14.4e}  {match:>6}")

    # --- Diagnostic: SEO swap kernel on initial positions ---
    _print("\nDirect SEO swap kernel test on x0:")
    swap_diag_key = jax.random.PRNGKey(99)
    _, swap_rr_test = K_seo(swap_diag_key, x0)
    _print(f"  {'pair':>6}  {'beta_i':>8}  {'beta_j':>8}  {'rejection_rate':>14}")
    for i, rr in enumerate(np.array(swap_rr_test)):
        _print(f"  {i:>3}<->{i+1:>2}  {float(betas[i]):>8.4f}  {float(betas[i+1]):>8.4f}  {rr:>14.4f}")

    # --- Diagnostic: local kernel on initial positions (one step, all chains) ---
    _print("\nLocal kernel one-step test on x0:")
    local_diag_key = jax.random.PRNGKey(42)
    x_after_local = K_ind(local_diag_key, x0)
    moved = np.any(np.array(x_after_local) != np.array(x0), axis=1)
    _print(f"  Chains that moved: {np.where(moved)[0].tolist()} / {NUM_CHAINS}")
    _print(f"  Cold-chain position unchanged: {not bool(moved[-1])}")

    # --- Run SEO sampling loop ---
    _print(f"\nRunning SEO ({KERNEL.upper()} local kernel, {NUM_CHAINS} chains, "
           f"{NUM_SAMPLES} samples, {NUM_WARMUP} warmup, {ndim} params)...")

    samples, rejection_rates = pt_jax.swap.seo_sampling_loop(
        key=sample_key,
        x0=x0,
        kernel_local=K_ind,
        kernel_seo=K_seo,
        n_samples=NUM_SAMPLES,
        warmup=NUM_WARMUP,
    )
    # samples:         (NUM_SAMPLES, NUM_CHAINS, NDIM)
    # rejection_rates: (NUM_SAMPLES, NUM_CHAINS - 1)

    wall_time_s = time.perf_counter() - t0

    # Cold chain (β=1, target distribution) is the last chain.
    cold_samples = np.array(samples[:, -1, :])  # (NUM_SAMPLES, NDIM)
    mean_swap_rejection = np.array(rejection_rates.mean(axis=0))  # (NUM_CHAINS - 1,)

    # --- Diagnostic: did the cold chain move? Compare first vs last sample ---
    _print("\nCold-chain movement check (first vs last sample):")
    _print(f"  {'param':20s}  {'first':>12s}  {'last':>12s}  {'|Δ|':>12s}")
    for i, name in enumerate(PARAM_NAMES):
        first = float(cold_samples[0, i])
        last = float(cold_samples[-1, i])
        _print(f"  {name:20s}  {first:12.6f}  {last:12.6f}  {abs(last - first):12.6f}")
    if np.allclose(cold_samples[0], cold_samples[-1]):
        _print("  WARNING: cold chain first and last samples are identical — chain never moved.")

    # Fraction of unique samples in the cold chain (low → chain is stuck)
    unique_rows = np.unique(cold_samples, axis=0)
    frac_unique = len(unique_rows) / NUM_SAMPLES
    _print(f"\n  Unique cold-chain samples: {len(unique_rows)}/{NUM_SAMPLES} ({frac_unique:.1%})")

    if save_outputs:
        run_step_diagnostics(cold_samples, save_lcs=True, output_dir=SEO_OUTPUT_DIR)

    # --- Build constrained dicts for all cold-chain samples ---
    # This is the "constrained_samples" dict: {name: array of shape (NUM_SAMPLES,)}
    constrained_samples = {name: cold_samples[:, i] for i, name in enumerate(PARAM_NAMES)}
    # Add derived quantities so plot_bestfit_lightcurve / compute_lc_from_constrained can use them
    impact_param_arr = cold_samples[:, PARAM_NAMES.index("impact_param")]
    inclination_arr  = cold_samples[:, PARAM_NAMES.index("inclination")]
    constrained_samples["semimajor_axis"] = np.abs(impact_param_arr / np.cos(np.deg2rad(inclination_arr)))
    ecc_h = cold_samples[:, PARAM_NAMES.index("ecc_h")]
    ecc_k = cold_samples[:, PARAM_NAMES.index("ecc_k")]
    constrained_samples["eccentricity"]  = ecc_h ** 2 + ecc_k ** 2
    constrained_samples["arg_periapsis"] = np.arctan2(ecc_k, ecc_h)
    q1 = cold_samples[:, PARAM_NAMES.index("ldc_q1")]
    q2 = cold_samples[:, PARAM_NAMES.index("ldc_q2")]
    constrained_samples["ldc_u1"] = 2 * np.sqrt(q1) * q2
    constrained_samples["ldc_u2"] = np.sqrt(q1) * (1 - 2 * q2)

    # --- Diagnostics ---
    # MALA: one gradient eval per chain per local step. RWMH: one log-density eval.
    # Swap moves cost only log-density evals in either case.
    total_local_evals = NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)
    cost_label = "gradient_evals" if KERNEL == "mala" else "log_density_evals"

    posterior_dict = {PARAM_NAMES[i]: cold_samples[None, :, i] for i in range(ndim)}
    _az_log = logging.getLogger("arviz")
    _az_prev = _az_log.level
    _az_log.setLevel(logging.ERROR)  # suppress shape-validation warning (single cold chain)
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats={"swap_rejection_rate": np.mean(mean_swap_rejection)},
    )
    summary = az.summary(idata)
    _az_log.setLevel(_az_prev)

    total_bulk_ess = float(summary["ess_bulk"].sum())
    ess_per_local_eval = total_bulk_ess / total_local_evals

    gt_array = np.array([GROUND_TRUTH[p] for p in PARAM_NAMES])
    posterior_means = cold_samples.mean(axis=0)
    param_bias = posterior_means - gt_array

    _print("\n=== Diagnostics ===")
    _print(f"  Kernel:        {KERNEL.upper()}")
    _print(f"  Total {cost_label}: {int(total_local_evals)}")
    _print()
    _print("  Mean per-pair swap rejection rates:")
    for i, r in enumerate(mean_swap_rejection):
        _print(f"    chain {i} <-> {i+1}  (beta {betas[i]:.4f} <-> {betas[i+1]:.4f}): {r:.3f}")
    _print()
    _print("  Parameter recovery (posterior mean vs ground truth):")
    for name, pm, gt, bias in zip(PARAM_NAMES, posterior_means, gt_array, param_bias):
        _print(f"    {name:20s}  mean={pm:8.4f}  truth={gt:8.4f}  bias={bias:+.4f}")
    _print()
    _print("  ArviZ summary (R-hat, ESS, MCSE):")
    _print(summary.to_string())
    _print()
    _print(f"  Total bulk ESS: {total_bulk_ess:.1f}")
    _print(f"  Bulk ESS per {cost_label.replace('_', '-')}: {ess_per_local_eval:.4f}")
    _print(f"\n  Wall-clock time: {wall_time_s:.2f}s")

    # --- Results ---
    diagnostics = {
        "sampler": "SEO_ParallelTempering",
        "kernel": KERNEL,
        "num_chains": NUM_CHAINS,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "ndim": ndim,
        "step_size_hot": float(step_size_hot),
        "step_size_cold": float(step_size_cold),
        "annealing_base": float(ANNEALING_BASE),
        "reference_distribution": "joint_prior",
        "beta_schedule": np.array(betas).tolist(),
        "step_sizes": np.array(step_sizes).tolist(),
        "rwmh_proposal_scale": (
            np.array(proposal_scale).tolist() if proposal_scale is not None else None
        ),
        "mean_swap_rejection_rates": mean_swap_rejection.tolist(),
        cost_label: int(total_local_evals),
        "wall_time_s": float(wall_time_s),
        "total_bulk_ess": float(total_bulk_ess),
        f"bulk_ess_per_{cost_label}": float(ess_per_local_eval),
        "posterior_means": {name: float(pm) for name, pm in zip(PARAM_NAMES, posterior_means)},
        "ground_truth": {k: float(v) for k, v in GROUND_TRUTH.items()},
        "param_bias": {name: float(b) for name, b in zip(PARAM_NAMES, param_bias)},
        "arviz_summary": json.loads(summary.to_json()),
    }

    if save_outputs:
        SEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(SEO_OUTPUT_DIR / "idata.nc"))
        diag_path = SEO_OUTPUT_DIR / "diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        _print(f"\nSaved idata to {SEO_OUTPUT_DIR / 'idata.nc'}")
        _print(f"Saved diagnostics to {diag_path}")

    if not save_outputs:
        return diagnostics

    # --- Plots ---

    # 1. Trace plots for first 6 parameters
    axes = az.plot_trace(idata, var_names=PARAM_NAMES[:6], figsize=(14, 10))
    plt.tight_layout()
    trace_path = SEO_OUTPUT_DIR / "traces_subset.png"
    plt.savefig(trace_path, dpi=150, bbox_inches="tight")
    plt.close()
    _print(f"Saved trace plot to {trace_path}")

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

    corner_path = SEO_OUTPUT_DIR / "corner_all.png"
    fig.savefig(corner_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved full corner plot to {corner_path}")

    # 3. Best-fit light curve — delegate to model.py
    plot_bestfit_lightcurve(constrained_samples, SEO_OUTPUT_DIR, map_params=None)

    # 4. Per-parameter prior vs posterior plots
    plot_prior_posterior(constrained_samples, SEO_OUTPUT_DIR)

    # 5. Per-pair swap rejection rates
    pair_labels = [f"{betas[i]:.3f}↔{betas[i+1]:.3f}" for i in range(NUM_CHAINS - 1)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(NUM_CHAINS - 1), mean_swap_rejection, color="steelblue", alpha=0.8)
    ax.set_xticks(range(NUM_CHAINS - 1))
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean rejection rate")
    ax.set_xlabel("Adjacent chain pair (β values)")
    ax.set_title("SEO swap rejection rates per adjacent pair")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    swap_path = SEO_OUTPUT_DIR / "swap_rates.png"
    fig.savefig(swap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _print(f"Saved swap rates plot to {swap_path}")

    return diagnostics


if __name__ == "__main__":
    main()
