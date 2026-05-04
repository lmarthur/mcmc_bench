"""
Scaling study: parameter recovery vs. wall-clock time across algorithms and effort levels.

Runs repeated trials of each inference algorithm over a range of effort levels,
where effort is defined by a shared log-density-equivalent (LDE) budget.

Each algorithm back-computes its native effort parameter (NUM_SAMPLES,
NUM_PARTICLES, NUM_LIVE_POINTS) from the LDE budget using its cost model:

  RWMH       :  LDE = NUM_CHAINS   x (NUM_BURNIN  + NUM_SAMPLES)
  Affine Inv :  LDE = NUM_WALKERS  x NUM_SAMPLES  (burn-in is internal)
  SMC        :  LDE = EST_SMC_STEPS x NUM_PARTICLES x (NUM_MCMC_STEPS + 2)
  NS         :  LDE ≈ NUM_LIVE_POINTS x evals_per_live_point
  DEO / SEO  :  LDE = NUM_CHAINS  x (NUM_WARMUP  + NUM_SAMPLES)

Metrics:
  - Normalised parameter MAE: mean over parameters of |posterior_mean - truth| / prior_std
  - Per-parameter bias
  - ESS (where available)

Saves all raw per-trial results to JSON so plots can be regenerated without
re-running.

Usage:
    python scaling_study_compute.py                          # run trials if needed, then plot
    python scaling_study_compute.py --force                  # re-run all trials
    python scaling_study_compute.py --plot-only              # regenerate plots from saved JSON
    python scaling_study_compute.py --algorithms rwmh smc    # run/plot a subset of algorithms
"""

import argparse
import contextlib
import importlib
import io
import json
import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Suppress TF C++ backend noise before any tensorflow_probability import (used by ns.py)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import OUTPUT_DIR, PARAM_NAMES, GROUND_TRUTH, PRIOR_DISTRIBUTIONS

SCRIPTS_DIR     = Path(__file__).parent
SCALING_OUT_DIR = OUTPUT_DIR / "scaling_compute"
RESULTS_PATH    = SCALING_OUT_DIR / "results.json"

# ---------------------------------------------------------------------------
# Compute prior standard deviations for normalisation
# ---------------------------------------------------------------------------
# Used to normalise parameter recovery errors so they are dimensionless
# and comparable across parameters with very different physical scales.

def _prior_std(name):
    """Standard deviation of the prior distribution for parameter `name`."""
    d = PRIOR_DISTRIBUTIONS[name]
    return float(np.sqrt(float(d.variance)))

PRIOR_STDS = {name: _prior_std(name) for name in PARAM_NAMES}

# ---------------------------------------------------------------------------
# Shared oracle budget levels (log-density equivalents)
# ---------------------------------------------------------------------------
# Every algorithm receives the same LDE budget at each level.  The back-
# computation translates this into the algorithm's native effort parameter.

LOGP_BUDGETS = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]

NUM_TRIALS = 5  # seeds 0 .. NUM_TRIALS-1

# ---------------------------------------------------------------------------
# Pilot estimates for unknowns in the cost models
# ---------------------------------------------------------------------------
# These are rough estimates from default-setting pilot runs.  The actual oracle
# consumption is recorded per trial so any deviation is transparent.
#
# To update these from your own pilot runs, check:
#   SMC:   diagnostics.json -> num_smc_steps
#   NS:    diagnostics.json -> total_likelihood_evals / num_live_points

SMC_EST_NUM_STEPS       = 15     # typical adaptive tempering steps

NS_EVALS_PER_LIVE_POINT = 715    # total_likelihood_evals / NUM_LIVE_POINTS

# ---------------------------------------------------------------------------
# Fixed (non-effort) hyperparameters per algorithm
# ---------------------------------------------------------------------------
# These mirror the defaults in each script but are stated explicitly so the
# back-computation is self-contained.
#
# The effort parameter (NUM_SAMPLES, NUM_PARTICLES, or NUM_LIVE_POINTS) is
# computed from the budget; everything else is fixed.

ALGO_FIXED = {
    "rwmh":   {"NUM_CHAINS": 4,   "NUM_BURNIN": 1000},

    "affinv": {"NUM_WALKERS": 52, "NUM_BURNIN": 1000},

    "smc":    {"NUM_MCMC_STEPS": 25, "TARGET_ESS": 0.75, "MAX_STEPS": 500,
               "SIGMA_FACTOR": 1.0},

    "ns":     {"NUM_POSTERIOR_DRAWS": 5000, "NUM_SLICES": 25,
               "DLOGZ_THRESHOLD": 5.0},

    "deo":    {"NUM_CHAINS": 30, "NUM_WARMUP": 500, "KERNEL": "rwmh",
               "ANNEALING_BASE": 1.4142135623730951,
               "STEP_SIZE_HOT_RWMH": 0.5, "STEP_SIZE_COLD_RWMH": 0.002},

    "seo":    {"NUM_CHAINS": 30, "NUM_WARMUP": 500, "KERNEL": "rwmh",
               "ANNEALING_BASE": 1.4142135623730951,
               "STEP_SIZE_HOT_RWMH": 0.5, "STEP_SIZE_COLD_RWMH": 0.002},
}

# Minimum values to avoid degenerate runs
MIN_SAMPLES        = 100
MIN_PARTICLES      = 50
MIN_LIVE_POINTS    = 50


# ---------------------------------------------------------------------------
# Back-computation: LDE budget -> native effort parameter
# ---------------------------------------------------------------------------

def _back_compute(algo: str, budget: int) -> dict:
    """
    Given a log-density-equivalent budget, return a dict of
    {param: name, value: int, fixed: {...}} to monkey-patch into the module.

    Returns None for the effort param if the budget is too small to produce
    a valid run (below minimum thresholds).
    """
    f = ALGO_FIXED[algo]

    if algo == "rwmh":
        # LDE = NUM_CHAINS × (NUM_BURNIN + NUM_SAMPLES)
        n = budget // f["NUM_CHAINS"] - f["NUM_BURNIN"]
        n = max(n, MIN_SAMPLES)
        return {"param": "NUM_SAMPLES", "value": n,
                "fixed": {"NUM_BURNIN": f["NUM_BURNIN"], "NUM_CHAINS": f["NUM_CHAINS"]}}

    if algo == "affinv":
        # LDE = NUM_WALKERS × NUM_SAMPLES (total steps including burn-in period)
        n = budget // f["NUM_WALKERS"]
        n = max(n, f["NUM_BURNIN"] + MIN_SAMPLES)
        return {"param": "NUM_SAMPLES", "value": n,
                "fixed": {"NUM_BURNIN": f["NUM_BURNIN"], "NUM_WALKERS": f["NUM_WALKERS"]}}

    if algo == "smc":
        # LDE = est_steps × NUM_PARTICLES × (NUM_MCMC_STEPS + 2)
        cost_per_particle = SMC_EST_NUM_STEPS * (f["NUM_MCMC_STEPS"] + 2)
        p = budget // cost_per_particle
        p = max(p, MIN_PARTICLES)
        return {"param": "NUM_PARTICLES", "value": p,
                "fixed": {"NUM_MCMC_STEPS": f["NUM_MCMC_STEPS"],
                          "TARGET_ESS": f["TARGET_ESS"],
                          "MAX_STEPS": f["MAX_STEPS"],
                          "SIGMA_FACTOR": f["SIGMA_FACTOR"]}}

    if algo == "ns":
        # LDE ≈ NUM_LIVE_POINTS × evals_per_live_point
        nlp = budget // NS_EVALS_PER_LIVE_POINT
        nlp = max(nlp, MIN_LIVE_POINTS)
        return {"param": "NUM_LIVE_POINTS", "value": nlp,
                "fixed": {"MAX_SAMPLES": float(budget),
                          "NUM_POSTERIOR_DRAWS": f["NUM_POSTERIOR_DRAWS"],
                          "NUM_SLICES": f["NUM_SLICES"],
                          "DLOGZ_THRESHOLD": f["DLOGZ_THRESHOLD"]}}

    if algo in ("deo", "seo"):
        # LDE = NUM_CHAINS × (NUM_WARMUP + NUM_SAMPLES)
        n = budget // f["NUM_CHAINS"] - f["NUM_WARMUP"]
        n = max(n, MIN_SAMPLES)
        fixed = {
            "NUM_WARMUP": f["NUM_WARMUP"],
            "NUM_CHAINS": f["NUM_CHAINS"],
            "KERNEL": f["KERNEL"],
            "ANNEALING_BASE": f["ANNEALING_BASE"],
            "STEP_SIZE_HOT_RWMH": f["STEP_SIZE_HOT_RWMH"],
            "STEP_SIZE_COLD_RWMH": f["STEP_SIZE_COLD_RWMH"],
        }
        return {"param": "NUM_SAMPLES", "value": n, "fixed": fixed}

    raise ValueError(f"Unknown algorithm: {algo}")


# ---------------------------------------------------------------------------

ALL_ALGORITHMS = list(ALGO_FIXED.keys())

ALGO_LABELS = {
    "rwmh":   "RWMH",
    "affinv": "Affine Invariant",
    "smc":    "Adaptive SMC",
    "ns":     "Nested Sampling",
    "deo":    "DEO-PT",
    "seo":    "SEO-PT",
}

ALGO_COLORS = {
    "rwmh":   "#1f77b4",
    "affinv": "#2ca02c",
    "smc":    "#d62728",
    "ns":     "#9467bd",
    "deo":    "#8c564b",
    "seo":    "#e377c2",
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_normalised_mae(posterior_means: dict, ground_truth: dict) -> float:
    """
    Normalised mean absolute error: average over parameters of
    |posterior_mean - truth| / prior_std.

    Dimensionless metric ∈ [0, ∞). Values < 1 indicate the posterior mean is
    within one prior standard deviation of truth on average.
    """
    errors = []
    for name in PARAM_NAMES:
        if name in posterior_means and name in ground_truth:
            err = abs(float(posterior_means[name]) - float(ground_truth[name]))
            std = PRIOR_STDS.get(name, 1.0)
            errors.append(err / std if std > 0 else err)
    return float(np.mean(errors)) if errors else float("nan")


def compute_raw_mae(posterior_means: dict, ground_truth: dict) -> float:
    """Un-normalised mean absolute error (in physical units)."""
    errors = []
    for name in PARAM_NAMES:
        if name in posterior_means and name in ground_truth:
            errors.append(abs(float(posterior_means[name]) - float(ground_truth[name])))
    return float(np.mean(errors)) if errors else float("nan")


def compute_per_param_normalised_error(posterior_means: dict, ground_truth: dict) -> dict:
    """Per-parameter |bias| / prior_std."""
    out = {}
    for name in PARAM_NAMES:
        if name in posterior_means and name in ground_truth:
            err = abs(float(posterior_means[name]) - float(ground_truth[name]))
            std = PRIOR_STDS.get(name, 1.0)
            out[name] = float(err / std) if std > 0 else float(err)
    return out


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(algo: str, budget: int, seed: int) -> dict:
    """
    Back-compute effort from budget, monkey-patch the module, call
    main(seed=seed, save_outputs=False), and return a result record.

    Records include both the *target* budget and the *actual* oracle
    consumption (read from the diagnostics dict returned by main()).
    """
    bc = _back_compute(algo, budget)
    effort_param = bc["param"]
    effort_value = bc["value"]
    fixed_params = bc["fixed"]

    base = {
        "algorithm":               algo,
        "budget_lde":              budget,
        "effort_param":            effort_param,
        "effort_value":            effort_value,
        "trial":                   seed,
        "seed":                    seed,
        "wall_time_s":             None,
        "normalised_mae":          None,
        "raw_mae":                 None,
        "per_param_error":         None,
        "posterior_means":         None,
        "ground_truth":            None,
        "total_bulk_ess":          None,
        "actual_oracle_evals":     None,
        "actual_oracle_type":      None,
        "error":                   None,
    }
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        mod = importlib.import_module(algo)
        importlib.reload(mod)

        # Monkey-patch effort parameter
        setattr(mod, effort_param, effort_value)
        # Monkey-patch fixed hyperparameters
        for k, v in fixed_params.items():
            setattr(mod, k, v)

        # Silence noisy loggers
        noisy = ["arviz", "jaxns", "absl", "absl-py", "tensorflow",
                 "tensorflow_probability", "jax", "blackjax"]
        _loggers   = [logging.getLogger(n) for n in noisy] + [logging.root]
        _prev_lvls = [lg.level for lg in _loggers]
        for lg in _loggers:
            lg.setLevel(logging.CRITICAL)

        sink = io.StringIO()
        t_start = time.perf_counter()
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(sink), \
             contextlib.redirect_stderr(sink):
            warnings.simplefilter("ignore")
            diag = mod.main(seed=seed, save_outputs=False)
        t_total = time.perf_counter() - t_start

        for lg, lv in zip(_loggers, _prev_lvls):
            lg.setLevel(lv)

        # Extract metrics
        post_means = diag.get("posterior_means", {})
        gt         = diag.get("ground_truth", GROUND_TRUTH)

        norm_mae     = compute_normalised_mae(post_means, gt)
        raw_mae      = compute_raw_mae(post_means, gt)
        per_param    = compute_per_param_normalised_error(post_means, gt)
        bulk_ess     = diag.get("total_bulk_ess")

        # Extract actual oracle consumption from diagnostics
        actual_evals, oracle_type = _extract_actual_oracle(algo, diag)

        # Prefer the sampler's own wall_time_s (excludes model setup / JIT overhead)
        wall_time = diag.get("wall_time_s", t_total)

        base.update({
            "wall_time_s":          float(wall_time),
            "wall_time_total_s":    float(t_total),
            "normalised_mae":       norm_mae,
            "raw_mae":              raw_mae,
            "per_param_error":      per_param,
            "posterior_means":      {k: float(v) for k, v in post_means.items()},
            "ground_truth":         {k: float(v) for k, v in gt.items()},
            "total_bulk_ess":       float(bulk_ess) if bulk_ess is not None else None,
            "actual_oracle_evals":  actual_evals,
            "actual_oracle_type":   oracle_type,
        })
    except Exception as exc:
        import traceback
        base["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return base


def _extract_actual_oracle(algo: str, diag: dict):
    """
    Read the actual oracle eval count from a diagnostics dict.
    Returns (count, type_string).
    """
    if algo == "rwmh":
        return diag.get("total_log_density_evals"), "logp"
    if algo == "affinv":
        return diag.get("total_log_density_evals"), "logp"
    if algo == "smc":
        return diag.get("total_log_density_evals"), "logp"
    if algo == "ns":
        return diag.get("total_likelihood_evals"), "logp"
    if algo in ("deo", "seo"):
        for k in ("gradient_evals", "log_density_evals"):
            if k in diag:
                otype = "grad" if "gradient" in k else "logp"
                return diag[k], otype
        return None, None
    return None, None


# ---------------------------------------------------------------------------
# Batch runner with persistence
# ---------------------------------------------------------------------------

def run_all_trials(algorithms: list, force: bool = False) -> list:
    if RESULTS_PATH.exists() and not force:
        print(f"Results file found: {RESULTS_PATH}")
        print("Loading existing results. Use --force to re-run, --plot-only to just replot.")
        with open(RESULTS_PATH) as f:
            return json.load(f)

    # Print the back-computed effort table before running
    _print_effort_table(algorithms)

    total = len(algorithms) * len(LOGP_BUDGETS) * NUM_TRIALS
    records = []
    n = 0

    for algo in algorithms:
        for budget in LOGP_BUDGETS:
            bc = _back_compute(algo, budget)
            for trial in range(NUM_TRIALS):
                n += 1
                print(
                    f"[{n:3d}/{total}]  {algo.upper():<8}  "
                    f"budget={budget:<10,}  {bc['param']}={bc['value']:<8}  "
                    f"trial={trial}",
                    flush=True,
                )
                rec = run_trial(algo, budget, seed=trial)
                if rec["error"]:
                    print(f"  !! FAILED: {rec['error'][:200]}")
                else:
                    print(f"  -> norm_MAE={rec['normalised_mae']:.4f}  "
                          f"wall={rec['wall_time_s']:.1f}s  "
                          f"oracle={rec['actual_oracle_evals']}")
                records.append(rec)

                # Save incrementally (in case of crashes)
                SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(RESULTS_PATH, "w") as f:
                    json.dump(records, f, indent=2)

    n_failed = sum(1 for r in records if r["error"])
    print(f"\nCompleted {len(records)} records ({n_failed} failures) -> {RESULTS_PATH}")
    return records


def _print_effort_table(algorithms: list) -> None:
    """Print the budget -> effort mapping for each algorithm before running."""
    print("\n" + "=" * 90)
    print("  Budget -> effort parameter mapping")
    print("=" * 90)
    header = f"  {'Budget (LDE)':>14}"
    for algo in algorithms:
        bc = _back_compute(algo, LOGP_BUDGETS[0])
        label = f"{algo.upper()}({bc['param']})"
        header += f"  {label:>22}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for budget in LOGP_BUDGETS:
        row = f"  {budget:>14,}"
        for algo in algorithms:
            bc = _back_compute(algo, budget)
            row += f"  {bc['value']:>22,}"
        print(row)
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list) -> dict:
    """
    Aggregate the flat record list into:
      { algo: { budget: { wall_times, norm_maes, raw_maes, actual_evals, ess } } }
    """
    out = defaultdict(lambda: defaultdict(lambda: {
        "wall_times":   [],
        "norm_maes":    [],
        "raw_maes":     [],
        "actual_evals": [],
        "ess":          [],
    }))
    for rec in records:
        if rec.get("error") or rec["normalised_mae"] is None:
            continue
        a = rec["algorithm"]
        b = rec["budget_lde"]
        out[a][b]["wall_times"].append(rec["wall_time_s"])
        out[a][b]["norm_maes"].append(rec["normalised_mae"])
        out[a][b]["raw_maes"].append(rec["raw_mae"])
        if rec["actual_oracle_evals"] is not None:
            out[a][b]["actual_evals"].append(rec["actual_oracle_evals"])
        if rec["total_bulk_ess"] is not None:
            out[a][b]["ess"].append(rec["total_bulk_ess"])
    return out


# ---------------------------------------------------------------------------
# Publication-quality plots
# ---------------------------------------------------------------------------

_RC = {
    "font.family":         "serif",
    "font.size":           11,
    "axes.labelsize":      12,
    "axes.titlesize":      11,
    "axes.linewidth":      1.2,
    "legend.fontsize":     9,
    "legend.framealpha":   0.9,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}


def make_plots(records: list) -> None:
    """
    Generate publication-ready figures:
      1. norm_mae_vs_budget.png     -- Primary fair comparison
      2. norm_mae_vs_wall_time.png  -- Accuracy vs real cost
      3. ess_vs_budget.png          -- Sampling efficiency
      4. budget_utilisation.png     -- Sanity check: actual vs target evals
      5. per_param_recovery.png     -- Per-parameter breakdown at highest budget
    """
    agg = aggregate(records)
    algo_order = [a for a in ALL_ALGORITHMS if a in agg]

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Normalised MAE vs LDE budget (primary comparison) ----
    _plot_metric_vs_x(
        agg, algo_order,
        x_key="budget", y_key="norm_maes",
        fname="norm_mae_vs_budget.png",
        xlabel="Log-density-equivalent budget (LDE)",
        ylabel="Normalised parameter MAE\n(|posterior mean − truth| / prior σ)",
        title="Parameter recovery vs. compute budget",
    )

    # ---- Figure 2: Normalised MAE vs wall time ----
    _plot_metric_vs_x(
        agg, algo_order,
        x_key="wall_times", y_key="norm_maes",
        fname="norm_mae_vs_wall_time.png",
        xlabel="Wall-clock time (s)",
        ylabel="Normalised parameter MAE",
        title="Parameter recovery vs. wall-clock time",
    )

    # ---- Figure 3: ESS vs budget ----
    _plot_metric_vs_x(
        agg, algo_order,
        x_key="budget", y_key="ess",
        fname="ess_vs_budget.png",
        xlabel="Log-density-equivalent budget (LDE)",
        ylabel="Total bulk ESS (ArviZ)",
        title="Effective sample size vs. compute budget",
        logy=True,
    )

    # ---- Figure 4: Budget utilisation (sanity check) ----
    _plot_budget_utilisation(agg, algo_order)

    # ---- Figure 5: Per-parameter recovery at highest budget ----
    _plot_per_param_recovery(records, algo_order)

    print(f"\nSaved all plots to {SCALING_OUT_DIR}/")


def _plot_metric_vs_x(agg, algo_order, x_key, y_key, fname, xlabel, ylabel,
                       title, logy=False):
    """Shared helper for metric-vs-something plots."""
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 5))

        for algo in algo_order:
            effort_data = agg[algo]
            med_x, med_y, q25_y, q75_y = [], [], [], []

            for budget in sorted(effort_data.keys()):
                d = effort_data[budget]
                y_vals = np.array(d[y_key])
                if len(y_vals) == 0:
                    continue

                if x_key == "budget":
                    med_x.append(float(budget))
                else:
                    x_vals = np.array(d[x_key])
                    if len(x_vals) == 0:
                        continue
                    med_x.append(float(np.median(x_vals)))

                med_y.append(float(np.median(y_vals)))
                q25_y.append(float(np.percentile(y_vals, 25)))
                q75_y.append(float(np.percentile(y_vals, 75)))

            if not med_x:
                continue

            idx = np.argsort(med_x)
            mx  = np.array(med_x)[idx]
            my  = np.array(med_y)[idx]
            lo  = np.array(q25_y)[idx]
            hi  = np.array(q75_y)[idx]
            c   = ALGO_COLORS.get(algo, "gray")

            ax.plot(mx, my, marker="o", markersize=5, lw=1.5,
                    color=c, label=ALGO_LABELS.get(algo, algo), zorder=3)
            ax.fill_between(mx, lo, hi, alpha=0.15, color=c, zorder=2)

        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6,
                color="gray", alpha=0.4, zorder=1)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4,
                color="gray", alpha=0.2, zorder=1)
        ax.legend(loc="best")
        ax.annotate(
            f"Shaded band = IQR across {NUM_TRIALS} trials",
            xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=8, color="gray",
        )
        fig.tight_layout()
        fig.savefig(SCALING_OUT_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {SCALING_OUT_DIR / fname}")


def _plot_budget_utilisation(agg, algo_order):
    """Actual oracle evals vs target budget — sanity check for cost model accuracy."""
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 5))

        for algo in algo_order:
            effort_data = agg[algo]
            budgets_x, actual_y, lo_y, hi_y = [], [], [], []

            for budget in sorted(effort_data.keys()):
                d = effort_data[budget]
                evals = np.array(d["actual_evals"])
                if len(evals) == 0:
                    continue
                budgets_x.append(float(budget))
                actual_y.append(float(np.median(evals)))
                lo_y.append(float(np.percentile(evals, 25)))
                hi_y.append(float(np.percentile(evals, 75)))

            if not budgets_x:
                continue

            c = ALGO_COLORS.get(algo, "gray")
            ax.plot(budgets_x, actual_y, marker="o", markersize=5, lw=1.5,
                    color=c, label=ALGO_LABELS.get(algo, algo), zorder=3)
            ax.fill_between(budgets_x, lo_y, hi_y, alpha=0.1, color=c, zorder=2)

        # Ideal line (actual == budget)
        bmin = min(LOGP_BUDGETS)
        bmax = max(LOGP_BUDGETS)
        ax.plot([bmin, bmax], [bmin, bmax], "k--", lw=1, alpha=0.5, label="Ideal (1:1)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Target LDE budget")
        ax.set_ylabel("Actual oracle evaluations (median)")
        ax.set_title("Budget utilisation: actual vs. target oracle evals", pad=8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6,
                color="gray", alpha=0.4, zorder=1)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4,
                color="gray", alpha=0.2, zorder=1)
        ax.legend(loc="upper left", fontsize=8)
        ax.annotate(
            "Points above the dashed line used MORE compute than budgeted.\n"
            "Points below used LESS (pilot estimates were conservative).",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=7, color="gray", va="bottom",
        )
        fig.tight_layout()
        fname = SCALING_OUT_DIR / "budget_utilisation.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def _plot_per_param_recovery(records, algo_order):
    """
    Bar chart showing per-parameter normalised error at the highest budget level.
    Each algorithm gets a cluster of bars (one per parameter).
    """
    max_budget = max(LOGP_BUDGETS)

    # Collect per-parameter errors at max budget for each algorithm
    algo_param_errors = {}
    for algo in algo_order:
        errors_by_param = defaultdict(list)
        for rec in records:
            if rec.get("error") or rec["algorithm"] != algo:
                continue
            if rec["budget_lde"] != max_budget:
                continue
            if rec["per_param_error"] is None:
                continue
            for name, val in rec["per_param_error"].items():
                errors_by_param[name].append(val)
        # Take median across trials
        algo_param_errors[algo] = {
            name: float(np.median(vals))
            for name, vals in errors_by_param.items()
            if len(vals) > 0
        }

    if not algo_param_errors:
        return

    # Use only parameters present in all algorithms
    common_params = set(PARAM_NAMES)
    for errs in algo_param_errors.values():
        common_params &= set(errs.keys())
    common_params = [p for p in PARAM_NAMES if p in common_params]

    if not common_params:
        return

    with matplotlib.rc_context(_RC):
        n_algos  = len(algo_order)
        n_params = len(common_params)
        bar_width = 0.8 / n_algos
        x = np.arange(n_params)

        fig, ax = plt.subplots(figsize=(max(10, n_params * 0.8), 5))

        for i, algo in enumerate(algo_order):
            errs = algo_param_errors.get(algo, {})
            vals = [errs.get(p, 0.0) for p in common_params]
            c = ALGO_COLORS.get(algo, "gray")
            offset = (i - n_algos / 2 + 0.5) * bar_width
            ax.bar(x + offset, vals, bar_width, label=ALGO_LABELS.get(algo, algo),
                   color=c, alpha=0.8, edgecolor="white", linewidth=0.5)

        # Reference lines
        ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5,
                   label="1 prior σ")
        ax.axhline(0.0, color="gray", ls="-", lw=0.5, alpha=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(common_params, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("| posterior mean − truth | / prior σ")
        ax.set_title(
            f"Per-parameter recovery at max budget "
            f"(LDE = {max_budget:,}, median over {NUM_TRIALS} trials)",
            pad=8,
        )
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlim(-0.6, n_params - 0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fname = SCALING_OUT_DIR / "per_param_recovery.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(records: list) -> None:
    """Print a condensed table of median results per algorithm per budget level."""
    agg = aggregate(records)
    algo_order = [a for a in ALL_ALGORITHMS if a in agg]

    print("\n" + "=" * 100)
    print("  SUMMARY: Median normalised MAE and wall time per algorithm per budget")
    print("=" * 100)

    header = f"  {'Algorithm':<12} {'Budget':>10}"
    header += f"  {'Norm MAE':>10}  {'Wall (s)':>9}  {'ESS':>8}  {'Oracle':>10}  {'Util %':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for algo in algo_order:
        for budget in sorted(agg[algo].keys()):
            d = agg[algo][budget]
            n_mae = np.median(d["norm_maes"]) if d["norm_maes"] else float("nan")
            wall  = np.median(d["wall_times"]) if d["wall_times"] else float("nan")
            ess   = np.median(d["ess"]) if d["ess"] else float("nan")
            oracle = np.median(d["actual_evals"]) if d["actual_evals"] else float("nan")
            util  = (oracle / budget * 100) if not np.isnan(oracle) else float("nan")

            print(
                f"  {ALGO_LABELS.get(algo, algo):<12} {budget:>10,}"
                f"  {n_mae:>10.4f}  {wall:>9.1f}  {ess:>8.0f}  {oracle:>10.0f}  {util:>6.1f}%"
            )
        print()

    print("=" * 100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scaling study: parameter recovery vs compute budget (LDE-normalised)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all trials even if results.json already exists",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate plots from existing results.json without running any trials",
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        default=ALL_ALGORITHMS,
        choices=ALL_ALGORITHMS,
        metavar="ALGO",
        help=(
            "Subset of algorithms to run (default: all). "
            f"Choices: {ALL_ALGORITHMS}"
        ),
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int,
        default=None,
        help="Override budget levels (space-separated integers)",
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override number of trials per (algorithm, budget) pair",
    )
    args = parser.parse_args()

    # Allow overriding globals from CLI
    global LOGP_BUDGETS, NUM_TRIALS
    if args.budgets is not None:
        LOGP_BUDGETS = sorted(args.budgets)
    if args.trials is not None:
        NUM_TRIALS = args.trials

    if args.plot_only and args.force:
        print("WARNING: --force is ignored when --plot-only is set.")

    if args.plot_only:
        if not RESULTS_PATH.exists():
            print(f"ERROR: {RESULTS_PATH} not found. Run without --plot-only first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records from {RESULTS_PATH}")
    else:
        records = run_all_trials(args.algorithms, force=args.force)

    # Filter records to requested algorithms (for --plot-only with subset)
    if args.algorithms != ALL_ALGORITHMS:
        records = [r for r in records if r["algorithm"] in args.algorithms]

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print_summary_table(records)
    make_plots(records)


if __name__ == "__main__":
    main()