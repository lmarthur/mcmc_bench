"""
Scaling study: mode weight MAE vs. wall-clock time across algorithms and effort levels.

Runs repeated trials of each inference algorithm over a range of effort levels,
where effort is defined by a shared log-density-equivalent (LDE) budget.

Each algorithm back-computes its native effort parameter (NUM_SAMPLES,
NUM_PARTICLES, NUM_LIVE_POINTS) from the LDE budget using its cost model:

  RWMH       :  LDE = NUM_CHAINS   x (NUM_BURNIN  + NUM_SAMPLES)
  NUTS       :  LDE = [NUM_CHAINS  x NUM_SAMPLES x mean_tree_depth
                        + warmup_grads] x grad_to_logp_ratio
  Affine Inv :  LDE = NUM_WALKERS  x (NUM_BURNIN  + NUM_SAMPLES)
  SMC        :  LDE = EST_SMC_STEPS x NUM_PARTICLES x NUM_MCMC_STEPS
  NS         :  LDE ≈ NUM_LIVE_POINTS x evals_per_live_point
  DEO / SEO  :  LDE = NUM_CHAINS  x (NUM_WARMUP  + NUM_SAMPLES)

NUTS uses gradient evaluations, which are more expensive than log-density
evaluations. A measured conversion factor (GRAD_TO_LOGP_RATIO) converts
gradient evals to log-density equivalents so all algorithms share the same
budget axis.

Pilot estimates (mean NUTS tree depth, typical SMC step count, NS evals per
live point) are used for the back-computation. Actual oracle consumption is
recorded per trial and reported in the output, so any mismatch between the
estimate and reality is transparent.

Saves all raw per-trial results to JSON so plots can be regenerated without
re-running.

Usage:
    python scaling_study_compute.py                          # run trials if needed, then plot
    python scaling_study_compute.py --force                  # re-run all trials
    python scaling_study_compute.py --plot-only              # regenerate plots from saved JSON
    python scaling_study_compute.py --algorithms rwmh nuts   # run/plot a subset of algorithms
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
from model import OUTPUT_DIR

SCRIPTS_DIR     = Path(__file__).parent
SCALING_OUT_DIR = OUTPUT_DIR / "scaling_compute"
RESULTS_PATH    = SCALING_OUT_DIR / "results.json"

# ---------------------------------------------------------------------------
# Shared oracle budget levels (log-density equivalents)
# ---------------------------------------------------------------------------
# Every algorithm receives the same LDE budget at each level.  The back-
# computation translates this into the algorithm's native effort parameter.

LOGP_BUDGETS = [10_000, 25_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

NUM_TRIALS = 10  # seeds 0 .. NUM_TRIALS-1

# ---------------------------------------------------------------------------
# Pilot estimates for unknowns in the cost models
# ---------------------------------------------------------------------------
# These are rough estimates from default-setting pilot runs.  The actual oracle
# consumption is recorded per trial so any deviation is transparent.
#
# To update these from your own pilot runs, check:
#   NUTS:  diagnostics.json -> mean_integration_steps
#   SMC:   diagnostics.json -> num_smc_steps
#   NS:    diagnostics.json -> total_likelihood_evals / num_live_points
#
# GRAD_TO_LOGP_RATIO: for d=2, reverse-mode AD costs roughly 2-3× a forward
# eval.  We use 3.0 (conservative).  Measure with %timeit if you want precision.

NUTS_MEAN_TREE_DEPTH    = 7      # mean leapfrog steps per NUTS iteration (empirical average)
NUTS_WARMUP_DEPTH_EST   = 5      # same depth assumed during warmup (empirical average)
GRAD_TO_LOGP_RATIO      = 1.5    # 1 grad eval ≈ 1.5 logp evals

SMC_EST_NUM_STEPS       = 15     # typical adaptive tempering steps (empirical average)

NS_EVALS_PER_LIVE_POINT = 715    # total_likelihood_evals / NUM_LIVE_POINTS (empirical average)

# ---------------------------------------------------------------------------
# Fixed (non-effort) hyperparameters per algorithm
# ---------------------------------------------------------------------------
# These mirror the defaults in each script but are stated explicitly so the
# back-computation is self-contained.

ALGO_FIXED = {
    "rwmh":   {"NUM_CHAINS": 4,  "NUM_BURNIN": 500},
    "nuts":   {"NUM_CHAINS": 4,  "NUM_WARMUP": 1000},
    "affinv": {"NUM_WALKERS": 8, "NUM_BURNIN": 250},
    "smc":    {"NUM_MCMC_STEPS": 10, "TARGET_ESS": 0.75, "MAX_STEPS": 500},
    "ns":     {"NUM_POSTERIOR_DRAWS": 5000},
    "deo":    {"NUM_CHAINS": 8,  "NUM_WARMUP": 250},
    "seo":    {"NUM_CHAINS": 8,  "NUM_WARMUP": 250},
}

# Minimum values to avoid degenerate runs
MIN_SAMPLES        = 100
MIN_PARTICLES      = 100
MIN_LIVE_POINTS    = 100


# ---------------------------------------------------------------------------
# Back-computation: LDE budget -> native effort parameter
# ---------------------------------------------------------------------------

def _back_compute(algo: str, budget: int) -> dict:
    """
    Given a log-density-equivalent budget, return a dict of
    {param_name: value, **fixed_overrides} to monkey-patch into the module.

    Returns None for the effort param if the budget is too small to produce
    a valid run (below minimum thresholds).
    """
    f = ALGO_FIXED[algo]

    if algo == "rwmh":
        # LDE = NUM_CHAINS * (NUM_BURNIN + NUM_SAMPLES)
        n = budget // f["NUM_CHAINS"] - f["NUM_BURNIN"]
        n = max(n, MIN_SAMPLES)
        return {"param": "NUM_SAMPLES", "value": n, "fixed": {"NUM_BURNIN": f["NUM_BURNIN"]}}

    if algo == "nuts":
        # LDE = (warmup_grads + NUM_CHAINS * NUM_SAMPLES * depth) * ratio
        # => NUM_SAMPLES = (budget/ratio - warmup_grads) / (NUM_CHAINS * depth)
        warmup_grads = f["NUM_WARMUP"] * NUTS_WARMUP_DEPTH_EST
        effective = budget / GRAD_TO_LOGP_RATIO - warmup_grads
        n = int(effective / (f["NUM_CHAINS"] * NUTS_MEAN_TREE_DEPTH))
        n = max(n, MIN_SAMPLES)
        return {"param": "NUM_SAMPLES", "value": n, "fixed": {"NUM_WARMUP": f["NUM_WARMUP"]}}

    if algo == "affinv":
        # LDE = NUM_WALKERS * NUM_SAMPLES  (NUM_SAMPLES includes burn-in period)
        n = budget // f["NUM_WALKERS"]
        n = max(n, f["NUM_BURNIN"] + MIN_SAMPLES)  # must exceed burn-in
        return {"param": "NUM_SAMPLES", "value": n, "fixed": {"NUM_BURNIN": f["NUM_BURNIN"]}}

    if algo == "smc":
        # LDE = est_steps * NUM_PARTICLES * (NUM_MCMC_STEPS + 2)
        cost_per_particle = SMC_EST_NUM_STEPS * (f["NUM_MCMC_STEPS"] + 2)
        p = budget // cost_per_particle
        p = max(p, MIN_PARTICLES)
        return {"param": "NUM_PARTICLES", "value": p,
                "fixed": {"NUM_MCMC_STEPS": f["NUM_MCMC_STEPS"],
                          "TARGET_ESS": f["TARGET_ESS"],
                          "MAX_STEPS": f["MAX_STEPS"]}}

    if algo == "ns":
        # LDE ≈ NUM_LIVE_POINTS * evals_per_live_point
        nlp = budget // NS_EVALS_PER_LIVE_POINT
        nlp = max(nlp, MIN_LIVE_POINTS)
        return {"param": "NUM_LIVE_POINTS", "value": nlp,
                "fixed": {"MAX_SAMPLES": float(budget),
                          "NUM_POSTERIOR_DRAWS": f["NUM_POSTERIOR_DRAWS"]}}

    if algo in ("deo", "seo"):
        # LDE = NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)
        n = budget // f["NUM_CHAINS"] - f["NUM_WARMUP"]
        n = max(n, MIN_SAMPLES)
        return {"param": "NUM_SAMPLES", "value": n, "fixed": {"NUM_WARMUP": f["NUM_WARMUP"]}}

    raise ValueError(f"Unknown algorithm: {algo}")


# ---------------------------------------------------------------------------

ALL_ALGORITHMS = list(ALGO_FIXED.keys())

ALGO_LABELS = {
    "rwmh":   "RWMH",
    "nuts":   "NUTS",
    "affinv": "Affine Invariant",
    "smc":    "Adaptive SMC",
    "ns":     "Nested Sampling",
    "deo":    "DEO-PT",
    "seo":    "SEO-PT",
}


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(algo: str, budget: int, seed: int) -> dict:
    """
    Back-compute effort from budget, monkey-patch the module, call
    main(save_outputs=False), and return a result record.

    Records include both the *target* budget and the *actual* oracle
    consumption (read from the diagnostics dict returned by main()).
    """
    bc = _back_compute(algo, budget)
    effort_param = bc["param"]
    effort_value = bc["value"]
    fixed_params = bc["fixed"]

    base = {
        "algorithm":         algo,
        "budget_lde":        budget,
        "effort_param":      effort_param,
        "effort_value":      effort_value,
        "trial":             seed,
        "seed":              seed,
        "wall_time_core_s":  None,
        "wall_time_total_s": None,
        "mode_weight_mae":   None,
        "mode_weights":      None,
        "true_mode_weights": None,
        "actual_oracle_evals": None,
        "actual_oracle_type":  None,
        "error":             None,
    }
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        mod = importlib.import_module(algo)
        importlib.reload(mod)

        setattr(mod, effort_param, effort_value)
        for k, v in fixed_params.items():
            setattr(mod, k, v)

        # Silence noisy loggers
        noisy = ["arviz", "jaxns", "absl", "absl-py", "tensorflow", "tensorflow_probability"]
        _loggers   = [logging.getLogger(n) for n in noisy] + [logging.root]
        _prev_lvls = [lg.level for lg in _loggers]
        for lg in _loggers:
            lg.setLevel(logging.CRITICAL)

        sink = io.StringIO()
        with warnings.catch_warnings(), \
             contextlib.redirect_stdout(sink), \
             contextlib.redirect_stderr(sink):
            warnings.simplefilter("ignore")
            diag = mod.main(seed=seed, save_outputs=False)

        for lg, lv in zip(_loggers, _prev_lvls):
            lg.setLevel(lv)

        mw  = np.array(diag["mode_weights"])
        tw  = np.array(diag["true_mode_weights"])
        mae = float(np.mean(np.abs(mw - tw)))

        # Extract actual oracle consumption from diagnostics
        actual_evals, oracle_type = _extract_actual_oracle(algo, diag)

        base.update({
            "wall_time_core_s":    float(diag["wall_time_core_s"]),
            "wall_time_total_s":   float(diag["wall_time_total_s"]),
            "mode_weight_mae":     mae,
            "mode_weights":        mw.tolist(),
            "true_mode_weights":   tw.tolist(),
            "actual_oracle_evals": actual_evals,
            "actual_oracle_type":  oracle_type,
        })
    except Exception as exc:
        base["error"] = str(exc)
    return base


def _extract_actual_oracle(algo: str, diag: dict):
    """
    Read the actual oracle eval count from a diagnostics dict.
    Returns (count, type_string).
    """
    if algo == "rwmh":
        return diag.get("total_log_density_evals"), "logp"
    if algo == "nuts":
        return diag.get("total_grad_evals"), "grad"
    if algo == "affinv":
        return diag.get("total_log_density_evals"), "logp"
    if algo == "smc":
        return diag.get("total_log_density_evals"), "logp"
    if algo == "ns":
        return diag.get("total_likelihood_evals"), "logp"
    if algo in ("deo", "seo"):
        # DEO/SEO store under a kernel-dependent key
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
                    print(f"  !! FAILED: {rec['error']}")
                records.append(rec)

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(records, f, indent=2)

    n_failed = sum(1 for r in records if r["error"])
    print(f"\nSaved {len(records)} records ({n_failed} failures) -> {RESULTS_PATH}")
    return records


def _print_effort_table(algorithms: list) -> None:
    """Print the budget -> effort mapping for each algorithm before running."""
    print("\n" + "=" * 80)
    print("  Budget -> effort parameter mapping")
    print("=" * 80)
    header = f"  {'Budget (LDE)':>14}"
    for algo in algorithms:
        bc = _back_compute(algo, LOGP_BUDGETS[0])
        header += f"  {algo.upper() + ' (' + bc['param'] + ')':>22}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for budget in LOGP_BUDGETS:
        row = f"  {budget:>14,}"
        for algo in algorithms:
            bc = _back_compute(algo, budget)
            row += f"  {bc['value']:>22,}"
        print(row)
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list) -> dict:
    """
    Aggregate the flat record list into:
      { algo: { budget: { core_times, total_times, maes, actual_evals } } }
    """
    out = defaultdict(lambda: defaultdict(lambda: {
        "core_times":   [],
        "total_times":  [],
        "maes":         [],
        "actual_evals": [],
    }))
    for rec in records:
        if rec.get("error") or rec["mode_weight_mae"] is None:
            continue
        a = rec["algorithm"]
        b = rec["budget_lde"]
        out[a][b]["core_times"].append(rec["wall_time_core_s"])
        out[a][b]["total_times"].append(rec["wall_time_total_s"])
        out[a][b]["maes"].append(rec["mode_weight_mae"])
        if rec["actual_oracle_evals"] is not None:
            out[a][b]["actual_evals"].append(rec["actual_oracle_evals"])
    return out


# ---------------------------------------------------------------------------
# Publication-quality plots
# ---------------------------------------------------------------------------

def make_plots(records: list) -> None:
    """
    Generate three publication-ready figures:
      mae_vs_budget.png      -- ModeMAE vs LDE budget (the primary fair comparison)
      mae_vs_core_time.png   -- ModeMAE vs wall_time_core_s
      budget_utilisation.png -- Actual oracle evals vs target budget (sanity check)
    """
    _rc = {
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

    agg        = aggregate(records)
    algo_order = ["rwmh", "nuts", "affinv", "smc", "ns", "deo", "seo"]
    palette    = plt.cm.tab10(np.linspace(0, 0.85, len(algo_order)))
    color_map  = {a: palette[i] for i, a in enumerate(algo_order)}

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: ModeMAE vs LDE budget (primary comparison) ----
    _plot_mae_vs_x(
        agg, algo_order, color_map, _rc,
        x_key="budget",
        fname="mae_vs_budget.png",
        xlabel="Log-density-equivalent budget (LDE)",
        title="Mode weight MAE vs. compute budget (equal budget across algorithms)",
    )

    # ---- Figure 2: ModeMAE vs core wall time ----
    _plot_mae_vs_x(
        agg, algo_order, color_map, _rc,
        x_key="core_times",
        fname="mae_vs_core_time.png",
        xlabel="Wall-clock time, core (s)",
        title="Mode weight MAE vs. core sampling time",
    )

    # ---- Figure 3: Budget utilisation (sanity check) ----
    _plot_budget_utilisation(agg, algo_order, color_map, _rc)

    print(f"Saved plots to {SCALING_OUT_DIR}/")


def _plot_mae_vs_x(agg, algo_order, color_map, rc, x_key, fname, xlabel, title):
    """Shared helper for MAE-vs-something plots."""
    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7, 5))

        for algo in algo_order:
            if algo not in agg:
                continue
            effort_data = agg[algo]
            med_x, med_mae, q25_mae, q75_mae = [], [], [], []

            for budget in sorted(effort_data.keys()):
                d    = effort_data[budget]
                maes = np.array(d["maes"])
                if len(maes) == 0:
                    continue

                if x_key == "budget":
                    med_x.append(float(budget))
                else:
                    times = np.array(d[x_key])
                    med_x.append(float(np.median(times)))

                med_mae.append(float(np.median(maes)))
                q25_mae.append(float(np.percentile(maes, 25)))
                q75_mae.append(float(np.percentile(maes, 75)))

            if not med_x:
                continue

            idx = np.argsort(med_x)
            mx  = np.array(med_x)[idx]
            mm  = np.array(med_mae)[idx]
            lo  = np.array(q25_mae)[idx]
            hi  = np.array(q75_mae)[idx]
            c   = color_map[algo]

            ax.plot(mx, mm, marker="o", markersize=5, lw=1.5,
                    color=c, label=ALGO_LABELS.get(algo, algo), zorder=3)
            ax.fill_between(mx, lo, hi, alpha=0.15, color=c, zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mode weight MAE")
        ax.set_title(title, pad=8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6,
                color="gray", alpha=0.4, zorder=1)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.4,
                color="gray", alpha=0.2, zorder=1)
        ax.legend(loc="upper right")
        ax.annotate(
            f"Shaded band = IQR across {NUM_TRIALS} trials",
            xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=8, color="gray",
        )
        fig.tight_layout()
        fig.savefig(SCALING_OUT_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {SCALING_OUT_DIR / fname}")


def _plot_budget_utilisation(agg, algo_order, color_map, rc):
    """Actual oracle evals vs target budget — sanity check for cost model accuracy."""
    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7, 5))

        for algo in algo_order:
            if algo not in agg:
                continue
            effort_data = agg[algo]
            budgets_x, actual_y = [], []

            for budget in sorted(effort_data.keys()):
                d = effort_data[budget]
                evals = np.array(d["actual_evals"])
                if len(evals) == 0:
                    continue
                budgets_x.append(float(budget))
                actual_y.append(float(np.median(evals)))

            if not budgets_x:
                continue

            c = color_map[algo]
            ax.plot(budgets_x, actual_y, marker="o", markersize=5, lw=1.5,
                    color=c, label=ALGO_LABELS.get(algo, algo), zorder=3)

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
        print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scaling study: ModeMAE vs compute budget (LDE-normalised)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    args = parser.parse_args()

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

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_plots(records)


if __name__ == "__main__":
    main()