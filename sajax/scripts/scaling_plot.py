"""
Generate publication-quality plots from scaling study results.

Loads all results_{algo}.json files from the output directory and merges
them into a single record list for plotting.  Algorithms with no results
file are silently skipped, so you can plot a partially-completed study.

Usage:
    python scaling_plot.py                          # plot all available results
    python scaling_plot.py --algorithms rwmh smc    # plot a subset
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scaling_config import (
    ALL_ALGORITHMS, ALGO_COLORS, ALGO_LABELS,
    LOGP_BUDGETS, NUM_TRIALS, PARAM_NAMES,
    SCALING_OUT_DIR,
)


# ---------------------------------------------------------------------------
# Loading and aggregation
# ---------------------------------------------------------------------------

def load_records(algorithms: list) -> list:
    records = []
    for algo in algorithms:
        path = SCALING_OUT_DIR / f"results_{algo}.json"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        with open(path) as f:
            recs = json.load(f)
        print(f"  Loaded {len(recs):4d} records from {path.name}")
        records.extend(recs)
    return records


def aggregate(records: list) -> dict:
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
# Plot helpers
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


def _plot_metric_vs_x(agg, algo_order, x_key, y_key, fname, xlabel, ylabel,
                       title, logy=False):
    num_trials = max(
        (len(d[y_key]) for a in agg for d in agg[a].values()),
        default=NUM_TRIALS,
    )
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
            f"Shaded band = IQR across {num_trials} trials",
            xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=8, color="gray",
        )
        fig.tight_layout()
        out = SCALING_OUT_DIR / fname
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def _plot_budget_utilisation(agg, algo_order):
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
        out = SCALING_OUT_DIR / "budget_utilisation.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


def _plot_per_param_recovery(records, algo_order):
    max_budget = max(LOGP_BUDGETS)
    num_trials = NUM_TRIALS

    algo_param_errors = {}
    for algo in algo_order:
        errors_by_param = defaultdict(list)
        for rec in records:
            if rec.get("error") or rec["algorithm"] != algo:
                continue
            if rec["budget_lde"] != max_budget or rec["per_param_error"] is None:
                continue
            for name, val in rec["per_param_error"].items():
                errors_by_param[name].append(val)
            num_trials = max(num_trials, rec.get("trial", 0) + 1)
        algo_param_errors[algo] = {
            name: float(np.median(vals))
            for name, vals in errors_by_param.items()
            if vals
        }

    if not algo_param_errors:
        return

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

        ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5, label="1 prior σ")
        ax.axhline(0.0, color="gray",  ls="-",  lw=0.5, alpha=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(common_params, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("| posterior mean − truth | / prior σ")
        ax.set_title(
            f"Per-parameter recovery at max budget "
            f"(LDE = {max_budget:,}, median over {num_trials} trials)",
            pad=8,
        )
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlim(-0.6, n_params - 0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        out = SCALING_OUT_DIR / "per_param_recovery.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(records: list, algo_order: list) -> None:
    agg = aggregate(records)

    print("\n" + "=" * 100)
    print("  SUMMARY: Median normalised MAE and wall time per algorithm per budget")
    print("=" * 100)

    header = f"  {'Algorithm':<12} {'Budget':>10}"
    header += f"  {'Norm MAE':>10}  {'Wall (s)':>9}  {'ESS':>8}  {'Oracle':>10}  {'Util %':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for algo in algo_order:
        if algo not in agg:
            continue
        for budget in sorted(agg[algo].keys()):
            d = agg[algo][budget]
            n_mae  = np.median(d["norm_maes"])   if d["norm_maes"]   else float("nan")
            wall   = np.median(d["wall_times"])   if d["wall_times"]   else float("nan")
            ess    = np.median(d["ess"])           if d["ess"]           else float("nan")
            oracle = np.median(d["actual_evals"]) if d["actual_evals"] else float("nan")
            util   = (oracle / budget * 100) if not np.isnan(oracle) else float("nan")
            print(
                f"  {ALGO_LABELS.get(algo, algo):<12} {budget:>10,}"
                f"  {n_mae:>10.4f}  {wall:>9.1f}  {ess:>8.0f}  {oracle:>10.0f}  {util:>6.1f}%"
            )
        print()

    print("=" * 100)


# ---------------------------------------------------------------------------
# Top-level plot dispatcher
# ---------------------------------------------------------------------------

def make_plots(records: list, algo_order: list) -> None:
    agg = aggregate(records)
    present = [a for a in algo_order if a in agg]

    SCALING_OUT_DIR.mkdir(parents=True, exist_ok=True)

    _plot_metric_vs_x(
        agg, present,
        x_key="budget", y_key="norm_maes",
        fname="norm_mae_vs_budget.png",
        xlabel="Log-density-equivalent budget (LDE)",
        ylabel="Normalised parameter MAE\n(|posterior mean − truth| / prior σ)",
        title="Parameter recovery vs. compute budget",
    )
    _plot_metric_vs_x(
        agg, present,
        x_key="wall_times", y_key="norm_maes",
        fname="norm_mae_vs_wall_time.png",
        xlabel="Wall-clock time (s)",
        ylabel="Normalised parameter MAE",
        title="Parameter recovery vs. wall-clock time",
    )
    _plot_metric_vs_x(
        agg, present,
        x_key="budget", y_key="ess",
        fname="ess_vs_budget.png",
        xlabel="Log-density-equivalent budget (LDE)",
        ylabel="Total bulk ESS (ArviZ)",
        title="Effective sample size vs. compute budget",
        logy=True,
    )
    _plot_budget_utilisation(agg, present)
    _plot_per_param_recovery(records, present)

    print(f"\nSaved all plots to {SCALING_OUT_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot scaling study results from per-algorithm JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        default=ALL_ALGORITHMS,
        choices=ALL_ALGORITHMS,
        metavar="ALGO",
        help=f"Subset of algorithms to include (default: all). Choices: {ALL_ALGORITHMS}",
    )
    args = parser.parse_args()

    print(f"Loading results from {SCALING_OUT_DIR}/")
    records = load_records(args.algorithms)

    if not records:
        print("No records found. Run scaling_run_sampler.py first.")
        sys.exit(1)

    algo_order = [a for a in ALL_ALGORITHMS if a in args.algorithms]
    print_summary_table(records, algo_order)
    make_plots(records, algo_order)


if __name__ == "__main__":
    main()
